//! SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 and VEX-128 (AVX) instruction lifting.
//!
//! # XMM registers
//! `xmm0..xmm15` live in the guest state ([`STATE_XMM_OFFSET`](super::super::STATE_XMM_OFFSET),
//! 16 bytes each, little endian). Inside a function every XMM register the function touches gets
//! an `alloca <2 x i64>` (the canonical in-memory type of an XMM register) that is loaded from the
//! state at entry and after every call and stored back before every call, exit and trap, exactly
//! like the GPRs. Instructions `bitcast` the 128-bit value to the lane type they operate on
//! (`<4 x float>`, `<16 x i8>`, ...); LLVM promotes the allocas to SSA values and the bitcasts are
//! free on wasm (`v128`). The import thunks (`crate::stubs`) therefore always find float arguments
//! in the low lanes of the `xmm0..3` slots and leave float results in `xmm0`.
//!
//! # YMM, AVX and VEX
//! The guest state has 128-bit slots only; the upper halves of `ymm0..15` are not modelled.
//! VEX-128 encodings are lifted (non-destructive three-operand forms; they zero bits 128+ of the
//! destination YMM register, which is exact because those bits are never observable here).
//! Legacy-SSE encodings preserve bits 128+, which is equally exact. Every 256-bit (VEX.L=1) form
//! traps as unsupported, so the unmodelled upper halves can never be read or written; for the
//! same reason `vzeroupper` is a no-op and `vzeroall` zeroes `xmm0..15`. FMA, AVX-512 (EVEX),
//! XOP, MMX and x87 instructions trap as unsupported.
//!
//! # Supported instructions
//! Legacy-SSE and VEX-128 forms of:
//! - moves: `movaps/movups/movapd/movupd/movdqa/movdqu/lddqu`, `movnt{ps,pd,dq,dqa,i}`,
//!   `movss/movsd`, `movd/movq`, `mov{l,h}{ps,pd}`, `movlhps/movhlps`, `movmskps/movmskpd/pmovmskb`,
//!   `movsldup/movshdup/movddup`;
//! - float arithmetic: `add/sub/mul/div/min/max/sqrt` `ss/sd/ps/pd`, `rcp/rsqrt` `ss/ps`,
//!   `and/andn/or/xor` `ps/pd`, `hadd/hsub/addsub` `ps/pd`, `round{ss,sd,ps,pd}`, `dpps/dppd`;
//! - compares: `cmp{ss,sd,ps,pd}` (8 legacy / 32 VEX predicates), `(u)comis{s,d}`;
//! - conversions: `cvtsi2ss/sd`, `cvtss2sd/cvtsd2ss`, `cvt(t)ss2si/cvt(t)sd2si`,
//!   `cvtdq2ps/cvt(t)ps2dq/cvtdq2pd/cvt(t)pd2dq/cvtps2pd/cvtpd2ps`;
//! - shuffles: `shufps/shufpd`, `unpck{l,h}{ps,pd}`, `blend{ps,pd}`, `blendv{ps,pd}`,
//!   `insertps/extractps`;
//! - `ldmxcsr/stmxcsr`, `vzeroupper/vzeroall`.
//!
//! Everything else that touches vector registers traps with a reason (integer SIMD, x87, MMX,
//! FMA, AVX-512, string compares, AES/CLMUL, 256-bit forms, ...).
//!
//! # Floating point semantics
//! Arithmetic, square root, conversions and rounding are IEEE-754 binary32/binary64 with
//! round-to-nearest-even, which is what x86 computes under the default MXCSR (`0x1F80`), so the
//! numerical results are bit-exact. Deviations:
//! - **NaN payloads**: when an arithmetic result is NaN, its payload and sign are not guaranteed
//!   (x86 returns the first NaN operand quieted, or the default `-QNaN`; LLVM may commute operands
//!   and wasm leaves NaN bits nondeterministic). Whether the result is NaN is always exact. Data
//!   movement, shuffles, blends, bitwise logic and `min`/`max` (compare + select) are bit-exact
//!   for NaNs too.
//! - **`min`/`max`** follow x86, not IEEE `minNum`: `minps a, b` is `a < b ? a : b` (the second
//!   operand when either is NaN or both are zero), `maxps a, b` is `a > b ? a : b`.
//! - **float -> int** conversions return the x86 "integer indefinite" value (`0x80000000` /
//!   `0x8000000000000000`) for NaN and out-of-range inputs, exactly like hardware; the rounding
//!   forms (`cvtss2si`, `cvtps2dq`, ...) round half to even, the `cvtt*` forms truncate.
//! - **`rcpss/rcpps/rsqrtss/rsqrtps`**: hardware returns an implementation-specific approximation
//!   (relative error <= 1.5 * 2^-12). The lifter returns the correctly rounded `1/x` and
//!   `1/sqrt(x)` (within the same bound of every conforming CPU) and reproduces the hardware's
//!   special cases: denormal inputs count as zero (result `+-inf`) and `rcp` of `|x| >= 2^126`
//!   (a result that is not a normal number) flushes to `+-0`.
//! - **MXCSR**: the guest state has no MXCSR slot, so the lifted CPU always runs with the default
//!   `0x1F80` (round to nearest, all exceptions masked, FTZ/DAZ off). `stmxcsr` stores `0x1F80`.
//!   `ldmxcsr` accepts any value with the default rounding mode and no reserved bits, but ignores
//!   FTZ/DAZ and the exception masks: denormals are always computed per IEEE (games that enable
//!   FTZ/DAZ see slightly different results on denormal operands only), exception flags are never
//!   raised and unmasked exceptions never fault. A value selecting another rounding mode (or
//!   setting reserved bits, which raises `#GP` on hardware) traps at run time through
//!   `__playgta6_unimplemented` instead of silently computing with the wrong rounding.
//! - **Alignment**: the aligned forms (`movaps`, `movdqa`, legacy-SSE memory operands, ...) do not
//!   raise `#GP` on misaligned addresses; all accesses are `align 1`.
//! - Non-temporal hints (`movnt*`) are plain loads/stores.

mod float;
mod int;
mod moves;

use std::fmt::Write as _;

use iced_x86::{EncodingKind, Instruction, Mnemonic, OpKind, Register};

use super::super::{GUEST_STATE_TYPE, HELPER_UNIMPLEMENTED, Support};
use super::{FunctionLifter, R, imm, unsupported};

/// Canonical type of an XMM register value (allocas and guest state slots).
pub(super) const XMM_TYPE: &str = "<2 x i64>";

/// Default MXCSR value (round to nearest, all exceptions masked, FTZ/DAZ off).
const MXCSR_DEFAULT: u64 = 0x1F80;

/// Lane element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) enum Elem {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Elem {
    fn bits(self) -> u32 {
        match self {
            Elem::I8 => 8,
            Elem::I16 => 16,
            Elem::I32 | Elem::F32 => 32,
            Elem::I64 | Elem::F64 => 64,
        }
    }

    /// Lanes in a 128-bit register.
    fn lanes(self) -> u32 {
        128 / self.bits()
    }

    fn scalar(self) -> &'static str {
        match self {
            Elem::I8 => "i8",
            Elem::I16 => "i16",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
            Elem::F32 => "float",
            Elem::F64 => "double",
        }
    }

    /// The integer type of the same width.
    fn int(self) -> Elem {
        match self {
            Elem::F32 => Elem::I32,
            Elem::F64 => Elem::I64,
            e => e,
        }
    }

    fn of_bits(bits: u32) -> Elem {
        match bits {
            8 => Elem::I8,
            16 => Elem::I16,
            32 => Elem::I32,
            _ => Elem::I64,
        }
    }
}

/// An LLVM value type: a scalar (`n == 1`) or a vector of `n` lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) struct Ty {
    e: Elem,
    n: u32,
}

impl Ty {
    /// A full 128-bit vector of `e`.
    fn v(e: Elem) -> Ty {
        Ty { e, n: e.lanes() }
    }

    fn s(e: Elem) -> Ty {
        Ty { e, n: 1 }
    }

    fn vn(e: Elem, n: u32) -> Ty {
        Ty { e, n }
    }

    fn ir(self) -> String {
        if self.n == 1 {
            self.e.scalar().to_string()
        } else {
            format!("<{} x {}>", self.n, self.e.scalar())
        }
    }

    /// Same shape with `i1` lanes (comparison results).
    fn bool_ir(self) -> String {
        if self.n == 1 {
            "i1".to_string()
        } else {
            format!("<{} x i1>", self.n)
        }
    }

    /// Same shape with integer lanes of the same width.
    fn int(self) -> Ty {
        Ty {
            e: self.e.int(),
            n: self.n,
        }
    }

    fn with(self, e: Elem) -> Ty {
        Ty { e, n: self.n }
    }

    /// `value` in every lane (`value` is a literal valid for the element type).
    fn splat(self, value: &str) -> String {
        if self.n == 1 {
            return value.to_string();
        }
        let lane = format!("{} {value}", self.e.scalar());
        let lanes: Vec<&str> = (0..self.n).map(|_| lane.as_str()).collect();
        format!("<{}>", lanes.join(", "))
    }

    /// LLVM intrinsic type suffix (`f32`, `v4f32`, `v16i8`, ...).
    fn suffix(self) -> String {
        let e = match self.e {
            Elem::F32 => "f32",
            Elem::F64 => "f64",
            Elem::I8 => "i8",
            Elem::I16 => "i16",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
        };
        if self.n == 1 {
            e.to_string()
        } else {
            format!("v{}{e}", self.n)
        }
    }
}

/// A floating point literal (`v` must be exactly representable in the lane type).
fn fconst(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

/// `<k x i32> <i32 a, i32 b, ...>` shuffle mask.
fn mask(indices: &[u32]) -> String {
    let lanes: Vec<String> = indices.iter().map(|i| format!("i32 {i}")).collect();
    format!("<{} x i32> <{}>", indices.len(), lanes.join(", "))
}

/// LLVM intrinsics (and wasm builtins) the SIMD lifter calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) enum Intr {
    Sqrt,
    Rint,
    Floor,
    Ceil,
    Trunc,
}

/// One module-level declaration required by lifted SIMD code (see `Support::Simd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) struct Decl {
    intr: Intr,
    ty: Ty,
}

impl Decl {
    fn name(self) -> String {
        let base = match self.intr {
            Intr::Sqrt => "sqrt",
            Intr::Rint => "rint",
            Intr::Floor => "floor",
            Intr::Ceil => "ceil",
            Intr::Trunc => "trunc",
        };
        format!("llvm.{base}.{}", self.ty.suffix())
    }

    fn params(self) -> String {
        self.ty.ir()
    }

    /// The `declare` line.
    pub(in crate::lift) fn declaration(self) -> String {
        format!(
            "declare {} @{}({})\n",
            self.ty.ir(),
            self.name(),
            self.params()
        )
    }
}

/// Operand positions of an SSE instruction: legacy encodings are destructive
/// (`dst = op0 = a`, `b = op1`), VEX encodings take three operands (`dst = op0, a = op1,
/// b = op2`).
#[derive(Debug, Clone, Copy)]
struct Form {
    dst: u32,
    a: u32,
    b: u32,
}

impl Form {
    fn of(vex: bool) -> Form {
        if vex {
            Form { dst: 0, a: 1, b: 2 }
        } else {
            Form { dst: 0, a: 0, b: 1 }
        }
    }
}

/// Whether the instruction is handled by the SIMD lifter: it has a vector (or MMX) register
/// operand, or it is one of the few SSE/AVX instructions without one.
pub(super) fn is_simd(ins: &Instruction) -> bool {
    use Mnemonic as M;
    let vector_operand = (0..ins.op_count()).any(|i| {
        ins.op_kind(i) == OpKind::Register && {
            let r = ins.op_register(i);
            r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_mm()
        }
    });
    vector_operand
        || matches!(
            ins.mnemonic(),
            M::Ldmxcsr
                | M::Vldmxcsr
                | M::Stmxcsr
                | M::Vstmxcsr
                | M::Vzeroupper
                | M::Vzeroall
                | M::Movnti
                | M::Emms
                // `r32/r64, m32/m64` forms have no vector operand.
                | M::Cvtss2si
                | M::Vcvtss2si
                | M::Cvttss2si
                | M::Vcvttss2si
                | M::Cvtsd2si
                | M::Vcvtsd2si
                | M::Cvttsd2si
                | M::Vcvttsd2si
        )
}

/// Records the XMM registers an instruction uses (`reg` from its used-register info).
pub(super) fn note_register(reg: Register, xmms: &mut [bool; 16]) {
    if (reg.is_xmm() || reg.is_ymm() || reg.is_zmm()) && reg.number() < 16 {
        xmms[reg.number()] = true;
    }
}

/// Records XMM registers an instruction writes without listing them as operands.
pub(super) fn note_instruction(ins: &Instruction, xmms: &mut [bool; 16]) {
    if ins.mnemonic() == Mnemonic::Vzeroall {
        *xmms = [true; 16];
    }
}

impl FunctionLifter<'_> {
    // ----- state ---------------------------------------------------------------------------------

    pub(super) fn xmm_allocas(&self, ir: &mut String) {
        for (idx, used) in self.xmms.iter().enumerate() {
            if *used {
                let _ = writeln!(ir, "  %xmm{idx} = alloca {XMM_TYPE}");
            }
        }
    }

    fn state_xmm_ptr(&mut self, idx: usize) -> String {
        self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 5, i64 {idx}"
        ))
    }

    /// Loads the tracked XMM registers from `%state` into their allocas.
    pub(super) fn sync_xmm_in(&mut self) {
        for idx in 0..16 {
            if self.xmms[idx] {
                let ptr = self.state_xmm_ptr(idx);
                let v = self.val(&format!("load {XMM_TYPE}, ptr {ptr}, align 8"));
                self.emit(&format!("store {XMM_TYPE} {v}, ptr %xmm{idx}"));
            }
        }
    }

    /// Stores the tracked XMM registers back into `%state`.
    pub(super) fn sync_xmm_out(&mut self) {
        for idx in 0..16 {
            if self.xmms[idx] {
                let v = self.val(&format!("load {XMM_TYPE}, ptr %xmm{idx}"));
                let ptr = self.state_xmm_ptr(idx);
                self.emit(&format!("store {XMM_TYPE} {v}, ptr {ptr}, align 8"));
            }
        }
    }

    // ----- registers and operands ----------------------------------------------------------------

    fn xmm_index(&self, reg: Register) -> R<usize> {
        if !reg.is_xmm() {
            return unsupported(format!("operand {reg:?} is not an XMM register"));
        }
        let idx = reg.number();
        if idx >= 16 {
            return unsupported(format!(
                "register {reg:?} (only xmm0..xmm15 exist without EVEX)"
            ));
        }
        if !self.xmms[idx] {
            return unsupported(format!("internal: xmm{idx} is not tracked"));
        }
        Ok(idx)
    }

    fn cast(&mut self, v: &str, from: Ty, to: Ty) -> String {
        if from == to {
            v.to_string()
        } else {
            self.val(&format!("bitcast {} {v} to {}", from.ir(), to.ir()))
        }
    }

    /// An XMM register as a 128-bit vector of `e`.
    fn xget(&mut self, reg: Register, e: Elem) -> R<String> {
        let idx = self.xmm_index(reg)?;
        let raw = self.val(&format!("load {XMM_TYPE}, ptr %xmm{idx}"));
        Ok(self.cast(&raw, Ty::v(Elem::I64), Ty::v(e)))
    }

    /// Writes a 128-bit vector of `e` to an XMM register.
    fn xset(&mut self, reg: Register, v: &str, e: Elem) -> R<()> {
        let idx = self.xmm_index(reg)?;
        let raw = self.cast(v, Ty::v(e), Ty::v(Elem::I64));
        self.emit(&format!("store {XMM_TYPE} {raw}, ptr %xmm{idx}"));
        Ok(())
    }

    fn mem_bytes(ins: &Instruction) -> usize {
        ins.memory_size().size()
    }

    fn mem_ptr(&mut self, ins: &Instruction) -> R<String> {
        let ea = self.mem_ea(ins)?;
        Ok(self.guest_ptr(&ea))
    }

    /// Loads a value of type `ty` from the memory operand, which must be exactly that large.
    fn mem_load(&mut self, ins: &Instruction, ty: Ty) -> R<String> {
        let bytes = (ty.e.bits() * ty.n / 8) as usize;
        if Self::mem_bytes(ins) != bytes {
            return unsupported(format!(
                "internal: {}-byte memory operand where {bytes} bytes were expected",
                Self::mem_bytes(ins)
            ));
        }
        let ptr = self.mem_ptr(ins)?;
        Ok(self.val(&format!("load {}, ptr {ptr}, align 1", ty.ir())))
    }

    fn mem_store(&mut self, ins: &Instruction, v: &str, ty: Ty) -> R<()> {
        let bytes = (ty.e.bits() * ty.n / 8) as usize;
        if Self::mem_bytes(ins) != bytes {
            return unsupported(format!(
                "internal: {}-byte memory operand where {bytes} bytes were expected",
                Self::mem_bytes(ins)
            ));
        }
        let ptr = self.mem_ptr(ins)?;
        self.emit(&format!("store {} {v}, ptr {ptr}, align 1", ty.ir()));
        Ok(())
    }

    /// A 128-bit operand (XMM register or `m128`) as a vector of `e`.
    fn v128(&mut self, ins: &Instruction, i: u32, e: Elem) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => self.xget(ins.op_register(i), e),
            OpKind::Memory => self.mem_load(ins, Ty::v(e)),
            kind => unsupported(format!("vector operand kind {kind:?}")),
        }
    }

    /// Writes a 128-bit result to an XMM register or `m128` operand.
    fn set_v128(&mut self, ins: &Instruction, i: u32, v: &str, e: Elem) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Register => self.xset(ins.op_register(i), v, e),
            OpKind::Memory => self.mem_store(ins, v, Ty::v(e)),
            kind => unsupported(format!("vector destination kind {kind:?}")),
        }
    }

    /// The low `n` lanes of an XMM register, or a memory operand of that size, as `<n x e>`
    /// (a scalar when `n == 1`).
    fn low(&mut self, ins: &Instruction, i: u32, e: Elem, n: u32) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let v = self.xget(ins.op_register(i), e)?;
                Ok(self.take_low(&v, e, n))
            }
            OpKind::Memory => self.mem_load(ins, Ty::vn(e, n)),
            kind => unsupported(format!("vector operand kind {kind:?}")),
        }
    }

    /// The low `n` lanes of a full vector of `e` (a scalar when `n == 1`).
    fn take_low(&mut self, v: &str, e: Elem, n: u32) -> String {
        let t = Ty::v(e).ir();
        if n == 1 {
            self.val(&format!("extractelement {t} {v}, i32 0"))
        } else if n == e.lanes() {
            v.to_string()
        } else {
            let idx: Vec<u32> = (0..n).collect();
            self.val(&format!(
                "shufflevector {t} {v}, {t} poison, {}",
                mask(&idx)
            ))
        }
    }

    /// Widens `<n x e>` (or a scalar) to a full vector, zeroing the upper lanes.
    fn zext_lanes(&mut self, v: &str, e: Elem, n: u32) -> String {
        let full = Ty::v(e);
        if n == 1 {
            return self.val(&format!(
                "insertelement {} zeroinitializer, {} {v}, i32 0",
                full.ir(),
                e.scalar()
            ));
        }
        if n == e.lanes() {
            return v.to_string();
        }
        let t = Ty::vn(e, n).ir();
        let idx: Vec<u32> = (0..e.lanes()).map(|i| i.min(n)).collect();
        // Lane `n` of the second operand is zero.
        self.val(&format!(
            "shufflevector {t} {v}, {t} zeroinitializer, {}",
            mask(&idx)
        ))
    }

    /// Replaces lane `lane` of a full vector of `e`.
    fn insert(&mut self, vec: &str, e: Elem, value: &str, lane: u32) -> String {
        self.val(&format!(
            "insertelement {} {vec}, {} {value}, i32 {lane}",
            Ty::v(e).ir(),
            e.scalar()
        ))
    }

    fn extract(&mut self, vec: &str, e: Elem, lane: u32) -> String {
        self.val(&format!(
            "extractelement {} {vec}, i32 {lane}",
            Ty::v(e).ir()
        ))
    }

    /// `shufflevector` of two full vectors of `e`.
    fn shuffle(&mut self, a: &str, b: &str, e: Elem, idx: &[u32]) -> String {
        let t = Ty::v(e).ir();
        self.val(&format!("shufflevector {t} {a}, {t} {b}, {}", mask(idx)))
    }

    /// The instruction's last operand as an 8-bit immediate.
    fn imm8(ins: &Instruction) -> R<u32> {
        let last = ins.op_count().saturating_sub(1);
        if !super::is_immediate(ins.op_kind(last)) {
            return unsupported("internal: missing imm8 operand");
        }
        match ins.try_immediate(last) {
            Ok(v) => Ok((v & 0xFF) as u32),
            Err(_) => unsupported("unreadable immediate"),
        }
    }

    /// Calls an intrinsic, declaring it at module level.
    fn intr(&mut self, intr: Intr, ty: Ty, args: &[&str]) -> String {
        let decl = Decl { intr, ty };
        self.support.insert(Support::Simd(decl));
        let t = ty.ir();
        let list: Vec<String> = args.iter().map(|a| format!("{t} {a}")).collect();
        self.val(&format!("call {t} @{}({})", decl.name(), list.join(", ")))
    }

    // ----- dispatch ------------------------------------------------------------------------------

    /// Lifts an instruction for which [`is_simd`] holds.
    pub(super) fn lift_simd(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        match ins.mnemonic() {
            M::Ldmxcsr | M::Vldmxcsr => return self.ldmxcsr(ins),
            M::Stmxcsr | M::Vstmxcsr => {
                let v = imm(MXCSR_DEFAULT, 32);
                return self.mem_store(ins, &v, Ty::s(Elem::I32));
            }
            // The upper YMM halves are not modelled (every 256-bit form traps), so clearing
            // them is unobservable.
            M::Vzeroupper => return Ok(()),
            M::Vzeroall => {
                for idx in 0..16 {
                    self.xset(Register::XMM0 + idx, "zeroinitializer", Elem::I64)?;
                }
                return Ok(());
            }
            M::Movnti => {
                let bits = self.op_bits(ins, 1)?;
                let v = self.read_op(ins, 1, bits)?;
                return self.write_op(ins, 0, &v, bits);
            }
            M::Emms => return unsupported("MMX instructions are not supported"),
            _ => {}
        }
        let vex = self.check_encoding(ins)?;
        if self.lift_moves(ins, vex)? || self.lift_float(ins, vex)? || self.lift_int(ins, vex)? {
            return Ok(());
        }
        unsupported(format!(
            "instruction {:?} is not supported yet",
            ins.mnemonic()
        ))
    }

    /// Rejects encodings and operands outside the modelled 128-bit SSE/VEX state; returns whether
    /// the instruction is VEX-encoded.
    fn check_encoding(&self, ins: &Instruction) -> R<bool> {
        let vex = match ins.encoding() {
            EncodingKind::Legacy => false,
            EncodingKind::VEX => true,
            EncodingKind::EVEX => {
                return unsupported("AVX-512 (EVEX) instructions are not supported");
            }
            other => {
                return unsupported(format!("{other:?}-encoded instructions are not supported"));
            }
        };
        for i in 0..ins.op_count() {
            match ins.op_kind(i) {
                OpKind::Register => {
                    let r = ins.op_register(i);
                    if r.is_ymm() {
                        return unsupported(
                            "256-bit AVX form (the upper YMM halves are not modelled in the \
                             guest state)",
                        );
                    }
                    if r.is_mm() {
                        return unsupported("MMX instructions are not supported");
                    }
                    if r.is_zmm() || (r.is_xmm() && r.number() >= 16) {
                        return unsupported(format!("register {r:?} requires EVEX"));
                    }
                }
                OpKind::Memory => {
                    if ins.is_vsib() {
                        return unsupported("gather (VSIB) instructions are not supported");
                    }
                    if Self::mem_bytes(ins) > 16 {
                        return unsupported(
                            "256-bit AVX form (the upper YMM halves are not modelled in the \
                             guest state)",
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(vex)
    }

    /// `ldmxcsr m32`: only the default rounding mode is supported (see the module docs).
    fn ldmxcsr(&mut self, ins: &Instruction) -> R<()> {
        let v = self.mem_load(ins, Ty::s(Elem::I32))?;
        // Rounding control (bits 13-14) and the reserved bits 16-31 must be clear.
        let bad = self.val(&format!("and i32 {v}, {}", imm(0xFFFF_6000, 32)));
        let fault = self.val(&format!("icmp ne i32 {bad}, 0"));
        let trap = self.new_label("mxcsr");
        let ok = self.new_label("cont");
        self.emit(&format!("br i1 {fault}, label %{trap}, label %{ok}"));
        self.start_block(&trap);
        self.sync_out();
        self.emit(&format!(
            "call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 {})",
            imm(ins.ip(), 64)
        ));
        self.emit("unreachable");
        self.start_block(&ok);
        Ok(())
    }
}
