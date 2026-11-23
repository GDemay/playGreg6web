//! Data movement: full/partial register moves, GPR <-> XMM moves, sign masks and the
//! floating-point shuffles, unpacks, blends and inserts. Moves use integer lane types so that no
//! value is ever interpreted as a float (NaN bits are preserved exactly).

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::super::{FunctionLifter, R, unsupported};
use super::{Elem, Form, Ty};

impl FunctionLifter<'_> {
    pub(super) fn lift_moves(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{I32, I64};
        use Mnemonic as M;
        let f = Form::of(vex);
        match ins.mnemonic() {
            M::Movaps
            | M::Vmovaps
            | M::Movups
            | M::Vmovups
            | M::Movapd
            | M::Vmovapd
            | M::Movupd
            | M::Vmovupd
            | M::Movdqa
            | M::Vmovdqa
            | M::Movdqu
            | M::Vmovdqu
            | M::Lddqu
            | M::Vlddqu
            | M::Movntps
            | M::Vmovntps
            | M::Movntpd
            | M::Vmovntpd
            | M::Movntdq
            | M::Vmovntdq
            | M::Movntdqa
            | M::Vmovntdqa => {
                let v = self.v128(ins, 1, I64)?;
                self.set_v128(ins, 0, &v, I64)?;
            }
            M::Movss | M::Vmovss => self.move_scalar(ins, I32)?,
            M::Movsd | M::Vmovsd => self.move_scalar(ins, I64)?,
            M::Movd | M::Vmovd => self.move_gpr(ins, 32)?,
            M::Movq | M::Vmovq => self.move_gpr(ins, 64)?,
            M::Movlps | M::Vmovlps | M::Movlpd | M::Vmovlpd => self.move_half(ins, 0)?,
            M::Movhps | M::Vmovhps | M::Movhpd | M::Vmovhpd => self.move_half(ins, 1)?,
            M::Movlhps | M::Vmovlhps | M::Movhlps | M::Vmovhlps => {
                let a = self.xget(ins.op_register(f.a), I64)?;
                let b = self.v128(ins, f.b, I64)?;
                let idx = if matches!(ins.mnemonic(), M::Movlhps | M::Vmovlhps) {
                    [0, 2]
                } else {
                    [3, 1]
                };
                let r = self.shuffle(&a, &b, I64, &idx);
                self.xset(ins.op_register(f.dst), &r, I64)?;
            }
            M::Movmskps | M::Vmovmskps => self.move_mask(ins, I32)?,
            M::Movmskpd | M::Vmovmskpd => self.move_mask(ins, I64)?,
            M::Pmovmskb | M::Vpmovmskb => self.move_mask(ins, Elem::I8)?,
            M::Movsldup | M::Vmovsldup | M::Movshdup | M::Vmovshdup => {
                let v = self.v128(ins, 1, I32)?;
                let idx = if matches!(ins.mnemonic(), M::Movsldup | M::Vmovsldup) {
                    [0, 0, 2, 2]
                } else {
                    [1, 1, 3, 3]
                };
                let r = self.shuffle(&v, &v, I32, &idx);
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Movddup | M::Vmovddup => {
                // xmm, xmm/m64 ; ymm, ymm/m256 (even quadword of each lane)
                let v = if ins.op_kind(1) == OpKind::Memory && Self::mem_bytes(ins) == 8 {
                    let s = self.low(ins, 1, I64, 1)?;
                    self.zext_lanes(&s, I64, 1)
                } else {
                    self.v128(ins, 1, I64)?
                };
                let r = self.shuffle(&v, &v, I64, &[0, 0]);
                self.xset(ins.op_register(0), &r, I64)?;
            }
            M::Vpermilps | M::Vpermilpd => {
                let e = if ins.mnemonic() == M::Vpermilps {
                    I32
                } else {
                    I64
                };
                let n = e.lanes();
                let v = self.v128(ins, 1, e)?;
                let r = if super::super::is_immediate(ins.op_kind(2)) {
                    let i = Self::imm8(ins)?;
                    let idx: Vec<u32> = if e == I32 {
                        (0..4).map(|l| (i >> (2 * l)) & 3).collect()
                    } else {
                        let i = self.lane_imm(i, 2);
                        (0..2).map(|l| (i >> l) & 1).collect()
                    };
                    self.shuffle(&v, &v, e, &idx)
                } else {
                    // Variable control: bits 1:0 (ps) or bit 1 (pd) of each control element.
                    let ctl = self.v128(ins, 2, e)?;
                    let t = Ty::v(e).ir();
                    let mut r = "poison".to_string();
                    for l in 0..n {
                        let k = self.extract(&ctl, e, l);
                        let k = if e == I32 {
                            self.val(&format!("and i32 {k}, 3"))
                        } else {
                            let k = self.val(&format!("lshr i64 {k}, 1"));
                            self.val(&format!("and i64 {k}, 1"))
                        };
                        let x = self.val(&format!("extractelement {t} {v}, {} {k}", e.scalar()));
                        r = self.insert(&r, e, &x, l);
                    }
                    r
                };
                self.xset(ins.op_register(0), &r, e)?;
            }
            M::Shufps | M::Vshufps => {
                let i = Self::imm8(ins)?;
                let idx = [i & 3, (i >> 2) & 3, 4 + ((i >> 4) & 3), 4 + ((i >> 6) & 3)];
                self.shuffle_op(ins, f, I32, &idx)?;
            }
            M::Shufpd | M::Vshufpd => {
                let i = Self::imm8(ins)?;
                let i = self.lane_imm(i, 2);
                self.shuffle_op(ins, f, I64, &[i & 1, 2 + ((i >> 1) & 1)])?;
            }
            M::Unpcklps | M::Vunpcklps => self.shuffle_op(ins, f, I32, &[0, 4, 1, 5])?,
            M::Unpckhps | M::Vunpckhps => self.shuffle_op(ins, f, I32, &[2, 6, 3, 7])?,
            M::Unpcklpd | M::Vunpcklpd => self.shuffle_op(ins, f, I64, &[0, 2])?,
            M::Unpckhpd | M::Vunpckhpd => self.shuffle_op(ins, f, I64, &[1, 3])?,
            M::Blendps | M::Vblendps => self.blend_imm(ins, f, I32)?,
            M::Blendpd | M::Vblendpd => self.blend_imm(ins, f, I64)?,
            M::Blendvps | M::Vblendvps => self.blend_var(ins, f, vex, I32)?,
            M::Blendvpd | M::Vblendvpd => self.blend_var(ins, f, vex, I64)?,
            M::Insertps | M::Vinsertps => {
                let i = Self::imm8(ins)?;
                let a = self.xget(ins.op_register(f.a), I32)?;
                let s = match ins.op_kind(f.b) {
                    OpKind::Register => {
                        let b = self.xget(ins.op_register(f.b), I32)?;
                        self.extract(&b, I32, (i >> 6) & 3)
                    }
                    _ => self.mem_load(ins, Ty::s(I32))?,
                };
                let mut r = self.insert(&a, I32, &s, (i >> 4) & 3);
                if i & 0xF != 0 {
                    let idx: Vec<u32> = (0..4)
                        .map(|l| if i >> l & 1 == 1 { 4 } else { l })
                        .collect();
                    r = self.shuffle(&r, "zeroinitializer", I32, &idx);
                }
                self.xset(ins.op_register(f.dst), &r, I32)?;
            }
            M::Extractps | M::Vextractps => {
                let i = Self::imm8(ins)?;
                let v = self.xget(ins.op_register(1), I32)?;
                let s = self.extract(&v, I32, i & 3);
                self.write_gpr_or_mem(ins, 0, &s, 32)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `dst = shufflevector(a, b, idx)` with the operand layout `f`.
    pub(super) fn shuffle_op(&mut self, ins: &Instruction, f: Form, e: Elem, idx: &[u32]) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let r = self.shuffle(&a, &b, e, idx);
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// Writes an `bits`-bit lane value to a GPR (zero-extended to the register width) or memory.
    pub(super) fn write_gpr_or_mem(
        &mut self,
        ins: &Instruction,
        i: u32,
        v: &str,
        bits: u32,
    ) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Memory => self.mem_store(ins, v, Ty::s(Elem::of_bits(bits))),
            OpKind::Register => {
                let reg_bits = self.op_bits(ins, i)?;
                let v = if reg_bits > bits {
                    self.val(&format!("zext i{bits} {v} to i{reg_bits}"))
                } else if reg_bits == bits {
                    v.to_string()
                } else {
                    return unsupported("internal: destination register too narrow");
                };
                self.write_op(ins, i, &v, reg_bits)
            }
            kind => unsupported(format!("destination operand kind {kind:?}")),
        }
    }

    /// `movss`/`movsd` (lane type `I32`/`I64`): register merges keep the upper lanes of the
    /// first source, loads zero them, stores write the low lane only.
    fn move_scalar(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        if ins.op_count() == 3 {
            // vmovss xmm1, xmm2, xmm3
            let a = self.xget(ins.op_register(1), e)?;
            let b = self.xget(ins.op_register(2), e)?;
            let b0 = self.extract(&b, e, 0);
            let r = self.insert(&a, e, &b0, 0);
            return self.xset(ins.op_register(0), &r, e);
        }
        match (ins.op_kind(0), ins.op_kind(1)) {
            (OpKind::Memory, OpKind::Register) => {
                let v = self.xget(ins.op_register(1), e)?;
                let s = self.extract(&v, e, 0);
                self.mem_store(ins, &s, Ty::s(e))
            }
            (OpKind::Register, OpKind::Memory) => {
                let s = self.mem_load(ins, Ty::s(e))?;
                let r = self.zext_lanes(&s, e, 1);
                self.xset(ins.op_register(0), &r, e)
            }
            (OpKind::Register, OpKind::Register) => {
                let a = self.xget(ins.op_register(0), e)?;
                let b = self.xget(ins.op_register(1), e)?;
                let b0 = self.extract(&b, e, 0);
                let r = self.insert(&a, e, &b0, 0);
                self.xset(ins.op_register(0), &r, e)
            }
            _ => unsupported("movss/movsd operand form"),
        }
    }

    /// `movd`/`movq` between XMM and GPR/memory, and `movq xmm, xmm/m64`: writes to an XMM
    /// register zero the upper bits.
    fn move_gpr(&mut self, ins: &Instruction, bits: u32) -> R<()> {
        let dst_is_xmm = ins.op_kind(0) == OpKind::Register && ins.op_register(0).is_xmm();
        if dst_is_xmm {
            let v = if ins.op_kind(1) == OpKind::Register && ins.op_register(1).is_xmm() {
                let x = self.xget(ins.op_register(1), Elem::I64)?;
                self.extract(&x, Elem::I64, 0)
            } else {
                let v = self.read_op(ins, 1, bits)?;
                if bits == 32 {
                    self.val(&format!("zext i32 {v} to i64"))
                } else {
                    v
                }
            };
            let r = self.zext_lanes(&v, Elem::I64, 1);
            return self.xset(ins.op_register(0), &r, Elem::I64);
        }
        if !(ins.op_kind(1) == OpKind::Register && ins.op_register(1).is_xmm()) {
            return unsupported("movd/movq operand form");
        }
        let e = Elem::of_bits(bits);
        let x = self.xget(ins.op_register(1), e)?;
        let s = self.extract(&x, e, 0);
        match ins.op_kind(0) {
            OpKind::Memory => self.mem_store(ins, &s, Ty::s(e)),
            _ => self.write_op(ins, 0, &s, bits),
        }
    }

    /// `movlps/movhps/movlpd/movhpd`: 64-bit loads into (VEX: merges with the first source) and
    /// stores from quadword lane `lane`.
    fn move_half(&mut self, ins: &Instruction, lane: u32) -> R<()> {
        let e = Elem::I64;
        if ins.op_kind(0) == OpKind::Memory {
            let v = self.xget(ins.op_register(1), e)?;
            let s = self.extract(&v, e, lane);
            return self.mem_store(ins, &s, Ty::s(e));
        }
        let base = if ins.op_count() == 3 { 1 } else { 0 };
        let a = self.xget(ins.op_register(base), e)?;
        let s = self.mem_load(ins, Ty::s(e))?;
        let r = self.insert(&a, e, &s, lane);
        self.xset(ins.op_register(0), &r, e)
    }

    /// `movmskps/movmskpd/pmovmskb`: the sign bit of every lane, zero-extended into a GPR.
    fn move_mask(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let n = e.lanes();
        let t = Ty::v(e).ir();
        let v = self.xget(ins.op_register(1), e)?;
        let neg = self.val(&format!("icmp slt {t} {v}, zeroinitializer"));
        let bits = self.val(&format!("bitcast <{n} x i1> {neg} to i{n}"));
        let dst_bits = self.op_bits(ins, 0)?;
        let r = self.val(&format!("zext i{n} {bits} to i{dst_bits}"));
        self.write_op(ins, 0, &r, dst_bits)
    }

    /// `blendps/blendpd` (and `pblendw`): lane `i` comes from the second source when bit `i` of
    /// the immediate is set.
    pub(super) fn blend_imm(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)?;
        let n = e.lanes();
        // 256-bit forms take the upper lane's selectors from the next immediate bits.
        let i = self.lane_imm(i, n);
        let idx: Vec<u32> = (0..n)
            .map(|l| if i >> (l % 8) & 1 == 1 { n + l } else { l })
            .collect();
        self.shuffle_op(ins, f, e, &idx)
    }

    /// `blendvps/blendvpd/pblendvb`: lane from the second source when the mask lane's sign bit is
    /// set. The mask is the implicit `xmm0` (legacy) or the fourth operand (VEX).
    pub(super) fn blend_var(&mut self, ins: &Instruction, f: Form, vex: bool, e: Elem) -> R<()> {
        let mask_reg = if vex {
            ins.op_register(3)
        } else {
            Register::XMM0
        };
        let t = Ty::v(e).ir();
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let m = self.xget(mask_reg, e)?;
        let c = self.val(&format!("icmp slt {t} {m}, zeroinitializer"));
        let r = self.val(&format!(
            "select {} {c}, {t} {b}, {t} {a}",
            Ty::v(e).bool_ir()
        ));
        self.xset(ins.op_register(f.dst), &r, e)
    }
}
