//! Scalar and packed floating point: arithmetic, `min`/`max`, square root, reciprocal
//! approximations, bitwise logic, compares, conversions, rounding, horizontal ops and dot
//! products. See the parent module for the exactness guarantees and deviations.

use iced_x86::{Instruction, Mnemonic, OpKind};

use super::super::{Flag, FunctionLifter, R, imm, unsupported};
use super::{Elem, Form, Intr, Ty, fconst};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
    Rcp,
    Rsqrt,
}

/// `(operation, lane type, scalar form)` of the arithmetic instructions.
fn arith(m: Mnemonic) -> Option<(FOp, Elem, bool)> {
    use Elem::{F32, F64};
    use FOp::*;
    use Mnemonic as M;
    Some(match m {
        M::Addps | M::Vaddps => (Add, F32, false),
        M::Addpd | M::Vaddpd => (Add, F64, false),
        M::Addss | M::Vaddss => (Add, F32, true),
        M::Addsd | M::Vaddsd => (Add, F64, true),
        M::Subps | M::Vsubps => (Sub, F32, false),
        M::Subpd | M::Vsubpd => (Sub, F64, false),
        M::Subss | M::Vsubss => (Sub, F32, true),
        M::Subsd | M::Vsubsd => (Sub, F64, true),
        M::Mulps | M::Vmulps => (Mul, F32, false),
        M::Mulpd | M::Vmulpd => (Mul, F64, false),
        M::Mulss | M::Vmulss => (Mul, F32, true),
        M::Mulsd | M::Vmulsd => (Mul, F64, true),
        M::Divps | M::Vdivps => (Div, F32, false),
        M::Divpd | M::Vdivpd => (Div, F64, false),
        M::Divss | M::Vdivss => (Div, F32, true),
        M::Divsd | M::Vdivsd => (Div, F64, true),
        M::Minps | M::Vminps => (Min, F32, false),
        M::Minpd | M::Vminpd => (Min, F64, false),
        M::Minss | M::Vminss => (Min, F32, true),
        M::Minsd | M::Vminsd => (Min, F64, true),
        M::Maxps | M::Vmaxps => (Max, F32, false),
        M::Maxpd | M::Vmaxpd => (Max, F64, false),
        M::Maxss | M::Vmaxss => (Max, F32, true),
        M::Maxsd | M::Vmaxsd => (Max, F64, true),
        M::Sqrtps | M::Vsqrtps => (Sqrt, F32, false),
        M::Sqrtpd | M::Vsqrtpd => (Sqrt, F64, false),
        M::Sqrtss | M::Vsqrtss => (Sqrt, F32, true),
        M::Sqrtsd | M::Vsqrtsd => (Sqrt, F64, true),
        M::Rcpps | M::Vrcpps => (Rcp, F32, false),
        M::Rcpss | M::Vrcpss => (Rcp, F32, true),
        M::Rsqrtps | M::Vrsqrtps => (Rsqrt, F32, false),
        M::Rsqrtss | M::Vrsqrtss => (Rsqrt, F32, true),
        _ => return None,
    })
}

/// Operand order of an FMA3 instruction (`132`, `213`, `231`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmaOrder {
    O132,
    O213,
    O231,
}

/// Signs of an FMA3 instruction: `-(x*y)` and `-z` per lane parity, plus lane type and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fma {
    order: FmaOrder,
    /// Negate the product.
    neg_product: bool,
    /// Negate the addend in even / odd lanes (`fmaddsub`: even subtract, odd add).
    neg_addend: [bool; 2],
    e: Elem,
    scalar: bool,
}

fn fma(m: Mnemonic) -> Option<Fma> {
    use FmaOrder::*;
    let name = format!("{m:?}");
    let rest = name.strip_prefix("Vf")?;
    let (kind, rest) = ["maddsub", "msubadd", "nmadd", "nmsub", "madd", "msub"]
        .iter()
        .find_map(|k| rest.strip_prefix(k).map(|r| (*k, r)))?;
    let (order, suffix) = if let Some(r) = rest.strip_prefix("132") {
        (O132, r)
    } else if let Some(r) = rest.strip_prefix("213") {
        (O213, r)
    } else {
        (O231, rest.strip_prefix("231")?)
    };
    let (e, scalar) = match suffix {
        "ps" => (Elem::F32, false),
        "pd" => (Elem::F64, false),
        "ss" => (Elem::F32, true),
        "sd" => (Elem::F64, true),
        _ => return None,
    };
    let (neg_product, neg_addend) = match kind {
        "madd" => (false, [false, false]),
        "msub" => (false, [true, true]),
        "nmadd" => (true, [false, false]),
        "nmsub" => (true, [true, true]),
        "maddsub" => (false, [true, false]),
        _ => (false, [false, true]), // msubadd
    };
    Some(Fma {
        order,
        neg_product,
        neg_addend,
        e,
        scalar,
    })
}

/// LLVM `fcmp` predicate for an SSE/AVX comparison predicate (`imm8[3:0]`; bit 4 only selects
/// the signalling variant, which makes no difference without exception flags).
fn predicate(p: u32) -> &'static str {
    match p & 0xF {
        0 => "oeq",
        1 => "olt",
        2 => "ole",
        3 => "uno",
        4 => "une",
        5 => "uge",
        6 => "ugt",
        7 => "ord",
        8 => "ueq",
        9 => "ult",
        10 => "ule",
        11 => "false",
        12 => "one",
        13 => "oge",
        14 => "ogt",
        _ => "true",
    }
}

/// Whether [`FunctionLifter::lift_float`] handles `m` (mirroring its dispatch).
pub(super) fn is_float_supported(m: Mnemonic) -> bool {
    use Mnemonic as M;
    if arith(m).is_some() || fma(m).is_some() {
        return true;
    }
    matches!(
        m,
        M::Andps
            | M::Vandps
            | M::Andpd
            | M::Vandpd
            | M::Andnps
            | M::Vandnps
            | M::Andnpd
            | M::Vandnpd
            | M::Orps
            | M::Vorps
            | M::Orpd
            | M::Vorpd
            | M::Xorps
            | M::Vxorps
            | M::Xorpd
            | M::Vxorpd
            | M::Cmpps
            | M::Vcmpps
            | M::Cmppd
            | M::Vcmppd
            | M::Cmpss
            | M::Vcmpss
            | M::Cmpsd
            | M::Vcmpsd
            | M::Comiss
            | M::Vcomiss
            | M::Ucomiss
            | M::Vucomiss
            | M::Comisd
            | M::Vcomisd
            | M::Ucomisd
            | M::Vucomisd
            | M::Cvtsi2ss
            | M::Vcvtsi2ss
            | M::Cvtsi2sd
            | M::Vcvtsi2sd
            | M::Cvtss2sd
            | M::Vcvtss2sd
            | M::Cvtsd2ss
            | M::Vcvtsd2ss
            | M::Cvtss2si
            | M::Vcvtss2si
            | M::Cvttss2si
            | M::Vcvttss2si
            | M::Cvtsd2si
            | M::Vcvtsd2si
            | M::Cvttsd2si
            | M::Vcvttsd2si
            | M::Cvtdq2ps
            | M::Vcvtdq2ps
            | M::Cvtps2dq
            | M::Vcvtps2dq
            | M::Cvttps2dq
            | M::Vcvttps2dq
            | M::Cvtdq2pd
            | M::Vcvtdq2pd
            | M::Cvtpd2dq
            | M::Vcvtpd2dq
            | M::Cvttpd2dq
            | M::Vcvttpd2dq
            | M::Cvtps2pd
            | M::Vcvtps2pd
            | M::Cvtpd2ps
            | M::Vcvtpd2ps
            | M::Haddps
            | M::Vhaddps
            | M::Haddpd
            | M::Vhaddpd
            | M::Hsubps
            | M::Vhsubps
            | M::Hsubpd
            | M::Vhsubpd
            | M::Addsubps
            | M::Vaddsubps
            | M::Addsubpd
            | M::Vaddsubpd
            | M::Roundps
            | M::Vroundps
            | M::Roundpd
            | M::Vroundpd
            | M::Roundss
            | M::Vroundss
            | M::Roundsd
            | M::Vroundsd
            | M::Dpps
            | M::Vdpps
            | M::Dppd
            | M::Vdppd
    )
}

impl FunctionLifter<'_> {
    pub(super) fn lift_float(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{F32, F64, I32};
        use Mnemonic as M;
        let f = Form::of(vex);
        let m = ins.mnemonic();
        if let Some((op, e, scalar)) = arith(m) {
            self.arith(ins, f, op, e, scalar)?;
            return Ok(true);
        }
        if let Some(fma) = fma(m) {
            self.fma(ins, fma)?;
            return Ok(true);
        }
        match m {
            M::Andps | M::Vandps | M::Andpd | M::Vandpd => self.logic(ins, f, "and", false)?,
            M::Andnps | M::Vandnps | M::Andnpd | M::Vandnpd => self.logic(ins, f, "and", true)?,
            M::Orps | M::Vorps | M::Orpd | M::Vorpd => self.logic(ins, f, "or", false)?,
            M::Xorps | M::Vxorps | M::Xorpd | M::Vxorpd => self.logic(ins, f, "xor", false)?,
            M::Cmpps | M::Vcmpps => self.compare(ins, f, vex, F32, false)?,
            M::Cmppd | M::Vcmppd => self.compare(ins, f, vex, F64, false)?,
            M::Cmpss | M::Vcmpss => self.compare(ins, f, vex, F32, true)?,
            M::Cmpsd | M::Vcmpsd => self.compare(ins, f, vex, F64, true)?,
            M::Comiss | M::Vcomiss | M::Ucomiss | M::Vucomiss => self.comis(ins, F32)?,
            M::Comisd | M::Vcomisd | M::Ucomisd | M::Vucomisd => self.comis(ins, F64)?,
            M::Cvtsi2ss | M::Vcvtsi2ss => self.cvt_int_to_scalar(ins, f, F32)?,
            M::Cvtsi2sd | M::Vcvtsi2sd => self.cvt_int_to_scalar(ins, f, F64)?,
            M::Cvtss2sd | M::Vcvtss2sd => {
                let a = self.v128(ins, f.a, F64)?;
                let s = self.low(ins, f.b, F32, 1)?;
                let d = self.val(&format!("fpext float {s} to double"));
                let r = self.insert(&a, F64, &d, 0);
                self.xset(ins.op_register(f.dst), &r, F64)?;
            }
            M::Cvtsd2ss | M::Vcvtsd2ss => {
                let a = self.v128(ins, f.a, F32)?;
                let s = self.low(ins, f.b, F64, 1)?;
                let d = self.val(&format!("fptrunc double {s} to float"));
                let r = self.insert(&a, F32, &d, 0);
                self.xset(ins.op_register(f.dst), &r, F32)?;
            }
            M::Cvtss2si | M::Vcvtss2si => self.cvt_scalar_to_int(ins, F32, false)?,
            M::Cvttss2si | M::Vcvttss2si => self.cvt_scalar_to_int(ins, F32, true)?,
            M::Cvtsd2si | M::Vcvtsd2si => self.cvt_scalar_to_int(ins, F64, false)?,
            M::Cvttsd2si | M::Vcvttsd2si => self.cvt_scalar_to_int(ins, F64, true)?,
            M::Cvtdq2ps | M::Vcvtdq2ps => {
                let v = self.v128(ins, 1, I32)?;
                let r = self.val(&format!("sitofp <4 x i32> {v} to <4 x float>"));
                self.xset(ins.op_register(0), &r, F32)?;
            }
            M::Cvtps2dq | M::Vcvtps2dq | M::Cvttps2dq | M::Vcvttps2dq => {
                let truncate = matches!(m, M::Cvttps2dq | M::Vcvttps2dq);
                let v = self.v128(ins, 1, F32)?;
                let r = self.float_to_int(&v, Ty::v(F32), 32, truncate);
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Cvtdq2pd | M::Vcvtdq2pd => {
                let v = self.low(ins, 1, I32, 2)?;
                let r = self.val(&format!("sitofp <2 x i32> {v} to <2 x double>"));
                self.xset(ins.op_register(0), &r, F64)?;
            }
            M::Cvtpd2dq | M::Vcvtpd2dq | M::Cvttpd2dq | M::Vcvttpd2dq => {
                // Zero-extend the `<2 x i32>` `fptosi.sat` result to `<4 x i32>` before selecting
                // the indefinite lane value so LLVM emits native `i32x4.trunc_sat_f64x2_s_zero`.
                let truncate = matches!(m, M::Cvttpd2dq | M::Vcvttpd2dq);
                let v = self.v128(ins, 1, F64)?;
                let x = if truncate {
                    v
                } else {
                    self.intr(Intr::Rint, Ty::v(F64), &[&v])
                };
                let sat2 = self.intr(Intr::FpToSiSat(I32), Ty::v(F64), &[&x]);
                let sat4 = self.zext_lanes(&sat2, I32, 2);
                let limit = Ty::v(F64).splat(&fconst(2f64.powi(31)));
                let ok2 = self.val(&format!("fcmp olt <2 x double> {x}, {limit}"));
                let ok4 = self.val(&format!(
                    "shufflevector <2 x i1> {ok2}, <2 x i1> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>"
                ));
                let r = self.val(&format!(
                    "select <4 x i1> {ok4}, <4 x i32> {sat4}, <4 x i32> <i32 -2147483648, i32 -2147483648, i32 0, i32 0>"
                ));
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Cvtps2pd | M::Vcvtps2pd => {
                let v = self.low(ins, 1, F32, 2)?;
                let r = self.val(&format!("fpext <2 x float> {v} to <2 x double>"));
                self.xset(ins.op_register(0), &r, F64)?;
            }
            M::Cvtpd2ps | M::Vcvtpd2ps => {
                let v = self.v128(ins, 1, F64)?;
                let r = self.val(&format!("fptrunc <2 x double> {v} to <2 x float>"));
                let r = self.zext_lanes(&r, F32, 2);
                self.xset(ins.op_register(0), &r, F32)?;
            }
            M::Haddps | M::Vhaddps => self.horizontal(ins, f, F32, "fadd")?,
            M::Haddpd | M::Vhaddpd => self.horizontal(ins, f, F64, "fadd")?,
            M::Hsubps | M::Vhsubps => self.horizontal(ins, f, F32, "fsub")?,
            M::Hsubpd | M::Vhsubpd => self.horizontal(ins, f, F64, "fsub")?,
            M::Addsubps | M::Vaddsubps | M::Addsubpd | M::Vaddsubpd => {
                let e = if matches!(m, M::Addsubps | M::Vaddsubps) {
                    F32
                } else {
                    F64
                };
                let t = Ty::v(e).ir();
                let a = self.v128(ins, f.a, e)?;
                let b = self.v128(ins, f.b, e)?;
                let d = self.val(&format!("fsub {t} {a}, {b}"));
                let s = self.val(&format!("fadd {t} {a}, {b}"));
                let n = e.lanes();
                // even lanes subtract, odd lanes add
                let idx: Vec<u32> = (0..n).map(|l| if l % 2 == 1 { n + l } else { l }).collect();
                let r = self.shuffle(&d, &s, e, &idx);
                self.xset(ins.op_register(f.dst), &r, e)?;
            }
            M::Roundps | M::Vroundps => self.round(ins, f, F32, false)?,
            M::Roundpd | M::Vroundpd => self.round(ins, f, F64, false)?,
            M::Roundss | M::Vroundss => self.round(ins, f, F32, true)?,
            M::Roundsd | M::Vroundsd => self.round(ins, f, F64, true)?,
            M::Dpps | M::Vdpps => self.dot(ins, f, F32)?,
            M::Dppd | M::Vdppd => self.dot(ins, f, F64)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn arith(&mut self, ins: &Instruction, f: Form, op: FOp, e: Elem, scalar: bool) -> R<()> {
        let unary = matches!(op, FOp::Sqrt | FOp::Rcp | FOp::Rsqrt);
        if scalar {
            // Lane 0 from (a op b) or op(b); lanes 1.. from a.
            let a = self.v128(ins, f.a, e)?;
            let b = self.low(ins, f.b, e, 1)?;
            let r = if unary {
                self.fop1(op, &b, Ty::s(e))
            } else {
                let a0 = self.extract(&a, e, 0);
                self.fop2(op, &a0, &b, Ty::s(e))
            };
            let v = self.insert(&a, e, &r, 0);
            self.xset(ins.op_register(f.dst), &v, e)
        } else if unary {
            // sqrtps xmm1, xmm2/m128 (both encodings)
            let src = ins.op_count() - 1;
            let b = self.v128(ins, src, e)?;
            let r = self.fop1(op, &b, Ty::v(e));
            self.xset(ins.op_register(0), &r, e)
        } else {
            let a = self.v128(ins, f.a, e)?;
            let b = self.v128(ins, f.b, e)?;
            let r = self.fop2(op, &a, &b, Ty::v(e));
            self.xset(ins.op_register(f.dst), &r, e)
        }
    }

    fn fop2(&mut self, op: FOp, a: &str, b: &str, ty: Ty) -> String {
        let t = ty.ir();
        let opcode = match op {
            FOp::Add => "fadd",
            FOp::Sub => "fsub",
            FOp::Mul => "fmul",
            FOp::Div => "fdiv",
            FOp::Min | FOp::Max => {
                // x86: the second operand unless the comparison holds (NaNs, +-0 -> b).
                let pred = if op == FOp::Min { "olt" } else { "ogt" };
                let c = self.val(&format!("fcmp {pred} {t} {a}, {b}"));
                return self.val(&format!("select {} {c}, {t} {a}, {t} {b}", ty.bool_ir()));
            }
            _ => "fadd",
        };
        self.val(&format!("{opcode} {t} {a}, {b}"))
    }

    fn fop1(&mut self, op: FOp, x: &str, ty: Ty) -> String {
        let t = ty.ir();
        match op {
            FOp::Sqrt => self.intr(Intr::Sqrt, ty, &[x]),
            FOp::Rcp | FOp::Rsqrt => {
                // Correctly rounded 1/x or 1/sqrt(x) plus the hardware's special cases.
                let one = ty.splat(&fconst(1.0));
                let d = if op == FOp::Rcp {
                    x.to_string()
                } else {
                    self.intr(Intr::Sqrt, ty, &[x])
                };
                let q = self.val(&format!("fdiv {t} {one}, {d}"));
                let it = ty.int();
                let ti = it.ir();
                let bits = self.val(&format!("bitcast {t} {x} to {ti}"));
                let sign = self.val(&format!(
                    "and {ti} {bits}, {}",
                    it.splat(&imm(0x8000_0000, 32))
                ));
                let mag = self.val(&format!("and {ti} {bits}, {}", it.splat("2147483647")));
                // Denormal inputs count as zero: +-inf.
                let den = self.val(&format!("icmp ult {ti} {mag}, {}", it.splat("8388608")));
                let inf = self.val(&format!("or {ti} {sign}, {}", it.splat("2139095040")));
                let inf = self.val(&format!("bitcast {ti} {inf} to {t}"));
                let r = self.val(&format!(
                    "select {} {den}, {t} {inf}, {t} {q}",
                    ty.bool_ir()
                ));
                if op == FOp::Rsqrt {
                    return r;
                }
                // rcp results that are not normal (|x| >= 2^126, finite) flush to +-0.
                let big = self.val(&format!("icmp uge {ti} {mag}, {}", it.splat("2122317824")));
                let finite = self.val(&format!("icmp ult {ti} {mag}, {}", it.splat("2139095040")));
                let tiny = self.val(&format!("and {} {big}, {finite}", ty.bool_ir()));
                let zero = self.val(&format!("bitcast {ti} {sign} to {t}"));
                self.val(&format!(
                    "select {} {tiny}, {t} {zero}, {t} {r}",
                    ty.bool_ir()
                ))
            }
            _ => x.to_string(),
        }
    }

    /// Bitwise logic on 128 bits (`andn` complements the first source).
    pub(super) fn logic(&mut self, ins: &Instruction, f: Form, op: &str, not_a: bool) -> R<()> {
        if ((op == "xor" && !not_a) || (op == "and" && not_a))
            && ins.op_kind(f.a) == OpKind::Register
            && ins.op_kind(f.b) == OpKind::Register
            && ins.op_register(f.a) == ins.op_register(f.b)
        {
            return self.xset(ins.op_register(f.dst), "zeroinitializer", Elem::I64);
        }
        let t = Ty::v(Elem::I64).ir();
        let mut a = self.v128(ins, f.a, Elem::I64)?;
        let b = self.v128(ins, f.b, Elem::I64)?;
        if not_a {
            a = self.val(&format!("xor {t} {a}, <i64 -1, i64 -1>"));
        }
        let r = self.val(&format!("{op} {t} {a}, {b}"));
        self.xset(ins.op_register(f.dst), &r, Elem::I64)
    }

    /// `cmpps/cmppd/cmpss/cmpsd`: all-ones lanes where the predicate holds.
    fn compare(&mut self, ins: &Instruction, f: Form, vex: bool, e: Elem, scalar: bool) -> R<()> {
        let i = Self::imm8(ins)?;
        // Legacy SSE encodings use imm8[2:0]; VEX uses imm8[4:0].
        let pred = predicate(if vex { i & 0x1F } else { i & 7 });
        let ie = e.int();
        if scalar {
            let a = self.v128(ins, f.a, e)?;
            let b = self.low(ins, f.b, e, 1)?;
            let a0 = self.extract(&a, e, 0);
            let c = self.val(&format!("fcmp {pred} {} {a0}, {b}", e.scalar()));
            let m = self.val(&format!("sext i1 {c} to {}", ie.scalar()));
            let ai = self.cast(&a, Ty::v(e), Ty::v(ie));
            let r = self.insert(&ai, ie, &m, 0);
            self.xset(ins.op_register(f.dst), &r, ie)
        } else {
            let ty = Ty::v(e);
            let a = self.v128(ins, f.a, e)?;
            let b = self.v128(ins, f.b, e)?;
            let c = self.val(&format!("fcmp {pred} {} {a}, {b}", ty.ir()));
            let m = self.val(&format!("sext {} {c} to {}", ty.bool_ir(), ty.int().ir()));
            self.xset(ins.op_register(f.dst), &m, ie)
        }
    }

    /// `comiss/ucomiss/comisd/ucomisd`: ZF/PF/CF = 111 unordered, 000 greater, 001 less,
    /// 100 equal; OF, SF and AF are cleared.
    fn comis(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let t = e.scalar();
        let a = self.low(ins, 0, e, 1)?;
        let b = self.low(ins, 1, e, 1)?;
        let zf = self.val(&format!("fcmp ueq {t} {a}, {b}"));
        let pf = self.val(&format!("fcmp uno {t} {a}, {b}"));
        let cf = self.val(&format!("fcmp ult {t} {a}, {b}"));
        self.set_flag(Flag::Zf, &zf);
        self.set_flag(Flag::Pf, &pf);
        self.set_flag(Flag::Cf, &cf);
        self.set_flag(Flag::Of, "false");
        self.set_flag(Flag::Sf, "false");
        self.set_flag(Flag::Af, "false");
        Ok(())
    }

    /// `cvtsi2ss/cvtsi2sd xmm, r/m32|64`: lane 0 = (float) int, other lanes from the first source.
    fn cvt_int_to_scalar(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let bits = self.op_bits(ins, f.b)?;
        if bits != 32 && bits != 64 {
            return unsupported("cvtsi2ss/sd source size");
        }
        let v = self.read_op(ins, f.b, bits)?;
        let c = self.val(&format!("sitofp i{bits} {v} to {}", e.scalar()));
        let a = self.v128(ins, f.a, e)?;
        let r = self.insert(&a, e, &c, 0);
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `cvt(t)ss2si/cvt(t)sd2si r32/r64, xmm/m`.
    fn cvt_scalar_to_int(&mut self, ins: &Instruction, e: Elem, truncate: bool) -> R<()> {
        if ins.op_kind(0) != OpKind::Register {
            return unsupported("conversion destination");
        }
        let bits = self.op_bits(ins, 0)?;
        if bits != 32 && bits != 64 {
            return unsupported("conversion destination size");
        }
        let s = self.low(ins, 1, e, 1)?;
        let r = self.float_to_int(&s, Ty::s(e), bits, truncate);
        self.write_op(ins, 0, &r, bits)
    }

    /// x86 float -> signed int conversion: round half to even (or truncate), and the "integer
    /// indefinite" value `1 << (bits - 1)` for NaN and out-of-range inputs.
    /// `llvm.fptosi.sat` already saturates `x <= -2^(bits-1)` (including `-inf`) to the exact
    /// indefinite value `MIN_INT`, so a single ordered comparison `x < 2^(bits-1)` catches both
    /// positive overflow and NaN.
    pub(super) fn float_to_int(&mut self, v: &str, ty: Ty, bits: u32, truncate: bool) -> String {
        let t = ty.ir();
        let x = if truncate {
            v.to_string()
        } else {
            self.intr(Intr::Rint, ty, &[v])
        };
        let dst_elem = Elem::of_bits(bits);
        let sat = self.intr(Intr::FpToSiSat(dst_elem), ty, &[&x]);
        let limit = 2f64.powi(bits as i32 - 1);
        let ok = self.val(&format!("fcmp olt {t} {x}, {}", ty.splat(&fconst(limit))));
        let it = ty.with(dst_elem);
        let indefinite = it.splat(&imm(1u64 << (bits - 1), bits));
        self.val(&format!(
            "select {} {ok}, {} {sat}, {} {indefinite}",
            ty.bool_ir(),
            it.ir(),
            it.ir()
        ))
    }

    /// `haddps/hsubps/haddpd/hsubpd`: pairwise op of adjacent lanes of a, then of b.
    fn horizontal(&mut self, ins: &Instruction, f: Form, e: Elem, op: &str) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let n = e.lanes();
        let even: Vec<u32> = (0..n).map(|l| 2 * l).collect();
        let odd: Vec<u32> = (0..n).map(|l| 2 * l + 1).collect();
        let lo = self.shuffle(&a, &b, e, &even);
        let hi = self.shuffle(&a, &b, e, &odd);
        let r = self.val(&format!("{op} {} {lo}, {hi}", Ty::v(e).ir()));
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `roundps/roundpd/roundss/roundsd`: imm8[1:0] = nearest-even, floor, ceil, truncate;
    /// imm8[2] selects MXCSR.RC, which is always round-to-nearest-even here.
    fn round(&mut self, ins: &Instruction, f: Form, e: Elem, scalar: bool) -> R<()> {
        let i = Self::imm8(ins)?;
        let intr = if i & 4 != 0 {
            Intr::Rint
        } else {
            [Intr::Rint, Intr::Floor, Intr::Ceil, Intr::Trunc][(i & 3) as usize]
        };
        if scalar {
            let a = self.v128(ins, f.a, e)?;
            let s = self.low(ins, f.b, e, 1)?;
            let r = self.intr(intr, Ty::s(e), &[&s]);
            let v = self.insert(&a, e, &r, 0);
            self.xset(ins.op_register(f.dst), &v, e)
        } else {
            let v = self.v128(ins, 1, e)?;
            let r = self.intr(intr, Ty::v(e), &[&v]);
            self.xset(ins.op_register(0), &r, e)
        }
    }

    /// `dpps/dppd`: products of the lanes selected by imm8[7:4] (others +0.0), summed in the
    /// hardware's order, broadcast to the lanes selected by imm8[3:0] (others +0.0).
    fn dot(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)?;
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let t = e.scalar();
        let n = e.lanes();
        let zero = fconst(0.0);
        let mut terms = Vec::new();
        for l in 0..n {
            if i >> (4 + l) & 1 == 1 {
                let x = self.extract(&a, e, l);
                let y = self.extract(&b, e, l);
                terms.push(self.val(&format!("fmul {t} {x}, {y}")));
            } else {
                terms.push(zero.clone());
            }
        }
        let sum = if n == 4 {
            let s01 = self.val(&format!("fadd {t} {}, {}", terms[0], terms[1]));
            let s23 = self.val(&format!("fadd {t} {}, {}", terms[2], terms[3]));
            self.val(&format!("fadd {t} {s01}, {s23}"))
        } else {
            self.val(&format!("fadd {t} {}, {}", terms[0], terms[1]))
        };
        let mut r = "zeroinitializer".to_string();
        for l in 0..n {
            if i >> l & 1 == 1 {
                r = self.insert(&r, e, &sum, l);
            }
        }
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// FMA3: `dst = (+-x) * y + (+-z)` with a single rounding. Operands: 132 `x=op0, y=op2,
    /// z=op1`; 213 `x=op1, y=op0, z=op2`; 231 `x=op1, y=op2, z=op0`. Scalar forms merge lane 0
    /// into the destination.
    fn fma(&mut self, ins: &Instruction, f: Fma) -> R<()> {
        let e = f.e;
        let n = if f.scalar { 1 } else { e.lanes() };
        let ty = if f.scalar { Ty::s(e) } else { Ty::v(e) };
        let mut ops = Vec::new();
        for i in 0..3 {
            ops.push(if f.scalar {
                self.low(ins, i, e, 1)?
            } else {
                self.v128(ins, i, e)?
            });
        }
        let (x, y, z) = match f.order {
            FmaOrder::O132 => (&ops[0], &ops[2], &ops[1]),
            FmaOrder::O213 => (&ops[1], &ops[0], &ops[2]),
            FmaOrder::O231 => (&ops[1], &ops[2], &ops[0]),
        };
        let t = ty.ir();
        let x = if f.neg_product {
            self.val(&format!("fneg {t} {x}"))
        } else {
            x.clone()
        };
        let z = match f.neg_addend {
            [false, false] => z.clone(),
            [true, true] => self.val(&format!("fneg {t} {z}")),
            [neg_even, _] => {
                let nz = self.val(&format!("fneg {t} {z}"));
                let idx: Vec<u32> = (0..n)
                    .map(|l| if (l % 2 == 0) == neg_even { n + l } else { l })
                    .collect();
                self.shuffle(z, &nz, e, &idx)
            }
        };
        let r = if e == Elem::F32 {
            self.fma32(&x, y, &z, n)
        } else {
            self.fma64(&x, y, &z, n)
        };
        if f.scalar {
            let d = self.v128(ins, 0, e)?;
            let v = self.insert(&d, e, &r, 0);
            self.xset(ins.op_register(0), &v, e)
        } else {
            self.xset(ins.op_register(0), &r, e)
        }
    }

    /// Correctly rounded binary64 `x * y + z` through the integer helper `__playgta6_fma_f64`.
    fn fma64(&mut self, x: &str, y: &str, z: &str, n: u32) -> String {
        self.use_helper(Intr::FmaF64);
        let call = |l: &mut Self, a: &str, b: &str, c: &str| {
            l.val(&format!(
                "call double @__playgta6_fma_f64(double {a}, double {b}, double {c})"
            ))
        };
        if n == 1 {
            return call(self, x, y, z);
        }
        let mut r = "poison".to_string();
        for lane in 0..n {
            let a = self.extract(x, Elem::F64, lane);
            let b = self.extract(y, Elem::F64, lane);
            let c = self.extract(z, Elem::F64, lane);
            let v = call(self, &a, &b, &c);
            r = self.insert(&r, Elem::F64, &v, lane);
        }
        r
    }

    /// Correctly rounded binary32 `x * y + z` without a hardware FMA: the product is exact in
    /// binary64, the sum is rounded to odd in binary64 (TwoSum error term), and the final
    /// rounding to binary32 is then correct (53 >= 2 * 24 + 2).
    fn fma32(&mut self, x: &str, y: &str, z: &str, n: u32) -> String {
        let (ft, dt, it, bt) = if n == 1 {
            (
                "float".to_string(),
                "double".to_string(),
                "i64".to_string(),
                "i1".to_string(),
            )
        } else {
            (
                format!("<{n} x float>"),
                format!("<{n} x double>"),
                format!("<{n} x i64>"),
                format!("<{n} x i1>"),
            )
        };
        let dty = Ty::vn(Elem::F64, n);
        let ity = Ty::vn(Elem::I64, n);
        let a = self.val(&format!("fpext {ft} {x} to {dt}"));
        let b = self.val(&format!("fpext {ft} {y} to {dt}"));
        let c = self.val(&format!("fpext {ft} {z} to {dt}"));
        let p = self.val(&format!("fmul {dt} {a}, {b}"));
        let s = self.val(&format!("fadd {dt} {p}, {c}"));
        // TwoSum: e = (p - (s - bb)) + (c - bb), bb = s - p.
        let bb = self.val(&format!("fsub {dt} {s}, {p}"));
        let t1 = self.val(&format!("fsub {dt} {s}, {bb}"));
        let e1 = self.val(&format!("fsub {dt} {p}, {t1}"));
        let e2 = self.val(&format!("fsub {dt} {c}, {bb}"));
        let err = self.val(&format!("fadd {dt} {e1}, {e2}"));
        let zero = dty.splat(&fconst(0.0));
        let inexact = self.val(&format!("fcmp one {dt} {err}, {zero}"));
        let bits = self.val(&format!("bitcast {dt} {s} to {it}"));
        let lsb = self.val(&format!("and {it} {bits}, {}", ity.splat("1")));
        let even = self.val(&format!("icmp eq {it} {lsb}, {}", ity.splat("0")));
        let fix = self.val(&format!("and {bt} {inexact}, {even}"));
        // Step to the odd neighbour on the side of the exact value.
        let err_pos = self.val(&format!("fcmp ogt {dt} {err}, {zero}"));
        let s_neg = self.val(&format!("icmp slt {it} {bits}, {}", ity.splat("0")));
        let away = self.val(&format!("xor {bt} {err_pos}, {s_neg}"));
        let step = self.val(&format!(
            "select {bt} {away}, {it} {}, {it} {}",
            ity.splat("1"),
            ity.splat("-1")
        ));
        let step = self.val(&format!(
            "select {bt} {fix}, {it} {step}, {it} {}",
            ity.splat("0")
        ));
        let odd = self.val(&format!("add {it} {bits}, {step}"));
        let odd = self.val(&format!("bitcast {it} {odd} to {dt}"));
        self.val(&format!("fptrunc {dt} {odd} to {ft}"))
    }
}
