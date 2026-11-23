//! Integer SIMD (SSE2 .. SSE4.2, AVX2): lane arithmetic, saturation, multiplies, compares,
//! min/max, shifts, byte shifts, shuffles, packs, unpacks, blends, horizontal ops, sums of
//! absolute differences, element inserts/extracts. Every operation is exact (no float involved).
//! In 256-bit (AVX2) forms all of these act on each 128-bit lane independently.

use iced_x86::{Instruction, Mnemonic, OpKind};

use super::super::{FunctionLifter, R, imm, unsupported};
use super::{Elem, Form, Intr, Ty};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IOp {
    Add,
    Sub,
    Mul,
    AddSatS,
    AddSatU,
    SubSatS,
    SubSatU,
    MulHiS,
    MulHiU,
    MulHrs,
    CmpEq,
    CmpGt,
    MinS,
    MinU,
    MaxS,
    MaxU,
    Avg,
    Sign,
    MulEvenU,
    MulEvenS,
}

/// Lane-wise binary operations `(op, lane type)`.
fn binary(m: Mnemonic) -> Option<(IOp, Elem)> {
    use Elem::{I8, I16, I32, I64};
    use IOp::*;
    use Mnemonic as M;
    Some(match m {
        M::Paddb | M::Vpaddb => (Add, I8),
        M::Paddw | M::Vpaddw => (Add, I16),
        M::Paddd | M::Vpaddd => (Add, I32),
        M::Paddq | M::Vpaddq => (Add, I64),
        M::Psubb | M::Vpsubb => (Sub, I8),
        M::Psubw | M::Vpsubw => (Sub, I16),
        M::Psubd | M::Vpsubd => (Sub, I32),
        M::Psubq | M::Vpsubq => (Sub, I64),
        M::Paddsb | M::Vpaddsb => (AddSatS, I8),
        M::Paddsw | M::Vpaddsw => (AddSatS, I16),
        M::Paddusb | M::Vpaddusb => (AddSatU, I8),
        M::Paddusw | M::Vpaddusw => (AddSatU, I16),
        M::Psubsb | M::Vpsubsb => (SubSatS, I8),
        M::Psubsw | M::Vpsubsw => (SubSatS, I16),
        M::Psubusb | M::Vpsubusb => (SubSatU, I8),
        M::Psubusw | M::Vpsubusw => (SubSatU, I16),
        M::Pmullw | M::Vpmullw => (Mul, I16),
        M::Pmulld | M::Vpmulld => (Mul, I32),
        M::Pmulhw | M::Vpmulhw => (MulHiS, I16),
        M::Pmulhuw | M::Vpmulhuw => (MulHiU, I16),
        M::Pmulhrsw | M::Vpmulhrsw => (MulHrs, I16),
        M::Pmuludq | M::Vpmuludq => (MulEvenU, I64),
        M::Pmuldq | M::Vpmuldq => (MulEvenS, I64),
        M::Pcmpeqb | M::Vpcmpeqb => (CmpEq, I8),
        M::Pcmpeqw | M::Vpcmpeqw => (CmpEq, I16),
        M::Pcmpeqd | M::Vpcmpeqd => (CmpEq, I32),
        M::Pcmpeqq | M::Vpcmpeqq => (CmpEq, I64),
        M::Pcmpgtb | M::Vpcmpgtb => (CmpGt, I8),
        M::Pcmpgtw | M::Vpcmpgtw => (CmpGt, I16),
        M::Pcmpgtd | M::Vpcmpgtd => (CmpGt, I32),
        M::Pcmpgtq | M::Vpcmpgtq => (CmpGt, I64),
        M::Pminsb | M::Vpminsb => (MinS, I8),
        M::Pminsw | M::Vpminsw => (MinS, I16),
        M::Pminsd | M::Vpminsd => (MinS, I32),
        M::Pminub | M::Vpminub => (MinU, I8),
        M::Pminuw | M::Vpminuw => (MinU, I16),
        M::Pminud | M::Vpminud => (MinU, I32),
        M::Pmaxsb | M::Vpmaxsb => (MaxS, I8),
        M::Pmaxsw | M::Vpmaxsw => (MaxS, I16),
        M::Pmaxsd | M::Vpmaxsd => (MaxS, I32),
        M::Pmaxub | M::Vpmaxub => (MaxU, I8),
        M::Pmaxuw | M::Vpmaxuw => (MaxU, I16),
        M::Pmaxud | M::Vpmaxud => (MaxU, I32),
        M::Pavgb | M::Vpavgb => (Avg, I8),
        M::Pavgw | M::Vpavgw => (Avg, I16),
        M::Psignb | M::Vpsignb => (Sign, I8),
        M::Psignw | M::Vpsignw => (Sign, I16),
        M::Psignd | M::Vpsignd => (Sign, I32),
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftKind {
    Left,
    Logical,
    Arith,
}

/// `(kind, lane type)` of `psll/psrl/psra` (immediate or XMM count).
fn shift(m: Mnemonic) -> Option<(ShiftKind, Elem)> {
    use Elem::{I16, I32, I64};
    use Mnemonic as M;
    use ShiftKind::*;
    Some(match m {
        M::Psllw | M::Vpsllw => (Left, I16),
        M::Pslld | M::Vpslld => (Left, I32),
        M::Psllq | M::Vpsllq => (Left, I64),
        M::Psrlw | M::Vpsrlw => (Logical, I16),
        M::Psrld | M::Vpsrld => (Logical, I32),
        M::Psrlq | M::Vpsrlq => (Logical, I64),
        M::Psraw | M::Vpsraw => (Arith, I16),
        M::Psrad | M::Vpsrad => (Arith, I32),
        _ => return None,
    })
}

/// `(kind, lane type)` of the AVX2 per-element variable shifts.
fn variable_shift(m: Mnemonic) -> Option<(ShiftKind, Elem)> {
    use Elem::{I32, I64};
    use Mnemonic as M;
    use ShiftKind::*;
    Some(match m {
        M::Vpsllvd => (Left, I32),
        M::Vpsllvq => (Left, I64),
        M::Vpsrlvd => (Logical, I32),
        M::Vpsrlvq => (Logical, I64),
        M::Vpsravd => (Arith, I32),
        _ => return None,
    })
}

/// Unpack shuffle masks: interleave the low (`high == false`) or high halves of a and b.
fn unpack_mask(e: Elem, high: bool) -> Vec<u32> {
    let n = e.lanes();
    let base = if high { n / 2 } else { 0 };
    (0..n / 2).flat_map(|i| [base + i, n + base + i]).collect()
}

impl FunctionLifter<'_> {
    pub(super) fn lift_int(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{I8, I16, I32, I64};
        use Mnemonic as M;
        let f = Form::of(vex);
        let m = ins.mnemonic();
        if let Some((op, e)) = binary(m) {
            self.int_binary(ins, f, op, e)?;
            return Ok(true);
        }
        if let Some((kind, e)) = shift(m) {
            self.int_shift(ins, f, kind, e)?;
            return Ok(true);
        }
        if let Some((kind, e)) = variable_shift(m) {
            self.variable_shift(ins, f, kind, e)?;
            return Ok(true);
        }
        match m {
            M::Pand | M::Vpand => self.logic(ins, f, "and", false)?,
            M::Pandn | M::Vpandn => self.logic(ins, f, "and", true)?,
            M::Por | M::Vpor => self.logic(ins, f, "or", false)?,
            M::Pxor | M::Vpxor => self.logic(ins, f, "xor", false)?,
            M::Pslldq | M::Vpslldq | M::Psrldq | M::Vpsrldq => {
                // pslldq xmm, imm / vpslldq xmm1, xmm2, imm
                let i = Self::imm8(ins)?;
                let src = if ins.op_count() == 3 { 1 } else { 0 };
                let v = self.v128(ins, src, I8)?;
                let left = matches!(m, M::Pslldq | M::Vpslldq);
                // Index 16 of the second operand is a zero byte.
                let idx: Vec<u32> = (0..16)
                    .map(|b| {
                        if i > 15 {
                            16
                        } else if left {
                            if b >= i { b - i } else { 16 }
                        } else if b + i < 16 {
                            b + i
                        } else {
                            16
                        }
                    })
                    .collect();
                let r = self.shuffle(&v, "zeroinitializer", I8, &idx);
                self.xset(ins.op_register(0), &r, I8)?;
            }
            M::Pshufd | M::Vpshufd => {
                let i = Self::imm8(ins)?;
                let v = self.v128(ins, 1, I32)?;
                let idx: Vec<u32> = (0..4).map(|l| (i >> (2 * l)) & 3).collect();
                let r = self.shuffle(&v, &v, I32, &idx);
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Pshuflw | M::Vpshuflw | M::Pshufhw | M::Vpshufhw => {
                let i = Self::imm8(ins)?;
                let v = self.v128(ins, 1, I16)?;
                let high = matches!(m, M::Pshufhw | M::Vpshufhw);
                let idx: Vec<u32> = (0..8)
                    .map(|l| match (high, l < 4) {
                        (false, true) => (i >> (2 * l)) & 3,
                        (true, false) => 4 + ((i >> (2 * (l - 4))) & 3),
                        _ => l,
                    })
                    .collect();
                let r = self.shuffle(&v, &v, I16, &idx);
                self.xset(ins.op_register(0), &r, I16)?;
            }
            M::Pshufb | M::Vpshufb => {
                let a = self.v128(ins, f.a, I8)?;
                let b = self.v128(ins, f.b, I8)?;
                // Bit 7 set selects zero: keep it so the index is >= 16 (swizzle yields 0).
                let idx = self.val(&format!("and <16 x i8> {b}, {}", Ty::v(I8).splat("-113")));
                let r = self.intr(Intr::Swizzle, Ty::v(I8), &[&a, &idx]);
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Punpcklbw | M::Vpunpcklbw => self.shuffle_op(ins, f, I8, &unpack_mask(I8, false))?,
            M::Punpckhbw | M::Vpunpckhbw => self.shuffle_op(ins, f, I8, &unpack_mask(I8, true))?,
            M::Punpcklwd | M::Vpunpcklwd => {
                self.shuffle_op(ins, f, I16, &unpack_mask(I16, false))?
            }
            M::Punpckhwd | M::Vpunpckhwd => {
                self.shuffle_op(ins, f, I16, &unpack_mask(I16, true))?
            }
            M::Punpckldq | M::Vpunpckldq => {
                self.shuffle_op(ins, f, I32, &unpack_mask(I32, false))?
            }
            M::Punpckhdq | M::Vpunpckhdq => {
                self.shuffle_op(ins, f, I32, &unpack_mask(I32, true))?
            }
            M::Punpcklqdq | M::Vpunpcklqdq => {
                self.shuffle_op(ins, f, I64, &unpack_mask(I64, false))?
            }
            M::Punpckhqdq | M::Vpunpckhqdq => {
                self.shuffle_op(ins, f, I64, &unpack_mask(I64, true))?
            }
            M::Packsswb | M::Vpacksswb => self.pack(ins, f, I16, true)?,
            M::Packssdw | M::Vpackssdw => self.pack(ins, f, I32, true)?,
            M::Packuswb | M::Vpackuswb => self.pack(ins, f, I16, false)?,
            M::Packusdw | M::Vpackusdw => self.pack(ins, f, I32, false)?,
            M::Pabsb | M::Vpabsb => self.abs(ins, I8)?,
            M::Pabsw | M::Vpabsw => self.abs(ins, I16)?,
            M::Pabsd | M::Vpabsd => self.abs(ins, I32)?,
            M::Pmaddwd | M::Vpmaddwd => {
                let a = self.v128(ins, f.a, I16)?;
                let b = self.v128(ins, f.b, I16)?;
                let wa = self.val(&format!("sext <8 x i16> {a} to <8 x i32>"));
                let wb = self.val(&format!("sext <8 x i16> {b} to <8 x i32>"));
                let p = self.val(&format!("mul <8 x i32> {wa}, {wb}"));
                let r = self.pair_sum(&p, 8, I32, "add");
                self.xset(ins.op_register(f.dst), &r, I32)?;
            }
            M::Pmaddubsw | M::Vpmaddubsw => {
                let a = self.v128(ins, f.a, I8)?;
                let b = self.v128(ins, f.b, I8)?;
                let wa = self.val(&format!("zext <16 x i8> {a} to <16 x i16>"));
                let wb = self.val(&format!("sext <16 x i8> {b} to <16 x i16>"));
                // Each product fits in i16; the pair sum saturates.
                let p = self.val(&format!("mul <16 x i16> {wa}, {wb}"));
                let even = self.val(&format!(
                    "shufflevector <16 x i16> {p}, <16 x i16> poison, {}",
                    super::mask(&(0..8).map(|i| 2 * i).collect::<Vec<_>>())
                ));
                let odd = self.val(&format!(
                    "shufflevector <16 x i16> {p}, <16 x i16> poison, {}",
                    super::mask(&(0..8).map(|i| 2 * i + 1).collect::<Vec<_>>())
                ));
                let r = self.intr(Intr::SAddSat, Ty::v(I16), &[&even, &odd]);
                self.xset(ins.op_register(f.dst), &r, I16)?;
            }
            M::Psadbw | M::Vpsadbw => {
                let a = self.v128(ins, f.a, I8)?;
                let b = self.v128(ins, f.b, I8)?;
                let wa = self.val(&format!("zext <16 x i8> {a} to <16 x i16>"));
                let wb = self.val(&format!("zext <16 x i8> {b} to <16 x i16>"));
                let d = self.val(&format!("sub <16 x i16> {wa}, {wb}"));
                let ad = self.intr(Intr::Abs, Ty::vn(I16, 16), &[&d]);
                let mut r = "zeroinitializer".to_string();
                for q in 0..2 {
                    let mut sum = String::from("0");
                    for k in 0..8 {
                        let x = self.val(&format!(
                            "extractelement <16 x i16> {ad}, i32 {}",
                            8 * q + k
                        ));
                        sum = self.val(&format!("add i16 {sum}, {x}"));
                    }
                    let wide = self.val(&format!("zext i16 {sum} to i64"));
                    r = self.insert(&r, I64, &wide, q);
                }
                self.xset(ins.op_register(f.dst), &r, I64)?;
            }
            M::Palignr | M::Vpalignr => {
                let i = Self::imm8(ins)?;
                let hi = self.v128(ins, f.a, I8)?;
                let lo = self.v128(ins, f.b, I8)?;
                // (hi:lo) >> 8*imm: byte k is lo[k+i] (< 16), hi[k+i-16] (< 32), else 0.
                let cat = self.val(&format!(
                    "shufflevector <16 x i8> {lo}, <16 x i8> {hi}, {}",
                    super::mask(&(0..32).collect::<Vec<_>>())
                ));
                let idx: Vec<u32> = (0..16).map(|k| (k + i).min(32)).collect();
                let r = self.val(&format!(
                    "shufflevector <32 x i8> {cat}, <32 x i8> zeroinitializer, {}",
                    super::mask(&idx)
                ));
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Pblendw | M::Vpblendw => {
                let i = Self::imm8(ins)?;
                let idx: Vec<u32> = (0..8)
                    .map(|l| if i >> l & 1 == 1 { 8 + l } else { l })
                    .collect();
                self.shuffle_op(ins, f, I16, &idx)?;
            }
            M::Vpblendd => {
                let i = Self::imm8(ins)?;
                let i = self.lane_imm(i, 4);
                let idx: Vec<u32> = (0..4)
                    .map(|l| if i >> l & 1 == 1 { 4 + l } else { l })
                    .collect();
                self.shuffle_op(ins, f, I32, &idx)?;
            }
            M::Pblendvb | M::Vpblendvb => self.blend_var(ins, f, vex, I8)?,
            M::Phaddw | M::Vphaddw => self.int_horizontal(ins, f, I16, "add", false)?,
            M::Phaddd | M::Vphaddd => self.int_horizontal(ins, f, I32, "add", false)?,
            M::Phaddsw | M::Vphaddsw => self.int_horizontal(ins, f, I16, "add", true)?,
            M::Phsubw | M::Vphsubw => self.int_horizontal(ins, f, I16, "sub", false)?,
            M::Phsubd | M::Vphsubd => self.int_horizontal(ins, f, I32, "sub", false)?,
            M::Phsubsw | M::Vphsubsw => self.int_horizontal(ins, f, I16, "sub", true)?,
            M::Pextrb | M::Vpextrb => self.extract_elem(ins, I8)?,
            M::Pextrw | M::Vpextrw => self.extract_elem(ins, I16)?,
            M::Pextrd | M::Vpextrd => self.extract_elem(ins, I32)?,
            M::Pextrq | M::Vpextrq => self.extract_elem(ins, I64)?,
            M::Pinsrb | M::Vpinsrb => self.insert_elem(ins, f, I8)?,
            M::Pinsrw | M::Vpinsrw => self.insert_elem(ins, f, I16)?,
            M::Pinsrd | M::Vpinsrd => self.insert_elem(ins, f, I32)?,
            M::Pinsrq | M::Vpinsrq => self.insert_elem(ins, f, I64)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn int_binary(&mut self, ins: &Instruction, f: Form, op: IOp, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let r = match op {
            IOp::Add => self.val(&format!("add {t} {a}, {b}")),
            IOp::Sub => self.val(&format!("sub {t} {a}, {b}")),
            IOp::Mul => self.val(&format!("mul {t} {a}, {b}")),
            IOp::AddSatS => self.intr(Intr::SAddSat, ty, &[&a, &b]),
            IOp::AddSatU => self.intr(Intr::UAddSat, ty, &[&a, &b]),
            IOp::SubSatS => self.intr(Intr::SSubSat, ty, &[&a, &b]),
            IOp::SubSatU => self.intr(Intr::USubSat, ty, &[&a, &b]),
            IOp::MulHiS | IOp::MulHiU | IOp::MulHrs => {
                let ext = if op == IOp::MulHiU { "zext" } else { "sext" };
                let w = ty.with(Elem::I32).ir();
                let wa = self.val(&format!("{ext} {t} {a} to {w}"));
                let wb = self.val(&format!("{ext} {t} {b} to {w}"));
                let p = self.val(&format!("mul {w} {wa}, {wb}"));
                let hi = if op == IOp::MulHrs {
                    // ((a * b >> 14) + 1) >> 1
                    let s = self.val(&format!("ashr {w} {p}, {}", ty.with(Elem::I32).splat("14")));
                    let s = self.val(&format!("add {w} {s}, {}", ty.with(Elem::I32).splat("1")));
                    self.val(&format!("ashr {w} {s}, {}", ty.with(Elem::I32).splat("1")))
                } else {
                    self.val(&format!("lshr {w} {p}, {}", ty.with(Elem::I32).splat("16")))
                };
                self.val(&format!("trunc {w} {hi} to {t}"))
            }
            IOp::MulEvenU | IOp::MulEvenS => {
                // Low 32 bits of each quadword, zero/sign-extended, multiplied to 64 bits.
                let (wa, wb) = if op == IOp::MulEvenU {
                    let m = ty.splat("4294967295");
                    (
                        self.val(&format!("and {t} {a}, {m}")),
                        self.val(&format!("and {t} {b}, {m}")),
                    )
                } else {
                    let s = ty.splat("32");
                    let a1 = self.val(&format!("shl {t} {a}, {s}"));
                    let b1 = self.val(&format!("shl {t} {b}, {s}"));
                    (
                        self.val(&format!("ashr {t} {a1}, {s}")),
                        self.val(&format!("ashr {t} {b1}, {s}")),
                    )
                };
                self.val(&format!("mul {t} {wa}, {wb}"))
            }
            IOp::CmpEq | IOp::CmpGt => {
                let pred = if op == IOp::CmpEq { "eq" } else { "sgt" };
                let c = self.val(&format!("icmp {pred} {t} {a}, {b}"));
                self.val(&format!("sext {bt} {c} to {t}"))
            }
            IOp::MinS | IOp::MinU | IOp::MaxS | IOp::MaxU => {
                let pred = match op {
                    IOp::MinS => "slt",
                    IOp::MinU => "ult",
                    IOp::MaxS => "sgt",
                    _ => "ugt",
                };
                let c = self.val(&format!("icmp {pred} {t} {a}, {b}"));
                self.val(&format!("select {bt} {c}, {t} {a}, {t} {b}"))
            }
            IOp::Avg => {
                let w = ty.with(e.wide()).ir();
                let one = ty.with(e.wide()).splat("1");
                let wa = self.val(&format!("zext {t} {a} to {w}"));
                let wb = self.val(&format!("zext {t} {b} to {w}"));
                let s = self.val(&format!("add {w} {wa}, {wb}"));
                let s = self.val(&format!("add {w} {s}, {one}"));
                let s = self.val(&format!("lshr {w} {s}, {one}"));
                self.val(&format!("trunc {w} {s} to {t}"))
            }
            IOp::Sign => {
                let neg = self.val(&format!("sub {t} zeroinitializer, {a}"));
                let lt = self.val(&format!("icmp slt {t} {b}, zeroinitializer"));
                let eq = self.val(&format!("icmp eq {t} {b}, zeroinitializer"));
                let r = self.val(&format!("select {bt} {lt}, {t} {neg}, {t} {a}"));
                self.val(&format!("select {bt} {eq}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `psll/psrl/psra` by an immediate or by the low 64 bits of an XMM/`m128` count operand
    /// (also in 256-bit forms). Counts >= the lane width give 0 (logical) or the sign (arith).
    fn int_shift(&mut self, ins: &Instruction, f: Form, kind: ShiftKind, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        let t = ty.ir();
        let last = ins.op_count() - 1;
        // Immediate forms: `psllw xmm, imm` / `vpsllw xmm1, xmm2, imm`.
        let (src, count) = if super::super::is_immediate(ins.op_kind(last)) {
            let src = if ins.op_count() == 3 { 1 } else { 0 };
            (src, imm(u64::from(Self::imm8(ins)?), 64))
        } else {
            let count = match ins.op_kind(f.b) {
                OpKind::Register => {
                    // The count is always the low quadword of an XMM register, also for YMM data.
                    let v = self.xget_half(ins.op_register(f.b), 0, Elem::I64)?;
                    self.extract(&v, Elem::I64, 0)
                }
                OpKind::Memory => self.mem_load_at(ins, Ty::s(Elem::I64), 0, 16)?,
                k => return unsupported(format!("shift count operand {k:?}")),
            };
            (f.a, count)
        };
        let v = self.v128(ins, src, e)?;
        let width = e.bits();
        let big = self.val(&format!("icmp ugt i64 {count}, {}", width - 1));
        let r = match kind {
            ShiftKind::Arith => {
                let n = self.val(&format!("select i1 {big}, i64 {}, i64 {count}", width - 1));
                let n = self.narrow_count(&n, e);
                let s = self.splat_value(&n, e);
                self.val(&format!("ashr {t} {v}, {s}"))
            }
            _ => {
                let n = self.val(&format!("select i1 {big}, i64 0, i64 {count}"));
                let n = self.narrow_count(&n, e);
                let s = self.splat_value(&n, e);
                let op = if kind == ShiftKind::Left {
                    "shl"
                } else {
                    "lshr"
                };
                let r = self.val(&format!("{op} {t} {v}, {s}"));
                self.val(&format!("select i1 {big}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(0), &r, e)
    }

    /// An `i64` shift count (already below the lane width) as a lane-typed scalar.
    fn narrow_count(&mut self, n: &str, e: Elem) -> String {
        if e == Elem::I64 {
            n.to_string()
        } else {
            self.val(&format!("trunc i64 {n} to {}", e.scalar()))
        }
    }

    /// AVX2 `vpsllv/vpsrlv/vpsrav`: per-element counts.
    fn variable_shift(&mut self, ins: &Instruction, f: Form, kind: ShiftKind, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let v = self.v128(ins, f.a, e)?;
        let c = self.v128(ins, f.b, e)?;
        let max = ty.splat(&(e.bits() - 1).to_string());
        let big = self.val(&format!("icmp ugt {t} {c}, {max}"));
        let r = match kind {
            ShiftKind::Arith => {
                let n = self.val(&format!("select {bt} {big}, {t} {max}, {t} {c}"));
                self.val(&format!("ashr {t} {v}, {n}"))
            }
            _ => {
                let n = self.val(&format!("select {bt} {big}, {t} zeroinitializer, {t} {c}"));
                let op = if kind == ShiftKind::Left {
                    "shl"
                } else {
                    "lshr"
                };
                let r = self.val(&format!("{op} {t} {v}, {n}"));
                self.val(&format!("select {bt} {big}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// A scalar broadcast to every lane of a 128-bit vector of `e`.
    pub(super) fn splat_value(&mut self, s: &str, e: Elem) -> String {
        let t = Ty::v(e).ir();
        let one = self.val(&format!(
            "insertelement {t} poison, {} {s}, i32 0",
            e.scalar()
        ));
        self.val(&format!(
            "shufflevector {t} {one}, {t} poison, <{} x i32> zeroinitializer",
            e.lanes()
        ))
    }

    /// Sum (or difference) of adjacent lanes of a `<n x e>` vector -> `<n/2 x e>`.
    fn pair_sum(&mut self, v: &str, n: u32, e: Elem, op: &str) -> String {
        let t = Ty::vn(e, n).ir();
        let even: Vec<u32> = (0..n / 2).map(|i| 2 * i).collect();
        let odd: Vec<u32> = (0..n / 2).map(|i| 2 * i + 1).collect();
        let lo = self.val(&format!(
            "shufflevector {t} {v}, {t} poison, {}",
            super::mask(&even)
        ));
        let hi = self.val(&format!(
            "shufflevector {t} {v}, {t} poison, {}",
            super::mask(&odd)
        ));
        self.val(&format!("{op} {} {lo}, {hi}", Ty::vn(e, n / 2).ir()))
    }

    /// `packss*`/`packus*`: saturate both sources to half-width lanes, a's lanes first.
    fn pack(&mut self, ins: &Instruction, f: Form, e: Elem, signed: bool) -> R<()> {
        let n = e.lanes();
        let ty = Ty::vn(e, 2 * n);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let cat = self.val(&format!(
            "shufflevector {} {a}, {} {b}, {}",
            Ty::v(e).ir(),
            Ty::v(e).ir(),
            super::mask(&(0..2 * n).collect::<Vec<_>>())
        ));
        let half = e.bits() / 2;
        let (lo, hi) = if signed {
            (-(1i64 << (half - 1)), (1i64 << (half - 1)) - 1)
        } else {
            (0, (1i64 << half) - 1)
        };
        let lo_s = ty.splat(&lo.to_string());
        let hi_s = ty.splat(&hi.to_string());
        let c1 = self.val(&format!("icmp slt {t} {cat}, {lo_s}"));
        let v = self.val(&format!("select {bt} {c1}, {t} {lo_s}, {t} {cat}"));
        let c2 = self.val(&format!("icmp sgt {t} {v}, {hi_s}"));
        let v = self.val(&format!("select {bt} {c2}, {t} {hi_s}, {t} {v}"));
        let ne = Elem::of_bits(half);
        let r = self.val(&format!("trunc {t} {v} to {}", Ty::vn(ne, 2 * n).ir()));
        self.xset(ins.op_register(f.dst), &r, ne)
    }

    fn abs(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let v = self.v128(ins, 1, e)?;
        let r = self.intr(Intr::Abs, Ty::v(e), &[&v]);
        self.xset(ins.op_register(0), &r, e)
    }

    /// `phadd*/phsub*`: adjacent pairs of a, then of b (optionally saturating).
    fn int_horizontal(
        &mut self,
        ins: &Instruction,
        f: Form,
        e: Elem,
        op: &str,
        saturate: bool,
    ) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let n = e.lanes();
        let even: Vec<u32> = (0..n).map(|l| 2 * l).collect();
        let odd: Vec<u32> = (0..n).map(|l| 2 * l + 1).collect();
        let lo = self.shuffle(&a, &b, e, &even);
        let hi = self.shuffle(&a, &b, e, &odd);
        let r = if saturate {
            let intr = if op == "add" {
                Intr::SAddSat
            } else {
                Intr::SSubSat
            };
            self.intr(intr, Ty::v(e), &[&lo, &hi])
        } else {
            self.val(&format!("{op} {} {lo}, {hi}", Ty::v(e).ir()))
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `pextrb/w/d/q r/m, xmm, imm`: zero-extended into a GPR or stored.
    fn extract_elem(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let i = Self::imm8(ins)? % e.lanes();
        let v = self.xget(ins.op_register(1), e)?;
        let s = self.extract(&v, e, i);
        self.write_gpr_or_mem(ins, 0, &s, e.bits())
    }

    /// `pinsrb/w/d/q xmm, r/m, imm` (VEX: merges into the first source).
    fn insert_elem(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)? % e.lanes();
        let s = match ins.op_kind(f.b) {
            OpKind::Register => {
                let (v, bits) = self.read_reg(ins.op_register(f.b))?;
                if bits > e.bits() {
                    self.val(&format!("trunc i{bits} {v} to {}", e.scalar()))
                } else if bits == e.bits() {
                    v
                } else {
                    return unsupported("pinsr source register too narrow");
                }
            }
            OpKind::Memory => self.mem_load(ins, Ty::s(e))?,
            k => return unsupported(format!("pinsr source {k:?}")),
        };
        let a = self.xget(ins.op_register(f.a), e)?;
        let r = self.insert(&a, e, &s, i);
        self.xset(ins.op_register(f.dst), &r, e)
    }
}
