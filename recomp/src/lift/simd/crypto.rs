//! AES-NI (`aesenc/aesenclast/aesdec/aesdeclast/aesimc/aeskeygenassist`, also VEX and VAES),
//! carry-less multiplication (`pclmulqdq`, also VEX and VPCLMULQDQ), `phminposuw` and the SSE4a
//! bit-field instructions `extrq/insertq`. All exact integer computations.

use iced_x86::{Instruction, Mnemonic};

use super::super::{FunctionLifter, R};
use super::{Elem, Form, Intr, Ty, mask};
use crate::lift::{BitIntrinsic, Support};

/// AES state byte `4 * column + row` of an XMM register.
fn byte(column: u32, row: u32) -> u32 {
    4 * (column % 4) + row
}

impl FunctionLifter<'_> {
    pub(super) fn lift_crypto(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{I8, I16, I32, I64};
        use Mnemonic as M;
        let f = Form::of(vex);
        let m = ins.mnemonic();
        match m {
            M::Aesenc | M::Vaesenc | M::Aesenclast | M::Vaesenclast => {
                let state = self.v128(ins, f.a, I8)?;
                let key = self.v128(ins, f.b, I8)?;
                let s = self.shift_rows(&state, false);
                let s = self.sub_bytes(&s, false);
                let s = if matches!(m, M::Aesenc | M::Vaesenc) {
                    self.mix_columns(&s)
                } else {
                    s
                };
                let r = self.val(&format!("xor <16 x i8> {s}, {key}"));
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Aesdec | M::Vaesdec | M::Aesdeclast | M::Vaesdeclast => {
                let state = self.v128(ins, f.a, I8)?;
                let key = self.v128(ins, f.b, I8)?;
                let s = self.shift_rows(&state, true);
                let s = self.sub_bytes(&s, true);
                let s = if matches!(m, M::Aesdec | M::Vaesdec) {
                    self.inv_mix_columns(&s)
                } else {
                    s
                };
                let r = self.val(&format!("xor <16 x i8> {s}, {key}"));
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Aesimc | M::Vaesimc => {
                let v = self.v128(ins, 1, I8)?;
                let r = self.inv_mix_columns(&v);
                self.xset(ins.op_register(0), &r, I8)?;
            }
            M::Aeskeygenassist | M::Vaeskeygenassist => {
                let rcon = Self::imm8(ins)?;
                let v = self.v128(ins, 1, I8)?;
                let s = self.sub_bytes(&v, false);
                // [SubWord(X1), RotWord(SubWord(X1)) ^ rcon, SubWord(X3), RotWord(SubWord(X3)) ^ rcon]
                let idx = [4, 5, 6, 7, 5, 6, 7, 4, 12, 13, 14, 15, 13, 14, 15, 12];
                let r = self.shuffle(&s, &s, I8, &idx);
                let r = self.cast(&r, Ty::v(I8), Ty::v(I32));
                let r = self.val(&format!(
                    "xor <4 x i32> {r}, <i32 0, i32 {rcon}, i32 0, i32 {rcon}>"
                ));
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Pclmulqdq | M::Vpclmulqdq => {
                let i = Self::imm8(ins)?;
                let a = self.v128(ins, f.a, I64)?;
                let b = self.v128(ins, f.b, I64)?;
                let x = self.extract(&a, I64, i & 1);
                let y = self.extract(&b, I64, (i >> 4) & 1);
                self.use_helper(Intr::Clmul);
                let p = self.val(&format!(
                    "call {{ i64, i64 }} @__playgta6_clmul(i64 {x}, i64 {y})"
                ));
                let lo = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 0"));
                let hi = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 1"));
                let r = self.insert("poison", I64, &lo, 0);
                let r = self.insert(&r, I64, &hi, 1);
                self.xset(ins.op_register(f.dst), &r, I64)?;
            }
            M::Phminposuw | M::Vphminposuw => {
                // Vector butterfly reduction via `i16x8.min_u`, then `i16x8.eq` + `i16x8.bitmask`
                // + `cttz` to recover the lowest lane index matching the minimum.
                let v = self.v128(ins, 1, I16)?;
                let s1 = self.shuffle(&v, &v, I16, &[4, 5, 6, 7, 0, 1, 2, 3]);
                let m1 = self.intr(Intr::UMin, Ty::v(I16), &[&v, &s1]);
                let s2 = self.shuffle(&m1, &m1, I16, &[2, 3, 0, 1, 6, 7, 4, 5]);
                let m2 = self.intr(Intr::UMin, Ty::v(I16), &[&m1, &s2]);
                let s3 = self.shuffle(&m2, &m2, I16, &[1, 0, 3, 2, 5, 4, 7, 6]);
                let m3 = self.intr(Intr::UMin, Ty::v(I16), &[&m2, &s3]);
                let eq = self.val(&format!("icmp eq <8 x i16> {v}, {m3}"));
                let bits = self.val(&format!("bitcast <8 x i1> {eq} to i8"));
                let bits16 = self.val(&format!("zext i8 {bits} to i16"));
                self.support.insert(Support::BitIntrinsic(BitIntrinsic::Cttz, 16));
                let index = self.val(&format!("call i16 @llvm.cttz.i16(i16 {bits16}, i1 true)"));
                let min = self.extract(&m3, I16, 0);
                let r = self.insert("zeroinitializer", I16, &min, 0);
                let r = self.insert(&r, I16, &index, 1);
                self.xset(ins.op_register(0), &r, I16)?;
            }
            M::Extrq | M::Insertq => self.sse4a_bitfield(ins, m == M::Insertq)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// SSE4a `extrq`/`insertq` on the low quadword: a field of `len` bits (0 means 64) at bit
    /// `idx`, from immediates or from `xmm2` (bits 5:0 / 13:8 for `extrq`, 69:64 / 77:72 for
    /// `insertq`). The architecturally undefined parts are defined here as: the upper quadword
    /// is preserved, and field bits beyond bit 63 (`len + idx > 64`) are dropped.
    fn sse4a_bitfield(&mut self, ins: &Instruction, insert: bool) -> R<()> {
        use Elem::I64;
        let dst = self.xget(ins.op_register(0), I64)?;
        let lo = self.extract(&dst, I64, 0);
        let imm_form = ins.op_count() > 2;
        let (len, idx, src) = if insert {
            let src = self.xget(ins.op_register(1), I64)?;
            let s0 = self.extract(&src, I64, 0);
            if imm_form {
                let len = u64::from(ins.immediate8()) & 63;
                let idx = u64::from(ins.immediate8_2nd()) & 63;
                (len.to_string(), idx.to_string(), Some(s0))
            } else {
                let s1 = self.extract(&src, I64, 1);
                let len = self.val(&format!("and i64 {s1}, 63"));
                let idx = self.val(&format!("lshr i64 {s1}, 8"));
                let idx = self.val(&format!("and i64 {idx}, 63"));
                (len, idx, Some(s0))
            }
        } else if imm_form {
            let len = u64::from(ins.immediate8()) & 63;
            let idx = u64::from(ins.immediate8_2nd()) & 63;
            (len.to_string(), idx.to_string(), None)
        } else {
            let src = self.xget(ins.op_register(1), I64)?;
            let s0 = self.extract(&src, I64, 0);
            let len = self.val(&format!("and i64 {s0}, 63"));
            let idx = self.val(&format!("lshr i64 {s0}, 8"));
            let idx = self.val(&format!("and i64 {idx}, 63"));
            (len, idx, None)
        };
        let full = self.val(&format!("icmp eq i64 {len}, 0"));
        let one = self.val(&format!("shl i64 1, {len}"));
        let m = self.val(&format!("sub i64 {one}, 1"));
        let m = self.val(&format!("select i1 {full}, i64 -1, i64 {m}"));
        let r = match src {
            None => {
                let v = self.val(&format!("lshr i64 {lo}, {idx}"));
                self.val(&format!("and i64 {v}, {m}"))
            }
            Some(s0) => {
                let field = self.val(&format!("and i64 {s0}, {m}"));
                let field = self.val(&format!("shl i64 {field}, {idx}"));
                let hole = self.val(&format!("shl i64 {m}, {idx}"));
                let hole = self.val(&format!("xor i64 {hole}, -1"));
                let kept = self.val(&format!("and i64 {lo}, {hole}"));
                self.val(&format!("or i64 {kept}, {field}"))
            }
        };
        let out = self.insert(&dst, I64, &r, 0);
        self.xset(ins.op_register(0), &out, I64)
    }

    /// (Inv)ShiftRows: row `r` of the state rotates left (right) by `r` columns.
    fn shift_rows(&mut self, s: &str, inverse: bool) -> String {
        let idx: Vec<u32> = (0..16)
            .map(|i| {
                let (column, row) = (i / 4, i % 4);
                if inverse {
                    byte(column + 4 - row, row)
                } else {
                    byte(column + row, row)
                }
            })
            .collect();
        self.shuffle(s, s, Elem::I8, &idx)
    }

    /// (Inv)SubBytes: table lookups in the AES S-box.
    fn sub_bytes(&mut self, s: &str, inverse: bool) -> String {
        self.use_helper(Intr::AesTables);
        let table = if inverse {
            "@__playgta6_aes_inv_sbox"
        } else {
            "@__playgta6_aes_sbox"
        };
        let mut r = "poison".to_string();
        for l in 0..16 {
            let b = self.extract(s, Elem::I8, l);
            let i = self.val(&format!("zext i8 {b} to i64"));
            let p = self.val(&format!(
                "getelementptr inbounds [256 x i8], ptr {table}, i64 0, i64 {i}"
            ));
            let x = self.val(&format!("load i8, ptr {p}"));
            r = self.insert(&r, Elem::I8, &x, l);
        }
        r
    }

    /// Multiplication by x in GF(2^8) of every byte.
    fn xtime(&mut self, v: &str) -> String {
        let t = Ty::v(Elem::I8);
        let hi = self.val(&format!("ashr <16 x i8> {v}, {}", t.splat("7")));
        let red = self.val(&format!("and <16 x i8> {hi}, {}", t.splat("27")));
        let sh = self.val(&format!("shl <16 x i8> {v}, {}", t.splat("1")));
        self.val(&format!("xor <16 x i8> {sh}, {red}"))
    }

    /// Rotates every column of the state by `k` rows (byte `row` takes row `row + k`).
    fn rotate_columns(&mut self, v: &str, k: u32) -> String {
        let idx: Vec<u32> = (0..16).map(|i| byte(i / 4, (i % 4 + k) % 4)).collect();
        self.val(&format!(
            "shufflevector <16 x i8> {v}, <16 x i8> poison, {}",
            mask(&idx)
        ))
    }

    /// MixColumns: `s'[r] = 2 s[r] ^ 3 s[r+1] ^ s[r+2] ^ s[r+3]` per column.
    fn mix_columns(&mut self, s: &str) -> String {
        let x2 = self.xtime(s);
        let x3 = self.val(&format!("xor <16 x i8> {x2}, {s}"));
        let r1 = self.rotate_columns(&x3, 1);
        let r2 = self.rotate_columns(s, 2);
        let r3 = self.rotate_columns(s, 3);
        let a = self.val(&format!("xor <16 x i8> {x2}, {r1}"));
        let b = self.val(&format!("xor <16 x i8> {r2}, {r3}"));
        self.val(&format!("xor <16 x i8> {a}, {b}"))
    }

    /// InvMixColumns: `s'[r] = 14 s[r] ^ 11 s[r+1] ^ 13 s[r+2] ^ 9 s[r+3]` per column.
    fn inv_mix_columns(&mut self, s: &str) -> String {
        let x2 = self.xtime(s);
        let x4 = self.xtime(&x2);
        let x8 = self.xtime(&x4);
        let x8s = self.val(&format!("xor <16 x i8> {x8}, {s}"));
        let m9 = x8s.clone();
        let m11 = self.val(&format!("xor <16 x i8> {x8s}, {x2}"));
        let m13 = self.val(&format!("xor <16 x i8> {x8s}, {x4}"));
        let x84 = self.val(&format!("xor <16 x i8> {x8}, {x4}"));
        let m14 = self.val(&format!("xor <16 x i8> {x84}, {x2}"));
        let r1 = self.rotate_columns(&m11, 1);
        let r2 = self.rotate_columns(&m13, 2);
        let r3 = self.rotate_columns(&m9, 3);
        let a = self.val(&format!("xor <16 x i8> {m14}, {r1}"));
        let b = self.val(&format!("xor <16 x i8> {r2}, {r3}"));
        self.val(&format!("xor <16 x i8> {a}, {b}"))
    }
}
