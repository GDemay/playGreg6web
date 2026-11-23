//! Integer SIMD.

use iced_x86::Instruction;

use super::super::{FunctionLifter, R};

impl FunctionLifter<'_> {
    pub(super) fn lift_int(&mut self, _ins: &Instruction, _vex: bool) -> R<bool> {
        Ok(false)
    }
}
