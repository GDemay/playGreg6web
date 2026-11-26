//! Execution tests for the x87 control/status instructions `setjmp`/`longjmp` rely on
//! (tool-gated like [`exec`](super::exec)).

use iced_x86::code_asm::*;

use super::asm_function;
use super::exec::{CODE_BASE, Case, DATA_ADDR, RAX, RCX, build, exec_or_skip, run};
use crate::lift::lift_module;

fn x87_control_function() -> crate::cfg::Function {
    asm_function(CODE_BASE, |a| {
        a.fnstcw(word_ptr(rcx))?;
        a.mov(word_ptr(rcx + 2), 0x1234u32)?;
        a.fldcw(word_ptr(rcx + 2))?;
        a.fnstcw(word_ptr(rcx + 4))?;
        a.fnstsw(word_ptr(rcx + 6))?;
        a.fnstsw(ax)?;
        a.fnclex()?;
        a.fninit()?;
        a.wait()?;
        a.ret()
    })
}

#[test]
fn x87_control_ops_are_lifted() {
    let module = lift_module([&x87_control_function()]).expect("lift");
    assert!(
        module.functions[0].unsupported.is_empty(),
        "{:?}",
        module.functions[0].unsupported
    );
}

#[test]
fn x87_control_ops_execute() {
    if !exec_or_skip("x87_control_ops_execute") {
        return;
    }
    let built = build("x87_control", &[x87_control_function()]);
    let buf = DATA_ADDR;
    let out = run(
        &built,
        &[Case::new(CODE_BASE)
            .reg(RCX, buf)
            .reg(RAX, 0xFFFF_FFFF)
            .mem(buf, &[0xAA; 16])
            .read(buf, 16)],
    );
    let o = &out[0];
    assert_eq!(o.error, "");
    // fnstcw reports the reset control word, fnstsw an empty status word; fldcw has no effect.
    assert_eq!(&o.mem[0][0..2], &0x027Fu16.to_le_bytes());
    assert_eq!(&o.mem[0][2..4], &0x1234u16.to_le_bytes());
    assert_eq!(&o.mem[0][4..6], &0x027Fu16.to_le_bytes());
    assert_eq!(&o.mem[0][6..8], &0u16.to_le_bytes());
    assert_eq!(o.regs[RAX] & 0xFFFF, 0);
    assert_eq!(o.regs[RAX] >> 16, 0xFFFF);
}
