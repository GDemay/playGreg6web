// Writes InputState (40 bytes, docs/wasm_abi.md) into a SharedArrayBuffer.
export class InputBridge {
  constructor(canvas) {
    this.sab = new SharedArrayBuffer(40);
    this.i32 = new Int32Array(this.sab);
    this.f32 = new Float32Array(this.sab);
    this.buttons = 0;

    canvas.addEventListener('click', () => {
      if (document.pointerLockElement !== canvas && canvas.requestPointerLock) {
        canvas.requestPointerLock();
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement !== canvas) return;
      Atomics.add(this.i32, 2, e.movementX | 0);
      Atomics.add(this.i32, 3, e.movementY | 0);
      Atomics.add(this.i32, 0, 1);
    });

    const keyMask = (code) => {
      switch (code) {
        case 'KeyW': return 1 << 0;
        case 'KeyS': return 1 << 1;
        case 'KeyA': return 1 << 2;
        case 'KeyD': return 1 << 3;
        case 'Space': return 1 << 4;
        case 'ShiftLeft': return 1 << 5;
        default: return 0;
      }
    };

    window.addEventListener('keydown', (e) => {
      const m = keyMask(e.code);
      if (!m) return;
      this.buttons |= m;
      Atomics.store(this.i32, 1, this.buttons);
      Atomics.add(this.i32, 0, 1);
    });

    window.addEventListener('keyup', (e) => {
      const m = keyMask(e.code);
      if (!m) return;
      this.buttons &= ~m;
      Atomics.store(this.i32, 1, this.buttons);
      Atomics.add(this.i32, 0, 1);
    });
  }
}
