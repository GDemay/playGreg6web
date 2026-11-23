#include "gpu/cmd_encoder.h"
#include "os/os_shim.h"

int main() {
    playgta6::os::log("playgta6web Wasm64 runtime initialized (ABI v1)");

    playgta6::gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.08f, 0.12f, 0.22f, 1.0f);
    enc.present();
    playgta6::os::wgpu_submit(enc.data(), enc.size());

    return 0;
}
