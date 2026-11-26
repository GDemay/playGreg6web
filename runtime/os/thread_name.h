// Names the calling host thread (visible in debuggers, ETW and per-thread CPU tools). Diagnostic
// only: nothing the title can observe depends on it.
#pragma once

namespace playgta6::os {

void name_this_thread(const wchar_t* name);

}  // namespace playgta6::os
