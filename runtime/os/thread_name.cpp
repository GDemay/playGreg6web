#include "os/thread_name.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace playgta6::os {

void name_this_thread(const wchar_t* name) {
#ifdef _WIN32
    SetThreadDescription(GetCurrentThread(), name);
#else
    (void)name;
#endif
}

}  // namespace playgta6::os
