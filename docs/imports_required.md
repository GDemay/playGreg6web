# PE64 Inspection Report — `game_scarlett_master.exe`

> Sanitized static analysis output: headers, section metadata, addresses, and symbol names only (zero raw binary bytes).

## Headers

| Field | Value |
| --- | --- |
| File Size | 120181760 bytes |
| DOS `e_lfanew` | `0x00000078` |
| Machine | `AMD64` (`0x8664`) |
| Subsystem | `WINDOWS_GUI` (`0x0002`) |
| Optional Magic | `0x020B` (PE32+) |
| Linker Version | 14.0 |
| Timestamp | `0x6AF783D4` |
| Image Base | `0x0000000140000000` |
| Entry Point RVA | `0x056A1EC8` (file offset `0x56A12C8`) |
| Base of Code | `0x00001000` |
| Size of Image | `0x0B06D000` (184995840 bytes) |
| Size of Headers | `0x00000400` |
| Alignment (Section / File) | `0x1000` / `0x200` |
| COFF Characteristics | `0x0022` |
| DLL Characteristics | `0x8160` |
| Checksum | `0x072AA97F` |
| Sections | 10 |
| Standard Imports | 68 DLL(s), 587 symbol(s) |
| Delay-Load Imports | 0 DLL(s), 0 symbol(s) |
| Exports | 1 symbol(s) |
| TLS Callbacks | 2 |

## Sections

| Name | Virtual Address | Virtual Size | Raw Offset | Raw Size | Perms | Characteristics |
| --- | --- | --- | --- | --- | --- | --- |
| `.text` | `0x00001000` | `0x05F3F966` | `0x00000400` | `0x05F3FA00` | `r-x` | `0x60000020` |
| `.rdata` | `0x05F41000` | `0x008A0C0C` | `0x05F3FE00` | `0x008A0E00` | `r--` | `0x40000040` |
| `.data` | `0x067E2000` | `0x043EC398` | `0x067E0C00` | `0x00622E00` | `rw-` | `0xC0000040` |
| `.pdata` | `0x0ABCF000` | `0x0036CBAC` | `0x06E03A00` | `0x0036CC00` | `r--` | `0x40000040` |
| `.retplne` | `0x0AF3C000` | `0x000000E4` | `0x07170600` | `0x00000200` | `---` | `0x00000000` |
| `.tls` | `0x0AF3D000` | `0x00001481` | `0x07170800` | `0x00001600` | `rw-` | `0xC0000040` |
| `.voltbl` | `0x0AF3F000` | `0x00000724` | `0x07171E00` | `0x00000800` | `---` | `0x00000000` |
| `.xbld` | `0x0AF40000` | `0x000006B9` | `0x07172600` | `0x00000800` | `rw-` | `0xC0000040` |
| `.reloc` | `0x0AF41000` | `0x0012A350` | `0x07172E00` | `0x0012A400` | `r--` | `0x42000040` |
| `.xbld` | `0x0B06C000` | `0x000000DC` | `0x0729D200` | `0x00000200` | `rw-` | `0xC2000040` |

## TLS Callbacks

- Raw Data Range: `0x000000014AF3D000` .. `0x000000014AF3E480` (zero-fill: 0 bytes, characteristics: `0x00500000`)
- Index VA: `0x0000000146E08B2C` | Callbacks Array VA: `0x0000000146791A18`

| # | Callback RVA | Callback VA |
| --- | --- | --- |
| 0 | `0x055DD4F4` | `0x00000001455DD4F4` |
| 1 | `0x055E2EBC` | `0x00000001455E2EBC` |

## Imports

### `AcpHal.dll` (9 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `AcpHalAllocateShapeContexts` | hint `0x0000` |
| name | `AcpHalCreate` | hint `0x0001` |
| name | `AcpHalReleaseShapeContexts` | hint `0x0002` |
| name | `ApuAlloc` | hint `0x0003` |
| name | `ApuCreateHeap` | hint `0x0004` |
| name | `ApuFree` | hint `0x0005` |
| name | `ApuIsVirtualAddressValid` | hint `0x0007` |
| name | `ApuMapApuAddress` | hint `0x0008` |
| name | `ApuMapVirtualAddress` | hint `0x0009` |

### `ADVAPI32.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `RegCloseKey` | hint `0x0251` |
| name | `RegOpenKeyW` | hint `0x0285` |
| name | `RegQueryValueExW` | hint `0x028F` |

### `amd_fidelityfx_loader_xbox.dll` (4 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `ffxCreateContext` | hint `0x0001` |
| name | `ffxDestroyContext` | hint `0x0002` |
| name | `ffxDispatch` | hint `0x0003` |
| name | `ffxQuery` | hint `0x0004` |

### `api-ms-win-core-com-l1-1-0.dll` (7 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CoCreateGuid` | hint `0x0008` |
| name | `CoCreateInstance` | hint `0x0009` |
| name | `CoInitializeEx` | hint `0x0028` |
| name | `CoTaskMemFree` | hint `0x0043` |
| name | `CoUninitialize` | hint `0x0046` |
| name | `PropVariantClear` | hint `0x0050` |
| name | `StringFromGUID2` | hint `0x0054` |

### `api-ms-win-core-datetime-l1-1-1.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetDateFormatEx` | hint `0x0001` |
| name | `GetTimeFormatEx` | hint `0x0005` |

### `api-ms-win-core-debug-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `IsDebuggerPresent` | hint `0x0005` |
| name | `OutputDebugStringA` | hint `0x0006` |

### `api-ms-win-core-errorhandling-l1-1-0.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetLastError` | hint `0x0005` |
| name | `SetUnhandledExceptionFilter` | hint `0x000F` |
| name | `UnhandledExceptionFilter` | hint `0x0011` |

### `api-ms-win-core-file-l1-1-0.dll` (22 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateDirectoryW` | hint `0x0006` |
| name | `CreateFileW` | hint `0x000A` |
| name | `DeleteFileW` | hint `0x000F` |
| name | `FileTimeToLocalFileTime` | hint `0x0011` |
| name | `FindClose` | hint `0x0012` |
| name | `FindFirstFileW` | hint `0x001A` |
| name | `FindNextFileW` | hint `0x0020` |
| name | `FlushFileBuffers` | hint `0x0024` |
| name | `GetDiskFreeSpaceExW` | hint `0x0029` |
| name | `GetFileAttributesA` | hint `0x002F` |
| name | `GetFileAttributesExW` | hint `0x0031` |
| name | `GetFileAttributesW` | hint `0x0032` |
| name | `GetFileSizeEx` | hint `0x0035` |
| name | `GetFileTime` | hint `0x0036` |
| name | `LocalFileTimeToFileTime` | hint `0x004D` |
| name | `ReadFile` | hint `0x0051` |
| name | `RemoveDirectoryW` | hint `0x0057` |
| name | `SetEndOfFile` | hint `0x0058` |
| name | `SetFileAttributesW` | hint `0x005C` |
| name | `SetFilePointerEx` | hint `0x0060` |
| name | `SetFileTime` | hint `0x0061` |
| name | `WriteFile` | hint `0x0065` |

### `api-ms-win-core-file-l2-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `MoveFileExW` | hint `0x0009` |

### `api-ms-win-core-handle-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CloseHandle` | hint `0x0000` |

### `api-ms-win-core-heap-l2-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `LocalFree` | hint `0x000A` |

### `api-ms-win-core-interlocked-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `InitializeSListHead` | hint `0x0000` |

### `api-ms-win-core-libraryloader-l1-2-0.dll` (8 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `FreeLibrary` | hint `0x000D` |
| name | `GetModuleFileNameA` | hint `0x0010` |
| name | `GetModuleFileNameW` | hint `0x0011` |
| name | `GetModuleHandleA` | hint `0x0012` |
| name | `GetModuleHandleW` | hint `0x0015` |
| name | `GetProcAddress` | hint `0x0016` |
| name | `LoadLibraryExA` | hint `0x0018` |
| name | `LoadLibraryExW` | hint `0x0019` |

### `api-ms-win-core-libraryloader-l1-2-1.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `LoadLibraryW` | hint `0x0040` |

### `api-ms-win-core-localization-l1-2-0.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `FormatMessageA` | hint `0x0008` |
| name | `GetUserDefaultLocaleName` | hint `0x0025` |
| name | `LocaleNameToLCID` | hint `0x0036` |

### `api-ms-win-core-localization-l1-2-3.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetUserDefaultGeoName` | hint `0x0022` |

### `api-ms-win-core-memory-l1-1-0.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `VirtualAlloc` | hint `0x002F` |
| name | `VirtualFree` | hint `0x0035` |
| name | `VirtualQueryEx` | hint `0x003C` |

### `api-ms-win-core-namedpipe-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `SetNamedPipeHandleState` | hint `0x000A` |
| name | `WaitNamedPipeW` | hint `0x000C` |

### `api-ms-win-core-processenvironment-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetCommandLineA` | hint `0x0004` |

### `api-ms-win-core-processthreads-l1-1-0.dll` (12 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateThread` | hint `0x0006` |
| name | `GetCurrentProcess` | hint `0x000C` |
| name | `GetCurrentProcessId` | hint `0x000D` |
| name | `GetCurrentThreadId` | hint `0x0011` |
| name | `GetProcessId` | hint `0x001A` |
| name | `GetStartupInfoW` | hint `0x0022` |
| name | `GetThreadPriority` | hint `0x002B` |
| name | `OpenThread` | hint `0x0035` |
| name | `ResumeThread` | hint `0x003C` |
| name | `SetThreadPriority` | hint `0x004D` |
| name | `SuspendThread` | hint `0x0053` |
| name | `TerminateProcess` | hint `0x0055` |

### `api-ms-win-core-processthreads-l1-1-1.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetCurrentThreadStackLimits` | hint `0x0012` |
| name | `IsProcessorFeaturePresent` | hint `0x0032` |

### `api-ms-win-core-processthreads-l1-1-2.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `SetProcessPriorityBoost` | hint `0x0045` |

### `api-ms-win-core-processtopology-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetThreadGroupAffinity` | hint `0x0001` |
| name | `SetThreadGroupAffinity` | hint `0x0002` |

### `api-ms-win-core-profile-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `QueryPerformanceCounter` | hint `0x0000` |
| name | `QueryPerformanceFrequency` | hint `0x0001` |

### `api-ms-win-core-psm-appnotify-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `RegisterAppStateChangeNotification` | hint `0x0001` |
| name | `UnregisterAppStateChangeNotification` | hint `0x0003` |

### `api-ms-win-core-psm-appnotify-l1-1-1.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `RegisterAppConstrainedChangeNotification` | hint `0x0000` |
| name | `UnregisterAppConstrainedChangeNotification` | hint `0x0002` |

### `api-ms-win-core-rtlsupport-l1-1-0.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `RtlCaptureContext` | hint `0x0002` |
| name | `RtlLookupFunctionEntry` | hint `0x000B` |
| name | `RtlVirtualUnwind` | hint `0x0011` |

### `api-ms-win-core-string-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `MultiByteToWideChar` | hint `0x0006` |
| name | `WideCharToMultiByte` | hint `0x0007` |

### `api-ms-win-core-string-obsolete-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `lstrcmpW` | hint `0x0003` |

### `api-ms-win-core-synch-l1-1-0.dll` (17 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `AcquireSRWLockExclusive` | hint `0x0000` |
| name | `CreateEventA` | hint `0x0003` |
| name | `CreateEventW` | hint `0x0006` |
| name | `CreateMutexA` | hint `0x0007` |
| name | `CreateSemaphoreExW` | hint `0x000B` |
| name | `DeleteCriticalSection` | hint `0x000F` |
| name | `EnterCriticalSection` | hint `0x0011` |
| name | `InitializeCriticalSection` | hint `0x0018` |
| name | `InitializeCriticalSectionAndSpinCount` | hint `0x0019` |
| name | `LeaveCriticalSection` | hint `0x001D` |
| name | `ReleaseMutex` | hint `0x0023` |
| name | `ReleaseSRWLockExclusive` | hint `0x0024` |
| name | `ReleaseSemaphore` | hint `0x0026` |
| name | `ResetEvent` | hint `0x0027` |
| name | `SetEvent` | hint `0x0029` |
| name | `WaitForSingleObject` | hint `0x0036` |
| name | `WaitForSingleObjectEx` | hint `0x0037` |

### `api-ms-win-core-synch-l1-2-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `Sleep` | hint `0x002D` |

### `api-ms-win-core-synch-l1-2-1.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateSemaphoreW` | hint `0x0010` |
| name | `WaitForMultipleObjects` | hint `0x0065` |

### `api-ms-win-core-sysinfo-l1-1-0.dll` (6 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetLocalTime` | hint `0x0006` |
| name | `GetLogicalProcessorInformation` | hint `0x0007` |
| name | `GetSystemTime` | hint `0x0015` |
| name | `GetSystemTimeAsFileTime` | hint `0x0018` |
| name | `GetTickCount64` | hint `0x001D` |
| name | `GlobalMemoryStatusEx` | hint `0x0023` |

### `api-ms-win-core-sysinfo-l1-2-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `GetSystemTimePreciseAsFileTime` | hint `0x0019` |

### `api-ms-win-core-timezone-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `FileTimeToSystemTime` | hint `0x0001` |
| name | `SystemTimeToFileTime` | hint `0x000A` |

### `api-ms-win-core-windowserrorreporting-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `WerRegisterFile` | hint `0x0009` |

### `api-ms-win-crt-convert-l1-1-0.dll` (12 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_itoa` | hint `0x0017` |
| name | `_strtoi64` | hint `0x0021` |
| name | `_strtoui64` | hint `0x0027` |
| name | `atof` | hint `0x004F` |
| name | `atoi` | hint `0x0050` |
| name | `atoll` | hint `0x0052` |
| name | `strtod` | hint `0x005E` |
| name | `strtof` | hint `0x005F` |
| name | `strtol` | hint `0x0061` |
| name | `strtoll` | hint `0x0063` |
| name | `strtoul` | hint `0x0064` |
| name | `strtoull` | hint `0x0065` |

### `api-ms-win-crt-filesystem-l1-1-0.dll` (4 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_lock_file` | hint `0x0016` |
| name | `_unlock_file` | hint `0x0024` |
| name | `_wstat64` | hint `0x003C` |
| name | `remove` | hint `0x003F` |

### `api-ms-win-crt-heap-l1-1-0.dll` (5 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_set_new_mode` | hint `0x0016` |
| name | `calloc` | hint `0x0017` |
| name | `free` | hint `0x0018` |
| name | `malloc` | hint `0x0019` |
| name | `realloc` | hint `0x001A` |

### `api-ms-win-crt-locale-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_configthreadlocale` | hint `0x0008` |
| name | `localeconv` | hint `0x0012` |

### `api-ms-win-crt-math-l1-1-0.dll` (41 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `__setusermatherr` | hint `0x0009` |
| name | `_dtest` | hint `0x0019` |
| name | `_fdsign` | hint `0x0025` |
| name | `_fdtest` | hint `0x0027` |
| name | `acos` | hint `0x004A` |
| name | `acosf` | hint `0x004B` |
| name | `asin` | hint `0x004F` |
| name | `asinf` | hint `0x0050` |
| name | `atan` | hint `0x0054` |
| name | `atan2` | hint `0x0055` |
| name | `atan2f` | hint `0x0056` |
| name | `atanf` | hint `0x0057` |
| name | `cbrtf` | hint `0x0074` |
| name | `ceilf` | hint `0x007D` |
| name | `cos` | hint `0x0090` |
| name | `cosf` | hint `0x0091` |
| name | `erff` | hint `0x00B0` |
| name | `exp` | hint `0x00B2` |
| name | `exp2` | hint `0x00B3` |
| name | `exp2f` | hint `0x00B4` |
| name | `expf` | hint `0x00B6` |
| name | `fmod` | hint `0x00C9` |
| name | `fmodf` | hint `0x00CA` |
| name | `log` | hint `0x00DA` |
| name | `log10f` | hint `0x00DC` |
| name | `log2` | hint `0x00E0` |
| name | `log2f` | hint `0x00E1` |
| name | `logf` | hint `0x00E6` |
| name | `lroundf` | hint `0x00EB` |
| name | `modf` | hint `0x00ED` |
| name | `modff` | hint `0x00EE` |
| name | `nextafterf` | hint `0x00F6` |
| name | `nexttowardf` | hint `0x00F9` |
| name | `pow` | hint `0x00FE` |
| name | `powf` | hint `0x00FF` |
| name | `remainderf` | hint `0x0101` |
| name | `sin` | hint `0x0112` |
| name | `sinf` | hint `0x0113` |
| name | `sinhf` | hint `0x0115` |
| name | `tan` | hint `0x0118` |
| name | `tanf` | hint `0x0119` |

### `api-ms-win-crt-runtime-l1-1-0.dll` (25 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `__p___argc` | hint `0x0004` |
| name | `__p___argv` | hint `0x0005` |
| name | `_c_exit` | hint `0x0015` |
| name | `_cexit` | hint `0x0016` |
| name | `_configure_narrow_argv` | hint `0x0018` |
| name | `_crt_atexit` | hint `0x001E` |
| name | `_errno` | hint `0x0021` |
| name | `_exit` | hint `0x0023` |
| name | `_get_errno` | hint `0x0027` |
| name | `_get_narrow_winmain_command_line` | hint `0x002B` |
| name | `_initialize_narrow_environment` | hint `0x0033` |
| name | `_initialize_onexit_table` | hint `0x0034` |
| name | `_initterm` | hint `0x0036` |
| name | `_initterm_e` | hint `0x0037` |
| name | `_invalid_parameter_noinfo` | hint `0x0038` |
| name | `_invalid_parameter_noinfo_noreturn` | hint `0x0039` |
| name | `_register_onexit_function` | hint `0x003C` |
| name | `_register_thread_local_exe_atexit_callback` | hint `0x003D` |
| name | `_seh_filter_exe` | hint `0x0040` |
| name | `_set_app_type` | hint `0x0042` |
| name | `_set_errno` | hint `0x0045` |
| name | `_set_invalid_parameter_handler` | hint `0x0047` |
| name | `abort` | hint `0x0054` |
| name | `exit` | hint `0x0055` |
| name | `terminate` | hint `0x0067` |

### `api-ms-win-crt-stdio-l1-1-0.dll` (29 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `__acrt_iob_func` | hint `0x0000` |
| name | `__p__commode` | hint `0x0001` |
| name | `__stdio_common_vfprintf` | hint `0x0003` |
| name | `__stdio_common_vfwprintf` | hint `0x0007` |
| name | `__stdio_common_vsnwprintf_s` | hint `0x000C` |
| name | `__stdio_common_vsprintf` | hint `0x000D` |
| name | `__stdio_common_vsprintf_s` | hint `0x000F` |
| name | `__stdio_common_vsscanf` | hint `0x0010` |
| name | `__stdio_common_vswprintf` | hint `0x0011` |
| name | `__stdio_common_vswprintf_s` | hint `0x0013` |
| name | `__stdio_common_vswscanf` | hint `0x0014` |
| name | `_fseeki64` | hint `0x002F` |
| name | `_get_stream_buffer_pointers` | hint `0x0039` |
| name | `_set_fmode` | hint `0x0054` |
| name | `_wfopen` | hint `0x0062` |
| name | `fclose` | hint `0x0074` |
| name | `fflush` | hint `0x0077` |
| name | `fgetc` | hint `0x0078` |
| name | `fgetpos` | hint `0x0079` |
| name | `fopen` | hint `0x007D` |
| name | `fputc` | hint `0x007F` |
| name | `fread` | hint `0x0083` |
| name | `fseek` | hint `0x0087` |
| name | `fsetpos` | hint `0x0088` |
| name | `ftell` | hint `0x0089` |
| name | `fwrite` | hint `0x008A` |
| name | `rewind` | hint `0x0096` |
| name | `setvbuf` | hint `0x0098` |
| name | `ungetc` | hint `0x009D` |

### `api-ms-win-crt-string-l1-1-0.dll` (31 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_stricmp` | hint `0x002A` |
| name | `_strnicmp` | hint `0x0034` |
| name | `_wcsicoll` | hint `0x004C` |
| name | `_wcsnicmp` | hint `0x0054` |
| name | `isalnum` | hint `0x0064` |
| name | `isalpha` | hint `0x0065` |
| name | `isdigit` | hint `0x0068` |
| name | `islower` | hint `0x006B` |
| name | `isspace` | hint `0x006E` |
| name | `isxdigit` | hint `0x007E` |
| name | `strcat` | hint `0x0084` |
| name | `strcat_s` | hint `0x0085` |
| name | `strcmp` | hint `0x0086` |
| name | `strcpy` | hint `0x0088` |
| name | `strcpy_s` | hint `0x0089` |
| name | `strcspn` | hint `0x008A` |
| name | `strlen` | hint `0x008B` |
| name | `strncat` | hint `0x008C` |
| name | `strncmp` | hint `0x008E` |
| name | `strncpy` | hint `0x008F` |
| name | `strncpy_s` | hint `0x0090` |
| name | `strpbrk` | hint `0x0092` |
| name | `strtok` | hint `0x0094` |
| name | `strtok_s` | hint `0x0095` |
| name | `tolower` | hint `0x0097` |
| name | `toupper` | hint `0x0098` |
| name | `wcscat_s` | hint `0x009D` |
| name | `wcscmp` | hint `0x009E` |
| name | `wcscoll` | hint `0x009F` |
| name | `wcslen` | hint `0x00A3` |
| name | `wcsncpy` | hint `0x00A7` |

### `api-ms-win-crt-time-l1-1-0.dll` (9 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_difftime64` | hint `0x0011` |
| name | `_ftime64` | hint `0x0014` |
| name | `_gmtime64` | hint `0x001F` |
| name | `_gmtime64_s` | hint `0x0020` |
| name | `_localtime64_s` | hint `0x0024` |
| name | `_mkgmtime64` | hint `0x0026` |
| name | `_mktime64` | hint `0x0028` |
| name | `_time64` | hint `0x0030` |
| name | `strftime` | hint `0x0046` |

### `api-ms-win-crt-utility-l1-1-0.dll` (4 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `bsearch` | hint `0x0010` |
| name | `qsort` | hint `0x0019` |
| name | `rand` | hint `0x001B` |
| name | `srand` | hint `0x001D` |

### `bcrypt.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `BCryptCloseAlgorithmProvider` | hint `0x0002` |
| name | `BCryptGenRandom` | hint `0x001F` |
| name | `BCryptOpenAlgorithmProvider` | hint `0x0029` |

### `Bink2Scarlett.dll` (23 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `BinkClose` | hint `0x0001` |
| name | `BinkDoFrame` | hint `0x0006` |
| name | `BinkDoFrameAsyncMulti` | hint `0x0008` |
| name | `BinkDoFrameAsyncWait` | hint `0x0009` |
| name | `BinkFreeGlobals` | hint `0x000C` |
| name | `BinkGetFrameBuffersInfo` | hint `0x000E` |
| name | `BinkGetKeyFrame` | hint `0x0010` |
| name | `BinkGetPlatformInfo` | hint `0x0011` |
| name | `BinkGoto` | hint `0x0019` |
| name | `BinkNextFrame` | hint `0x001B` |
| name | `BinkOpenWithOptions` | hint `0x001E` |
| name | `BinkPause` | hint `0x0020` |
| name | `BinkRegisterFrameBuffers` | hint `0x0021` |
| name | `BinkRequestStopAsyncThreadsMulti` | hint `0x0024` |
| name | `BinkSetMemory` | hint `0x002B` |
| name | `BinkSetOSFileCallbacks` | hint `0x002C` |
| name | `BinkSetSoundSystem` | hint `0x0030` |
| name | `BinkSetVolume` | hint `0x0035` |
| name | `BinkSetWillLoop` | hint `0x0036` |
| name | `BinkShouldSkip` | hint `0x0037` |
| name | `BinkStartAsyncThread` | hint `0x0038` |
| name | `BinkWait` | hint `0x0043` |
| name | `BinkWaitStopAsyncThreadsMulti` | hint `0x0045` |

### `chad.dll` (5 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `XDspActivate` | hint `0x0000` |
| name | `XDspConnectWithMaximumStreamLimit` | hint `0x0002` |
| name | `XDspDeactivate` | hint `0x0003` |
| name | `XDspDisconnect` | hint `0x0004` |
| name | `XDspSubmitCommand` | hint `0x0005` |

### `d3d12_xs.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `D3D12ConfigureVirtualMemoryX` | hint `0x0001` |
| name | `D3D12SerializeVersionedRootSignature` | hint `0x0007` |
| name | `D3D12XboxCreateDevice` | hint `0x0009` |

### `ext-ms-win-rtcore-ntuser-message-ansi-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `DispatchMessageA` | hint `0x0000` |
| name | `PeekMessageA` | hint `0x0002` |

### `ext-ms-win-rtcore-ntuser-message-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `TranslateMessage` | hint `0x0013` |

### `ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll` (3 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateWindowExA` | hint `0x0001` |
| name | `DefWindowProcA` | hint `0x0002` |
| name | `RegisterClassExA` | hint `0x0006` |

### `ext-ms-win-rtcore-ntuser-window-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `ShowWindow` | hint `0x0011` |

### `GameInput.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| ordinal | `#1` | ordinal `1` |

### `libHttpClient.GDK.dll` (38 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `HCCleanupAsync` | hint `0x0003` |
| name | `HCHttpCallCloseHandle` | hint `0x0009` |
| name | `HCHttpCallCreate` | hint `0x000A` |
| name | `HCHttpCallDuplicateHandle` | hint `0x000B` |
| name | `HCHttpCallGetRequestUrl` | hint `0x000F` |
| name | `HCHttpCallPerformAsync` | hint `0x0010` |
| name | `HCHttpCallRequestGetHeaderAtIndex` | hint `0x0015` |
| name | `HCHttpCallRequestGetNumHeaders` | hint `0x0017` |
| name | `HCHttpCallRequestGetRequestBodyBytes` | hint `0x0018` |
| name | `HCHttpCallRequestGetRetryAllowed` | hint `0x001B` |
| name | `HCHttpCallRequestGetRetryCacheId` | hint `0x001C` |
| name | `HCHttpCallRequestGetRetryDelay` | hint `0x001D` |
| name | `HCHttpCallRequestGetTimeout` | hint `0x001E` |
| name | `HCHttpCallRequestGetTimeoutWindow` | hint `0x001F` |
| name | `HCHttpCallRequestGetUrl` | hint `0x0020` |
| name | `HCHttpCallRequestSetHeader` | hint `0x0022` |
| name | `HCHttpCallRequestSetRequestBodyBytes` | hint `0x0025` |
| name | `HCHttpCallRequestSetRequestBodyString` | hint `0x0027` |
| name | `HCHttpCallRequestSetRetryAllowed` | hint `0x0028` |
| name | `HCHttpCallRequestSetRetryCacheId` | hint `0x0029` |
| name | `HCHttpCallRequestSetRetryDelay` | hint `0x002A` |
| name | `HCHttpCallRequestSetTimeout` | hint `0x002C` |
| name | `HCHttpCallRequestSetTimeoutWindow` | hint `0x002D` |
| name | `HCHttpCallRequestSetUrl` | hint `0x002E` |
| name | `HCHttpCallResponseGetHeader` | hint `0x0032` |
| name | `HCHttpCallResponseGetHeaderAtIndex` | hint `0x0033` |
| name | `HCHttpCallResponseGetNetworkErrorCode` | hint `0x0034` |
| name | `HCHttpCallResponseGetNumHeaders` | hint `0x0035` |
| name | `HCHttpCallResponseGetPlatformNetworkErrorMessage` | hint `0x0036` |
| name | `HCHttpCallResponseGetResponseBodyBytes` | hint `0x0037` |
| name | `HCHttpCallResponseGetResponseBodyBytesSize` | hint `0x0038` |
| name | `HCHttpCallResponseGetResponseString` | hint `0x003A` |
| name | `HCHttpCallResponseGetStatusCode` | hint `0x003B` |
| name | `HCHttpCallSetTracing` | hint `0x0046` |
| name | `HCInitialize` | hint `0x0048` |
| name | `HCSettingsGetTraceLevel` | hint `0x005C` |
| name | `HCTraceImplMessage` | hint `0x005F` |
| name | `HCTraceSetEtwEnabled` | hint `0x0064` |

### `MFPlat.DLL` (6 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `MFCreateAttributes` | hint `0x001C` |
| name | `MFCreateMediaType` | hint `0x002B` |
| name | `MFCreateSourceResolver` | hint `0x0032` |
| name | `MFGetSystemTime` | hint `0x0054` |
| name | `MFShutdown` | hint `0x007B` |
| name | `MFStartup` | hint `0x007C` |

### `MFReadWrite.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `MFCreateSourceReaderFromMediaSource` | hint `0x0005` |

### `MSVCP140.dll` (113 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `??0?$basic_ios@DU?$char_traits@D@std@@@std@@IEAA@XZ` | hint `0x000C` |
| name | `??0?$basic_iostream@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z` | hint `0x0013` |
| name | `??0?$basic_iostream@_WU?$char_traits@_W@std@@@std@@QEAA@PEAV?$basic_streambuf@_WU?$char_traits@_W@std@@@1@@Z` | hint `0x0017` |
| name | `??0?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@_N@Z` | hint `0x001A` |
| name | `??0?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@_N@Z` | hint `0x0025` |
| name | `??0?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAA@XZ` | hint `0x002F` |
| name | `??0?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@IEAA@XZ` | hint `0x0035` |
| name | `??0_Locinfo@std@@QEAA@HPEBD@Z` | hint `0x006B` |
| name | `??0_Lockit@std@@QEAA@H@Z` | hint `0x006D` |
| name | `??0ios_base@std@@IEAA@XZ` | hint `0x0077` |
| name | `??1?$basic_ios@DU?$char_traits@D@std@@@std@@UEAA@XZ` | hint `0x007D` |
| name | `??1?$basic_ios@_WU?$char_traits@_W@std@@@std@@UEAA@XZ` | hint `0x007F` |
| name | `??1?$basic_iostream@DU?$char_traits@D@std@@@std@@UEAA@XZ` | hint `0x0080` |
| name | `??1?$basic_iostream@_WU?$char_traits@_W@std@@@std@@UEAA@XZ` | hint `0x0082` |
| name | `??1?$basic_istream@DU?$char_traits@D@std@@@std@@UEAA@XZ` | hint `0x0083` |
| name | `??1?$basic_ostream@DU?$char_traits@D@std@@@std@@UEAA@XZ` | hint `0x0086` |
| name | `??1?$basic_streambuf@DU?$char_traits@D@std@@@std@@UEAA@XZ` | hint `0x0089` |
| name | `??1?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@UEAA@XZ` | hint `0x008B` |
| name | `??1_Locinfo@std@@QEAA@XZ` | hint `0x00A4` |
| name | `??1_Lockit@std@@QEAA@XZ` | hint `0x00A5` |
| name | `??1ios_base@std@@UEAA@XZ` | hint `0x00AC` |
| name | `??4?$_Yarn@D@std@@QEAAAEAV01@PEBD@Z` | hint `0x00B1` |
| name | `??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z` | hint `0x00CC` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z` | hint `0x00FF` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@I@Z` | hint `0x0100` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@J@Z` | hint `0x0101` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z` | hint `0x0103` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z` | hint `0x010A` |
| name | `??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@_K@Z` | hint `0x010C` |
| name | `??Bid@locale@std@@QEAA_KXZ` | hint `0x0131` |
| name | `??_7?$basic_ios@DU?$char_traits@D@std@@@std@@6B@` | hint `0x0133` |
| name | `??_7?$basic_ios@_WU?$char_traits@_W@std@@@std@@6B@` | hint `0x0135` |
| name | `??_7ios_base@std@@6B@` | hint `0x015B` |
| name | `?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z` | hint `0x01AB` |
| name | `?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z` | hint `0x01AD` |
| name | `?_Getcat@?$codecvt@DDU_Mbstatet@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z` | hint `0x01B1` |
| name | `?_Getcat@?$ctype@_W@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z` | hint `0x01B8` |
| name | `?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ` | hint `0x01D5` |
| name | `?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ` | hint `0x01E8` |
| name | `?_Id_cnt@id@locale@std@@0HA` | hint `0x01EE` |
| name | `?_Init@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAXXZ` | hint `0x01F5` |
| name | `?_Init@locale@std@@CAPEAV_Locimp@12@_N@Z` | hint `0x020F` |
| name | `?_Ios_base_dtor@ios_base@std@@CAXPEAV12@@Z` | hint `0x0218` |
| name | `?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z` | hint `0x0219` |
| name | `?_Lock@?$basic_streambuf@DU?$char_traits@D@std@@@std@@UEAAXXZ` | hint `0x0228` |
| name | `?_Lock@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@UEAAXXZ` | hint `0x022A` |
| name | `?_Makeloc@_Locimp@locale@std@@CAPEAV123@AEBV_Locinfo@3@HPEAV123@PEBV23@@Z` | hint `0x023A` |
| name | `?_New_Locimp@_Locimp@locale@std@@CAPEAV123@_N@Z` | hint `0x0243` |
| name | `?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ` | hint `0x0244` |
| name | `?_Osfx@?$basic_ostream@_WU?$char_traits@_W@std@@@std@@QEAAXXZ` | hint `0x0246` |
| name | `?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ` | hint `0x024A` |
| name | `?_Pninc@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@IEAAPEA_WXZ` | hint `0x024C` |
| name | `?_Random_device@std@@YAIXZ` | hint `0x025D` |
| name | `?_Syserror_map@std@@YAPEBDH@Z` | hint `0x0273` |
| name | `?_Throw_C_error@std@@YAXH@Z` | hint `0x0274` |
| name | `?_Unlock@?$basic_streambuf@DU?$char_traits@D@std@@@std@@UEAAXXZ` | hint `0x027F` |
| name | `?_Unlock@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@UEAAXXZ` | hint `0x0281` |
| name | `?_Winerror_map@std@@YAHH@Z` | hint `0x0285` |
| name | `?_Xbad_alloc@std@@YAXXZ` | hint `0x028B` |
| name | `?_Xbad_function_call@std@@YAXXZ` | hint `0x028C` |
| name | `?_Xinvalid_argument@std@@YAXPEBD@Z` | hint `0x028D` |
| name | `?_Xlength_error@std@@YAXPEBD@Z` | hint `0x028E` |
| name | `?_Xout_of_range@std@@YAXPEBD@Z` | hint `0x028F` |
| name | `?_Xruntime_error@std@@YAXPEBD@Z` | hint `0x0292` |
| name | `?always_noconv@codecvt_base@std@@QEBA_NXZ` | hint `0x029D` |
| name | `?classic@locale@std@@SAAEBV12@XZ` | hint `0x02A4` |
| name | `?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z` | hint `0x02A6` |
| name | `?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ` | hint `0x0368` |
| name | `?flush@?$basic_ostream@_WU?$char_traits@_W@std@@@std@@QEAAAEAV12@XZ` | hint `0x036A` |
| name | `?getloc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEBA?AVlocale@2@XZ` | hint `0x03C0` |
| name | `?getloc@ios_base@std@@QEBA?AVlocale@2@XZ` | hint `0x03C3` |
| name | `?good@ios_base@std@@QEBA_NXZ` | hint `0x03C5` |
| name | `?id@?$codecvt@DDU_Mbstatet@@@std@@2V0locale@2@A` | hint `0x03C9` |
| name | `?id@?$ctype@_W@std@@2V0locale@2@A` | hint `0x03D1` |
| name | `?imbue@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAA?AVlocale@2@AEBV32@@Z` | hint `0x03F3` |
| name | `?imbue@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAAXAEBVlocale@2@@Z` | hint `0x03F6` |
| name | `?imbue@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAAXAEBVlocale@2@@Z` | hint `0x03F8` |
| name | `?in@?$codecvt@DDU_Mbstatet@@@std@@QEBAHAEAU_Mbstatet@@PEBD1AEAPEBDPEAD3AEAPEAD@Z` | hint `0x03FA` |
| name | `?init@?$basic_ios@DU?$char_traits@D@std@@@std@@IEAAXPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@_N@Z` | hint `0x0402` |
| name | `?out@?$codecvt@DDU_Mbstatet@@@std@@QEBAHAEAU_Mbstatet@@PEBD1AEAPEBDPEAD3AEAPEAD@Z` | hint `0x0433` |
| name | `?read@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@PEAD_J@Z` | hint `0x048D` |
| name | `?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ` | hint `0x0495` |
| name | `?seekg@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@V?$fpos@U_Mbstatet@@@2@@Z` | hint `0x049E` |
| name | `?seekg@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@_JH@Z` | hint `0x049F` |
| name | `?setbuf@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAAPEAV12@PEAD_J@Z` | hint `0x04B5` |
| name | `?setbuf@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAAPEAV12@PEA_W_J@Z` | hint `0x04B7` |
| name | `?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z` | hint `0x04C5` |
| name | `?setstate@?$basic_ios@_WU?$char_traits@_W@std@@@std@@QEAAXH_N@Z` | hint `0x04C9` |
| name | `?showmanyc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAA_JXZ` | hint `0x04D5` |
| name | `?showmanyc@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAA_JXZ` | hint `0x04D7` |
| name | `?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z` | hint `0x04DE` |
| name | `?sputc@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@QEAAG_W@Z` | hint `0x04E0` |
| name | `?sync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAAHXZ` | hint `0x04FD` |
| name | `?sync@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAAHXZ` | hint `0x04FF` |
| name | `?tellg@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA?AV?$fpos@U_Mbstatet@@@2@XZ` | hint `0x0503` |
| name | `?tellp@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA?AV?$fpos@U_Mbstatet@@@2@XZ` | hint `0x0506` |
| name | `?uflow@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAAHXZ` | hint `0x051B` |
| name | `?uflow@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAAGXZ` | hint `0x051D` |
| name | `?uncaught_exception@std@@YA_NXZ` | hint `0x051E` |
| name | `?unshift@?$codecvt@DDU_Mbstatet@@@std@@QEBAHAEAU_Mbstatet@@PEAD1AEAPEAD@Z` | hint `0x0527` |
| name | `?widen@?$ctype@_W@std@@QEBA_WD@Z` | hint `0x053C` |
| name | `?write@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@PEBD_J@Z` | hint `0x053F` |
| name | `?xsgetn@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAA_JPEAD_J@Z` | hint `0x0543` |
| name | `?xsgetn@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAA_JPEA_W_J@Z` | hint `0x0545` |
| name | `?xsputn@?$basic_streambuf@DU?$char_traits@D@std@@@std@@MEAA_JPEBD_J@Z` | hint `0x0546` |
| name | `?xsputn@?$basic_streambuf@_WU?$char_traits@_W@std@@@std@@MEAA_JPEB_W_J@Z` | hint `0x0548` |
| name | `_Mtx_destroy_in_situ` | hint `0x0581` |
| name | `_Mtx_init_in_situ` | hint `0x0584` |
| name | `_Mtx_lock` | hint `0x0585` |
| name | `_Mtx_unlock` | hint `0x0589` |
| name | `_Query_perf_counter` | hint `0x0590` |
| name | `_Query_perf_frequency` | hint `0x0591` |
| name | `_Xtime_get_ticks` | hint `0x05CC` |

### `VCRUNTIME140.dll` (23 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `_CxxThrowException` | hint `0x0001` |
| name | `__C_specific_handler` | hint `0x0008` |
| name | `__CxxFrameHandler3` | hint `0x000E` |
| name | `__RTDynamicCast` | hint `0x0018` |
| name | `__current_exception` | hint `0x001B` |
| name | `__current_exception_context` | hint `0x001C` |
| name | `__intrinsic_setjmp` | hint `0x001D` |
| name | `__std_exception_copy` | hint `0x0021` |
| name | `__std_exception_destroy` | hint `0x0022` |
| name | `__std_terminate` | hint `0x0023` |
| name | `__std_type_info_compare` | hint `0x0024` |
| name | `_purecall` | hint `0x0036` |
| name | `_set_purecall_handler` | hint `0x0037` |
| name | `longjmp` | hint `0x0039` |
| name | `memchr` | hint `0x003A` |
| name | `memcmp` | hint `0x003B` |
| name | `memcpy` | hint `0x003C` |
| name | `memmove` | hint `0x003D` |
| name | `memset` | hint `0x003E` |
| name | `strchr` | hint `0x0040` |
| name | `strrchr` | hint `0x0041` |
| name | `strstr` | hint `0x0042` |
| name | `wcsrchr` | hint `0x0045` |

### `VCRUNTIME140_1.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `__CxxFrameHandler4` | hint `0x0000` |

### `WINHTTP.dll` (15 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `WinHttpAddRequestHeaders` | hint `0x0004` |
| name | `WinHttpCloseHandle` | hint `0x0008` |
| name | `WinHttpConnect` | hint `0x0009` |
| name | `WinHttpCrackUrl` | hint `0x0018` |
| name | `WinHttpOpen` | hint `0x0031` |
| name | `WinHttpOpenRequest` | hint `0x0032` |
| name | `WinHttpQueryDataAvailable` | hint `0x003A` |
| name | `WinHttpQueryHeaders` | hint `0x003B` |
| name | `WinHttpReadData` | hint `0x003E` |
| name | `WinHttpReceiveResponse` | hint `0x0042` |
| name | `WinHttpSendRequest` | hint `0x0048` |
| name | `WinHttpSetOption` | hint `0x004B` |
| name | `WinHttpSetStatusCallback` | hint `0x004E` |
| name | `WinHttpSetTimeouts` | hint `0x004F` |
| name | `WinHttpWriteData` | hint `0x0059` |

### `WS2_32.dll` (25 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| ordinal | `#111` | ordinal `111` |
| ordinal | `#115` | ordinal `115` |
| ordinal | `#151` | ordinal `151` |
| ordinal | `#1` | ordinal `1` |
| ordinal | `#2` | ordinal `2` |
| ordinal | `#3` | ordinal `3` |
| ordinal | `#4` | ordinal `4` |
| name | `freeaddrinfo` | hint `0x00A5` |
| name | `getaddrinfo` | hint `0x00A6` |
| ordinal | `#52` | ordinal `52` |
| ordinal | `#57` | ordinal `57` |
| ordinal | `#6` | ordinal `6` |
| ordinal | `#7` | ordinal `7` |
| ordinal | `#9` | ordinal `9` |
| ordinal | `#11` | ordinal `11` |
| ordinal | `#10` | ordinal `10` |
| ordinal | `#13` | ordinal `13` |
| ordinal | `#16` | ordinal `16` |
| ordinal | `#17` | ordinal `17` |
| ordinal | `#18` | ordinal `18` |
| ordinal | `#19` | ordinal `19` |
| ordinal | `#20` | ordinal `20` |
| ordinal | `#21` | ordinal `21` |
| ordinal | `#22` | ordinal `22` |
| ordinal | `#23` | ordinal `23` |

### `xapu.dll` (5 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `XApuConnect` | hint `0x0000` |
| name | `XApuDequeueResult` | hint `0x0001` |
| name | `XApuDisconnect` | hint `0x0002` |
| name | `XApuEnqueueCommand` | hint `0x0003` |
| name | `XApuGetResultCount` | hint `0x0004` |

### `XboxBvhBuilder.dll` (7 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z` | hint `0x0000` |
| name | `??1XboxBvhBuilder@@QEAA@XZ` | hint `0x0001` |
| name | `?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z` | hint `0x0003` |
| name | `?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z` | hint `0x0004` |
| name | `?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z` | hint `0x0006` |
| name | `?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z` | hint `0x0007` |
| name | `?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z` | hint `0x000A` |

### `XCRDAPI.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `DStorageGetFactory` | hint `0x0000` |

### `xg_xs.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `XGCreateTextureComputer` | hint `0x000C` |
| name | `XGSuggestSwizzleMode` | hint `0x0017` |

### `xmem.dll` (12 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `XMemAlloc` | hint `0x0000` |
| name | `XMemAllocDefault` | hint `0x0001` |
| name | `XMemAllocatePhysicalPages` | hint `0x0002` |
| name | `XMemFree` | hint `0x0005` |
| name | `XMemFreeDefault` | hint `0x0006` |
| name | `XMemFreePhysicalPages` | hint `0x0007` |
| name | `XMemGetWorkingSetStatistics` | hint `0x0010` |
| name | `XMemIsToolsMemAvailable` | hint `0x0013` |
| name | `XMemMapPhysicalPages` | hint `0x0015` |
| name | `XMemMapPhysicalPagesScatter` | hint `0x0016` |
| name | `XMemTransferMemory` | hint `0x0021` |
| name | `XMemVirtualAlloc` | hint `0x0022` |

## Delay-Load Imports

_No delay-load imports._

## Exports

- Module Name: `game_scarlett_master.exe` | Ordinal Base: 1

| Ordinal | RVA | Name | Forwarder |
| --- | --- | --- | --- |
| 1 | `0x06844380` | `__XMemAllocationHooks__` | - |
