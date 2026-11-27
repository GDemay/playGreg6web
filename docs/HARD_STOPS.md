# Hard stops: store, licence, entitlement, attestation, sign-in and user-account entry points

Status: inventory only. **Nothing here is to be implemented by agents.** The repo owner decides policy
for every row marked "owner decision". Sources: `runtime/os/xgameruntime_provider.cpp` (COM slot tables,
header comment lines ~62-110), `runtime/os/xbox_offline.cpp` (C-ABI shim exports), `runtime/os/xgameruntime.cpp`,
and the comments on issue #663. No game bytes and no extracted or decrypted data are recorded here.

"Stub" in the provider means a trapping stub: if the title calls the slot, the guest stops with an
unimplemented-import report. "Offline HRESULT" means `0x800704CF` (`HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE)`,
`kOfflineHResult` in `runtime/os/xbox_offline.h`).

## 1. XStore (`xgameruntime.XStore`, 96-slot table)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 3 | XStoreCreateContext | Implemented: local dummy context handle | #235 (O16) |
| 4 | XStoreCloseContextHandle | Implemented | #235 (O16) |
| 28 | XStoreQueryGameLicenseAsync | Synchronous offline HRESULT `0x800704CF`, no argument touched, no completion queued (title runs as the full game). Decided 2026-11-27 by owner (7.3 A) | #2031 |
| 65 | XStoreQueryPackageIdentifier | `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`, no argument touched (add-on unavailable). Decided 2026-11-27 by owner (7.3 A) | #2031 |
| 66 / 67 | XStoreRegisterGameLicenseChanged / Unregister | Trapping stub (the C-ABI shim in `xbox_offline.cpp` has a silent version, but the title path does not use it) | #235 |
| 68 / 69 | XStoreRegisterPackageLicenseLost / Unregister | Trapping stub | #235 |
| other (about 40 used slots, e.g. 20/21 AcquireLicenseForPackage, 22 IsLicenseValid, 29 QueryGameLicenseResult) | various | Trapping stub, "nothing about licences is invented" (O8 offline policy); a tripwire per 7.3 A | #235 / #77 |

C-ABI shim (`xbox_offline.cpp`, XGR exports): XStoreQueryGameLicense, QueryAddOnLicenses, QueryAssociatedProducts,
QueryProductForCurrentGame, Register/UnregisterGameLicenseChanged all complete with the offline HRESULT (no
licence data). Set by #77 (O8).

## 2. XStoreExt (`xgameruntime.XStoreExt`)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 71 | XStoreAcquireLicenseForDurablesAsync(ctx, storeId, async) | Synchronous `E_NOTIMPL`, logs "offline mode", claims zero ownership; the title maps it to state 4 (unavailable) | #856 (O37 / #849); owner sign-off "Option A" on #663 |
| 72 | XStoreAcquireLicenseForDurablesResult | Same, clears the out handle | #856 |
| all other | | Trapping stub | |

Open question for the owner: the research comment on #663 lists alternatives (`E_GAMESTORE_NETWORK_ERROR 0x89245301`,
`0x803F9009`). Codes that assert ownership or sign-in state (`0x803F8001`, `0x803F9006`, `0x803F8008`) or return a
licence handle are entitlement claims and must not be used without an explicit decision.

## 3. XUser (`xgameruntime.XUser`, `XUserExt`)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 3 / 4 | XUserDuplicateHandle / CloseHandle | Implemented | #235 |
| 5 | XUserCompare | Implemented: 0 when both handles are the local user (by XUID, so duplicates match), non-zero otherwise. Decided 2026-11-27 by owner (7.2) | #2031 |
| 7 / 8 | XUserAddAsync / AddResult | Implemented: one local offline user ("OFFLINEPLAYER"), no account picker, no real sign-in | #36 (O5), #235 |
| 9 | XUserGetLocalId | Implemented | #235 |
| 10 | XUserFindUserByLocalId | Implemented: a new handle when the id is our local id and the user was added, else `E_GAMEUSER_USER_NOT_FOUND`. Decided 2026-11-27 by owner (7.2) | #2031 |
| 11 | XUserGetId (XUID) | Implemented (local value) | #235 |
| 12 | XUserFindUserById | Implemented: same, by our local XUID. Decided 2026-11-27 by owner (7.2) | #2031 |
| 13 | XUserGetIsGuest | Implemented: not a guest | #487 (R23) |
| 14 | XUserGetState | Implemented: SignedIn for the single local offline user, `E_INVALIDARG` for an invalid handle; no token, no online access. Decided 2026-11-27 by owner (7.2) | #2031 |
| 19 | XUserGetAgeGroup | Implemented: `--user-age-group`, default Unknown | #514 (R23) |
| 20 | XUserCheckPrivilege | Implemented: no privilege, deny reason Unknown | #235 / #77 |
| 21 / 22 | XUserResolvePrivilegeWithUiAsync / Result | Implemented: offline HRESULT | #235 / #77 |
| 23 | XUserGetTokenAndSignatureAsync | Implemented: completes with offline HRESULT, no token | #235 / #77 |
| 24 | XUserGetTokenAndSignatureResultSize | Implemented: offline HRESULT | #235 / #77 |
| 25 | XUserGetTokenAndSignatureResult | **Trapping stub** in the provider (the C-ABI shim version returns a null buffer) | #235 |
| 33 / 34 | XUserRegisterForChangeEvent / Unregister | Implemented (no events fire) | #235 |
| XUserExt 3 | XUserGetGamertag | Implemented (local name) | #235 |

Xbox Live (XSAPI, `XblContextCreateHandle` etc.) answers with the offline HRESULT through `xbox_offline.cpp` (#77).

## 4. XSystem, XGame, XPackage

| Slot | Function | Today | Set by |
|---|---|---|---|
| XSystem 3 | XSystemGetConsoleId | Implemented | #235 |
| XSystem 4 | XSystemGetXboxLiveSandboxId | Configured sandbox, else `E_NOTIMPL` | #235 |
| XSystem 5 | XSystemGetAppSpecificDeviceId | `E_NOTIMPL`, nothing written (no device identity). Decided 2026-11-27 by owner (7.1 A) | #2031 |
| XGame 3 | XGameGetXboxTitleId | Implemented | #235 |
| XPackage 3/4 | GetCurrentProcessPackageIdentifier / IsPackagedProcess | Implemented: deterministic local id (32 zero digits), packaged | #235 |
| XPackage 5/6/7/9 | Installation monitor and progress | Implemented: always complete and launchable | #235 |
| XPackage 39 | XPackageEnumeratePackages | Content kind: none; game-package kind: `E_NOTIMPL` | #235 |
| XPackage 11/12/14/16/17/18/19 | UserLocale, FindChunkAvailability, ChangeChunkInstallOrder, InstallChunks(+Result), EstimateDownloadSize, UninstallChunks | Implemented for a fully installed package: locale en-US, chunks Ready, install completes at once, estimate 0 bytes, uninstall accepted (nothing removed) | R-xpackage |
| XPackage 34 | bool(name) "already installed" query (call site: one string, low byte of result, true skips a start-install branch; not an uninstall, public API not named with certainty) | true only for the title's own package identifier, false for null/other names; no licence or ownership claim | R-xpackage |
| XPackage 37/38/25/26 | MountWithUiAsync(+Result), GetMountPath, CloseMountHandle (slots read from the title's mount wrapper: identifier + XAsyncBlock, mount handle, 0x100-byte path buffer) | Implemented for the title's own installed package only: mounts at the game-data root `G:\`, no licence asserted; any other identifier is `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`. Not served (and not claimed): enumerating owned/other packages as entitlement | R-xpackage |
| XPersistentLocalStorage 3/5/6/7 | GetPathSize, GetSpaceInfo (used from real contents of the storage dir, 2048 MB quota), EnsureSpace(+Result) (E_ABORT when it does not fit) | Implemented; a too-small path buffer is `E_FAIL` | R-xpackage |
| XNetworking 3/4/5 | QueryPreferredLocalUdpMultiplayerPort (sync, Async, AsyncResult) | Offline: `0x800704CF` (network unreachable), no argument touched, no completion queued; no port is claimed | R-xpackage |
| XNetworking 8/9/10 | QuerySecurityInformationForUrlAsync, ...ResultSize, ...Result | **Hard stop:** security/certificate information is never invented and no URL is contacted; `0x800704CF` | R-xpackage |
| XNetworking 14 | VerifyServerCertificate | **Hard stop:** never vouches for a server; `0x800704CF` | R-xpackage |
| XGameRuntimeIsFeatureAvailable | static survey: the title asks only ids 10 (XNetworking) and 20 (Events, telemetry) through one wrapper with two call sites; ids use the public zero-based enum | true only for 10 (GetConnectivityHint, offline). **Always false (hard stop, owner decides):** 20 Events, store 14, system 15, thread 17, user 18 (sign-in state), game streaming, game UI, invites, launcher, app capture, accessibility, any unknown id | R-xpackage |

XGame 4 XLaunchNewGame is no longer listed: relaunching the running title is a process restart, not a store,
licence or entitlement call. When the target is the title itself the guest stops with `kRelaunchSelfExitCode`
(`runtime/os/win32_compat.h`) and `tools/run_native_boot.py` / the browser boot relaunch it with the same
arguments; launching any other title returns `E_NOTIMPL`.

There is no package-licence or entitlement answer anywhere: no slot returns a licence handle or "owned".

## 5. What the boot actually hits (from #663)

- **XStoreExt slot 71**: the first hard stop, after about 14 s (18.4 s in the run where the memory-model fixes
  landed). After #856 it fails synchronously and the boot continues (next stop was an unrelated title assertion
  at 27-29 s, fixed in #892).
- **XUserAddAsync and XGameSaveInitializeProviderAsync**: reached past the Start Screen (the "OFFLINEPLAYER (A)" entry),
  both answered locally.
- **No store or licence call later in the boot.** A native menu-flow run to 240 s states "No store/licence call".
  I found no comment on #663 reporting a store/licence call at about 73 s; if one exists elsewhere, add it here.
- **GetAppSpecificDeviceId (XSystem slot 5)**: every boot now reaches it and stops there, at about 62-67 s (see 7.1).
- Token/signature: no run reports the boot reaching it (slot 25 cannot be reached while slot 24 fails, see 7.2).
  Their stubs would trap on first call, so a stop at them would be visible as an unimplemented-import fault.

## 6. Owner decisions pending

1. Whether XUserGetTokenAndSignatureResult (slot 25) should match the offline HRESULT of slots 23/24 rather than trap.
2. ~~What XSystemGetAppSpecificDeviceId should return~~: decided 2026-11-27 by owner, implemented in #2031 (7.1).
3. Which HRESULT slot 71/72 should settle on long term (see section 2).
4. XStore slots: 28 and 65 decided 2026-11-27 by owner, implemented in #2031 (7.3); 20/21/22/68 stay trapping tripwires.
5. S7: decided 2026-11-27, implemented in #2029: Media Foundation object-creating imports (MFCreateAttributes, MFCreateMediaType, MFCreateSourceResolver,
   MFCreateSourceReaderFromMediaSource) are `ok` failing shims in `runtime/os/shim_audit.h` (section 7.4).
6. S8: CRT `fopen` / `_wfopen` / `remove` (and the msvcp140 `_fsopen` / `_wfsopen`) are `ok` since #1998; no owner
   decision is left (section 7.5).

## 7. Owner decisions: device id, XUser sign-in state and XStore licence

Behaviour-only summary of a static analysis of the title's call sites. No addresses, bytes or extracted data
are recorded. 7.1, 7.2 and 7.3 were decided 2026-11-27 by the owner (option A / the recommended rows) and implemented in #2031.

### 7.1 XSystemGetAppSpecificDeviceId (XSystem slot 5)

What the title does:
- One call site, inside a step that builds a single text value from four parts. The fourth part is the console
  id if that equals the title's all-zero placeholder, otherwise the app-specific device id.
- Any error return (including "not implemented") or a missing interface: the step returns quietly and adds
  nothing. No retry, no abort, no fallback.
- On success it adds the id only if it equals an all-zero placeholder; any other value is read and discarded.
- No hashing, network send, save or key use was found for the id (static analysis; an indirect read cannot be
  fully excluded). Reference runtimes return fixed or per-configuration ids, which the title discards.

| Option | Consequence |
|---|---|
| A. Return a documented error, write nothing (recommended) | Boot continues, no device identity is claimed; matches the offline "nothing to report" pattern. |
| B. Fixed fake id | Same visible result; puts a made-up identity value into the runtime, identical for every player. |
| C. Per-installation derived id | Same visible result; introduces a stable machine identifier for no benefit. |
| D. All-zero placeholder | No benefit; feeds a value nobody reads. |
| E. Leave the trap | Boot stops (not viable). |

Recommendation: **A**. The only caller ignores errors by design.

**Decided 2026-11-27 by owner, implemented in #2031:** option A, `E_NOTIMPL`, nothing written.

#### 7.1 Evidence (static re-check of the real executable, read-only)

The boot now stops here every run, at about 62-67 s.
- **Call site.** The slot 5 wrapper (QueryApiImpl, one vtable call, Release) has exactly **one** caller. That caller
  passes a zeroed 45-byte buffer and a "bytes used" out value.
- **Branch.** The return value is tested for `== 0`, not just for its sign. Any non-zero HRESULT skips the step
  silently: `E_NOTIMPL`, any other error, `S_FALSE`, or the wrapper's own QueryApiImpl failure. On `S_OK` the title
  compares `min(used, 45)` bytes with a 44-digit all-zero constant and appends the id only when they match exactly.
  The step does not retry, log, raise a dialog, quit, or take any other path.
- **Why it is called at all.** The step first asks XSystem slot 3 (console id, 39-byte buffer). It appends the
  console id and skips the device id only when the console id equals the title's *dotted* all-zero placeholder
  (zero digit groups joined by dots). Our slot 3 returns 38 zeros with no dots, which never matches, so the device id is
  always requested.
- **Consumer.** The step's output (3 or 4 parts, `;`-separated, together with a per-user value and two small
  numbers) goes through a base-36-style text encoder and is stored in **one** global. That global has exactly four
  code references: a "built yet?" check, the store, and a shutdown routine that frees it and clears it. Nothing else
  reads it, so "the built text is never read" now has **high** confidence. The only thing static analysis cannot
  rule out is a read through pointer arithmetic.
- **When.** The value is built lazily, once, at the tail of the main game loop's iteration. That fits a stop about
  60 s in: the first full main-loop pass after the boot.
- **What comes next.** After the step returns, the loop moves to its next iteration. The loop makes no other
  XSystem call; slots 3 and 4 are already implemented. The next trap will come from another subsystem. The
  candidates that are statically reachable are the XUser slots in 7.2.
- **Consequence per option.** A: three parts, encoded, stored, never read; the boot continues. B/C: the value is
  compared with the constant and dropped, the same as A. D: one more part that nobody reads. E: the boot stops.
  None of the options leads to a fatal or "offline" UI path.
- Not an option to use instead of A, listed only for completeness: if slot 3 returned the dotted placeholder format,
  the device id would never be asked for. That would change a different API's value just to steer the title's
  control flow.

### 7.2 XUser sign-in state for an offline local user (slots 14, 10, 12, 5; backlog item 17)

What the title does (all four are purely local lookups; none leads to a network call, and every online path
needs a token from slot 23, which already fails offline):
- Slot 14 (get sign-in state): polled from the user manager's update routine. Success with "signed in" keeps
  the user; an error or "signed out" makes the title treat the user as signed out and clear that user's
  pending items.
- Slots 10 / 12 (find user by local id / by XUID): return codes are ignored; an empty reference is tolerated.
- Slot 5 (compare users): "same user" on 0. Our duplicate-handle call yields a new handle object, so the
  answer must compare the underlying user, not the handle value.

| Item | Option | Consequence |
|---|---|---|
| Slot 14 | A. "Signed in" for the single local user, invalid-handle error otherwise (recommended) | Local session claim only; no tokens, no online access (slot 23 still fails offline). Matches all reference runtimes. |
| | B. "Signed out" or an error | Title keeps treating the user as signed out and clearing its items; likely breaks saves/settings (not verified dynamically). |
| | C. Leave the trap | Boot stops soon after the user is added. |
| Slots 10 / 12 | A. Handle on match with our local id / XUID, else "no such user" (recommended) | Consistent with slots 9 and 11. |
| | B. Always "no such user" | Tolerated, but the user manager cannot rebuild the record for our own user. |
| Slot 5 | A. "Same" when both handles are the same local user, by id (recommended) | Correct with duplicate handles. |
| | B. Compare by handle value | Duplicates count as different users; possible spurious "user changed" handling. |

Recommendation: slot 14 = signed in for the single local offline user, 10/12/5 as in the recommended rows;
no tokens, no online services.

**Decided 2026-11-27 by owner, implemented in #2031:** slot 14 SignedIn for the single local offline user; slots 10/12 find by our local id / XUID; slot 5 compares by XUID; no tokens (slot 24 keeps failing offline).

Confidence: high for the call sites and error handling; medium for the exact meaning of the cleared list
("the built text is never read" is now high, see 7.1 evidence). No dynamic run backs this.

#### 7.2 Evidence (static re-check of the real executable, read-only)

Slot numbers follow the public header order from slot 3: Duplicate, Close, Compare, GetMaxUsers, AddAsync,
AddResult, GetLocalId, FindUserByLocalId, GetId, FindUserById, GetIsGuest, **GetState (14)**. GetGamertag lives on
the XUser extension.
- **Slot 14, XUserGetState.** One routine, four call sites, one for each of the user manager's four local-user
  slots. At each site: if the handle is valid, the state is preset to 2 (SignedOut) and the call is made. If the
  HRESULT is negative **or** the state is 2, the routine releases every pending item in that user's list (Release
  on each, count set to 0). States 0 (SignedIn) and 1 (SigningOut) keep the user. The routine raises no dialog,
  does not quit, and does not call AddAsync again. A "signed out" or error answer therefore stays silent, but it
  repeats on every poll.
- **When slot 14 runs.** It is polled from the user manager's per-frame update, only after a deadline that one
  virtual method of the manager sets to "now + 500 ms". Static analysis did not resolve who calls that method. The
  same vtable holds the change-event registration (slot 33) and the change-event handler, so user-change handling
  probably arms the poll. Our runtime never fires change events, so slot 14 may not be reached in today's boot at
  all. If it is reached, it comes from this routine.
- **Slots 10 / 12.** Each is called from one small "look up a handle into a reference" helper. A null input skips
  the call, the output is preset to 0, and the HRESULT is ignored. The result is stored only when it differs from
  the current handle; a non-null handle is then re-read with slots 11 (GetId) and 9 (GetLocalId). An error gives an
  empty reference, never a fatal path. Slot 12 is reached from the change-event handler path and one other site.
  Slot 10 is reached from one path that rebuilds a user record.
- **Slot 5, XUserCompare.** Called with two handles and no out parameter; it returns a plain int, where 0 means
  "same". The caller decides some cases itself without calling slot 5: equal handle values count as "same", and an
  invalid handle on either side counts as "different". Our duplicate handles have different values, so they do reach
  slot 5. Any non-zero return, including an error HRESULT, counts as "different user". If QueryApiImpl fails, the
  wrapper returns 0, which counts as "same". The helper is used by the change-event handler and by about five other
  sites.
- **Token (23 / 24 / 25).** Slot 25 has two call sites. Both run only after slot 24 succeeded; a negative slot 24
  takes the site's failure path. Slot 24 already answers with the offline HRESULT, so the slot 25 trap cannot be
  reached today, and slot 25 needs no decision unless slot 24 changes.
- **No fatal path.** No call site of slots 5, 10, 12, 14, 24 or 25 leads to a dialog, a quit or a sign-in UI.
  Every error answer degrades silently.
- **Next likely stops after 7.1.** Slot 5, 10 or 12 when a user record is compared or rebuilt; slot 14 if the poll
  is armed. Nothing further can be predicted statically.
- **Consequence per option (unchanged).** Slot 14 A keeps the local user and its pending items. Slot 14 B
  silently drops that user's pending items on every poll, which probably affects saves and settings (not verified
  dynamically). Slot 5 B, or an error HRESULT from slot 5, makes duplicate handles of our one user count as
  different users.

### 7.3 XStore licence slots (20, 22, 23, 28, 65, 67, 69)

Store, licence and entitlement are owner hard stops. Behaviour only: no addresses or bytes are recorded.
**Decided 2026-11-27 by owner, implemented in #2031:** option A. Slot 28 returns `0x800704CF`, slot 65 returns `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`; every other XStore slot keeps trapping as a tripwire.

Which slots the title reaches:
- Game licence: slot 28 (query), with follow-ups 29, 66 and 67. The title only reads "is disc", "is trial",
  "trial owned by this user" and "trial seconds left".
- Add-on licences: slot 65, then 20 (package) or XStoreExt 71 (durable), then 21/72, 22 and 68. XStoreExt 71/72 is
  already answered.
- Slots 22, 23 and 66-69 are only reached after a success (a licence handle or a received game licence exists).
  If the runtime never claims a success, the title never gets there.
- No store call is made until a local user is active; with no active user the title marks the request
  "unavailable" itself. This decision therefore follows the XUser decision (7.2): the stops appear once a user
  is active.

What the title does on failure (synchronous or asynchronous; no error screen found in the traced code):
- Game licence: it behaves as a full, non-trial, non-disc game and retries quietly in the background
  (roughly 120 s, interval not certain). There is nothing to gain from claiming a licence.
- Add-on licences: the item is treated as unavailable. Most errors are retried up to 4 more times, then
  "unavailable"; a failed start or failed slot 65 goes to "unavailable" at once.

| Option | Consequence |
|---|---|
| A. Truthful offline answers: slot 28 returns the offline code 0x800704CF, slot 65 returns "not found"; all other slots keep trapping (recommended) | Title runs as the full game, add-ons unavailable, no claim about ownership or sign-in. Trapping slots stay as a tripwire: a hit there means something claimed a success. Same pattern as the slot 71 decision. |
| B. Fail the whole store context at creation (xcl-style) | Title skips both flows itself; fewest store calls. XStoreCreateContext is implemented and has other callers (product and catalogue paths) that were not traced, so other stops may move. |
| C. Anything that claims ownership, licence or entitlement (success from 28/29, licence handles, slot 22 true, active add-ons) | Rejected. The title does not need it, and it is a hard stop. |

Recommendation: **A**.

Confidence: high for the state machines and the result codes they test; medium for "no error screen" (not every
consumer of the add-on callback was traced) and the exact retry interval.

Evidence (static, re-checked against main after #1998):
- The surface survey on #663 (`tools/surface_survey.py`) marks 20, 22, 23, 28, 65, 67 and 69 as tier A (call site
  reachable from the entry); the other 33 called XStore slots are tier C (no evidence). No recorded run has hit any
  of them yet: the boot stops earlier at the device-id and XUser decisions (7.1, 7.2).
- When: after XUserAddAsync has produced an active local user (past the Start Screen, "OFFLINEPLAYER (A)"), and only
  once slot 14 reports that user as signed in. Game licence first (28), add-on licences (65, then 20) as the title
  asks for each item.
- Result handling: every failure path is a local state change (full game / item unavailable) plus a quiet retry;
  no call site treats a store error as fatal. With option A only 28 and 65 are ever answered, and the tripwire
  slots (20, 22, 23, 67, 69) stay unreachable because no success is ever claimed.

### 7.4 S7: Media Foundation imports (MFPlat / MFReadWrite)

**Decided 2026-11-27 (option A), implemented in #2029.** The four imports are `ok` in `shim_audit.h` and fail cleanly (null out pointer, `E_NOTIMPL`, logged once each). The text below is the original analysis.

State on main: MFStartup, MFShutdown and MFGetSystemTime are `ok` (MFStartup returns `S_OK`). MFCreateAttributes,
MFCreateMediaType, MFCreateSourceResolver and MFReadWrite MFCreateSourceReaderFromMediaSource are `deny` ("host COM:
IUnknown stub with a host vtable"), so the first call traps as an unresolved import. Not a store or licence item; it
is listed here because the bridge verdict needs a decision.

What the title does (static, behaviour only):
- All seven MF imports are used by one small media-playback module that decodes a file into NV12 video and PCM audio
  (the audio path only accepts 16-bit, 48 kHz, at most 2 channels). It runs on its own worker thread, created
  lazily on the first playback request; nothing in start-up calls it.
- Open sequence: CoInitializeEx, MFStartup, MFCreateAttributes, MFCreateSourceResolver (open by URL), the source
  reader, then media types per stream. Every step tests the HRESULT; any failure releases what was created and
  marks the request failed. No retry, no assertion, no abort in the module. Who consumes the "failed" state was
  not traced beyond the module.
- When: tier C in the surface survey; no recorded native or browser run has reached the module.

| Option | Consequence |
|---|---|
| A. Bind the four denied imports as failing shims: write a null object, return an MF error (for example `MF_E_UNSUPPORTED_SERVICE` or `E_NOTIMPL`) (recommended) | Playback requests fail cleanly through the title's own error path; no host COM object reaches the guest. |
| B. Make MFStartup fail instead | Same outcome one step earlier, but MFStartup is already `ok` and shared with MFShutdown bookkeeping. |
| C. Real decoding (guest-callable IMFAttributes / IMFSourceReader) | Large; needs a guest-vtable COM layer and a decoder. Only worth it if a run shows the content matters. |
| D. Leave `deny` | Boot traps the first time the title plays such a file. |

Recommendation: **A** now (cheap, truthful "unavailable"), revisit C only if a run shows missing content.

### 7.5 S8: CRT `fopen` / `_wfopen` / `remove`

State on main: resolved, no owner decision left. #1998 (Marc) changed `pg6_crt_fopen`, `pg6_crt_wfopen`,
`pg6_crt_remove`, `pg6_crt_fsopen` and `pg6_crt_wfsopen` from `deny` to `ok` in `runtime/os/shim_audit.h`: they go
through the sandboxed virtual `FILE*` over the writable `/userdata` overlay and the read-only game root and never
touch the host file system. The surface survey lines on #663 that still list them as denied predate #1998.

What the title does with them (static):
- `_wfopen` (two call sites): only the crash-report writer, which saves a `.crash.log` file after a fault and logs
  "failed to save" if the open fails. Never reached on a healthy boot.
- `fopen` (one call site): a generic file-stream open that picks the mode from flags and records errno on a null
  result; the caller handles failure.
- `remove` (one call site): the local-storage service's delete; failure is logged with errno and reported to the
  caller.

Recommendation: nothing to decide; keep the sandbox rules (writes only under `/userdata`) as the guard.
