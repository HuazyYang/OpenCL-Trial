#include "page_allocator.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#if defined(_WIN32)

namespace inplace {
namespace test {

BOOL _AdjustProcessAccessToken(LPWSTR pszPrivilege, BOOL bEnable) {

  HANDLE hToken;
  TOKEN_PRIVILEGES tp;
  BOOL status;
  DWORD error;

  // open process token
  if (!OpenProcessToken(GetCurrentProcess(),
                        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
    return FALSE;

  // get the luid
  if (!LookupPrivilegeValueW(NULL, pszPrivilege, &tp.Privileges[0].Luid))
    return FALSE;

  tp.PrivilegeCount = 1;

  // enable or disable privilege
  if (bEnable)
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
  else
    tp.Privileges[0].Attributes = 0;

  // enable or disable privilege
  status =
      AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

  // It is possible for AdjustTokenPrivileges to return TRUE and still not
  // succeed.
  // So always check for the last error value.
  error = GetLastError();
  if (!status || (error != ERROR_SUCCESS))
    return FALSE;

  // close the handle
  if (!CloseHandle(hToken))
    return FALSE;
  return TRUE;
}

void *large_page_alloc(size_t req_size) {

  SIZE_T dwLargePageMini = GetLargePageMinimum();
  DWORD dwPageSize = req_size;
  DWORD dwAllocFlags = MEM_COMMIT|MEM_RESERVE;
  SYSTEM_INFO SysInfo;

  if (req_size >= dwLargePageMini && _AdjustProcessAccessToken(L"SeLockMemoryPrivilege", TRUE)) {
    dwAllocFlags |= MEM_LARGE_PAGES;
    dwPageSize = ((dwPageSize + dwLargePageMini - 1) / dwLargePageMini) *
                 dwLargePageMini;
  } else {
    GetSystemInfo(&SysInfo);
    dwPageSize = (dwPageSize + SysInfo.dwPageSize - 1) & ~(SysInfo.dwPageSize - 1);
  }

  return VirtualAlloc(NULL, dwPageSize, dwAllocFlags, PAGE_READWRITE);
}

void large_page_dealloc(void *p) {
  if(p) VirtualFree(p, 0, MEM_RELEASE);
}

int get_last_error() {
  return (int)GetLastError();
}

}
}

#endif /** _WIN32 */