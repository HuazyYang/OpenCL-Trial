#pragma once

namespace inplace {
namespace test {

  void *large_page_alloc(size_t req_size);
  void large_page_dealloc(void *p);

  int get_last_error();
}
}