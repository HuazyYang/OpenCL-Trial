#include "decomposition_tr.h"
#include <algorithm>

namespace decomposition_tr {

namespace internal {

template<typename T> T __gcd(T a, T b) {

  T r = 1;
  while(b > 0) {
    r = a % b;
    a = b;
    b = r;
  }

  return a;
}


};

};