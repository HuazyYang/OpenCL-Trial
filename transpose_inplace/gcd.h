#pragma once

namespace tr_inplace { namespace details {

template <typename T> T gcd(T a, T b) {
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }

  return a;
}

/**
 * return the gcd of a and b followed by
 * the pair x and y of equation ax + by = gcd(a, b)
 */
template <typename T> void extended_gcd(T a, T b, T &gcd, T &mmi) {
  T x = 0;
  T lastx = 1;
  T y = 1;
  T lasty = 0;
  T origb = b;
  T q, r, m, n;

  while (b != 0) {
    q = a / b;
    r = a % b;
    a = b;
    b = r;
    m = lastx - q * x;
    lastx = x;
    x = m;
    n = lasty - q * y;
    lasty = y;
    y = n;
  }
  gcd = a;

  mmi = 0;
  if(gcd == 1) {
    if (lastx & (T(1) << ((sizeof(T) << 3) - 1)))
      mmi = (lastx + origb);
    else
      mmi = lastx;
  }

  return;
}

}; };