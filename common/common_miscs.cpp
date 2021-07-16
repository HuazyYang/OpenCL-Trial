#include "common_miscs.h"

std::default_random_engine g_RandomEngine{[]() -> std::random_device::result_type {
  std::random_device rdev;
  return rdev();
}()};

template <> fmilliseconds fmilliseconds_cast<hp_timer::duration>(const hp_timer::duration &dur) {
  return std::chrono::duration_cast<fmilliseconds>(dur);
}