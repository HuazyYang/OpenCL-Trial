#pragma once
#include <random>
#include <chrono>

using hp_timer = std::chrono::high_resolution_clock;
using fmilliseconds = std::chrono::duration<float, std::milli>;

template<typename Duration> extern fmilliseconds fmilliseconds_cast(const Duration &);

template<> fmilliseconds fmilliseconds_cast<hp_timer::duration>(const hp_timer::duration &dur);

extern std::default_random_engine g_RandomEngine;