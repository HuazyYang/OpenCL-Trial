#include <common.cl.h>

// #define NATIVE_DIVIDE

#ifdef NATIVE_DIVIDE
#define DIV_IMPL(a, b) native_divide((a), (b))
#else
#define DIV_IMPL(a, b) (a) / (b)
#endif