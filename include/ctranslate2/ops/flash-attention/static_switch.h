// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once
#include <cutlass/numeric_types.h>

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

// When FLASH_ATTN_HDIMS is restricted via cmake, only instantiate selected
// head dimensions. Others throw at runtime instead of generating link-time
// symbol references. CT2_FLASH_ATTN_HDIM_N is defined per compiled hdim.
#define _HEADDIM_DISPATCH(DIM, ...)                           \
    constexpr static int kHeadDim = DIM;                      \
    return __VA_ARGS__();

#define _HEADDIM_UNSUPPORTED(DIM)                             \
    throw std::runtime_error(                                 \
      "Flash attention head dim " #DIM " not compiled. "      \
      "Rebuild CTranslate2 with FLASH_ATTN_HDIMS including " #DIM);

#ifndef CT2_FLASH_ATTN_HDIM_32
#define _HEADDIM_CASE_32(...) _HEADDIM_UNSUPPORTED(32)
#else
#define _HEADDIM_CASE_32(...) _HEADDIM_DISPATCH(32, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_64
#define _HEADDIM_CASE_64(...) _HEADDIM_UNSUPPORTED(64)
#else
#define _HEADDIM_CASE_64(...) _HEADDIM_DISPATCH(64, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_96
#define _HEADDIM_CASE_96(...) _HEADDIM_UNSUPPORTED(96)
#else
#define _HEADDIM_CASE_96(...) _HEADDIM_DISPATCH(96, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_128
#define _HEADDIM_CASE_128(...) _HEADDIM_UNSUPPORTED(128)
#else
#define _HEADDIM_CASE_128(...) _HEADDIM_DISPATCH(128, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_160
#define _HEADDIM_CASE_160(...) _HEADDIM_UNSUPPORTED(160)
#else
#define _HEADDIM_CASE_160(...) _HEADDIM_DISPATCH(160, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_192
#define _HEADDIM_CASE_192(...) _HEADDIM_UNSUPPORTED(192)
#else
#define _HEADDIM_CASE_192(...) _HEADDIM_DISPATCH(192, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_224
#define _HEADDIM_CASE_224(...) _HEADDIM_UNSUPPORTED(224)
#else
#define _HEADDIM_CASE_224(...) _HEADDIM_DISPATCH(224, __VA_ARGS__)
#endif
#ifndef CT2_FLASH_ATTN_HDIM_256
#define _HEADDIM_CASE_256(...) _HEADDIM_UNSUPPORTED(256)
#else
#define _HEADDIM_CASE_256(...) _HEADDIM_DISPATCH(256, __VA_ARGS__)
#endif

// When all hdims are compiled (no FLASH_ATTN_HDIMS set), all CT2_FLASH_ATTN_HDIM_*
// macros are undefined and _HEADDIM_CASE_* defaults to _HEADDIM_UNSUPPORTED.
// Fix: when not restricted, define all as dispatching.
#ifndef CT2_FLASH_ATTN_HDIMS_RESTRICTED
#undef _HEADDIM_CASE_32
#undef _HEADDIM_CASE_64
#undef _HEADDIM_CASE_96
#undef _HEADDIM_CASE_128
#undef _HEADDIM_CASE_160
#undef _HEADDIM_CASE_192
#undef _HEADDIM_CASE_224
#undef _HEADDIM_CASE_256
#define _HEADDIM_CASE_32(...)  _HEADDIM_DISPATCH(32, __VA_ARGS__)
#define _HEADDIM_CASE_64(...)  _HEADDIM_DISPATCH(64, __VA_ARGS__)
#define _HEADDIM_CASE_96(...)  _HEADDIM_DISPATCH(96, __VA_ARGS__)
#define _HEADDIM_CASE_128(...) _HEADDIM_DISPATCH(128, __VA_ARGS__)
#define _HEADDIM_CASE_160(...) _HEADDIM_DISPATCH(160, __VA_ARGS__)
#define _HEADDIM_CASE_192(...) _HEADDIM_DISPATCH(192, __VA_ARGS__)
#define _HEADDIM_CASE_224(...) _HEADDIM_DISPATCH(224, __VA_ARGS__)
#define _HEADDIM_CASE_256(...) _HEADDIM_DISPATCH(256, __VA_ARGS__)
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      _HEADDIM_CASE_32(__VA_ARGS__)        \
    } else if (HEADDIM <= 64) {            \
      _HEADDIM_CASE_64(__VA_ARGS__)        \
    } else if (HEADDIM <= 96) {            \
      _HEADDIM_CASE_96(__VA_ARGS__)        \
    } else if (HEADDIM <= 128) {           \
      _HEADDIM_CASE_128(__VA_ARGS__)       \
    } else if (HEADDIM <= 160) {           \
      _HEADDIM_CASE_160(__VA_ARGS__)       \
    } else if (HEADDIM <= 192) {           \
      _HEADDIM_CASE_192(__VA_ARGS__)       \
    } else if (HEADDIM <= 224) {           \
      _HEADDIM_CASE_224(__VA_ARGS__)       \
    } else if (HEADDIM <= 256) {           \
      _HEADDIM_CASE_256(__VA_ARGS__)       \
    }                                      \
  }()
