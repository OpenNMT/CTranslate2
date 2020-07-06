#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {

  // Inspired by
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.h
  // for matching enum values and C++ types.

  // Validates type T for whether it is a supported DataType.
  template <class T>
  struct IsValidDataType;

  // DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
  // constants for T, e.g. DataTypeToEnum<float>::v() is DataType::FLOAT.
  template <class T>
  struct DataTypeToEnum {
    static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
  };  // Specializations below

  // EnumToDataType<DataType::VALUE>::Type is the type for DataType constant VALUE, e.g.
  // EnumToDataType<DataType::FLOAT>::Type is float.
  template <DataType VALUE>
  struct EnumToDataType {}; // Specializations below

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM, NAME)           \
  template<>                                            \
  struct DataTypeToEnum<TYPE> {                         \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template<>                                            \
  struct IsValidDataType<TYPE> {                        \
    static constexpr bool value = true;                 \
  };                                                    \
  template<>                                            \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

  MATCH_TYPE_AND_ENUM(float, DataType::FLOAT, "float");
  MATCH_TYPE_AND_ENUM(int8_t, DataType::INT8, "int8");
  MATCH_TYPE_AND_ENUM(int16_t, DataType::INT16, "int16");
  MATCH_TYPE_AND_ENUM(int32_t, DataType::INT32, "int32");

  // All types not specialized are marked invalid.
  template <class T>
  struct IsValidDataType {
    static constexpr bool value = false;
  };

#define TYPE_CASE(TYPE, STMTS)                  \
  case DataTypeToEnum<TYPE>::value: {           \
    typedef TYPE T;                             \
    STMTS;                                      \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#define TYPE_DISPATCH(TYPE_ENUM, STMTS) \
  switch (TYPE_ENUM) {                  \
    TYPE_CASE(float, SINGLE_ARG(STMTS)) \
    TYPE_CASE(int8_t, SINGLE_ARG(STMTS))            \
    TYPE_CASE(int16_t, SINGLE_ARG(STMTS))           \
    TYPE_CASE(int32_t, SINGLE_ARG(STMTS))           \
  }

#define DECLARE_ALL_TYPES(FUNC)                 \
  FUNC(float)                                   \
  FUNC(int8_t)                                  \
  FUNC(int16_t)                                 \
  FUNC(int32_t)                                 \

}
