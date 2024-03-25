#include <unordered_set>
#include "ctranslate2/primitives.h"
#include "utils.h"
#include "type_dispatch.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  template<>
  template <typename T>
  T primitives<Device::CANN>::at(const T* x, dim_t index) {
      T val = T();
      cross_device_primitives<Device::CANN, Device::CPU>::copy(x + index, &val, 1);
      return val;
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::fill(T* x, T a, dim_t size) {
    ctranslate2::cann::CannPreparation prepare;

    const aclDataType aclType = cann::getACLType<T>();
    const ctranslate2::Shape x_shape = {size};

    cann_prepare_inputdesc(prepare, aclType, x_shape.size(), x_shape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, aclType, x_shape.size(), x_shape.data(), ACL_FORMAT_ND);

    ACL_CALL(aclopSetAttrFloat(prepare._opAttr, "value", static_cast<float>(a)));

    const dim_t size_of_x = size*sizeof(T);
    cann_prepare_inputbuffer(prepare, x, size_of_x);
    cann_prepare_outputbuffer(prepare, x, size_of_x);

    ACL_CALL(aclopCompileAndExecute("Fills",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::zero(T* x, dim_t size, bool synchronous) {
    ctranslate2::cann::CannPreparation prepare;

    const aclDataType aclType = cann::getACLType<T>();
    const ctranslate2::Shape x_shape = {size};

    cann_prepare_inputdesc(prepare, aclType, x_shape.size(), x_shape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, aclType, x_shape.size(), x_shape.data(), ACL_FORMAT_ND);

    const dim_t size_of_x = size*sizeof(T);
    cann_prepare_inputbuffer(prepare, x, size_of_x);
    cann_prepare_outputbuffer(prepare, x, size_of_x);

    ACL_CALL(aclopCompileAndExecute("ZerosLike",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    if (synchronous) {
      ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices, dim_t size) {
    if (size <= 0) {
      THROW_RUNTIME_ERROR("Input 'size' of 'indexed_fill' primitive should be positive");
    }

    // Using a unique pointer to a bool array instead of a 'std::vector<bool>' because in the case of the latter, it's
    // not possible to get a raw pointer to the underlying data.
    auto mask = std::make_unique<bool[]>(size);
    std::vector<int32_t> indices_cpu(num_indices);

    // Moving the 'indices' from the NPU to the CPU in order to iterate over them.
    cross_device_primitives<Device::CANN, Device::CPU>::copy(indices, indices_cpu.data(), num_indices);
    for (dim_t i = 0; i < num_indices; ++i) {
        mask[indices_cpu[i]] = true;
    }

    ctranslate2::cann::CannPreparation prepare;
    const aclDataType aclType  = cann::getACLType<T>();
    const Shape input_and_mask_shape = {size}, value_shape = {1};

    cann_prepare_inputdesc(prepare, aclType, input_and_mask_shape.size(), input_and_mask_shape.data(), ACL_FORMAT_ND);
    cann_prepare_inputdesc(prepare, ACL_BOOL, input_and_mask_shape.size(), input_and_mask_shape.data(), ACL_FORMAT_ND);
    cann_prepare_inputdesc(prepare, aclType, value_shape.size(), value_shape.data(), ACL_FORMAT_ND);

    cann_prepare_outputdesc(prepare, aclType, input_and_mask_shape.size(), input_and_mask_shape.data(), ACL_FORMAT_ND);

    const dim_t size_of_input_in_bytes = sizeof(T)*size;
    cann_prepare_inputbuffer(prepare, const_cast<T*>(x), size_of_input_in_bytes);

    // Temporary way to allocate memory for the 'mask' input
    void *mask_dev_ptr = nullptr;
    const dim_t size_of_mask_in_bytes = sizeof(bool)*size;
    ACL_CALL(aclrtMalloc(&mask_dev_ptr, size_of_mask_in_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMemcpyAsync(mask_dev_ptr, size_of_mask_in_bytes, mask.get(), size_of_mask_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE, cann::get_aclrt_stream()));
    cann_prepare_inputbuffer(prepare, mask_dev_ptr, size_of_mask_in_bytes);

    // Temporary way to allocate memory for the 'value' input
    void *value_dev_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&value_dev_ptr, sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMemcpyAsync(value_dev_ptr, sizeof(T), const_cast<T*>(&a), sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE, cann::get_aclrt_stream()));
    cann_prepare_inputbuffer(prepare, value_dev_ptr, sizeof(T));
    cann_prepare_outputbuffer(prepare, x, size_of_input_in_bytes);

    ACL_CALL(aclopCompileAndExecute("MaskedFill",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));

    // Temporary way to free the allocated memory for inputs
    ACL_CALL(aclrtFree(mask_dev_ptr));
    ACL_CALL(aclrtFree(value_dev_ptr));
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::copy(const T* x, T* y, dim_t size) {
      const auto size_in_bytes = size * sizeof (T);
      ACL_CALL(aclrtMemcpy(y, size_in_bytes, x, size_in_bytes,
                                ACL_MEMCPY_DEVICE_TO_DEVICE));
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::CANN>::convert(const U* x, V* y, dim_t size) {      
      cann::CannPreparation prepare;
      // Assume the shape as if the tensor was one-dimensional
      const Shape shape_1d = {size};
      const auto in_type = cann::getACLType<U>();
      const auto out_type = cann::getACLType<V>();

      ACL_CALL(aclopSetAttrDataType(prepare._opAttr, "dst_type", out_type));
      aclFormat format = ACL_FORMAT_ND;

      cann_prepare_inputdesc(prepare, in_type, shape_1d.size(), shape_1d.data(), format);
      cann_prepare_outputdesc(prepare, out_type, shape_1d.size(), shape_1d.data(), format);

      cann_prepare_inputbuffer(prepare, const_cast<U*>(x), size*sizeof(U));
      cann_prepare_outputbuffer(prepare, y, size*sizeof(V));

      ACL_CALL(aclopCompileAndExecute("Cast",
                                      prepare._inputDesc.size(),
                                      prepare._inputDesc.data(),
                                      prepare._inputBuffers.data(),
                                      prepare._outputDesc.size(),
                                      prepare._outputDesc.data(),
                                      prepare._outputBuffers.data(),
                                      prepare._opAttr,
                                      ACL_ENGINE_SYS,
                                      ACL_COMPILE_SYS,
                                      NULL,
                                      cann::get_aclrt_stream()));
      ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template void primitives<Device::CANN>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::CANN>::convert(const float16_t*, float*, dim_t);
  template<>
  template<>
  void primitives<Device::CANN>::convert(const float*, bfloat16_t*, dim_t) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: float to bfloat16_t");
  }
  template<>
  template<>
  void primitives<Device::CANN>::convert(const bfloat16_t*, float*, dim_t) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: bfloat16_t to float");
  }
  template<>
  template<>
  void primitives<Device::CANN>::convert(const float16_t* x, bfloat16_t* y, dim_t size) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: float16_t to bfloat16_t");
  }
  template<>
  template<>
  void primitives<Device::CANN>::convert(const bfloat16_t* x, float16_t* y, dim_t size) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: bfloat16_t to float16_t");
  }

  template<>
  template <typename T>
  T primitives<Device::CANN>::sum(const T* array, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CANN>::max_element(const T* array, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  T primitives<Device::CANN>::max(const T* array, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::add(T a, const T* x, T* y, dim_t size) {
    const aclDataType aclType  = cann::getACLType<T>();

    static std::unordered_set<aclDataType> supportedTypes{ACL_INT64, ACL_INT32, ACL_FLOAT, ACL_FLOAT16};
    if(supportedTypes.find(aclType) == supportedTypes.end())
      THROW_RUNTIME_ERROR("Unsupported ACL type for Add-scalar: " + std::to_string(aclType));

    Shape arrayShape = {size};

    ctranslate2::cann::CannPreparation prepare;
    cann_prepare_inputdesc(prepare, aclType, arrayShape.size(), arrayShape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, aclType, arrayShape.size(), arrayShape.data(), ACL_FORMAT_ND);

    cann_prepare_inputbuffer(prepare, const_cast<T*>(x), sizeof(T)*size);
    cann_prepare_outputbuffer(prepare, y, sizeof(T)*size);

    // 'value' must be a float according to the documentation
    ACL_CALL(aclopSetAttrFloat(prepare._opAttr, "value", static_cast<float>(a)));

    ACL_CALL(aclopCompileAndExecute("Adds",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::add(const T* a, const T* b, T* c, dim_t size) {
    ctranslate2::cann::CannPreparation prepare;

    const aclDataType aclType  = cann::getACLType<T>();
    Shape arrayShape = {size};

    cann_prepare_inputdesc(prepare, aclType, arrayShape.size(), arrayShape.data(), ACL_FORMAT_ND);
    cann_prepare_inputdesc(prepare, aclType, arrayShape.size(), arrayShape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, aclType, arrayShape.size(), arrayShape.data(), ACL_FORMAT_ND);

    cann_prepare_inputbuffer(prepare, const_cast<T*>(a), sizeof(T)*size);
    cann_prepare_inputbuffer(prepare, const_cast<T*>(b), sizeof(T)*size);
    cann_prepare_outputbuffer(prepare, c, sizeof(T)*size);

    ACL_CALL(aclopCompileAndExecute("Add",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size, bool synchronous) {
    const aclDataType aclType  = cann::getACLType<T>();

    // According to the documentation, 'bias' should have length that is equal to the last dimension of 'value'
    Shape bias_shape = {a_size}, value_shape = {b_size/a_size, a_size};

    ctranslate2::cann::CannPreparation prepare;

    cann_prepare_inputdesc(prepare, aclType, value_shape.size(), value_shape.data(), ACL_FORMAT_ND);
    cann_prepare_inputdesc(prepare, aclType, bias_shape.size(), bias_shape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, aclType, value_shape.size(), value_shape.data(), ACL_FORMAT_ND);

    cann_prepare_inputbuffer(prepare, const_cast<T*>(b), sizeof(T)*b_size);
    cann_prepare_inputbuffer(prepare, const_cast<T*>(a), sizeof(T)*a_size);
    cann_prepare_outputbuffer(prepare, c, sizeof(T)*b_size);

    // We skipped the 'data_format' optional attribute, because it defaults to "NHWC"

    ACL_CALL(aclopCompileAndExecute("BiasAdd",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    if (synchronous){
      ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    cann::CannPreparation prepare;
    const aclDataType aclType = cann::getACLType<T>();
    const aclFormat format = ACL_FORMAT_ND;

    // 'bias' should have length that is equal to the first dimension of 'value'.
    // 'value' can have any shape, but we can safely assume that it's a 2-D vector where the 1st dimension matches
    // the length of 'bias'.
    Shape bias_shape = {a_size}, value_shape = {a_size, b_size/a_size};

    cann_prepare_inputdesc(prepare, aclType, value_shape.size(), value_shape.data(), format);
    cann_prepare_inputdesc(prepare, aclType, bias_shape.size(), bias_shape.data(), format);
    cann_prepare_outputdesc(prepare, aclType, value_shape.size(), value_shape.data(), format);

    // Instruct the "Bias" operator to match the 1st axis of the 'values' vector with the 'bias' vector.
    ACL_CALL(aclopSetAttrInt(prepare._opAttr, "axis", 0));

    const dim_t value_size = b_size*sizeof(T);
    cann_prepare_inputbuffer(prepare, const_cast<T*>(b), value_size);
    cann_prepare_inputbuffer(prepare, const_cast<T*>(a), a_size*sizeof(T));
    cann_prepare_outputbuffer(prepare, c, value_size);

    ACL_CALL(aclopCompileAndExecute("Bias",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::sub(const T* a, const T* b, T* c, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::min(T a, const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::min(const T* a, const T* b, T* c, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::max(T a, const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::max(const T* a, const T* b, T* c, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::mul(T a, const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("CANN case is handled in StorageView level");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::mul(const T* a, const T* b, T* c, dim_t size) {
      THROW_RUNTIME_ERROR("CANN case is handled in StorageView level");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::relu(const T* x, T* y, dim_t size) {
    cann::CannPreparation prepare;

    // Assume the shape as if the tensor was one-dimensional
    const ctranslate2::Shape shape_1d = {size};
    const aclDataType aclType = cann::getACLType<T>();

    if(aclType == ACL_BF16) {
      THROW_RUNTIME_ERROR("Unsupported ACL type: " + std::to_string(aclType));
    }

    aclFormat format = ACL_FORMAT_ND;
    const dim_t size_in_bytes =  size*sizeof(T);

    cann_prepare_inputdesc(prepare, aclType, shape_1d.size(), shape_1d.data(), format);
    cann_prepare_outputdesc(prepare, aclType, shape_1d.size(), shape_1d.data(), format);

    cann_prepare_inputbuffer(prepare, const_cast<T*>(x), size_in_bytes);
    cann_prepare_outputbuffer(prepare, y, size_in_bytes);

    ACL_CALL(aclopCompileAndExecute("Relu",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template void primitives<Device::CANN>::relu(const float*, float*, dim_t);
  template void primitives<Device::CANN>::relu(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::relu(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::gelu(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::gelu(const float*, float*, dim_t);
  template void primitives<Device::CANN>::gelu(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::gelu(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::gelu_tanh(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::gelu_tanh(const float*, float*, dim_t);
  template void primitives<Device::CANN>::gelu_tanh(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::gelu_tanh(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::gelu_sigmoid(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::gelu_sigmoid(const float*, float*, dim_t);
  template void primitives<Device::CANN>::gelu_sigmoid(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::gelu_sigmoid(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::swish(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::swish(const float*, float*, dim_t);
  template void primitives<Device::CANN>::swish(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::swish(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::penalize_previous_tokens(T* scores,
                                                          const T* previous_scores,
                                                          const int32_t* previous_ids,
                                                          T penalty,
                                                          dim_t batch_size,
                                                          dim_t length,
                                                          dim_t vocabulary_size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  /**
   * Broadcasts vector 'x' of length 'x_length' across 'num_rows' rows and produces a vector 'y' of shape {num_rows, x_length}.
   * This is achieved through the "BroadcastTo" CANN operator with:
   * Inputs:
   *    x:     A tensor.
   *    shape: A 1D tensor of type int32, for the shape of the desired output.
   * Outputs:
   *    y:     A tensor of shape 'shape' and type same as 'x'.
   */
  void broadcast_to(int32_t* x, int32_t num_rows, int32_t x_length, int32_t* y) {
    cann::CannPreparation prepare;
    const Shape shape_of_x = {x_length};
    const Shape shape_of_y = {num_rows, x_length};
    const Shape shape_of_shape = {static_cast<dim_t>(shape_of_y.size())};
    cann_prepare_inputdesc(prepare, ACL_INT32, shape_of_x.size(), shape_of_x.data(), ACL_FORMAT_ND);
    cann_prepare_inputdesc(prepare, ACL_INT32, shape_of_shape.size(), shape_of_shape.data(), ACL_FORMAT_ND);
    cann_prepare_outputdesc(prepare, ACL_INT32, shape_of_y.size(), shape_of_y.data(), ACL_FORMAT_ND);

    std::vector<int32_t> shape = {num_rows, x_length};
    dim_t shape_size_in_bytes = shape.size()*sizeof(int32_t);
    dim_t x_size_in_bytes = x_length*sizeof(int32_t);
    void *x_dev_ptr = nullptr, *shape_dev_ptr = nullptr;
    // Temporary way to allocate memory on the NPU for inputs 'x' and 'shape'
    ACL_CALL(aclrtMalloc(&x_dev_ptr, x_size_in_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMemcpyAsync(x_dev_ptr, x_size_in_bytes, x, x_size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE, cann::get_aclrt_stream()));
    ACL_CALL(aclrtMalloc(&shape_dev_ptr, shape_size_in_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMemcpyAsync(shape_dev_ptr, shape_size_in_bytes, shape.data(), shape_size_in_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE, cann::get_aclrt_stream()));

    cann_prepare_inputbuffer(prepare, x_dev_ptr, x_size_in_bytes);
    cann_prepare_inputbuffer(prepare, shape_dev_ptr, shape_size_in_bytes);
    cann_prepare_outputbuffer(prepare, y, num_rows*x_size_in_bytes);

    ACL_CALL(aclopCompileAndExecute("BroadcastTo",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));

    // Temporary way to free allocated memory for 'x' and 'shape'.
    ACL_CALL(aclrtFree(x_dev_ptr));
    ACL_CALL(aclrtFree(shape_dev_ptr));
  }

  int32_t* prepare_vector_to_broadcast(std::vector<int32_t> &x, const dim_t size, const int32_t max_value) {
    x.resize(size);
    for (int32_t j = 0; j < size; ++j) {
      x[j] = std::min(max_value, j+1);
    }
    return x.data();
  }

  template<>
  void primitives<Device::CANN>::prepare_length_mask(const int32_t* lengths,
                                                     dim_t batch_size,
                                                     dim_t num_heads,
                                                     dim_t num_queries,
                                                     bool mask_future,
                                                     bool multi_query,
                                                     int32_t* mask) {
    std::vector<int32_t> lengths_cpu(batch_size);
    cross_device_primitives<Device::CANN, Device::CPU>::copy(lengths, lengths_cpu.data(), batch_size);
    std::vector<int32_t> x;
    const uint64_t num_heads_and_queries = num_heads*num_queries;
    for (dim_t b = 0; b < batch_size; ++b) { // Iterate over the 1st dimension (batches) of the output 'mask'
      const int32_t length = lengths_cpu[b];
      int32_t* batch_mask = mask + b * num_heads_and_queries;
      if (mask_future) {
        if (multi_query) { // Shape of output 'mask' is: {batch_size, num_queries, num_heads}
          int32_t *row_ptr;
          for (dim_t i = 0; i < num_queries; ++i) { // Iterate over the 2nd dimension of the output 'mask'
            row_ptr = batch_mask + i*num_heads;
            // Fill each row of the current batch with value: min(length, i+1)
            primitives<Device::CANN>::fill(row_ptr, std::min(length, int32_t(i+1)), num_heads);
          }
        } else { // Shape of output 'mask' is: {batch_size, num_heads, num_queries}
          // Create a 1-D vector: {1, 2, ..., min(length, num_queries-1), min(length, num_queries)}.
          prepare_vector_to_broadcast(x, num_queries, length);
          // Broadcast the vector across all the rows of the current batch.
          broadcast_to(x.data(), num_heads, num_queries, batch_mask);
        }
      } else {
        primitives<Device::CANN>::fill(batch_mask, length, num_heads_and_queries);
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::transpose_2d(const T* a, const dim_t* dims, T* b) {
      THROW_RUNTIME_ERROR("CANN case is handled in StorageView level");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::transpose_3d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
      THROW_RUNTIME_ERROR("CANN case is handled in StorageView level");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::transpose_4d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
      THROW_RUNTIME_ERROR("CANN case is handled in StorageView level");
  }

  template<typename In, typename Out>
  void run_gemm_alpha_beta_in_device(bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     const float* alpha_dev_ptr, const In* a,
                                     const float* beta_dev_ptr, const In* b,
                                     Out* c) {
    aclFormat format = ACL_FORMAT_ND;
    aclDataType aclIn = ctranslate2::cann::getACLType<In>(), aclOut = ctranslate2::cann::getACLType<Out>();

    ctranslate2::cann::CannPreparation prepare;

    ACL_CALL(aclopSetAttrBool(prepare._opAttr, "transpose_a", transpose_a));
    ACL_CALL(aclopSetAttrBool(prepare._opAttr, "transpose_b", transpose_b));

    ctranslate2::Shape a_shape, b_shape;
    // The "GEMM" CANN operator expects different shapes for the 'a' and 'b' input vectors, based on whether they are
    // transpose or not.
    if (transpose_a) {
      a_shape = {k, m};
    } else {
      a_shape = {m, k};
    }
    if (transpose_b) {
      b_shape = {n, k};
    } else {
      b_shape = {k, n};
    }

    const ctranslate2::Shape c_shape = {m, n};
    // 'alpha' and 'beta' should be 1-D vectors of size 1, according to CANN documentation
    static const ctranslate2::Shape alpha_beta_shape = {1};

    cann_prepare_inputdesc(prepare, aclIn, a_shape.size(), a_shape.data(), format);
    cann_prepare_inputdesc(prepare, aclIn, b_shape.size(), b_shape.data(), format);
    cann_prepare_inputdesc(prepare, aclOut, c_shape.size(), c_shape.data(), format);
    cann_prepare_inputdesc(prepare, ACL_FLOAT, alpha_beta_shape.size(), alpha_beta_shape.data(), format);
    cann_prepare_inputdesc(prepare, ACL_FLOAT, alpha_beta_shape.size(), alpha_beta_shape.data(), format);
    cann_prepare_outputdesc(prepare, aclOut, c_shape.size(), c_shape.data(), format);

    const dim_t c_size_in_bytes = m*n*sizeof(Out);
    cann_prepare_inputbuffer(prepare, const_cast<In*>(a), m*k*sizeof(In));
    cann_prepare_inputbuffer(prepare, const_cast<In*>(b), k*n*sizeof(In));
    cann_prepare_inputbuffer(prepare, c, c_size_in_bytes);
    cann_prepare_inputbuffer(prepare, const_cast<float*>(alpha_dev_ptr), sizeof(float));
    cann_prepare_inputbuffer(prepare, const_cast<float*>(beta_dev_ptr), sizeof(float));
    cann_prepare_outputbuffer(prepare, c, c_size_in_bytes);

    ACL_CALL(aclopCompileAndExecute("GEMM",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    // Synchronizing the stream takes place in the StorageView level
  }
 
  template<>
  template<>
  void primitives<Device::CANN>::gemm_alpha_beta_in_device(bool, bool,
                                                           bool transpose_a, bool transpose_b,
                                                           dim_t m, dim_t n, dim_t k,
                                                           const float* alpha,
                                                           const float* a, dim_t,
                                                           const float* b, dim_t,
                                                           const float* beta,
                                                           float* c, dim_t,
                                                           const float*) {
    run_gemm_alpha_beta_in_device(transpose_a, transpose_b, m, n, k, alpha, a, beta, b, c);
  }

  template<>
  template<>
  void primitives<Device::CANN>::gemm_alpha_beta_in_device(bool, bool,
                                                           bool transpose_a, bool transpose_b,
                                                           dim_t m, dim_t n, dim_t k,
                                                           const float* alpha,
                                                           const float16_t* a, dim_t,
                                                           const float16_t* b, dim_t,
                                                           const float* beta,
                                                           float16_t* c, dim_t,
                                                           const float16_t*) {
    run_gemm_alpha_beta_in_device(transpose_a, transpose_b, m, n, k, alpha, a, beta, b, c);
  }

  template<>
  template<>
  void primitives<Device::CANN>::gemm_alpha_beta_in_device(bool, bool,
                                                           bool transpose_a, bool transpose_b,
                                                           dim_t m, dim_t n, dim_t k,
                                                           const float* alpha,
                                                           const bfloat16_t* a, dim_t,
                                                           const bfloat16_t* b, dim_t,
                                                           const float* beta,
                                                           bfloat16_t* c, dim_t,
                                                           const bfloat16_t*) {
    THROW_RUNTIME_ERROR("FP16 GEMM is not supported by CANN");
  }

  template<>
  template<>
  void primitives<Device::CANN>::gemm_alpha_beta_in_device(bool, bool,
                                                           bool transpose_a, bool transpose_b,
                                                           dim_t m, dim_t n, dim_t k,
                                                           const float* alpha,
                                                           const int8_t* a, dim_t,
                                                           const int8_t* b, dim_t,
                                                           const float* beta,
                                                           int32_t* c, dim_t,
                                                           const int32_t*) {
    run_gemm_alpha_beta_in_device(transpose_a, transpose_b, m, n, k, alpha, a, beta, b, c);
  } 

  template<>
  template <typename T>
  float primitives<Device::CANN>::logsumexp(const T* x, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template float primitives<Device::CANN>::logsumexp(const float*, dim_t);
  template float primitives<Device::CANN>::logsumexp(const float16_t*, dim_t);
  template float primitives<Device::CANN>::logsumexp(const bfloat16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CANN>::sin(const T* x, T* y, dim_t size) {
    const auto aclType = cann::getACLType<T>();

    aclFormat format = ACL_FORMAT_ND;
    cann::CannPreparation prepare;
    const ctranslate2::Shape x_y_shape = {size};
    const dim_t size_in_bytes = size * sizeof(T);

    cann_prepare_inputdesc(prepare, aclType, x_y_shape.size(), x_y_shape.data(), format);
    cann_prepare_outputdesc(prepare, aclType, x_y_shape.size(), x_y_shape.data(), format);

    cann_prepare_inputbuffer(prepare, const_cast<T*>(x), size_in_bytes);
    cann_prepare_outputbuffer(prepare, y, size_in_bytes);

    ACL_CALL(aclopCompileAndExecute("Sin",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template void primitives<Device::CANN>::sin(const float*, float*, dim_t);
  template void primitives<Device::CANN>::sin(const float16_t*, float16_t*, dim_t);
  template<>
  template<>
  void primitives<Device::CANN>::sin(const bfloat16_t*, bfloat16_t*, dim_t) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: bfloat16_t");
  }

  template<>
  template <typename T>
  void primitives<Device::CANN>::cos(const T* x, T* y, dim_t size) {
    const auto aclType = cann::getACLType<T>(); 
    aclFormat format = ACL_FORMAT_ND;
    cann::CannPreparation prepare;

    const ctranslate2::Shape shape_1d = {size};
    cann_prepare_inputdesc(prepare, aclType, shape_1d.size(), shape_1d.data(), format);
    cann_prepare_outputdesc(prepare, aclType, shape_1d.size(), shape_1d.data(), format);

    const auto in_out_size_in_bytes = size*sizeof(T);
    cann_prepare_inputbuffer(prepare, const_cast<T*>(x), in_out_size_in_bytes);
    cann_prepare_outputbuffer(prepare, y, in_out_size_in_bytes);

    ACL_CALL(aclopCompileAndExecute("Cos",
                                    prepare._inputDesc.size(),
                                    prepare._inputDesc.data(),
                                    prepare._inputBuffers.data(),
                                    prepare._outputDesc.size(),
                                    prepare._outputDesc.data(),
                                    prepare._outputBuffers.data(),
                                    prepare._opAttr,
                                    ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS,
                                    NULL,
                                    cann::get_aclrt_stream()));
    ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
  }

  template void primitives<Device::CANN>::cos(const float*, float*, dim_t);
  template void primitives<Device::CANN>::cos(const float16_t*, float16_t*, dim_t);
  template<>
  template<>
  void primitives<Device::CANN>::cos(const bfloat16_t*, bfloat16_t*, dim_t) {
    THROW_RUNTIME_ERROR("Unsupported ACL type: bfloat16_t");
  }


  template<>
  template <typename T>
  void primitives<Device::CANN>::tanh(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::tanh(const float*, float*, dim_t);
  template void primitives<Device::CANN>::tanh(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::tanh(const bfloat16_t*, bfloat16_t*, dim_t);

  template<>
  template<>
  void primitives<Device::CANN>::exp(const float* x, float* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }
  
  template<>
  template <typename T>
  void primitives<Device::CANN>::log(const T* x, T* y, dim_t size) {
      THROW_RUNTIME_ERROR("not implemented in CANN");
  }

  template void primitives<Device::CANN>::log(const float*, float*, dim_t);
  template void primitives<Device::CANN>::log(const float16_t*, float16_t*, dim_t);
  template void primitives<Device::CANN>::log(const bfloat16_t*, bfloat16_t*, dim_t);
    
  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CANN>::copy(const T* x, T* y, dim_t size) {
    const auto size_in_bytes = size * sizeof (T);
    ACL_CALL(aclrtMemcpy(y, size_in_bytes, x, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CANN, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    const auto size_in_bytes = size * sizeof (T);
    ACL_CALL(aclrtMemcpy(y, size_in_bytes, x, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  }

#define DECLARE_IMPL(T) \
  template T                                                            \
  primitives<Device::CANN>::at(const T* x, dim_t index);                \
  template void                                                         \
  primitives<Device::CANN>::fill(T* x, T a, dim_t size);                \
  template void                                                         \
  primitives<Device::CANN>::zero(T* x, dim_t size, bool synchronous);   \
  template void                                                         \
  primitives<Device::CANN>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::indexed_fill(T*, T, const int32_t*, dim_t, dim_t); \
  template void                                                         \
  primitives<Device::CANN>::copy<T>(const T* x, T* y, dim_t size);      \
  template T                                                            \
  primitives<Device::CANN>::sum(const T* array, dim_t size);            \
  template dim_t                                                        \
  primitives<Device::CANN>::max_element(const T* array, dim_t size);    \
  template T                                                            \
  primitives<Device::CANN>::max(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CANN>::add(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CANN>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::add_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size,     \
                                                dim_t b_size,           \
                                                bool synchronous);      \
  template void                                                         \
  primitives<Device::CANN>::add_depth_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CANN>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CANN>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::max(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CANN>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::mul(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CANN>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CANN>::mul_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CANN>::penalize_previous_tokens(T*,                \
                                                     const T*,          \
                                                     const int32_t*,    \
                                                     T,                 \
                                                     dim_t,             \
                                                     dim_t,             \
                                                     dim_t);            \
  template void                                                         \
  primitives<Device::CANN>::transpose_2d(const T* a,                    \
                                         const dim_t* dims,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CANN>::transpose_3d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CANN>::transpose_4d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::CANN>::copy<T>(const T*, T*, dim_t); \
  template void                                                         \
  cross_device_primitives<Device::CANN, Device::CPU>::copy<T>(const T*, T*, dim_t);
  DECLARE_ALL_TYPES(DECLARE_IMPL)
}
