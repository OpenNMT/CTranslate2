#include "./utils.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include <spdlog/spdlog.h>

namespace ctranslate2 {
    namespace cann {
        CannPreparation::CannPreparation() {
          _opAttr = aclopCreateAttr();
          if(_opAttr == nullptr)
            THROW_RUNTIME_ERROR("aclopCreateAttr out of memory");
        }

        CannPreparation::~CannPreparation() {
          for (auto desc : _inputDesc)
            aclDestroyTensorDesc(desc);

          for (auto desc : _outputDesc)
            aclDestroyTensorDesc(desc);

          try {
            for (auto buf : _inputBuffers)
              ACL_CALL(aclDestroyDataBuffer(buf));

            for (auto buf : _outputBuffers)
              ACL_CALL(aclDestroyDataBuffer(buf));
          }
          catch (const std::exception& e) {
            // Log that CannPreparation deallocation failed and swallow the exception
            spdlog::error(e.what());
          }
          aclopDestroyAttr(_opAttr);
        }

        template <typename T>
        aclDataType getACLType() {
            return ACL_DT_UNDEFINED;
        }

        #define GET_ACL_TYPE(ctranslate2_type, cann_type) \
          template <>                                     \
          aclDataType getACLType<ctranslate2_type>() {    \
            return cann_type;                             \
          }

        GET_ACL_TYPE(int8_t, ACL_INT8);
        GET_ACL_TYPE(int16_t, ACL_INT16);
        GET_ACL_TYPE(int32_t, ACL_INT32);
        GET_ACL_TYPE(int64_t, ACL_INT64);
        GET_ACL_TYPE(uint8_t, ACL_UINT8);
        GET_ACL_TYPE(uint16_t, ACL_UINT16);
        GET_ACL_TYPE(uint32_t, ACL_UINT32);
        GET_ACL_TYPE(uint64_t, ACL_UINT64);
        GET_ACL_TYPE(float, ACL_FLOAT);
        GET_ACL_TYPE(float16_t, ACL_FLOAT16);
        GET_ACL_TYPE(bfloat16_t, ACL_BF16);
        GET_ACL_TYPE(double, ACL_DOUBLE);
        GET_ACL_TYPE(bool, ACL_BOOL);
    }
}

namespace ctranslate2 {
    namespace cann {
        void AclDeviceEnabler::acl_initialize() {
          static std::once_flag initialize_flag;
          std::call_once(initialize_flag, [](){
            spdlog::debug("aclInit");
            ACL_CALL(aclInit(nullptr));
          });
        }

        struct AclInitializer {
          AclInitializer() {
            AclDeviceEnabler::acl_initialize();
          }
        };
        // Initializes AscendCL. It can be called only once per execution.
        // aclInit must be called before the use of AscendCL APIs.
        const static AclInitializer aclInitializer;

        void AclDeviceEnabler::acl_finalize() {
          if(!_finalize_enabled)
            return;

          static std::once_flag finalize_flag;
          std::call_once(finalize_flag, [](){
            try {
              // Make sure all streams are destroyed before AscendCL deinitializing
              AclrtStreamHandler::destroy_steams();
              spdlog::debug("aclFinalize");
              ACL_CALL(aclFinalize());
            }
            catch (const std::exception& e) {
              // acl_finalize is called in ReplicaPool dtor
              // Log that deinitialization failed and swallow the exception
              spdlog::error(e.what());
            }
          });
        }
        
        void AclDeviceEnabler::set_allow_acl_finalize(const bool enable) {
          _finalize_enabled = enable;
        }

        void AclrtStreamHandler::store(const int32_t device, const aclrtStream stream) {
            const std::lock_guard<std::mutex> lock(_mutex);
            _streams.emplace_back(device, stream);
        }

        void AclrtStreamHandler::destroy_steams() {
            const std::lock_guard<std::mutex> lock(_mutex);
            for(const auto& [device, stream] : _streams) {
                ScopedDeviceSetter scoped_device_setter(Device::CANN, device);
                // Synchronize stream to ensure that all tasks in the stream have completed before destroying it
                ACL_CALL(aclrtSynchronizeStream(stream));
                ACL_CALL(aclrtDestroyStream(stream));
            }
        }

        class AclrtStream {
          public:
              AclrtStream() {
                  ACL_CALL(aclrtGetDevice(&_device));
                  ACL_CALL(aclrtCreateStream(&_stream));
                  AclrtStreamHandler::store(_device, _stream);
              }

              // Place the stream destruction responsibility to AclrtStreamHandler to ensure
              // streams are destroyed just before AscendCL deinitialization

              aclrtStream get() const {
                  return _stream;
              }

          private:
              int32_t _device;
              aclrtStream _stream;
        };

        // We create one aclrt handle per host thread. The handle is destroyed when the thread exits.
        aclrtStream get_aclrt_stream() {
            static thread_local AclrtStream aclrt_stream;
            return aclrt_stream.get();
        }

        uint32_t get_npu_count() {
            uint32_t npu_count = 0;
            ACL_CALL(aclrtGetDeviceCount(&npu_count));
            return npu_count;
        }

        bool has_npu() {
            return get_npu_count() > 0;
        }
    }
}
