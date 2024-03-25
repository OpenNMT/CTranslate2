#pragma once

#include "./cann_inc.h"
#include "ctranslate2/utils.h"
#include <mutex>

namespace ctranslate2 {
    namespace cann {
#define ACL_CALL(ans)                                                            \
    {                                                                            \
      aclError code = (ans);                                                     \
      if (code != ACL_SUCCESS)                                                   \
        THROW_RUNTIME_ERROR("CANN failed with error " + std::to_string(code));   \
    }
    }
}

namespace ctranslate2 {
  namespace cann {
    struct CannPreparation {
      CannPreparation();
      ~CannPreparation();

      std::vector<aclDataBuffer*> _inputBuffers;
      std::vector<aclDataBuffer*> _outputBuffers;
      std::vector<aclTensorDesc*> _inputDesc;
      std::vector<aclTensorDesc*> _outputDesc;
      aclopAttr* _opAttr;
    };

    template<typename... Args>
    inline void cann_prepare_inputdesc(CannPreparation& prepare, Args... args) {
      auto _rPtr = aclCreateTensorDesc(args...);
      if (_rPtr == nullptr)
        THROW_RUNTIME_ERROR("aclCreateTensorDesc run failed");
      else
        prepare._inputDesc.emplace_back(_rPtr);
    }

    template<typename... Args>
    inline void cann_prepare_outputdesc(CannPreparation& prepare, Args... args) {
      auto _rPtr = aclCreateTensorDesc(args...);
      if (_rPtr == nullptr)
        THROW_RUNTIME_ERROR("aclCreateTensorDesc run failed");
      else
        prepare._outputDesc.emplace_back(_rPtr);
    }

    template<typename... Args>
    inline void cann_prepare_inputbuffer(CannPreparation& prepare, Args... args) {
      auto _rPtr = aclCreateDataBuffer(args...);
      if (_rPtr == nullptr)
        THROW_RUNTIME_ERROR("aclCreateDataBuffer run failed");
      else
        prepare._inputBuffers.emplace_back(_rPtr);
    }

    template<typename... Args>
    inline void cann_prepare_outputbuffer(CannPreparation& prepare, Args... args) {
      auto _rPtr = aclCreateDataBuffer(args...);
      if (_rPtr == nullptr)
        THROW_RUNTIME_ERROR("aclCreateDataBuffer run failed");
      else
        prepare._outputBuffers.emplace_back(_rPtr);
    }

    template<typename... Args>
    inline void cann_const_inputdesc(CannPreparation& prepare, size_t index, Args... args) {
      auto _rPtr = aclSetTensorConst(prepare._inputDesc[index], args...);
      if (_rPtr != ACL_SUCCESS)
        THROW_RUNTIME_ERROR("aclSetTensorConst run failed");
    }

    inline void cann_prepare_inputdescname(CannPreparation& prepare, size_t index, const char* name) {
      aclSetTensorDescName(prepare._inputDesc[index], name);
    }

    inline void cann_prepare_outputdescname(CannPreparation& prepare, size_t index, const char* name) {
      aclSetTensorDescName(prepare._outputDesc[index], name);
    }

    inline void cann_tensor_placement(CannPreparation& prepare, size_t index, aclMemType memType) {
      auto _rPtr = aclSetTensorPlaceMent(prepare._inputDesc[index], memType);
      if (_rPtr != ACL_SUCCESS)
        THROW_RUNTIME_ERROR("aclSetTensorDescName run failed");
    }

    template <typename T>
    aclDataType getACLType();
  }
}

namespace ctranslate2 {
  namespace cann {
    class AclDeviceEnabler {
      public:
        static void acl_initialize();
        static void acl_finalize();
        static void set_allow_acl_finalize(bool enable);

        AclDeviceEnabler() = delete;
      private:
        static inline bool _finalize_enabled = true; // False value only during testing
    };
  }
}

namespace ctranslate2 {
  namespace cann {
    aclrtStream get_aclrt_stream();
    uint32_t get_npu_count();
    bool has_npu();
  }
}

namespace ctranslate2 {
  namespace cann {
    class AclrtStreamHandler {
      public:
          static void store(int32_t device, aclrtStream stream);
          static void destroy_steams();
      private:
          inline static std::mutex _mutex;
          inline static std::vector<std::pair<int32_t, aclrtStream>> _streams;
    };
  }
}
