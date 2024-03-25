#include "ctranslate2/ops/matmul.h"
#ifdef CT2_WITH_CANN
#include "../cann/utils.h"
#include "ctranslate2/ops/mul.h"
#endif
#include "dispatch.h"

namespace ctranslate2 {
    namespace ops {

        MatMul::MatMul(bool trans_a, bool trans_b, float alpha)
                : _trans_a(trans_a)
                , _trans_b(trans_b)
                , _alpha(alpha) {
        }

        void MatMul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
            PROFILE("MatMul");
            DEVICE_AND_FLOAT_DISPATCH("MatMul", a.device(), a.dtype(), (compute<D, T>(a, b, c)));
        }

        template<Device D, typename T>
        void MatMul::handleNonCann(const StorageView &a, const StorageView &b, StorageView &c, dim_t m, dim_t n,
                                   const dim_t k,
                                   const dim_t a_batch_size) const {
            const dim_t batch_size = a_batch_size;
            const dim_t lda = _trans_a ? m : k;
            const dim_t ldb = _trans_b ? k : n;
            const dim_t ldc = n;
            const float beta = 0;

            if (batch_size > 1) {
                const dim_t stridea = m * k;
                const dim_t strideb = k * n;
                const dim_t stridec = m * n;
                primitives<D>::gemm_batch_strided(_trans_a, _trans_b,
                                                  m, n, k,
                                                  _alpha,
                                                  a.data<T>(), lda, stridea,
                                                  b.data<T>(), ldb, strideb,
                                                  beta,
                                                  c.data<T>(), ldc, stridec,
                                                  batch_size);
            } else {
                primitives<D>::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                                                    _trans_a, _trans_b,
                                                    m, n, k,
                                                    _alpha,
                                                    a.data<T>(), lda,
                                                    b.data<T>(), ldb,
                                                    beta,
                                                    c.data<T>(), ldc);
            }
        }

        template<typename T>
        void MatMul::handleCann(const StorageView &a, const StorageView &b, StorageView &c) const {
#ifdef CT2_WITH_CANN
            const auto aclType = cann::getACLType<T>();
            aclFormat format = ACL_FORMAT_ND;

            cann::CannPreparation prepare;

            ACL_CALL(aclopSetAttrBool(prepare._opAttr, "adj_x1", _trans_a));
            ACL_CALL(aclopSetAttrBool(prepare._opAttr, "adj_x2", _trans_b));

            cann_prepare_inputdesc(prepare, aclType, a.shape().size(), a.shape().data(), format);
            cann_prepare_inputdesc(prepare, aclType, b.shape().size(), b.shape().data(), format);
            cann_prepare_outputdesc(prepare, aclType, c.shape().size(), c.shape().data(), format);

            cann_prepare_inputbuffer(prepare, const_cast<T*>(a.data<T>()), a.size_in_bytes());
            cann_prepare_inputbuffer(prepare, const_cast<T*>(b.data<T>()), b.size_in_bytes());
            cann_prepare_outputbuffer(prepare, c.data<T>(), c.size_in_bytes());

            ACL_CALL(aclopCompileAndExecute("BatchMatMul",
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
                                  ctranslate2::cann::get_aclrt_stream()));
            if (_alpha != 1) {
              // The Mul operator will synchronize the stream.
              ops::Mul()(c, StorageView(_alpha), c);
            } else {
              ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
            }
#endif
        }

        template <Device D, typename T>
        void MatMul::compute(const StorageView& a, const StorageView& b, StorageView& c) const {
            dim_t m, k_a;
            if (_trans_a) {
                m = a.dim(-1);
                k_a = a.dim(-2);
            } else {
                m = a.dim(-2);
                k_a = a.dim(-1);
            }

            dim_t k_b, n;
            if (_trans_b) {
                n = b.dim(-2);
                k_b = b.dim(-1);
            } else {
                n = b.dim(-1);
                k_b = b.dim(-2);
            }

            if (k_a != k_b)
                throw std::invalid_argument("MatMul: k dimension of inputs a and b should match");

            const dim_t k = k_a;
            const dim_t a_batch_size = a.size() / (m * k);
            const dim_t b_batch_size = b.size() / (k * n);

            if (a_batch_size != b_batch_size)
                throw std::invalid_argument("MatMul: batch dimension of inputs a and b should match");

            {
                Shape output_shape(a.shape());
                output_shape[output_shape.size() - 1] = n;
                output_shape[output_shape.size() - 2] = m;
                c.resize(std::move(output_shape));
            }

            if constexpr (D == Device::CANN) {
                // Employ BatchMatMul directly instead of Gemm since it is significantly faster in CANN
                handleCann<T>(a, b, c);
            }
            else {
                handleNonCann<D,T>(a, b, c, m, n, k, a_batch_size);
            }
        }
    }
}
