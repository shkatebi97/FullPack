#ifndef _KERNELS_INTERFACE_Float32_H_
#define _KERNELS_INTERFACE_Float32_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace Float32 {
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const float32_t* input, LowPrecision::Shape shape, float32_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status UnpackOutput(LowPrecision::Method method, const float32_t* input, LowPrecision::Shape shape, float32_t* output);
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const float32_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                float32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::PreprocessType InputPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType FilterPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType OutputPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType OutputPostProcess(LowPrecision::Method method);
            LowPrecision::GEMMType GEMMSupport(LowPrecision::Method method);
        }
    }
}

#endif // _KERNELS_INTERFACE_Float32_H_