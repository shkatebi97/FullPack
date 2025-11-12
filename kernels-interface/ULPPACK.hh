#ifndef _KERNELS_INTERFACE_ULPPACK_H_
#define _KERNELS_INTERFACE_ULPPACK_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout, size_t Wb, size_t Ab);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout, size_t Wb, size_t Ab);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const LowPrecision::Params params);
            LowPrecision::PreprocessType InputPreProcess();
            LowPrecision::PreprocessType FilterPreProcess();
            LowPrecision::PreprocessType OutputPreProcess();
            LowPrecision::PreprocessType OutputPostProcess();
            LowPrecision::GEMMType GEMMSupport();
        }
    }
}

#endif // _KERNELS_INTERFACE_ULPPACK_H_