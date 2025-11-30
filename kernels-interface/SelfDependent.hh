#ifndef _KERNELS_INTERFACE_SelfDependent_H_
#define _KERNELS_INTERFACE_SelfDependent_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        LowPrecision::SelfDependentType IsSelfDependent(Method method);
        namespace SelfDependent {
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                LowPrecision::Method method, 
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                LowPrecision::Method method, 
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const LowPrecision::Params params
            );
            LowPrecision::PreprocessType InputPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType FilterPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType OutputPreProcess(LowPrecision::Method method);
            LowPrecision::PreprocessType OutputPostProcess(LowPrecision::Method method);
            LowPrecision::GEMMType GEMMSupport(LowPrecision::Method method);
            LowPrecision::LeastSize MethodLeaseSize(LowPrecision::Method method);
            namespace W4A4{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
                Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const LowPrecision::Params params);
                void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
                LowPrecision::LeastSize MethodLeaseSize();
            }
            namespace W4A8{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
                Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const LowPrecision::Params params);
                void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
            }
            namespace W8A4{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
                Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const LowPrecision::Params params);
                void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
            }
            namespace W2A2{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
                Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                );
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const LowPrecision::Params params);
                void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
            }
        }
        }
}

#endif // _KERNELS_INTERFACE_SelfDependent_H_