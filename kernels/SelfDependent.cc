#include "../low_precision_fully_connected.h"


#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::QuantizeFilter(input, k_shape, output, layout);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::QuantizeFilter(input, k_shape, output, layout);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::QuantizeInput(input, shape, output, layout);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::QuantizeInput(input, shape, output, layout);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            Status MultiplyInt8SingleBatch(
                LowPrecision::Method method, 
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            ){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::MultiplyInt8SingleBatch(input, input_shape, kernel, kernel_shape, output, output_shape);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params
            ){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params
            ){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                LowPrecision::Method method, 
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){
                switch (method)
                {
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::W4A4::MultiplyInt8MultiBatchedBlock(input, kernel, output, params);
                default:
                    return LowPrecision::Status::NotSupported;
                }
            }
            LowPrecision::PreprocessType InputPreProcess()  { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType FilterPreProcess() { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType OutputPreProcess() { return OutputPostProcess(); }
            LowPrecision::PreprocessType OutputPostProcess(){ return LowPrecision::PreprocessType::PaddingIfNeccessery;}
            LowPrecision::GEMMType GEMMSupport(){ return LowPrecision::GEMMType::SupportsGEMM; }
        }
    }
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            ){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            ){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            ){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params){ return LowPrecision::Status::NotImplemented; }
            LowPrecision::PreprocessType InputPreProcess()  { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType FilterPreProcess() { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType OutputPreProcess() { return OutputPostProcess(); }
            LowPrecision::PreprocessType OutputPostProcess(){ return LowPrecision::PreprocessType::PaddingIfNeccessery;}
            LowPrecision::GEMMType GEMMSupport(){ return LowPrecision::GEMMType::SupportsGEMMAndGEMV; }
        }
    }
}
#endif









