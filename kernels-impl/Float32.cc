#include "../low_precision_fully_connected.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace Float32 {
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
                return LowPrecision::Status::NotImplemented;
            }
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const float32_t* input, LowPrecision::Shape shape, float32_t* output, LowPrecision::MemLayout layout){
                return LowPrecision::Status::NotImplemented;
            }
            LowPrecision::Status UnpackOutput(LowPrecision::Method method, const float32_t* input, LowPrecision::Shape shape, float32_t* output){
                return LowPrecision::Status::NotImplemented;
            }
            LowPrecision::Status MultiplyInt8MultiBatched(
                LowPrecision::Method method, 
                const float32_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                float32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params
            ){
                return LowPrecision::Status::NotImplemented;
            }
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                LowPrecision::Method method, 
                const float32_t* input, const int8_t* kernel,
                float32_t* output, const Params params
            ){
                return LowPrecision::Status::NotImplemented;
            }
            LowPrecision::PreprocessType InputPreProcess(LowPrecision::Method method) {
                return LowPrecision::PreprocessType::PaddingAndPacking;
            }
            LowPrecision::PreprocessType FilterPreProcess(LowPrecision::Method method){ return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType OutputPreProcess(LowPrecision::Method method){ return LowPrecision::FullyConnected::BSM::OutputPostProcess(method); }
            LowPrecision::PreprocessType OutputPostProcess(LowPrecision::Method method){
                return LowPrecision::PreprocessType::PaddingIfNeccessery;
            }
            LowPrecision::GEMMType GEMMSupport(LowPrecision::Method method){ return LowPrecision::GEMMType::SupportsGEMM; }
        }
    }
}









