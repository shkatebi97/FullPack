#include "../low_precision_fully_connected.h"
#include "ULPPACK/test.h"
#include "ULPPACK/ULPPACK.h"
#ifdef IS_ARM64
namespace LowPrecision{
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                bool is_multibatch = n_dims > 1 && shape[n_dims - 2] > 1;
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout, size_t Wb, size_t Ab){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[1] % 8)
                    return Status::SizesMisMatch;
                if (k_shape.size[0] % 8)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    std::copy(input, input + k_shape.flatsize, output);
                }
                else {
                    // doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1]);
                    uint8_t* input_c = const_cast<uint8_t*>(get_pointer_as<uint8_t>(input));
                    uint8_t* output_c = get_pointer_as<uint8_t>(output);
                    pack_qnnpack4x8multi_type2_rhs(input_c, output_c, k_shape.size[0], k_shape.size[1], std::max(Wb,Ab));
                }
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout, size_t Wb, size_t Ab){
                if (shape.size[shape.number_dims - 1] % 8)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                if (is_multibatched && shape.size[0] % 8)
                    return Status::SizesMisMatch; 
                
                uint8_t* output_c = get_pointer_as<uint8_t>(output);
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    std::copy(input, input + shape.flatsize, output);
                }
                else {
                    if (is_multibatched){
                        uint8_t* input_casted = const_cast<uint8_t*>(get_pointer_as<uint8_t>(input));
                        // std::copy(input, input + shape.flatsize, t + 0 * shape.flatsize);
                        // ULPPACK-Type2 LHS packing function works inplace!
                        pack_qnnpack4x8multi_type2_lhs(input_casted, output_c, shape.size[0], shape.size[1], std::max(Wb,Ab));
                    }
                    else{
                        uint8_t* t = allocate<uint8_t>(shape.flatsize * 8);
                        std::copy(input, input + shape.flatsize, t + 0 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 1 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 2 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 3 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 4 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 5 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 6 * shape.flatsize);
                        std::copy(input, input + shape.flatsize, t + 7 * shape.flatsize);
                        pack_qnnpack4x8multi_type2_lhs(t, output_c, 8, shape.size[shape.number_dims - 1], std::max(Wb,Ab));
                    }
                }
                return Status::Success;
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            ){
                /*
                 * N: Batch Size
                 * M: Output Size
                 * K: Input Size
                 */
                size_t N = 8, M = kernel_shape.size[0], K = input_shape.size[input_shape.number_dims - 1];
                // size_t Wb = 3, Ab = 3;

                if (Wb > 7 || Ab > 7)
                    return Status::NotSupported;
                if (M != kernel_shape.size[1])
                    return Status::SizesMisMatch;
                if(M == 0 || K == 0)
                    return Status::Success;

                if (N % 8)
                    return Status::SizesMisMatch;
                if (M % 4)
                    return Status::SizesMisMatch;
                // if (K % 512)
                //     return Status::SizesMisMatch;

                uint8_t* A = const_cast<uint8_t*>(get_pointer_as<uint8_t>(kernel));
                uint8_t* B_before_pack = const_cast<uint8_t*>(get_pointer_as<uint8_t>(input));
                int32_t* C = output;
                
                size_t mr_block_size = 4, nr_block_size = 8, kr_block_size = 512;
                size_t iter = iter_cnt[Wb-1][Ab-1];

                void (*ukernel)(size_t,size_t,size_t,const uint8_t*,size_t,const uint8_t*,int32_t*,size_t);
                if (iter == 8) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter8;
                else if (iter == 4) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter4;
                else if (iter == 2) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter2;
                else if (iter == 1) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter1;
                else return Status::NotSupported;

                mat_initialize(C,M,N,-1);

                uint8_t *B = B_before_pack;
                pack_qnnpack4x8multi(B_before_pack,B,K,N);

                for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                    for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                        for (size_t kr_block_start = 0; kr_block_start < K; kr_block_start += kr_block_size) {
                            ukernel(
                            mr_block_size,
                            nr_block_size,
                            std::min(kr_block_size, K-kr_block_start),
                            A + mr_block_start * K + kr_block_start,
                            K,
                            B + nr_block_start * K + nr_block_size * kr_block_start,
                            C + mr_block_start * N + nr_block_start,
                            N
                            );
                        }
                    }
                }

                return Status::Success;
            }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            ){
                /*
                 * N: Batch Size
                 * M: Output Size
                 * K: Input Size
                 */
                size_t N = input_shape.size[0], M = kernel_shape.size[1], K = input_shape.size[input_shape.number_dims - 1];

                if (Wb > 7 || Ab > 7)
                    return Status::NotSupported;
                if (K != kernel_shape.size[0])
                    return Status::SizesMisMatch;
                if(M == 0 || K == 0)
                    return Status::Success;
                if (N % 8)
                    return Status::SizesMisMatch;
                if (K % 4)
                    return Status::SizesMisMatch;

                uint8_t* A = const_cast<uint8_t*>(get_pointer_as<uint8_t>(kernel));
                uint8_t* B = const_cast<uint8_t*>(get_pointer_as<uint8_t>(input));
                int32_t* C = output;
                
                size_t mr_block_size = 4, nr_block_size = 8, kr_block_size = 512;
                size_t iter = iter_cnt_type2[Wb-1][Ab-1];
                const int shift = 16-std::max(Wb,Ab);

                void (*ukernel)(size_t,size_t,size_t,const uint8_t*,size_t,const uint8_t*,int32_t*,size_t,int);
                if (iter == 8) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter8;
                else if (iter == 4) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter4;
                else if (iter == 2) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter2;
                else if (iter == 1) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter1;
                else return Status::NotSupported;
                
                // mat_initialize(C,M,N,-1,1);

                for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                    for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                        // for (size_t kr_block_start = 0; kr_block_start < K; kr_block_start += kr_block_size) {
                            ukernel(
                            mr_block_size,
                            nr_block_size,
                            K,
                            A + mr_block_start * K,
                            K,
                            B + nr_block_start * K,
                            C + mr_block_start * N + nr_block_start,
                            N,
                            shift
                            );
                        // }
                    }
                }

                return Status::Success;
            }
            LowPrecision::PreprocessType InputPreProcess() { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType FilterPreProcess(){ return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType OutputPreProcess(){ return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::OutputPostProcess(); }
            LowPrecision::PreprocessType OutputPostProcess(){ return LowPrecision::PreprocessType::PaddingIfNeccessery; }
            LowPrecision::GEMMType GEMMSupport(){ return LowPrecision::GEMMType::SupportsGEMM; }
        }
    }
}
#endif
#ifdef IS_ARM32
namespace LowPrecision{
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                bool is_multibatch = n_dims > 1 && shape[n_dims - 2] > 1;
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                return Status::NotImplemented;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                return Status::NotImplemented;
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            ){
                return Status::NotImplemented;
            }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            ){
                return Status::NotImplemented;
            }
        }
    }
}
#endif
