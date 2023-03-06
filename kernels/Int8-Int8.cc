#include "../low_precision_fully_connected.h"
#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        using ::LowPrecision::MulParams;
        namespace Int8InputsInt8WeightsBarrelShiftMul{
            #define Int8InputsInt8WeightsBarrelShiftMul_SimpleUnpack 0
            #define Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack 1
            #define Int8InputsInt8WeightsBarrelShiftMul_UnpackWithSmallStore 1
            #define Int8InputsInt8WeightsBarrelShiftMul_UnpackWithTLB 0 // This is not implemented!
            #define Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad 0
            size_t TransformFilterShape(int* shape, int n_dims){
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            template <> LowPrecision::Status QuantizeFilter<uint8_t>(const uint8_t* input, Shape k_shape, uint8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 8)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                size_t N = k_shape.size[1],
                       K = k_shape.size[0];
                size_t p = 0;
                for (size_t j=0;j<N;j+=8)
                    for (size_t i=0;i<K;i++)
                        for (size_t k=j;k<j+8;k++)
                            output[p++] = input[i*N+k];
                return Status::Success;
            }
            template <> LowPrecision::Status QuantizeFilter<int8_t>(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 8)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                size_t N = k_shape.size[1],
                       K = k_shape.size[0];
                size_t p = 0;
                for (size_t j=0;j<N;j+=8)
                    for (size_t i=0;i<K;i++)
                        for (size_t k=j;k<j+8;k++)
                            output[p++] = input[i*N+k];
                return Status::Success;
            }
            template <> LowPrecision::Status QuantizeInput<uint8_t>(const uint8_t* input, Shape shape, uint8_t* output, MemLayout layout){
                if (shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                size_t K = shape.size[1],
                       M = shape.size[0];
                size_t p = 0;
                for (size_t j=0;j<M;j+=8)
                    for (size_t i=0;i<K;i++)
                        for (size_t k=j;k<j+8;k++)
                            output[p++] = input[k*K+i];
                return Status::Success;
            }
            template <> LowPrecision::Status QuantizeInput<int8_t>(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                size_t K = shape.size[1],
                       M = shape.size[0];
                size_t p = 0;
                for (size_t j = 0 ; j < M ; j += 8)
                    for (size_t i = 0 ; i < K ; i++)
                        for (size_t k = j ; k < j + 8 ; k++)
                            output[p++] = input[k * K + i];
                return Status::Success;
            }
            LowPrecision::Status UnpackOutput(const int32_t* input, Shape shape, int32_t* output){
                #if Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack
                return Status::NotNeeded;
                #else
                if (shape.number_dims != 2 || shape.size[1] <= 1)
                    return Status::DimensionsMisMatch;
                size_t N = shape.size[1],
                       M = shape.size[0];
                size_t p = 0;
                for (size_t i = 0 ; i < M ; i += 8)
                    for (size_t j = 0 ; j < N ; j += 8)
                        unpack_8x8_block_barrelshift(&input[i * N + j], &output[i * N + j], N);
                return Status::Success;
                #endif
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                return Status::NotImplemented;
            }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                MulParams params
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];

                int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                
                if (lhs_columns != rhs_rows)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_columns == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 8)
                    return Status::NotSupported;

                const int M        = lhs_batches,
                          K        = lhs_columns,
                          N        = rhs_columns,
                          a_stride = lhs_columns,
                          c_stride = N;
                
                const int8_t* w = kernel;
                
                #if Int8InputsInt8WeightsBarrelShiftMul_UnpackWithSmallStore == 1 && Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack == 1
                    int16x8_t vACC_Ar76543210_x_Wc76543210 = veorq_s32(vACC_Ar76543210_x_Wc76543210, vACC_Ar76543210_x_Wc76543210); 
                    int16x8_t vACC_Ar76543210_x_Wc07654321 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc10765432 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc21076543 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc32107654 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc43210765 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc54321076 = vACC_Ar76543210_x_Wc76543210; 
                    int16x8_t vACC_Ar76543210_x_Wc65432107 = vACC_Ar76543210_x_Wc76543210;
                    #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                    int8x16_t vWc15T0    = vld1q_s8((int8_t*)w); w += 16;
                    int8x8_t vWc76543210 = vget_low_u8(vWc15T0);
                    #else
                    int8x8_t vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                    #endif
                    int8x8_t vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                    int8x8_t vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                    int8x8_t vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                    int8x8_t vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                    int8x8_t vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                    int8x8_t vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                    int8x8_t vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);
                #endif

                int mr_block_size = 8, nr_block_size = 8;
                for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                    for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                                      w = kernel + nr_block_start * K;
                        const int8_t* a = input  + mr_block_start * K;
                        int32_t*      c = output + mr_block_start * N + nr_block_start;
                        int k = K;
                        #if Int8InputsInt8WeightsBarrelShiftMul_UnpackWithSmallStore == 1 && Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack == 1
                        #else
                        int16x8_t vACC_Ar76543210_x_Wc76543210 = veorq_s32(vACC_Ar76543210_x_Wc76543210, vACC_Ar76543210_x_Wc76543210); 
                        int16x8_t vACC_Ar76543210_x_Wc07654321 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc10765432 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc21076543 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc32107654 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc43210765 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc54321076 = vACC_Ar76543210_x_Wc76543210; 
                        int16x8_t vACC_Ar76543210_x_Wc65432107 = vACC_Ar76543210_x_Wc76543210;
                        #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                        int8x16_t vWc15T0    = vld1q_s8((int8_t*)w); w += 16;
                        int8x8_t vWc76543210 = vget_low_u8(vWc15T0);
                        #else
                        int8x8_t vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                        #endif
                        int8x8_t vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                        int8x8_t vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                        int8x8_t vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                        int8x8_t vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                        int8x8_t vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                        int8x8_t vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                        int8x8_t vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);
                        #if Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack == 1 && Int8InputsInt8WeightsBarrelShiftMul_UnpackWithTLB == 1
                        uint8x16_t vidxs;
                        asm volatile(
                            "mov x0, 0xFFFF1110FFFF0100\n\t"
                            "ins %[vidxs].d[0], x0\n\t"
                            "mov x0, 0xFFFF3130FFFF2120\n\t"
                            "ins %[vidxs].d[1], x0\n\t"
                            :[ vidxs ]"=w"( vidxs )::"x0"
                        );
                        #endif
                        #endif

                        for (; k >= 8; k -= 8) { // 158 - 52 = 106
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                int8x16_t vAr15T0 = vld1q_s8((int8_t*)a); a += 16;
                                int8x8_t vAr76543210 = vget_low_u8(vAr15T0);
                            #else
                                int8x8_t vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_high_u8(vWc15T0);
                                vWc15T0     = vld1q_s8((int8_t*)w); w += 16;
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);

                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_high_u8(vAr15T0);
                                vAr15T0     = vld1q_s8((int8_t*)a); a += 16;
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_low_u8(vWc15T0);
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);

                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_low_u8(vAr15T0);
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_high_u8(vWc15T0);
                                vWc15T0     = vld1q_s8((int8_t*)w); w += 16;
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);


                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_high_u8(vAr15T0);
                                vAr15T0     = vld1q_s8((int8_t*)a); a += 16;
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_low_u8(vWc15T0);
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);

                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_low_u8(vAr15T0);
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_high_u8(vWc15T0);
                                vWc15T0     = vld1q_s8((int8_t*)w); w += 16;
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);

                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_high_u8(vAr15T0);
                                vAr15T0     = vld1q_s8((int8_t*)a); a += 16;
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_low_u8(vWc15T0);
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);

                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_low_u8(vAr15T0);
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_high_u8(vWc15T0);
                                vWc15T0     = vld1q_s8((int8_t*)w); w += 16;
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);
    
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vAr76543210 = vget_high_u8(vAr15T0);
                            #else
                                vAr76543210 = vld1_s8((int8_t*)a); a += 8;
                            #endif
                            vACC_Ar76543210_x_Wc76543210 = vmlal_s8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            #if Int8InputsInt8WeightsBarrelShiftMul_UseUInt8x16VectorsForLoad
                                vWc76543210 = vget_low_u8(vWc15T0);
                            #else
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;
                            #endif
                            vACC_Ar76543210_x_Wc07654321 = vmlal_s8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_s8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_s8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_s8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_s8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_s8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_s8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);
                        }

                        #if Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack == 1 && Int8InputsInt8WeightsBarrelShiftMul_UnpackWithSmallStore == 1
                        #else
                        int32_t* c0 = c;
                        int32_t* c1 = c0 + c_stride;
                        int32_t* c2 = c1 + c_stride;
                        int32_t* c3 = c2 + c_stride;
                        int32_t* c4 = c3 + c_stride;
                        int32_t* c5 = c4 + c_stride;
                        int32_t* c6 = c5 + c_stride;
                        int32_t* c7 = c6 + c_stride;
                        #endif

                        #if Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack
                            #if Int8InputsInt8WeightsBarrelShiftMul_UnpackWithSmallStore
                                int16_t* cp0 = get_pointer_as<int16_t>(c);
                                int16_t* cp1 = cp0 + 2 * c_stride; // get_pointer_as<int16_t>(c0 + c_stride);
                                int16_t* cp2 = cp1 + 2 * c_stride; // get_pointer_as<int16_t>(c1 + c_stride);
                                int16_t* cp3 = cp2 + 2 * c_stride; // get_pointer_as<int16_t>(c2 + c_stride);
                                int16_t* cp4 = cp3 + 2 * c_stride; // get_pointer_as<int16_t>(c3 + c_stride);
                                int16_t* cp5 = cp4 + 2 * c_stride; // get_pointer_as<int16_t>(c4 + c_stride);
                                int16_t* cp6 = cp5 + 2 * c_stride; // get_pointer_as<int16_t>(c5 + c_stride);
                                int16_t* cp7 = cp6 + 2 * c_stride; // get_pointer_as<int16_t>(c6 + c_stride);

                                vst1q_lane_s16(cp0 + 0 , vACC_Ar76543210_x_Wc76543210, 0);
                                vst1q_lane_s16(cp0 + 2 , vACC_Ar76543210_x_Wc07654321, 0);
                                vst1q_lane_s16(cp0 + 4 , vACC_Ar76543210_x_Wc10765432, 0);
                                vst1q_lane_s16(cp0 + 6 , vACC_Ar76543210_x_Wc21076543, 0);
                                
                                vWc76543210 = vld1_s8((int8_t*)w); w += 8;

                                vst1q_lane_s16(cp0 + 8 , vACC_Ar76543210_x_Wc32107654, 0);
                                vst1q_lane_s16(cp0 + 10, vACC_Ar76543210_x_Wc43210765, 0);
                                vst1q_lane_s16(cp0 + 12, vACC_Ar76543210_x_Wc54321076, 0);
                                vst1q_lane_s16(cp0 + 14, vACC_Ar76543210_x_Wc65432107, 0);


                                vWc07654321 = vext_s8(vWc76543210, vWc76543210, 1);
                                vst1q_lane_s16(cp1 + 0 , vACC_Ar76543210_x_Wc65432107, 1);
                                vst1q_lane_s16(cp1 + 2 , vACC_Ar76543210_x_Wc76543210, 1);
                                vst1q_lane_s16(cp1 + 4 , vACC_Ar76543210_x_Wc07654321, 1);
                                vst1q_lane_s16(cp1 + 6 , vACC_Ar76543210_x_Wc10765432, 1);

                                vst1q_lane_s16(cp1 + 8 , vACC_Ar76543210_x_Wc21076543, 1);
                                vst1q_lane_s16(cp1 + 10, vACC_Ar76543210_x_Wc32107654, 1);
                                vst1q_lane_s16(cp1 + 12, vACC_Ar76543210_x_Wc43210765, 1);
                                vst1q_lane_s16(cp1 + 14, vACC_Ar76543210_x_Wc54321076, 1);


                                vWc10765432 = vext_s8(vWc76543210, vWc76543210, 2);
                                vst1q_lane_s16(cp2 + 0 , vACC_Ar76543210_x_Wc54321076, 2);
                                vst1q_lane_s16(cp2 + 2 , vACC_Ar76543210_x_Wc65432107, 2);
                                vst1q_lane_s16(cp2 + 4 , vACC_Ar76543210_x_Wc76543210, 2);
                                vst1q_lane_s16(cp2 + 6 , vACC_Ar76543210_x_Wc07654321, 2);

                                vst1q_lane_s16(cp2 + 8 , vACC_Ar76543210_x_Wc10765432, 2);
                                vst1q_lane_s16(cp2 + 10, vACC_Ar76543210_x_Wc21076543, 2);
                                vst1q_lane_s16(cp2 + 12, vACC_Ar76543210_x_Wc32107654, 2);
                                vst1q_lane_s16(cp2 + 14, vACC_Ar76543210_x_Wc43210765, 2);


                                vWc21076543 = vext_s8(vWc76543210, vWc76543210, 3);
                                vst1q_lane_s16(cp3 + 0 , vACC_Ar76543210_x_Wc43210765, 3);
                                vst1q_lane_s16(cp3 + 2 , vACC_Ar76543210_x_Wc54321076, 3);
                                vst1q_lane_s16(cp3 + 4 , vACC_Ar76543210_x_Wc65432107, 3);
                                vst1q_lane_s16(cp3 + 6 , vACC_Ar76543210_x_Wc76543210, 3);

                                vst1q_lane_s16(cp3 + 8 , vACC_Ar76543210_x_Wc07654321, 3);
                                vst1q_lane_s16(cp3 + 10, vACC_Ar76543210_x_Wc10765432, 3);
                                vst1q_lane_s16(cp3 + 12, vACC_Ar76543210_x_Wc21076543, 3);
                                vst1q_lane_s16(cp3 + 14, vACC_Ar76543210_x_Wc32107654, 3);


                                vWc32107654 = vext_s8(vWc76543210, vWc76543210, 4);
                                vst1q_lane_s16(cp4 + 0 , vACC_Ar76543210_x_Wc32107654, 4);
                                vst1q_lane_s16(cp4 + 2 , vACC_Ar76543210_x_Wc43210765, 4);
                                vst1q_lane_s16(cp4 + 4 , vACC_Ar76543210_x_Wc54321076, 4);
                                vst1q_lane_s16(cp4 + 6 , vACC_Ar76543210_x_Wc65432107, 4);

                                vst1q_lane_s16(cp4 + 8 , vACC_Ar76543210_x_Wc76543210, 4);
                                vst1q_lane_s16(cp4 + 10, vACC_Ar76543210_x_Wc07654321, 4);
                                vst1q_lane_s16(cp4 + 12, vACC_Ar76543210_x_Wc10765432, 4);
                                vst1q_lane_s16(cp4 + 14, vACC_Ar76543210_x_Wc21076543, 4);


                                vWc43210765 = vext_s8(vWc76543210, vWc76543210, 5);
                                vst1q_lane_s16(cp5 + 0 , vACC_Ar76543210_x_Wc21076543, 5);
                                vst1q_lane_s16(cp5 + 2 , vACC_Ar76543210_x_Wc32107654, 5);
                                vst1q_lane_s16(cp5 + 4 , vACC_Ar76543210_x_Wc43210765, 5);
                                vst1q_lane_s16(cp5 + 6 , vACC_Ar76543210_x_Wc54321076, 5);

                                vst1q_lane_s16(cp5 + 8 , vACC_Ar76543210_x_Wc65432107, 5);
                                vst1q_lane_s16(cp5 + 10, vACC_Ar76543210_x_Wc76543210, 5);
                                vst1q_lane_s16(cp5 + 12, vACC_Ar76543210_x_Wc07654321, 5);
                                vst1q_lane_s16(cp5 + 14, vACC_Ar76543210_x_Wc10765432, 5);


                                vWc54321076 = vext_s8(vWc76543210, vWc76543210, 6);
                                vst1q_lane_s16(cp6 + 0 , vACC_Ar76543210_x_Wc10765432, 6);
                                vst1q_lane_s16(cp6 + 2 , vACC_Ar76543210_x_Wc21076543, 6);
                                vst1q_lane_s16(cp6 + 4 , vACC_Ar76543210_x_Wc32107654, 6);
                                vst1q_lane_s16(cp6 + 6 , vACC_Ar76543210_x_Wc43210765, 6);

                                vst1q_lane_s16(cp6 + 8 , vACC_Ar76543210_x_Wc54321076, 6);
                                vst1q_lane_s16(cp6 + 10, vACC_Ar76543210_x_Wc65432107, 6);
                                vst1q_lane_s16(cp6 + 12, vACC_Ar76543210_x_Wc76543210, 6);
                                vst1q_lane_s16(cp6 + 14, vACC_Ar76543210_x_Wc07654321, 6);


                                vWc65432107 = vext_s8(vWc76543210, vWc76543210, 7);
                                vst1q_lane_s16(cp7 + 0 , vACC_Ar76543210_x_Wc07654321, 7);
                                vACC_Ar76543210_x_Wc07654321 = veorq_s32(vACC_Ar76543210_x_Wc07654321, vACC_Ar76543210_x_Wc07654321);
                                vst1q_lane_s16(cp7 + 2 , vACC_Ar76543210_x_Wc10765432, 7);
                                vACC_Ar76543210_x_Wc10765432 = vACC_Ar76543210_x_Wc07654321;
                                vst1q_lane_s16(cp7 + 4 , vACC_Ar76543210_x_Wc21076543, 7);
                                vACC_Ar76543210_x_Wc21076543 = vACC_Ar76543210_x_Wc07654321;
                                vst1q_lane_s16(cp7 + 6 , vACC_Ar76543210_x_Wc32107654, 7);
                                vACC_Ar76543210_x_Wc32107654 = vACC_Ar76543210_x_Wc07654321;

                                vst1q_lane_s16(cp7 + 8 , vACC_Ar76543210_x_Wc43210765, 7);
                                vACC_Ar76543210_x_Wc43210765 = vACC_Ar76543210_x_Wc07654321;
                                vst1q_lane_s16(cp7 + 10, vACC_Ar76543210_x_Wc54321076, 7);
                                vACC_Ar76543210_x_Wc54321076 = vACC_Ar76543210_x_Wc07654321;
                                vst1q_lane_s16(cp7 + 12, vACC_Ar76543210_x_Wc65432107, 7);
                                vACC_Ar76543210_x_Wc65432107 = vACC_Ar76543210_x_Wc07654321;
                                vst1q_lane_s16(cp7 + 14, vACC_Ar76543210_x_Wc76543210, 7);
                                vACC_Ar76543210_x_Wc76543210 = vACC_Ar76543210_x_Wc07654321;
                            #else
                                #if Int8InputsInt8WeightsBarrelShiftMul_UnpackWithTLB
                                    // TODO: Compelete this!
                                    // vACC_Ar76543210_x_Wc76543210
                                    // vACC_Ar76543210_x_Wc07654321
                                    // vACC_Ar76543210_x_Wc10765432
                                    // vACC_Ar76543210_x_Wc21076543
                                    int8x16x4_t vACC_Ar76543210_x_Wc_76543210_07654321_10765432_21076543 = {
                                        vreinterpretq_s8_s16(vACC_Ar76543210_x_Wc76543210),
                                        vreinterpretq_s8_s16(vACC_Ar76543210_x_Wc07654321),
                                        vreinterpretq_s8_s16(vACC_Ar76543210_x_Wc10765432),
                                        vreinterpretq_s8_s16(vACC_Ar76543210_x_Wc21076543)
                                    };
                                    int8x16_t o = vqtbl4q_s8(vACC_Ar76543210_x_Wc_76543210_07654321_10765432_21076543, vidxs);
                                #else
                                    uint16x8_t vACC_Or0C0123,
                                            vACC_Or0C4567,
                                            vACC_Or1C0123,
                                            vACC_Or1C4567,
                                            vACC_Or2C0123,
                                            vACC_Or2C4567,
                                            vACC_Or3C0123,
                                            vACC_Or3C4567,
                                            vACC_Or4C0123,
                                            vACC_Or4C4567,
                                            vACC_Or5C0123,
                                            vACC_Or5C4567,
                                            vACC_Or6C0123,
                                            vACC_Or6C4567,
                                            vACC_Or7C0123,
                                            vACC_Or7C4567;

                                    vACC_Or0C0123 = veorq_u16(vACC_Or0C0123, vACC_Or0C0123);
                                    vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 0, vACC_Ar76543210_x_Wc76543210, 0);
                                    vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 2, vACC_Ar76543210_x_Wc07654321, 0);
                                    vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 4, vACC_Ar76543210_x_Wc10765432, 0);
                                    vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 6, vACC_Ar76543210_x_Wc21076543, 0);
                                    vACC_Or0C4567 = veorq_u16(vACC_Or0C4567, vACC_Or0C4567);
                                    vst1q_s32(c0, vACC_Or0C0123);
                                    vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 0, vACC_Ar76543210_x_Wc32107654, 0);
                                    vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 2, vACC_Ar76543210_x_Wc43210765, 0);
                                    vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 4, vACC_Ar76543210_x_Wc54321076, 0);
                                    vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 6, vACC_Ar76543210_x_Wc65432107, 0);
                                    vACC_Or1C0123 = veorq_u16(vACC_Or1C0123, vACC_Or1C0123);
                                    vst1q_s32(c0+4, vACC_Or0C4567);

                                    vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 0, vACC_Ar76543210_x_Wc65432107, 1);
                                    vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 2, vACC_Ar76543210_x_Wc76543210, 1);
                                    vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 4, vACC_Ar76543210_x_Wc07654321, 1);
                                    vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 6, vACC_Ar76543210_x_Wc10765432, 1);
                                    vACC_Or1C4567 = veorq_u16(vACC_Or1C4567, vACC_Or1C4567);
                                    vst1q_s32(c1, vACC_Or1C0123);
                                    vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 0, vACC_Ar76543210_x_Wc21076543, 1);
                                    vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 2, vACC_Ar76543210_x_Wc32107654, 1);
                                    vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 4, vACC_Ar76543210_x_Wc43210765, 1);
                                    vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 6, vACC_Ar76543210_x_Wc54321076, 1);
                                    vACC_Or2C0123 = veorq_u16(vACC_Or2C0123, vACC_Or2C0123);
                                    vst1q_s32(c1+4, vACC_Or1C4567);

                                    vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 0, vACC_Ar76543210_x_Wc54321076, 2);
                                    vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 2, vACC_Ar76543210_x_Wc65432107, 2);
                                    vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 4, vACC_Ar76543210_x_Wc76543210, 2);
                                    vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 6, vACC_Ar76543210_x_Wc07654321, 2);
                                    vACC_Or2C4567 = veorq_u16(vACC_Or2C4567, vACC_Or2C4567);
                                    vst1q_s32(c2, vACC_Or2C0123);
                                    vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 0, vACC_Ar76543210_x_Wc10765432, 2);
                                    vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 2, vACC_Ar76543210_x_Wc21076543, 2);
                                    vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 4, vACC_Ar76543210_x_Wc32107654, 2);
                                    vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 6, vACC_Ar76543210_x_Wc43210765, 2);
                                    vACC_Or3C0123 = veorq_u16(vACC_Or3C0123, vACC_Or3C0123);
                                    vst1q_s32(c2+4, vACC_Or2C4567);

                                    vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C0123, 0, vACC_Ar76543210_x_Wc43210765, 3);
                                    vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 2, vACC_Ar76543210_x_Wc54321076, 3);
                                    vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 4, vACC_Ar76543210_x_Wc65432107, 3);
                                    vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 6, vACC_Ar76543210_x_Wc76543210, 3);
                                    vACC_Or3C4567 = veorq_u16(vACC_Or3C4567, vACC_Or3C4567);
                                    vst1q_s32(c3, vACC_Or3C0123);
                                    vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 0, vACC_Ar76543210_x_Wc07654321, 3);
                                    vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 2, vACC_Ar76543210_x_Wc10765432, 3);
                                    vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 4, vACC_Ar76543210_x_Wc21076543, 3);
                                    vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 6, vACC_Ar76543210_x_Wc32107654, 3);
                                    vACC_Or4C0123 = veorq_u16(vACC_Or4C0123, vACC_Or4C0123);
                                    vst1q_s32(c3+4, vACC_Or3C4567);

                                    vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 0, vACC_Ar76543210_x_Wc32107654, 4);
                                    vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 2, vACC_Ar76543210_x_Wc43210765, 4);
                                    vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 4, vACC_Ar76543210_x_Wc54321076, 4);
                                    vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 6, vACC_Ar76543210_x_Wc65432107, 4);
                                    vACC_Or4C4567 = veorq_u16(vACC_Or4C4567, vACC_Or4C4567);
                                    vst1q_s32(c4, vACC_Or4C0123);
                                    vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 0, vACC_Ar76543210_x_Wc76543210, 4);
                                    vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 2, vACC_Ar76543210_x_Wc07654321, 4);
                                    vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 4, vACC_Ar76543210_x_Wc10765432, 4);
                                    vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 6, vACC_Ar76543210_x_Wc21076543, 4);
                                    vACC_Or5C0123 = veorq_u16(vACC_Or5C0123, vACC_Or5C0123);
                                    vst1q_s32(c4+4, vACC_Or4C4567);

                                    vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 0, vACC_Ar76543210_x_Wc21076543, 5);
                                    vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 2, vACC_Ar76543210_x_Wc32107654, 5);
                                    vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 4, vACC_Ar76543210_x_Wc43210765, 5);
                                    vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 6, vACC_Ar76543210_x_Wc54321076, 5);
                                    vACC_Or5C4567 = veorq_u16(vACC_Or5C4567, vACC_Or5C4567);
                                    vst1q_s32(c5, vACC_Or5C0123);
                                    vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 0, vACC_Ar76543210_x_Wc65432107, 5);
                                    vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 2, vACC_Ar76543210_x_Wc76543210, 5);
                                    vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 4, vACC_Ar76543210_x_Wc07654321, 5);
                                    vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 6, vACC_Ar76543210_x_Wc10765432, 5);
                                    vACC_Or6C0123 = veorq_u16(vACC_Or6C0123, vACC_Or6C0123);
                                    vst1q_s32(c5+4, vACC_Or5C4567);

                                    vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 0, vACC_Ar76543210_x_Wc10765432, 6);
                                    vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 2, vACC_Ar76543210_x_Wc21076543, 6);
                                    vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 4, vACC_Ar76543210_x_Wc32107654, 6);
                                    vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 6, vACC_Ar76543210_x_Wc43210765, 6);
                                    vACC_Or6C4567 = veorq_u16(vACC_Or6C4567, vACC_Or6C4567);
                                    vst1q_s32(c6, vACC_Or6C0123);
                                    vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 0, vACC_Ar76543210_x_Wc54321076, 6);
                                    vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 2, vACC_Ar76543210_x_Wc65432107, 6);
                                    vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 4, vACC_Ar76543210_x_Wc76543210, 6);
                                    vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 6, vACC_Ar76543210_x_Wc07654321, 6);
                                    vACC_Or7C0123 = veorq_u16(vACC_Or7C0123, vACC_Or7C0123);
                                    vst1q_s32(c6+4, vACC_Or6C4567);

                                    vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 0, vACC_Ar76543210_x_Wc07654321, 7);
                                    vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 2, vACC_Ar76543210_x_Wc10765432, 7);
                                    vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 4, vACC_Ar76543210_x_Wc21076543, 7);
                                    vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 6, vACC_Ar76543210_x_Wc32107654, 7);
                                    vACC_Or7C4567 = veorq_u16(vACC_Or7C4567, vACC_Or7C4567);
                                    vst1q_s32(c7, vACC_Or7C0123);
                                    vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 0, vACC_Ar76543210_x_Wc43210765, 7);
                                    vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 2, vACC_Ar76543210_x_Wc54321076, 7);
                                    vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 4, vACC_Ar76543210_x_Wc65432107, 7);
                                    vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 6, vACC_Ar76543210_x_Wc76543210, 7);
                                    vst1q_s32(c7+4, vACC_Or7C4567);
                                #endif
                            #endif
                        #else
                            int32x4_t vACC_Ar76543210_x_Wc76543210_high, vACC_Ar76543210_x_Wc76543210_low,
                                      vACC_Ar76543210_x_Wc07654321_high, vACC_Ar76543210_x_Wc07654321_low,
                                      vACC_Ar76543210_x_Wc10765432_high, vACC_Ar76543210_x_Wc10765432_low,
                                      vACC_Ar76543210_x_Wc21076543_high, vACC_Ar76543210_x_Wc21076543_low,
                                      vACC_Ar76543210_x_Wc32107654_high, vACC_Ar76543210_x_Wc32107654_low,
                                      vACC_Ar76543210_x_Wc43210765_high, vACC_Ar76543210_x_Wc43210765_low,
                                      vACC_Ar76543210_x_Wc54321076_high, vACC_Ar76543210_x_Wc54321076_low,
                                      vACC_Ar76543210_x_Wc65432107_high, vACC_Ar76543210_x_Wc65432107_low;
                            
                            vACC_Ar76543210_x_Wc76543210_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc76543210), 0);
                            vst1q_s32(c0, vACC_Ar76543210_x_Wc76543210_low);
                            vACC_Ar76543210_x_Wc76543210_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc76543210, 0);
                            vst1q_s32(c0+4, vACC_Ar76543210_x_Wc76543210_high);

                            vACC_Ar76543210_x_Wc07654321_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc07654321), 0);
                            vst1q_s32(c1, vACC_Ar76543210_x_Wc07654321_low);
                            vACC_Ar76543210_x_Wc07654321_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc07654321, 0);
                            vst1q_s32(c1+4, vACC_Ar76543210_x_Wc07654321_high);

                            vACC_Ar76543210_x_Wc10765432_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc10765432), 0);
                            vst1q_s32(c2, vACC_Ar76543210_x_Wc10765432_low);
                            vACC_Ar76543210_x_Wc10765432_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc10765432, 0);
                            vst1q_s32(c2+4, vACC_Ar76543210_x_Wc10765432_high);

                            vACC_Ar76543210_x_Wc21076543_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc21076543), 0);
                            vst1q_s32(c3, vACC_Ar76543210_x_Wc21076543_low);
                            vACC_Ar76543210_x_Wc21076543_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc21076543, 0);
                            vst1q_s32(c3+4, vACC_Ar76543210_x_Wc21076543_high);

                            vACC_Ar76543210_x_Wc32107654_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc32107654), 0);
                            vst1q_s32(c4, vACC_Ar76543210_x_Wc32107654_low);
                            vACC_Ar76543210_x_Wc32107654_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc32107654, 0);
                            vst1q_s32(c4+4, vACC_Ar76543210_x_Wc32107654_high);

                            vACC_Ar76543210_x_Wc43210765_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc43210765), 0);
                            vst1q_s32(c5, vACC_Ar76543210_x_Wc43210765_low);
                            vACC_Ar76543210_x_Wc43210765_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc43210765, 0);
                            vst1q_s32(c5+4, vACC_Ar76543210_x_Wc43210765_high);

                            vACC_Ar76543210_x_Wc54321076_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc54321076), 0);
                            vst1q_s32(c6, vACC_Ar76543210_x_Wc54321076_low);
                            vACC_Ar76543210_x_Wc54321076_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc54321076, 0);
                            vst1q_s32(c6+4, vACC_Ar76543210_x_Wc54321076_high);

                            vACC_Ar76543210_x_Wc65432107_low = vshll_n_s16(vget_low_u16(vACC_Ar76543210_x_Wc65432107), 0);
                            vst1q_s32(c7, vACC_Ar76543210_x_Wc65432107_low);
                            vACC_Ar76543210_x_Wc65432107_high = vshll_high_n_s16(vACC_Ar76543210_x_Wc65432107, 0);
                            vst1q_s32(c7+4, vACC_Ar76543210_x_Wc65432107_high);
                        #endif
                    }
                }
                return Status::Success;
            }
            Status MultiplyInt8MultiBatched(
                const uint8_t* input, Shape input_shape,
                const uint8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                MulParams params
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
                int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                
                if (lhs_columns != rhs_rows)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 4)
                    return Status::NotSupported;

                const int M        = lhs_batches,
                          K        = lhs_columns,
                          N        = rhs_columns,
                          a_stride = lhs_columns,
                          c_stride = N;
                bool unpack = Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack;

                if (unpack)
                    std::cout << "Unpacking In Kernel" << std::endl;
                
                int mr_block_size = 8, nr_block_size = 8;
                for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                    for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                        const uint8_t* w = kernel + nr_block_start * K;
                        const uint8_t* a = input  + mr_block_start * K;
                        int32_t*       c = output + mr_block_start * N + nr_block_start;
                        int k = K;

                        uint16x8_t vACC_Ar76543210_x_Wc76543210 = veorq_u32(vACC_Ar76543210_x_Wc76543210, vACC_Ar76543210_x_Wc76543210); 
                        uint16x8_t vACC_Ar76543210_x_Wc07654321 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc10765432 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc21076543 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc32107654 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc43210765 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc54321076 = vACC_Ar76543210_x_Wc76543210; 
                        uint16x8_t vACC_Ar76543210_x_Wc65432107 = vACC_Ar76543210_x_Wc76543210;

                        uint8x8_t vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                        uint8x8_t vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                        uint8x8_t vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                        uint8x8_t vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                        uint8x8_t vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                        uint8x8_t vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                        uint8x8_t vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                        uint8x8_t vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);
                        
                        for (; k >= 8; k -= 8) {
                            uint8x8_t vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);

                            vAr76543210 = vld1_u8((uint8_t*)a); a += 8;
                            vACC_Ar76543210_x_Wc76543210 = vmlal_u8(vACC_Ar76543210_x_Wc76543210, vWc76543210, vAr76543210);
                            vWc76543210 = vld1_u8((uint8_t*)w); w += 8;
                            vACC_Ar76543210_x_Wc07654321 = vmlal_u8(vACC_Ar76543210_x_Wc07654321, vWc07654321, vAr76543210);
                            vWc07654321 = vext_u8(vWc76543210, vWc76543210, 1);
                            vACC_Ar76543210_x_Wc10765432 = vmlal_u8(vACC_Ar76543210_x_Wc10765432, vWc10765432, vAr76543210);
                            vWc10765432 = vext_u8(vWc76543210, vWc76543210, 2);
                            vACC_Ar76543210_x_Wc21076543 = vmlal_u8(vACC_Ar76543210_x_Wc21076543, vWc21076543, vAr76543210);
                            vWc21076543 = vext_u8(vWc76543210, vWc76543210, 3);
                            vACC_Ar76543210_x_Wc32107654 = vmlal_u8(vACC_Ar76543210_x_Wc32107654, vWc32107654, vAr76543210);
                            vWc32107654 = vext_u8(vWc76543210, vWc76543210, 4);
                            vACC_Ar76543210_x_Wc43210765 = vmlal_u8(vACC_Ar76543210_x_Wc43210765, vWc43210765, vAr76543210);
                            vWc43210765 = vext_u8(vWc76543210, vWc76543210, 5);
                            vACC_Ar76543210_x_Wc54321076 = vmlal_u8(vACC_Ar76543210_x_Wc54321076, vWc54321076, vAr76543210);
                            vWc54321076 = vext_u8(vWc76543210, vWc76543210, 6);
                            vACC_Ar76543210_x_Wc65432107 = vmlal_u8(vACC_Ar76543210_x_Wc65432107, vWc65432107, vAr76543210);
                            vWc65432107 = vext_u8(vWc76543210, vWc76543210, 7);
                        }

                        int32_t* c0 = c;
                        int32_t* c1 = c0 + c_stride;
                        int32_t* c2 = c1 + c_stride;
                        int32_t* c3 = c2 + c_stride;
                        int32_t* c4 = c3 + c_stride;
                        int32_t* c5 = c4 + c_stride;
                        int32_t* c6 = c5 + c_stride;
                        int32_t* c7 = c6 + c_stride;

                        if (Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack){
                            uint16x8_t vACC_Or0C0123,
                                    vACC_Or0C4567,
                                    vACC_Or1C0123,
                                    vACC_Or1C4567,
                                    vACC_Or2C0123,
                                    vACC_Or2C4567,
                                    vACC_Or3C0123,
                                    vACC_Or3C4567,
                                    vACC_Or4C0123,
                                    vACC_Or4C4567,
                                    vACC_Or5C0123,
                                    vACC_Or5C4567,
                                    vACC_Or6C0123,
                                    vACC_Or6C4567,
                                    vACC_Or7C0123,
                                    vACC_Or7C4567;

                            vACC_Or0C0123 = veorq_u16(vACC_Or0C0123, vACC_Or0C0123);
                            vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 0, vACC_Ar76543210_x_Wc76543210, 0);
                            vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 2, vACC_Ar76543210_x_Wc07654321, 0);
                            vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 4, vACC_Ar76543210_x_Wc10765432, 0);
                            vACC_Or0C0123 = vcopyq_laneq_u16(vACC_Or0C0123, 6, vACC_Ar76543210_x_Wc21076543, 0);
                            vACC_Or0C4567 = veorq_u16(vACC_Or0C4567, vACC_Or0C4567);
                            vst1q_s32(c0, vACC_Or0C0123);
                            vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 0, vACC_Ar76543210_x_Wc32107654, 0);
                            vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 2, vACC_Ar76543210_x_Wc43210765, 0);
                            vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 4, vACC_Ar76543210_x_Wc54321076, 0);
                            vACC_Or0C4567 = vcopyq_laneq_u16(vACC_Or0C4567, 6, vACC_Ar76543210_x_Wc65432107, 0);
                            vACC_Or1C0123 = veorq_u16(vACC_Or1C0123, vACC_Or1C0123);
                            vst1q_s32(c0+4, vACC_Or0C4567);

                            vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 0, vACC_Ar76543210_x_Wc65432107, 1);
                            vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 2, vACC_Ar76543210_x_Wc76543210, 1);
                            vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 4, vACC_Ar76543210_x_Wc07654321, 1);
                            vACC_Or1C0123 = vcopyq_laneq_u16(vACC_Or1C0123, 6, vACC_Ar76543210_x_Wc10765432, 1);
                            vACC_Or1C4567 = veorq_u16(vACC_Or1C4567, vACC_Or1C4567);
                            vst1q_s32(c1, vACC_Or1C0123);
                            vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 0, vACC_Ar76543210_x_Wc21076543, 1);
                            vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 2, vACC_Ar76543210_x_Wc32107654, 1);
                            vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 4, vACC_Ar76543210_x_Wc43210765, 1);
                            vACC_Or1C4567 = vcopyq_laneq_u16(vACC_Or1C4567, 6, vACC_Ar76543210_x_Wc54321076, 1);
                            vACC_Or2C0123 = veorq_u16(vACC_Or2C0123, vACC_Or2C0123);
                            vst1q_s32(c1+4, vACC_Or1C4567);

                            vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 0, vACC_Ar76543210_x_Wc54321076, 2);
                            vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 2, vACC_Ar76543210_x_Wc65432107, 2);
                            vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 4, vACC_Ar76543210_x_Wc76543210, 2);
                            vACC_Or2C0123 = vcopyq_laneq_u16(vACC_Or2C0123, 6, vACC_Ar76543210_x_Wc07654321, 2);
                            vACC_Or2C4567 = veorq_u16(vACC_Or2C4567, vACC_Or2C4567);
                            vst1q_s32(c2, vACC_Or2C0123);
                            vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 0, vACC_Ar76543210_x_Wc10765432, 2);
                            vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 2, vACC_Ar76543210_x_Wc21076543, 2);
                            vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 4, vACC_Ar76543210_x_Wc32107654, 2);
                            vACC_Or2C4567 = vcopyq_laneq_u16(vACC_Or2C4567, 6, vACC_Ar76543210_x_Wc43210765, 2);
                            vACC_Or3C0123 = veorq_u16(vACC_Or3C0123, vACC_Or3C0123);
                            vst1q_s32(c2+4, vACC_Or2C4567);

                            vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C0123, 0, vACC_Ar76543210_x_Wc43210765, 3);
                            vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 2, vACC_Ar76543210_x_Wc54321076, 3);
                            vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 4, vACC_Ar76543210_x_Wc65432107, 3);
                            vACC_Or3C0123 = vcopyq_laneq_u16(vACC_Or3C0123, 6, vACC_Ar76543210_x_Wc76543210, 3);
                            vACC_Or3C4567 = veorq_u16(vACC_Or3C4567, vACC_Or3C4567);
                            vst1q_s32(c3, vACC_Or3C0123);
                            vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 0, vACC_Ar76543210_x_Wc07654321, 3);
                            vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 2, vACC_Ar76543210_x_Wc10765432, 3);
                            vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 4, vACC_Ar76543210_x_Wc21076543, 3);
                            vACC_Or3C4567 = vcopyq_laneq_u16(vACC_Or3C4567, 6, vACC_Ar76543210_x_Wc32107654, 3);
                            vACC_Or4C0123 = veorq_u16(vACC_Or4C0123, vACC_Or4C0123);
                            vst1q_s32(c3+4, vACC_Or3C4567);

                            vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 0, vACC_Ar76543210_x_Wc32107654, 4);
                            vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 2, vACC_Ar76543210_x_Wc43210765, 4);
                            vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 4, vACC_Ar76543210_x_Wc54321076, 4);
                            vACC_Or4C0123 = vcopyq_laneq_u16(vACC_Or4C0123, 6, vACC_Ar76543210_x_Wc65432107, 4);
                            vACC_Or4C4567 = veorq_u16(vACC_Or4C4567, vACC_Or4C4567);
                            vst1q_s32(c4, vACC_Or4C0123);
                            vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 0, vACC_Ar76543210_x_Wc76543210, 4);
                            vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 2, vACC_Ar76543210_x_Wc07654321, 4);
                            vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 4, vACC_Ar76543210_x_Wc10765432, 4);
                            vACC_Or4C4567 = vcopyq_laneq_u16(vACC_Or4C4567, 6, vACC_Ar76543210_x_Wc21076543, 4);
                            vACC_Or5C0123 = veorq_u16(vACC_Or5C0123, vACC_Or5C0123);
                            vst1q_s32(c4+4, vACC_Or4C4567);

                            vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 0, vACC_Ar76543210_x_Wc21076543, 5);
                            vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 2, vACC_Ar76543210_x_Wc32107654, 5);
                            vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 4, vACC_Ar76543210_x_Wc43210765, 5);
                            vACC_Or5C0123 = vcopyq_laneq_u16(vACC_Or5C0123, 6, vACC_Ar76543210_x_Wc54321076, 5);
                            vACC_Or5C4567 = veorq_u16(vACC_Or5C4567, vACC_Or5C4567);
                            vst1q_s32(c5, vACC_Or5C0123);
                            vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 0, vACC_Ar76543210_x_Wc65432107, 5);
                            vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 2, vACC_Ar76543210_x_Wc76543210, 5);
                            vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 4, vACC_Ar76543210_x_Wc07654321, 5);
                            vACC_Or5C4567 = vcopyq_laneq_u16(vACC_Or5C4567, 6, vACC_Ar76543210_x_Wc10765432, 5);
                            vACC_Or6C0123 = veorq_u16(vACC_Or6C0123, vACC_Or6C0123);
                            vst1q_s32(c5+4, vACC_Or5C4567);

                            vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 0, vACC_Ar76543210_x_Wc10765432, 6);
                            vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 2, vACC_Ar76543210_x_Wc21076543, 6);
                            vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 4, vACC_Ar76543210_x_Wc32107654, 6);
                            vACC_Or6C0123 = vcopyq_laneq_u16(vACC_Or6C0123, 6, vACC_Ar76543210_x_Wc43210765, 6);
                            vACC_Or6C4567 = veorq_u16(vACC_Or6C4567, vACC_Or6C4567);
                            vst1q_s32(c6, vACC_Or6C0123);
                            vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 0, vACC_Ar76543210_x_Wc54321076, 6);
                            vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 2, vACC_Ar76543210_x_Wc65432107, 6);
                            vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 4, vACC_Ar76543210_x_Wc76543210, 6);
                            vACC_Or6C4567 = vcopyq_laneq_u16(vACC_Or6C4567, 6, vACC_Ar76543210_x_Wc07654321, 6);
                            vACC_Or7C0123 = veorq_u16(vACC_Or7C0123, vACC_Or7C0123);
                            vst1q_s32(c6+4, vACC_Or6C4567);

                            vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 0, vACC_Ar76543210_x_Wc07654321, 7);
                            vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 2, vACC_Ar76543210_x_Wc10765432, 7);
                            vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 4, vACC_Ar76543210_x_Wc21076543, 7);
                            vACC_Or7C0123 = vcopyq_laneq_u16(vACC_Or7C0123, 6, vACC_Ar76543210_x_Wc32107654, 7);
                            vACC_Or7C4567 = veorq_u16(vACC_Or7C4567, vACC_Or7C4567);
                            vst1q_s32(c7, vACC_Or7C0123);
                            vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 0, vACC_Ar76543210_x_Wc43210765, 7);
                            vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 2, vACC_Ar76543210_x_Wc54321076, 7);
                            vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 4, vACC_Ar76543210_x_Wc65432107, 7);
                            vACC_Or7C4567 = vcopyq_laneq_u16(vACC_Or7C4567, 6, vACC_Ar76543210_x_Wc76543210, 7);
                            vst1q_s32(c7+4, vACC_Or7C4567);
                        } else {
                            uint32x4_t vACC_Ar76543210_x_Wc76543210_high, vACC_Ar76543210_x_Wc76543210_low,
                                    vACC_Ar76543210_x_Wc07654321_high, vACC_Ar76543210_x_Wc07654321_low,
                                    vACC_Ar76543210_x_Wc10765432_high, vACC_Ar76543210_x_Wc10765432_low,
                                    vACC_Ar76543210_x_Wc21076543_high, vACC_Ar76543210_x_Wc21076543_low,
                                    vACC_Ar76543210_x_Wc32107654_high, vACC_Ar76543210_x_Wc32107654_low,
                                    vACC_Ar76543210_x_Wc43210765_high, vACC_Ar76543210_x_Wc43210765_low,
                                    vACC_Ar76543210_x_Wc54321076_high, vACC_Ar76543210_x_Wc54321076_low,
                                    vACC_Ar76543210_x_Wc65432107_high, vACC_Ar76543210_x_Wc65432107_low;
                            
                            vACC_Ar76543210_x_Wc76543210_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc76543210), 0);
                            vst1q_s32(c0, vACC_Ar76543210_x_Wc76543210_low);
                            vACC_Ar76543210_x_Wc76543210_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc76543210, 0);
                            vst1q_s32(c0+4, vACC_Ar76543210_x_Wc76543210_high);

                            vACC_Ar76543210_x_Wc07654321_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc07654321), 0);
                            vst1q_s32(c1, vACC_Ar76543210_x_Wc07654321_low);
                            vACC_Ar76543210_x_Wc07654321_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc07654321, 0);
                            vst1q_s32(c1+4, vACC_Ar76543210_x_Wc07654321_high);

                            vACC_Ar76543210_x_Wc10765432_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc10765432), 0);
                            vst1q_s32(c2, vACC_Ar76543210_x_Wc10765432_low);
                            vACC_Ar76543210_x_Wc10765432_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc10765432, 0);
                            vst1q_s32(c2+4, vACC_Ar76543210_x_Wc10765432_high);

                            vACC_Ar76543210_x_Wc21076543_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc21076543), 0);
                            vst1q_s32(c3, vACC_Ar76543210_x_Wc21076543_low);
                            vACC_Ar76543210_x_Wc21076543_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc21076543, 0);
                            vst1q_s32(c3+4, vACC_Ar76543210_x_Wc21076543_high);

                            vACC_Ar76543210_x_Wc32107654_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc32107654), 0);
                            vst1q_s32(c4, vACC_Ar76543210_x_Wc32107654_low);
                            vACC_Ar76543210_x_Wc32107654_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc32107654, 0);
                            vst1q_s32(c4+4, vACC_Ar76543210_x_Wc32107654_high);

                            vACC_Ar76543210_x_Wc43210765_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc43210765), 0);
                            vst1q_s32(c5, vACC_Ar76543210_x_Wc43210765_low);
                            vACC_Ar76543210_x_Wc43210765_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc43210765, 0);
                            vst1q_s32(c5+4, vACC_Ar76543210_x_Wc43210765_high);

                            vACC_Ar76543210_x_Wc54321076_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc54321076), 0);
                            vst1q_s32(c6, vACC_Ar76543210_x_Wc54321076_low);
                            vACC_Ar76543210_x_Wc54321076_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc54321076, 0);
                            vst1q_s32(c6+4, vACC_Ar76543210_x_Wc54321076_high);

                            vACC_Ar76543210_x_Wc65432107_low = vshll_n_u16(vget_low_u16(vACC_Ar76543210_x_Wc65432107), 0);
                            vst1q_s32(c7, vACC_Ar76543210_x_Wc65432107_low);
                            vACC_Ar76543210_x_Wc65432107_high = vshll_high_n_u16(vACC_Ar76543210_x_Wc65432107, 0);
                            vst1q_s32(c7+4, vACC_Ar76543210_x_Wc65432107_high);
                        }
                    }
                }
                return Status::Success;
            }
            Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){ return Status::NotImplemented; }
            inline void unpack_8x8_block_barrelshift_mul(
                uint16x8_t& vACC_Ar76543210_x_Wc76543210,
                uint16x8_t& vACC_Ar76543210_x_Wc07654321,
                uint16x8_t& vACC_Ar76543210_x_Wc10765432,
                uint16x8_t& vACC_Ar76543210_x_Wc21076543,
                uint16x8_t& vACC_Ar76543210_x_Wc32107654,
                uint16x8_t& vACC_Ar76543210_x_Wc43210765,
                uint16x8_t& vACC_Ar76543210_x_Wc54321076,
                uint16x8_t& vACC_Ar76543210_x_Wc65432107
            ){
                asm volatile(
                    "ins v0.h[0], %[vACC_Ar76543210_x_Wc76543210].h[0]\n\t"
                    "ins v1.h[1], %[vACC_Ar76543210_x_Wc76543210].h[1]\n\t"
                    "ins v2.h[2], %[vACC_Ar76543210_x_Wc76543210].h[2]\n\t"
                    "ins v3.h[3], %[vACC_Ar76543210_x_Wc76543210].h[3]\n\t"
                    "ins v4.h[4], %[vACC_Ar76543210_x_Wc76543210].h[4]\n\t"
                    "ins v5.h[5], %[vACC_Ar76543210_x_Wc76543210].h[5]\n\t"
                    "ins v6.h[6], %[vACC_Ar76543210_x_Wc76543210].h[6]\n\t"
                    "ins v7.h[7], %[vACC_Ar76543210_x_Wc76543210].h[7]\n\t"

                    "ins v0.h[1], %[vACC_Ar76543210_x_Wc07654321].h[0]\n\t"
                    "ins v1.h[2], %[vACC_Ar76543210_x_Wc07654321].h[1]\n\t"
                    "ins v2.h[3], %[vACC_Ar76543210_x_Wc07654321].h[2]\n\t"
                    "ins v3.h[4], %[vACC_Ar76543210_x_Wc07654321].h[3]\n\t"
                    "ins v4.h[5], %[vACC_Ar76543210_x_Wc07654321].h[4]\n\t"
                    "ins v5.h[6], %[vACC_Ar76543210_x_Wc07654321].h[5]\n\t"
                    "ins v6.h[7], %[vACC_Ar76543210_x_Wc07654321].h[6]\n\t"
                    "ins v7.h[0], %[vACC_Ar76543210_x_Wc07654321].h[7]\n\t"

                    "ins v0.h[2], %[vACC_Ar76543210_x_Wc10765432].h[0]\n\t"
                    "ins v1.h[3], %[vACC_Ar76543210_x_Wc10765432].h[1]\n\t"
                    "ins v2.h[4], %[vACC_Ar76543210_x_Wc10765432].h[2]\n\t"
                    "ins v3.h[5], %[vACC_Ar76543210_x_Wc10765432].h[3]\n\t"
                    "ins v4.h[6], %[vACC_Ar76543210_x_Wc10765432].h[4]\n\t"
                    "ins v5.h[7], %[vACC_Ar76543210_x_Wc10765432].h[5]\n\t"
                    "ins v6.h[0], %[vACC_Ar76543210_x_Wc10765432].h[6]\n\t"
                    "ins v7.h[1], %[vACC_Ar76543210_x_Wc10765432].h[7]\n\t"

                    "ins v0.h[3], %[vACC_Ar76543210_x_Wc21076543].h[0]\n\t"
                    "ins v1.h[4], %[vACC_Ar76543210_x_Wc21076543].h[1]\n\t"
                    "ins v2.h[5], %[vACC_Ar76543210_x_Wc21076543].h[2]\n\t"
                    "ins v3.h[6], %[vACC_Ar76543210_x_Wc21076543].h[3]\n\t"
                    "ins v4.h[7], %[vACC_Ar76543210_x_Wc21076543].h[4]\n\t"
                    "ins v5.h[0], %[vACC_Ar76543210_x_Wc21076543].h[5]\n\t"
                    "ins v6.h[1], %[vACC_Ar76543210_x_Wc21076543].h[6]\n\t"
                    "ins v7.h[2], %[vACC_Ar76543210_x_Wc21076543].h[7]\n\t"

                    "ins v0.h[4], %[vACC_Ar76543210_x_Wc32107654].h[0]\n\t"
                    "ins v1.h[5], %[vACC_Ar76543210_x_Wc32107654].h[1]\n\t"
                    "ins v2.h[6], %[vACC_Ar76543210_x_Wc32107654].h[2]\n\t"
                    "ins v3.h[7], %[vACC_Ar76543210_x_Wc32107654].h[3]\n\t"
                    "ins v4.h[0], %[vACC_Ar76543210_x_Wc32107654].h[4]\n\t"
                    "ins v5.h[1], %[vACC_Ar76543210_x_Wc32107654].h[5]\n\t"
                    "ins v6.h[2], %[vACC_Ar76543210_x_Wc32107654].h[6]\n\t"
                    "ins v7.h[3], %[vACC_Ar76543210_x_Wc32107654].h[7]\n\t"

                    "ins v0.h[5], %[vACC_Ar76543210_x_Wc43210765].h[0]\n\t"
                    "ins v1.h[6], %[vACC_Ar76543210_x_Wc43210765].h[1]\n\t"
                    "ins v2.h[7], %[vACC_Ar76543210_x_Wc43210765].h[2]\n\t"
                    "ins v3.h[0], %[vACC_Ar76543210_x_Wc43210765].h[3]\n\t"
                    "ins v4.h[1], %[vACC_Ar76543210_x_Wc43210765].h[4]\n\t"
                    "ins v5.h[2], %[vACC_Ar76543210_x_Wc43210765].h[5]\n\t"
                    "ins v6.h[3], %[vACC_Ar76543210_x_Wc43210765].h[6]\n\t"
                    "ins v7.h[4], %[vACC_Ar76543210_x_Wc43210765].h[7]\n\t"

                    "ins v0.h[6], %[vACC_Ar76543210_x_Wc54321076].h[0]\n\t"
                    "ins v1.h[7], %[vACC_Ar76543210_x_Wc54321076].h[1]\n\t"
                    "ins v2.h[0], %[vACC_Ar76543210_x_Wc54321076].h[2]\n\t"
                    "ins v3.h[1], %[vACC_Ar76543210_x_Wc54321076].h[3]\n\t"
                    "ins v4.h[2], %[vACC_Ar76543210_x_Wc54321076].h[4]\n\t"
                    "ins v5.h[3], %[vACC_Ar76543210_x_Wc54321076].h[5]\n\t"
                    "ins v6.h[4], %[vACC_Ar76543210_x_Wc54321076].h[6]\n\t"
                    "ins v7.h[5], %[vACC_Ar76543210_x_Wc54321076].h[7]\n\t"

                    "ins v0.h[7], %[vACC_Ar76543210_x_Wc65432107].h[0]\n\t"
                    "ins v1.h[0], %[vACC_Ar76543210_x_Wc65432107].h[1]\n\t"
                    "ins v2.h[1], %[vACC_Ar76543210_x_Wc65432107].h[2]\n\t"
                    "ins v3.h[2], %[vACC_Ar76543210_x_Wc65432107].h[3]\n\t"
                    "ins v4.h[3], %[vACC_Ar76543210_x_Wc65432107].h[4]\n\t"
                    "ins v5.h[4], %[vACC_Ar76543210_x_Wc65432107].h[5]\n\t"
                    "ins v6.h[5], %[vACC_Ar76543210_x_Wc65432107].h[6]\n\t"
                    "ins v7.h[6], %[vACC_Ar76543210_x_Wc65432107].h[7]\n\t"

                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v0.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v1.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v2.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v3.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v4.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v5.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v6.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v7.16b\n\t"

                    : [ vACC_Ar76543210_x_Wc76543210 ] "=w" ( vACC_Ar76543210_x_Wc76543210 ),
                      [ vACC_Ar76543210_x_Wc07654321 ] "=w" ( vACC_Ar76543210_x_Wc07654321 ),
                      [ vACC_Ar76543210_x_Wc10765432 ] "=w" ( vACC_Ar76543210_x_Wc10765432 ),
                      [ vACC_Ar76543210_x_Wc21076543 ] "=w" ( vACC_Ar76543210_x_Wc21076543 ),
                      [ vACC_Ar76543210_x_Wc32107654 ] "=w" ( vACC_Ar76543210_x_Wc32107654 ),
                      [ vACC_Ar76543210_x_Wc43210765 ] "=w" ( vACC_Ar76543210_x_Wc43210765 ),
                      [ vACC_Ar76543210_x_Wc54321076 ] "=w" ( vACC_Ar76543210_x_Wc54321076 ),
                      [ vACC_Ar76543210_x_Wc65432107 ] "=w" ( vACC_Ar76543210_x_Wc65432107 )
                    :
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                );
            }
            inline void unpack_8x8_block_barrelshift_mul(
                int16x8_t& vACC_Ar76543210_x_Wc76543210,
                int16x8_t& vACC_Ar76543210_x_Wc07654321,
                int16x8_t& vACC_Ar76543210_x_Wc10765432,
                int16x8_t& vACC_Ar76543210_x_Wc21076543,
                int16x8_t& vACC_Ar76543210_x_Wc32107654,
                int16x8_t& vACC_Ar76543210_x_Wc43210765,
                int16x8_t& vACC_Ar76543210_x_Wc54321076,
                int16x8_t& vACC_Ar76543210_x_Wc65432107
            ){
                asm volatile(
                    "ins v0.h[0], %[vACC_Ar76543210_x_Wc76543210].h[0]\n\t"
                    "ins v1.h[1], %[vACC_Ar76543210_x_Wc76543210].h[1]\n\t"
                    "ins v2.h[2], %[vACC_Ar76543210_x_Wc76543210].h[2]\n\t"
                    "ins v3.h[3], %[vACC_Ar76543210_x_Wc76543210].h[3]\n\t"
                    "ins v4.h[4], %[vACC_Ar76543210_x_Wc76543210].h[4]\n\t"
                    "ins v5.h[5], %[vACC_Ar76543210_x_Wc76543210].h[5]\n\t"
                    "ins v6.h[6], %[vACC_Ar76543210_x_Wc76543210].h[6]\n\t"
                    "ins v7.h[7], %[vACC_Ar76543210_x_Wc76543210].h[7]\n\t"

                    "ins v0.h[1], %[vACC_Ar76543210_x_Wc07654321].h[0]\n\t"
                    "ins v1.h[2], %[vACC_Ar76543210_x_Wc07654321].h[1]\n\t"
                    "ins v2.h[3], %[vACC_Ar76543210_x_Wc07654321].h[2]\n\t"
                    "ins v3.h[4], %[vACC_Ar76543210_x_Wc07654321].h[3]\n\t"
                    "ins v4.h[5], %[vACC_Ar76543210_x_Wc07654321].h[4]\n\t"
                    "ins v5.h[6], %[vACC_Ar76543210_x_Wc07654321].h[5]\n\t"
                    "ins v6.h[7], %[vACC_Ar76543210_x_Wc07654321].h[6]\n\t"
                    "ins v7.h[0], %[vACC_Ar76543210_x_Wc07654321].h[7]\n\t"

                    "ins v0.h[2], %[vACC_Ar76543210_x_Wc10765432].h[0]\n\t"
                    "ins v1.h[3], %[vACC_Ar76543210_x_Wc10765432].h[1]\n\t"
                    "ins v2.h[4], %[vACC_Ar76543210_x_Wc10765432].h[2]\n\t"
                    "ins v3.h[5], %[vACC_Ar76543210_x_Wc10765432].h[3]\n\t"
                    "ins v4.h[6], %[vACC_Ar76543210_x_Wc10765432].h[4]\n\t"
                    "ins v5.h[7], %[vACC_Ar76543210_x_Wc10765432].h[5]\n\t"
                    "ins v6.h[0], %[vACC_Ar76543210_x_Wc10765432].h[6]\n\t"
                    "ins v7.h[1], %[vACC_Ar76543210_x_Wc10765432].h[7]\n\t"

                    "ins v0.h[3], %[vACC_Ar76543210_x_Wc21076543].h[0]\n\t"
                    "ins v1.h[4], %[vACC_Ar76543210_x_Wc21076543].h[1]\n\t"
                    "ins v2.h[5], %[vACC_Ar76543210_x_Wc21076543].h[2]\n\t"
                    "ins v3.h[6], %[vACC_Ar76543210_x_Wc21076543].h[3]\n\t"
                    "ins v4.h[7], %[vACC_Ar76543210_x_Wc21076543].h[4]\n\t"
                    "ins v5.h[0], %[vACC_Ar76543210_x_Wc21076543].h[5]\n\t"
                    "ins v6.h[1], %[vACC_Ar76543210_x_Wc21076543].h[6]\n\t"
                    "ins v7.h[2], %[vACC_Ar76543210_x_Wc21076543].h[7]\n\t"

                    "ins v0.h[4], %[vACC_Ar76543210_x_Wc32107654].h[0]\n\t"
                    "ins v1.h[5], %[vACC_Ar76543210_x_Wc32107654].h[1]\n\t"
                    "ins v2.h[6], %[vACC_Ar76543210_x_Wc32107654].h[2]\n\t"
                    "ins v3.h[7], %[vACC_Ar76543210_x_Wc32107654].h[3]\n\t"
                    "ins v4.h[0], %[vACC_Ar76543210_x_Wc32107654].h[4]\n\t"
                    "ins v5.h[1], %[vACC_Ar76543210_x_Wc32107654].h[5]\n\t"
                    "ins v6.h[2], %[vACC_Ar76543210_x_Wc32107654].h[6]\n\t"
                    "ins v7.h[3], %[vACC_Ar76543210_x_Wc32107654].h[7]\n\t"

                    "ins v0.h[5], %[vACC_Ar76543210_x_Wc43210765].h[0]\n\t"
                    "ins v1.h[6], %[vACC_Ar76543210_x_Wc43210765].h[1]\n\t"
                    "ins v2.h[7], %[vACC_Ar76543210_x_Wc43210765].h[2]\n\t"
                    "ins v3.h[0], %[vACC_Ar76543210_x_Wc43210765].h[3]\n\t"
                    "ins v4.h[1], %[vACC_Ar76543210_x_Wc43210765].h[4]\n\t"
                    "ins v5.h[2], %[vACC_Ar76543210_x_Wc43210765].h[5]\n\t"
                    "ins v6.h[3], %[vACC_Ar76543210_x_Wc43210765].h[6]\n\t"
                    "ins v7.h[4], %[vACC_Ar76543210_x_Wc43210765].h[7]\n\t"

                    "ins v0.h[6], %[vACC_Ar76543210_x_Wc54321076].h[0]\n\t"
                    "ins v1.h[7], %[vACC_Ar76543210_x_Wc54321076].h[1]\n\t"
                    "ins v2.h[0], %[vACC_Ar76543210_x_Wc54321076].h[2]\n\t"
                    "ins v3.h[1], %[vACC_Ar76543210_x_Wc54321076].h[3]\n\t"
                    "ins v4.h[2], %[vACC_Ar76543210_x_Wc54321076].h[4]\n\t"
                    "ins v5.h[3], %[vACC_Ar76543210_x_Wc54321076].h[5]\n\t"
                    "ins v6.h[4], %[vACC_Ar76543210_x_Wc54321076].h[6]\n\t"
                    "ins v7.h[5], %[vACC_Ar76543210_x_Wc54321076].h[7]\n\t"

                    "ins v0.h[7], %[vACC_Ar76543210_x_Wc65432107].h[0]\n\t"
                    "ins v1.h[0], %[vACC_Ar76543210_x_Wc65432107].h[1]\n\t"
                    "ins v2.h[1], %[vACC_Ar76543210_x_Wc65432107].h[2]\n\t"
                    "ins v3.h[2], %[vACC_Ar76543210_x_Wc65432107].h[3]\n\t"
                    "ins v4.h[3], %[vACC_Ar76543210_x_Wc65432107].h[4]\n\t"
                    "ins v5.h[4], %[vACC_Ar76543210_x_Wc65432107].h[5]\n\t"
                    "ins v6.h[5], %[vACC_Ar76543210_x_Wc65432107].h[6]\n\t"
                    "ins v7.h[6], %[vACC_Ar76543210_x_Wc65432107].h[7]\n\t"

                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v0.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v1.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v2.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v3.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v4.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v5.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v6.16b\n\t"
                    "mov %[vACC_Ar76543210_x_Wc65432107].16b, v7.16b\n\t"

                    : [ vACC_Ar76543210_x_Wc76543210 ] "=w" ( vACC_Ar76543210_x_Wc76543210 ),
                      [ vACC_Ar76543210_x_Wc07654321 ] "=w" ( vACC_Ar76543210_x_Wc07654321 ),
                      [ vACC_Ar76543210_x_Wc10765432 ] "=w" ( vACC_Ar76543210_x_Wc10765432 ),
                      [ vACC_Ar76543210_x_Wc21076543 ] "=w" ( vACC_Ar76543210_x_Wc21076543 ),
                      [ vACC_Ar76543210_x_Wc32107654 ] "=w" ( vACC_Ar76543210_x_Wc32107654 ),
                      [ vACC_Ar76543210_x_Wc43210765 ] "=w" ( vACC_Ar76543210_x_Wc43210765 ),
                      [ vACC_Ar76543210_x_Wc54321076 ] "=w" ( vACC_Ar76543210_x_Wc54321076 ),
                      [ vACC_Ar76543210_x_Wc65432107 ] "=w" ( vACC_Ar76543210_x_Wc65432107 )
                    :
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                );
            }
            inline void unpack_8x8_block_barrelshift(const int32_t* O, int32_t* O_unpack, size_t offset){
                #if Int8InputsInt8WeightsBarrelShiftMul_SimpleUnpack
                for (size_t i = 0 ; i < 8 ; i++)
                    for (size_t j = 0 ; j < 8 ; j++)
                        O_unpack[j * offset + ((j + i) % 8)] = O[i * offset + j];
                #else
                int32_t* O_unpack_0 = O_unpack + 0 * offset;
                int32_t* O_unpack_1 = O_unpack + 1 * offset;
                int32_t* O_unpack_2 = O_unpack + 2 * offset;
                int32_t* O_unpack_3 = O_unpack + 3 * offset;
                int32_t* O_unpack_4 = O_unpack + 4 * offset;
                int32_t* O_unpack_5 = O_unpack + 5 * offset;
                int32_t* O_unpack_6 = O_unpack + 6 * offset;
                int32_t* O_unpack_7 = O_unpack + 7 * offset;

                const int32_t* O_0 = O + 0 * offset;
                const int32_t* O_1 = O + 1 * offset;
                const int32_t* O_2 = O + 2 * offset;
                const int32_t* O_3 = O + 3 * offset;
                const int32_t* O_4 = O + 4 * offset;
                const int32_t* O_5 = O + 5 * offset;
                const int32_t* O_6 = O + 6 * offset;
                const int32_t* O_7 = O + 7 * offset;

                asm volatile(
                    "ld2 {v16.4s, v17.4s}, [%[O_0]], #32\n\t"
                    "ld2 {v18.4s, v19.4s}, [%[O_1]], #32\n\t"
                    "ld2 {v20.4s, v21.4s}, [%[O_2]], #32\n\t"
                    "ld2 {v22.4s, v23.4s}, [%[O_3]], #32\n\t"
                    "ld2 {v24.4s, v25.4s}, [%[O_4]], #32\n\t"
                    "ld2 {v26.4s, v27.4s}, [%[O_5]], #32\n\t"
                    "ld2 {v28.4s, v29.4s}, [%[O_6]], #32\n\t"
                    "ld2 {v30.4s, v31.4s}, [%[O_7]], #32\n\t"

                    "ins v0.s[0],  v16.s[0]\n\t"
                    "ins v2.s[1],  v16.s[1]\n\t"
                    "ins v4.s[2],  v16.s[2]\n\t"
                    "ins v6.s[3],  v16.s[3]\n\t"
                    "ins v9.s[0],  v17.s[0]\n\t"
                    "ins v11.s[1], v17.s[1]\n\t"
                    "ins v13.s[2], v17.s[2]\n\t"
                    "ins v15.s[3], v17.s[3]\n\t"

                    "ins v0.s[1],  v18.s[0]\n\t"
                    "ins v2.s[2],  v18.s[1]\n\t"
                    "ins v4.s[3],  v18.s[2]\n\t"
                    "ins v7.s[0],  v18.s[3]\n\t"
                    "ins v9.s[1],  v19.s[0]\n\t"
                    "ins v11.s[2], v19.s[1]\n\t"
                    "ins v13.s[3], v19.s[2]\n\t"
                    "ins v14.s[0], v19.s[3]\n\t"

                    "ins v0.s[2],  v20.s[0]\n\t"
                    "ins v2.s[3],  v20.s[1]\n\t"
                    "ins v5.s[0],  v20.s[2]\n\t"
                    "ins v7.s[1],  v20.s[3]\n\t"
                    "ins v9.s[2],  v21.s[0]\n\t"
                    "ins v11.s[3], v21.s[1]\n\t"
                    "ins v12.s[0], v21.s[2]\n\t"
                    "ins v14.s[1], v21.s[3]\n\t"

                    "ins v0.s[3],  v22.s[0]\n\t"
                    "ins v3.s[0],  v22.s[1]\n\t"
                    "ins v5.s[1],  v22.s[2]\n\t"
                    "ins v7.s[2],  v22.s[3]\n\t"
                    "ins v9.s[3],  v23.s[0]\n\t"
                    "ins v10.s[0], v23.s[1]\n\t"
                    "ins v12.s[1], v23.s[2]\n\t"
                    "ins v14.s[2], v23.s[3]\n\t"

                    "ins v1.s[0],  v24.s[0]\n\t"
                    "ins v3.s[1],  v24.s[1]\n\t"
                    "ins v5.s[2],  v24.s[2]\n\t"
                    "ins v7.s[3],  v24.s[3]\n\t"
                    "ins v8.s[0],  v25.s[0]\n\t"
                    "ins v10.s[1], v25.s[1]\n\t"
                    "ins v12.s[2], v25.s[2]\n\t"
                    "ins v14.s[3], v25.s[3]\n\t"

                    "ins v1.s[1],  v26.s[0]\n\t"
                    "ins v3.s[2],  v26.s[1]\n\t"
                    "ins v5.s[3],  v26.s[2]\n\t"
                    "ins v6.s[0],  v26.s[3]\n\t"
                    "ins v8.s[1],  v27.s[0]\n\t"
                    "ins v10.s[2], v27.s[1]\n\t"
                    "ins v12.s[3], v27.s[2]\n\t"
                    "ins v15.s[0], v27.s[3]\n\t"

                    "ins v1.s[2],  v28.s[0]\n\t"
                    "ins v3.s[3],  v28.s[1]\n\t"
                    "ins v4.s[0],  v28.s[2]\n\t"
                    "ins v6.s[1],  v28.s[3]\n\t"
                    "ins v8.s[2],  v29.s[0]\n\t"
                    "ins v10.s[3], v29.s[1]\n\t"
                    "ins v13.s[0], v29.s[2]\n\t"
                    "ins v15.s[1], v29.s[3]\n\t"

                    "ins v1.s[3],  v30.s[0]\n\t"
                    "ins v2.s[0],  v30.s[1]\n\t"
                    "ins v4.s[1],  v30.s[2]\n\t"
                    "ins v6.s[2],  v30.s[3]\n\t"
                    "ins v8.s[3],  v31.s[0]\n\t"
                    "ins v11.s[0], v31.s[1]\n\t"
                    "ins v13.s[1], v31.s[2]\n\t"
                    "ins v15.s[2], v31.s[3]\n\t"

                    "st2 {v0.4s , v1.4s}, [%[O_unpack_0]]\n\t"
                    "st2 {v2.4s , v3.4s}, [%[O_unpack_1]]\n\t"
                    "st2 {v4.4s , v5.4s}, [%[O_unpack_2]]\n\t"
                    "st2 {v6.4s , v7.4s}, [%[O_unpack_3]]\n\t"
                    "st2 {v8.4s , v9.4s}, [%[O_unpack_4]]\n\t"
                    "st2 {v10.4s, v11.4s}, [%[O_unpack_5]]\n\t"
                    "st2 {v12.4s, v13.4s}, [%[O_unpack_6]]\n\t"
                    "st2 {v14.4s, v15.4s}, [%[O_unpack_7]]\n\t"

                    : [ O_unpack_0 ] "+r" ( O_unpack_0 ),
                      [ O_unpack_1 ] "+r" ( O_unpack_1 ),
                      [ O_unpack_2 ] "+r" ( O_unpack_2 ),
                      [ O_unpack_3 ] "+r" ( O_unpack_3 ),
                      [ O_unpack_4 ] "+r" ( O_unpack_4 ),
                      [ O_unpack_5 ] "+r" ( O_unpack_5 ),
                      [ O_unpack_6 ] "+r" ( O_unpack_6 ),
                      [ O_unpack_7 ] "+r" ( O_unpack_7 )
                    : [    O_0     ] "r"  (    O_0     ),
                      [    O_1     ] "r"  (    O_1     ),
                      [    O_2     ] "r"  (    O_2     ),
                      [    O_3     ] "r"  (    O_3     ),
                      [    O_4     ] "r"  (    O_4     ),
                      [    O_5     ] "r"  (    O_5     ),
                      [    O_6     ] "r"  (    O_6     ),
                      [    O_7     ] "r"  (    O_7     )
                    : "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
                      "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                #endif
            }
            LowPrecision::PreprocessType InputPreProcess() { return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType FilterPreProcess(){ return LowPrecision::PreprocessType::PaddingAndPacking; }
            LowPrecision::PreprocessType OutputPreProcess(){ return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::OutputPostProcess(); }
            LowPrecision::PreprocessType OutputPostProcess(){
                #if Int8InputsInt8WeightsBarrelShiftMul_InKernelUnpack
                return LowPrecision::PreprocessType::PaddingIfNeccessery;
                #else
                return LowPrecision::PreprocessType::PaddingAndPacking;
                #endif
            }
            LowPrecision::GEMMType GEMMSupport(){ return LowPrecision::GEMMType::SupportsGEMM; }
        }
    }
}
#endif
