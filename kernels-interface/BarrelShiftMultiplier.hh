#ifndef _KERNELS_INTERFACE_BSM_H_
#define _KERNELS_INTERFACE_BSM_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace BSM { // Barrel Shift Multiplier
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status UnpackOutput(LowPrecision::Method method, const int32_t* input, LowPrecision::Shape shape, int32_t* output);
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
            namespace W8A8{
                size_t TransformFilterShape(int* shape, int n_dims);
                size_t TransformInputShape(int* shape, int n_dims);
                template <typename T>
                LowPrecision::Status QuantizeFilter(const T* input, LowPrecision::Shape k_shape, T* output, LowPrecision::MemLayout layout);
                template <typename T>
                LowPrecision::Status QuantizeInput(const T* input, LowPrecision::Shape shape, T* output, LowPrecision::MemLayout layout);
                LowPrecision::Status UnpackOutput(const int32_t* input, Shape shape, int32_t* output);
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
                #if IS_ARM
                void unpack_8x8_block_barrelshift_mul(
                    uint16x8_t& vACC_Ar76543210_x_Wc76543210,
                    uint16x8_t& vACC_Ar76543210_x_Wc07654321,
                    uint16x8_t& vACC_Ar76543210_x_Wc10765432,
                    uint16x8_t& vACC_Ar76543210_x_Wc21076543,
                    uint16x8_t& vACC_Ar76543210_x_Wc32107654,
                    uint16x8_t& vACC_Ar76543210_x_Wc43210765,
                    uint16x8_t& vACC_Ar76543210_x_Wc54321076,
                    uint16x8_t& vACC_Ar76543210_x_Wc65432107
                );
                void unpack_8x8_block_barrelshift_mul(
                    int16x8_t& vACC_Ar76543210_x_Wc76543210,
                    int16x8_t& vACC_Ar76543210_x_Wc07654321,
                    int16x8_t& vACC_Ar76543210_x_Wc10765432,
                    int16x8_t& vACC_Ar76543210_x_Wc21076543,
                    int16x8_t& vACC_Ar76543210_x_Wc32107654,
                    int16x8_t& vACC_Ar76543210_x_Wc43210765,
                    int16x8_t& vACC_Ar76543210_x_Wc54321076,
                    int16x8_t& vACC_Ar76543210_x_Wc65432107
                );
                #endif
                inline void unpack_8x8_block_barrelshift(const int32_t* O, int32_t* O_unpack, size_t offset);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
            }
            namespace W4A4{
                size_t TransformFilterShape(int* shape, int n_dims);
                size_t TransformInputShape(int* shape, int n_dims);
                template <typename T>
                LowPrecision::Status QuantizeFilter(const T* input, LowPrecision::Shape k_shape, T* output, LowPrecision::MemLayout layout);
                template <typename T>
                LowPrecision::Status QuantizeInput(const T* input, LowPrecision::Shape shape, T* output, LowPrecision::MemLayout layout);
                LowPrecision::Status UnpackOutput(const int32_t* input, Shape shape, int32_t* output);
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
                #if IS_ARM
                void unpack_8x8_block_barrelshift_mul(
                    uint16x8_t& vACC_Ar76543210_x_Wc76543210,
                    uint16x8_t& vACC_Ar76543210_x_Wc07654321,
                    uint16x8_t& vACC_Ar76543210_x_Wc10765432,
                    uint16x8_t& vACC_Ar76543210_x_Wc21076543,
                    uint16x8_t& vACC_Ar76543210_x_Wc32107654,
                    uint16x8_t& vACC_Ar76543210_x_Wc43210765,
                    uint16x8_t& vACC_Ar76543210_x_Wc54321076,
                    uint16x8_t& vACC_Ar76543210_x_Wc65432107
                );
                void unpack_8x8_block_barrelshift_mul(
                    int16x8_t& vACC_Ar76543210_x_Wc76543210,
                    int16x8_t& vACC_Ar76543210_x_Wc07654321,
                    int16x8_t& vACC_Ar76543210_x_Wc10765432,
                    int16x8_t& vACC_Ar76543210_x_Wc21076543,
                    int16x8_t& vACC_Ar76543210_x_Wc32107654,
                    int16x8_t& vACC_Ar76543210_x_Wc43210765,
                    int16x8_t& vACC_Ar76543210_x_Wc54321076,
                    int16x8_t& vACC_Ar76543210_x_Wc65432107
                );
                #endif
                inline void unpack_8x8_block_barrelshift(const int32_t* O, int32_t* O_unpack, size_t offset);
                LowPrecision::PreprocessType InputPreProcess();
                LowPrecision::PreprocessType FilterPreProcess();
                LowPrecision::PreprocessType OutputPreProcess();
                LowPrecision::PreprocessType OutputPostProcess();
                LowPrecision::GEMMType GEMMSupport();
            }
            
        }

    }
}

#endif // _KERNELS_INTERFACE_BSM_H_