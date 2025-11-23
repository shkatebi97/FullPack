#ifndef LOW_PRECISION_FULLY_CONNECTED_H_
#define LOW_PRECISION_FULLY_CONNECTED_H_
#include "common/types.h"
// #include "common/cvector.hpp"
#include "ops-implementations/mul/LowPrecisionPacking.h"
#include "common/flags.h"
#include "kernels-interface/kernels.hh"
#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cstring>
#include <stdint.h>
#include <tuple>

#if defined(IS_ARM)
#include <arm_neon.h>
#elif (defined(IS_X86) || defined(IS_X86_64)) && (defined(HAS_AVX512) || defined(HAS_AVX2))
#include "common/asmutility.h"
#include <sstream>
#include <immintrin.h>
#endif

// #define PRINT_VALUES true
// #define PRINT_VALUES_DETAILED false
namespace LowPrecision {
    static TimingManager timingManager;
    namespace FullyConnected {
        static long int id = 0;
        LowPrecision::Method get_default_method();
        void set_default_method(LowPrecision::Method method);
        LowPrecision::Method GetMethodFromEnv();
        std::string GetVariableFromEnv(std::string variable);
        LowPrecision::DataType GetDataType(int type);
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type,
            LowPrecision::DataType output_type, bool Is_FC);
        bool IncludesActivationCompression(LowPrecision::Method method);
        bool RequiresOutputUnpacking(LowPrecision::Method method);
        LowPrecision::PreprocessType    InputPreProcess(Method method);
        LowPrecision::PreprocessType    FilterPreProcess(Method method);
        LowPrecision::PreprocessType    OutputPreProcess(Method method);
        LowPrecision::PreprocessType    OutputPostProcess(Method method);
        LowPrecision::GEMMType          GEMMSupport(Method method);
        size_t CalcFlatSize(int* sizes, int num_dims);
        int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape, Method method);
        size_t TransformFilterShape(LowPrecision::Method method, int* shape, int n_dims);
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims);
        template<typename RHS_T> 
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const RHS_T* input, LowPrecision::Shape k_shape, RHS_T* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
        template<typename LHS_T> 
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const LHS_T* input, LowPrecision::Shape shape, LHS_T* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
        template<typename OUT_T> 
        LowPrecision::Status UnpackOutput(LowPrecision::Method method, const OUT_T* input, LowPrecision::Shape shape, OUT_T* output);
        LowPrecision::Status UnpackOutput(LowPrecision::Method method, const int32_t* input, LowPrecision::Shape shape, int32_t* output);
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        template<typename LHS_T, typename RHS_T, typename OUT_T>
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const LHS_T* input, LowPrecision::Shape input_shape,
            const RHS_T* kernel, LowPrecision::Shape kernel_shape,
            OUT_T* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        template<typename LHS_T, typename RHS_T, typename OUT_T>
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const LHS_T* input, LowPrecision::Shape input_shape,
            const RHS_T* kernel, LowPrecision::Shape kernel_shape,
            OUT_T* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        Shape GetPaddedShape(const LowPrecision::Method method, const Shape& input_shape, bool pad_rows_too = false, LowPrecision::MatrixType type = LowPrecision::MatrixType::Unknown);
        Status TransformShapeToPaddedShape(const LowPrecision::Method method, int* input_sizes, int num_dims, bool pad_rows_too = true);
        template <typename Ti, typename To>
        Status PadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape, const To pad_value = 0);
        template<typename Ti, typename To>
        Status DePadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape);
        Status ApplyDowncast(int32_t* input, int8_t* output, Shape shape, const int32_t downcast_coeff);
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n);
        LowPrecision::Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing=nullptr);
    }
    
    LowPrecision::PreprocessType    InputPreProcess(Method method);
    LowPrecision::PreprocessType    FilterPreProcess(Method method);
    LowPrecision::PreprocessType    OutputPreProcess(Method method);
    LowPrecision::PreprocessType    OutputPostProcess(Method method);
    LowPrecision::GEMMType          GEMMSupport(Method method);
    LowPrecision::SelfDependentType IsSelfDependent(Method method);

    LowPrecision::Status MultiplyBackend(
        LowPrecision::Method method,
        const int8_t* input, LowPrecision::Shape input_shape,
        const int8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params = LowPrecision::MulParams());
    LowPrecision::Status MultiplyBackend(
        LowPrecision::Method method,
        const uint8_t* input, LowPrecision::Shape input_shape,
        const uint8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params = LowPrecision::MulParams());
    
    template<typename LHS_T, typename RHS_T, typename OUT_T>
    LowPrecision::Status MultiplyBackend(
        LowPrecision::Method method,
        const LHS_T* input, LowPrecision::Shape input_shape,
        const RHS_T* kernel, LowPrecision::Shape kernel_shape,
        OUT_T* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params = LowPrecision::MulParams());

    LowPrecision::Status PrepareMatrixAsFilterForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PrepareMatrixAsInputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PrepareMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PostprocessMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);

    template<typename T>
    LowPrecision::Status PrepareMatrixAsFilterForMethod(Matrix_t<T>& matrix, Method method, TimingDetailes* timing=nullptr);
    template<typename T>
    LowPrecision::Status PrepareMatrixAsInputForMethod(Matrix_t<T>& matrix, Method method, TimingDetailes* timing=nullptr);
    template<typename T>
    LowPrecision::Status PrepareMatrixAsOutputForMethod(Matrix_t<T>& matrix, Method method, TimingDetailes* timing=nullptr);
    template<typename T>
    LowPrecision::Status PostprocessMatrixAsOutputForMethod(Matrix_t<T>& matrix, Method method, TimingDetailes* timing=nullptr);

    LowPrecision::ShapeList GetInputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape);
    LowPrecision::ShapeList GetFilterShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape);
    LowPrecision::ShapeList GetOutputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, LowPrecision::Shape output_shape);

    LowPrecision::Status GEMM(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing=nullptr);
    template<typename LHS_T, typename RHS_T, typename OUT_T>
    LowPrecision::Status GEMM(Matrix_t<LHS_T>& lhs, Matrix_t<RHS_T>& rhs, Matrix_t<OUT_T>& dst, Method method, TimingDetailes* timing=nullptr);

    void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                        int batch_n, int input_n);
}
#endif