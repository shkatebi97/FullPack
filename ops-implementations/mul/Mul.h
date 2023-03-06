#ifndef MUL_H
#define MUL_H
#include <arm_neon.h>
#include <iostream>
#include <algorithm>

using std::string;
using std::to_string;

#include "../../common/types.h"
#include "../../common/flags.h"
#include "ruy/ruy.h"
#include "ruy/context.h"
#include "Tensor.h"
#include "../../common/asmutility.h"
#include "../../profiler/profiler.h"
#include "4BitShift.h"

class Mul
{
protected:
    Tensor*       _kernel;
    void**        _temperories;
    Shape         _shape;
    Method        _multiplication_method;
    bool          _use_fused;
    bool          _shared_kernel;
    bool          _shared_kernels_copied;
    ruy::Context* _ruy_context;
public:
    Mul();
    Mul(Shape shape, Method multiplication_method, bool use_fused = false, void* context = nullptr);
    Mul(Tensor* kernel, Method multiplication_method, bool use_fused = false, bool copy = false, void* context = nullptr);
    virtual Mul& operator=(const Mul&);
    ~Mul();
    virtual Tensor* operator()(Tensor* input);
    Shape get_shape(){return _shape;}
    Shape get_kernel_shape(){return _kernel->get_shape();}
protected:
    void copy_shape(const Shape& other);
    
    // Multiplication Handlers
    Status do_int8_mul( int8_t* lhs, int32_t* lhs_mantises,
                        void* temperories, int8_t* rhs,  Sign_DataType* rhs_sign,
                        float* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns);
    Status do_int8_shift(int8_t* lhs,
                        int32_t* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns);
    Status do_float_mul(float* lhs, float* rhs, float* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns);
    Status do_ruy_float_mul(Tensor* input, Tensor* output);
    Status do_ruy_int8_mul(Tensor* input, Tensor* output);

    // Multiplication Main Function
    void int8_mul(int8_t* lhs_exponents, int32_t* lhs_mantises,
                  void* temperories, int8_t* rhs,  Sign_DataType* rhs_sign,
                  float* dst,
                  int lhs_columns, int rhs_columns);
    void float_mul(float* lhs, float* rhs, float* dst,
                   int lhs_columns, int rhs_columns);
    
    // Multiplication ASM implementations
    void vector_matrix_multiplication_accumulation_int8(
        int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
        int8_t* src_ptr2_1, int8_t* src_ptr2_2, int8_t* src_ptr2_3, int8_t* src_ptr2_4, 
        Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2, Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4, 
        int8_t *differ_from_max_1_ref,
        float& dst_1, float& dst_2, float& dst_3, float& dst_4,
        int size
    );
    void vector_matrix_multiplication_accumulation_int8_fused(
        int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
        int8_t* src_ptr2_1, int8_t* src_ptr2_2, int8_t* src_ptr2_3, int8_t* src_ptr2_4,
        Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2, Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4,  
        float& dst_1, float& dst_2, float& dst_3, float& dst_4,
        int size
    );
    void vector_matrix_multiplication_accumulation_int8_hybrid_fused(
        int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
        int8_t* src_ptr2_1, int8_t* src_ptr2_2, int8_t* src_ptr2_3, int8_t* src_ptr2_4,
        Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2, Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4,  
        float& dst_1, float& dst_2, float& dst_3, float& dst_4,
        int size
    );
    void vector_matrix_multiplication_accumulation_int8_shift(
        int8_t* src_ptr1_exp,
        int8_t* src_ptr2_ref,
        int32_t& dst_1, int32_t& dst_2, int32_t& dst_3, int32_t& dst_4,
        int size
    );
    void vector_matrix_multiplication_accumulation_float32(
        float* src_ptr1,
        float* src_ptr2_1, float* src_ptr2_2, float* src_ptr2_3, float* src_ptr2_4,
        float *dst_1, float *dst_2, float *dst_3, float *dst_4,
        int size
    );
    void weight_matrix_pack_int8_signed(
        int8_t* src, int8_t* src_sign,
        int8_t* packed_matrix,
        int rows, int columns
    );
    void activation_single_batched_vector_pack_int8(int8_t* src, int8_t* packed_matrix, int columns);
    void matrix_pack_int8_signed_impl(
        int8_t* src_ptr_1, int8_t* src_ptr_2,
        int8_t* src_ptr_3, int8_t* src_ptr_4,
        int8_t* src_ptr_s_ref_1, int8_t* src_ptr_s_ref_2,
        int8_t* src_ptr_s_ref_3, int8_t* src_ptr_s_ref_4,
        int8_t* dst_ptr,
        int size
    );
    void vector_pack_int8_impl(int8_t* src_ptr_ref, int8_t* dst_ptr, int size);

    // Binary Multiplication
    Status Prepare_binary(DataType data_type = DataType::Int8);
    Status Init_binary();
    Status Free_binary(DataType data_type);
    Status Eval_binary(int8_t* lhs,
                       int32_t* dst, 
                       int lhs_columns, int rhs_rows, int rhs_columns);
    Status Eval_binary(float* lhs,
                       float* dst, 
                       int lhs_columns, int rhs_rows, int rhs_columns);
    Status Eval_binary(float16* lhs,
                       float16* dst, 
                       int lhs_columns, int rhs_rows, int rhs_columns);
    // Ternary Multiplication
    Status Prepare_ternary(DataType data_type = DataType::Int8);
    Status Init_ternary();
    Status Free_ternary(DataType data_type);
    Status Eval_ternary(int8_t* lhs,
                        int32_t* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns);
    Status Eval_ternary(float* lhs,
                       float* dst, 
                       int lhs_columns, int rhs_rows, int rhs_columns);
    Status Eval_ternary(float16* lhs,
                       float16* dst, 
                       int lhs_columns, int rhs_rows, int rhs_columns);
    // Quaternary Multiplication
    Status Prepare_quaternary(DataType data_type = DataType::Int8);
    Status Init_quaternary();
    Status Free_quaternary(DataType data_type);
    Status Eval_quaternary(int8_t* lhs,
                           int32_t* dst, 
                           int lhs_columns, int rhs_rows, int rhs_columns);
    // 4Bit Multiplication
    Status Prepare_4Bit(DataType data_type = DataType::Int8);
    Status Init_4Bit();
    Status Free_4Bit(DataType data_type);
    Status Eval_4Bit(int8_t* lhs,
                     int32_t* dst, 
                     int lhs_columns, int rhs_rows, int rhs_columns);
    friend class LSTM;
};

#endif