#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "mul/Mul.h"
#include "mul/Tensor.h"
#include "../common/types.h"
#include <string>
#include <iostream>

using std::string;

class Fully_Connected: public Mul
{
private:
    bool    _is_multi_batched;
public:
    Fully_Connected();
    Fully_Connected(Shape shape, Method multiplication_method = Method::kFloatMultiplication, bool use_fused = false, void* context = nullptr);
    Fully_Connected(Tensor* kernel, Method multiplication_method = Method::kFloatMultiplication, bool use_fused = false, bool copy = false, void* context = nullptr);
    ~Fully_Connected();
    Fully_Connected& operator=(const Fully_Connected&);
    Tensor* operator()(Tensor* input);

    Status set_multi_batch();
    Status set_single_batch();
protected:
    Status do_float_mul_mul_batch(float* lhs, float* rhs, float* dst, 
                                  int lhs_rows, int lhs_columns,
                                  int rhs_rows, int rhs_columns);
    void float_mul_mul_batch(float* lhs, float* rhs, float* dst,
                             int lhs_rows, int lhs_columns, int rhs_columns);

    void matrix_matrix_multiplication_accumulation_float32(
        float* src_ptr1_1, float* src_ptr1_2, float* src_ptr1_3, float* src_ptr1_4,
        float* src_ptr2_1, float* src_ptr2_2, float* src_ptr2_3, float* src_ptr2_4,
        float* dst_1     , float* dst_2     , float* dst_3     , float* dst_4     ,
        int size
    );
};



#endif