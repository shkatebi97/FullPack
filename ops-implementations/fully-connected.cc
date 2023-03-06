#include "fully-connected.h"

Fully_Connected::Fully_Connected():
Mul(){
    _is_multi_batched = false;
}

Fully_Connected::Fully_Connected(Shape shape, 
                                 Method multiplication_method, 
                                 bool use_fused,
                                 void* context):
Mul(shape, multiplication_method, use_fused, context){
    _is_multi_batched = false;
}

Fully_Connected::Fully_Connected(Tensor* kernel, 
                                 Method multiplication_method, 
                                 bool use_fused, bool copy,
                                 void* context):
Mul(kernel, multiplication_method, use_fused, copy, context){
    _is_multi_batched = false;
}

Fully_Connected::~Fully_Connected(){}

Fully_Connected& Fully_Connected::operator=(const Fully_Connected& other){
    Mul::operator=(other);
    this->_is_multi_batched = other._is_multi_batched;
    return *this;
}

// Multi-batch inference only is supported
// on float multiplications for now
Status Fully_Connected::set_multi_batch(){
    if(_multiplication_method != Method::kFloatMultiplication)
        return Status::NotImplemented;
    if(!_is_multi_batched){
        if(_kernel->get_transpose())
            throw string("Kernel is transposed while operation is in single batch state\n");
        Status transpose_result = _kernel->Transpose();
        if(transpose_result != Status::Success)
            return transpose_result;
    }
    _is_multi_batched = true;
    return Status::Success;
}
Status Fully_Connected::set_single_batch(){
    if(_is_multi_batched){
        Status transpose_result = _kernel->Transpose();
        if(transpose_result != Status::Success)
            return transpose_result;
    }
    _is_multi_batched = false;
    return Status::Success;
}

Tensor* Fully_Connected::operator()(Tensor* input){
    if (!_is_multi_batched){
        return Mul::operator()(input);
    }
    else{
        Shape in_shape = input->get_shape();
        if (in_shape.number_dims != 2)
            throw string("Invalid number of dimensions in input. ") + to_string(in_shape.number_dims);
        if (in_shape.size[1] != _shape.size[0])
            throw string("Input size mismatch. Should be ") + to_string(_shape.size[0]) +
                  string(" but the passed input has the size of ") + to_string(in_shape.size[1]);
        if(_kernel->get_shape().number_dims != 2)
            throw string("Kernel Shape is not 2D, got: ") + 
                  get_shape_string(_kernel->get_shape());
        if(in_shape.size[1] != _kernel->get_shape().size[0])
            throw string("Kernel Shape is diffrent from the input verified Shape. Kernel: ") + 
                  get_shape_string(_kernel->get_shape()) +
                  string(" Input: ") + get_shape_string(input->get_shape());
        if (in_shape.size[0] % 4)
            throw string("Batch number can only be multiply of 4. But got ") + to_string(in_shape.size[0]);
        Shape* output_shape = new Shape;
        output_shape->number_dims = 2;
        output_shape->size = new int[2];
        output_shape->size[0] = in_shape.size[0];
        output_shape->size[1] = _shape.size[1];
        output_shape->flatsize = output_shape->size[0] * output_shape->size[1];
        if(output_shape->size[1] != _kernel->get_shape().size[1])
            throw string("Kernel Shape is diffrent from the output verified Shape. Kernel: ") + 
                  get_shape_string(_kernel->get_shape()) +
                  string(" Input: ") + get_shape_string(*output_shape);
        Tensor* output = new Tensor(*output_shape);
        output->Allocate();
        if(_multiplication_method == Method::kLogMultiplication || 
           _multiplication_method == Method::kHybridFusedLogMultiplication){
            throw string("Not Implemented: Multi-Batch Fully Connected is only supported for float operation");
            // Status in_extract = input->Extract();
            // if (in_extract != Status::Success)
            //     throw string("Input Extraction Failed. ") + to_string(in_extract);
            // do_int8_mul(input->get_exponents(), input->get_mantisas(),
            //             (_use_fused)?(nullptr):(_temperories->get_exponents()),
            //             _kernel->get_exponents(), _kernel->get_signs(),
            //             output->get_data(),
            //             in_shape.flatsize, _shape.size[0], output_shape->flatsize);
        }
        else if(_multiplication_method == Method::kFloatMultiplication){
            output->Fill_Zero();
            do_float_mul_mul_batch(input->get_data(), _kernel->get_data(),
                                   output->get_data(),
                                   in_shape.size[0], in_shape.size[1],
                                   _shape.size[0], _shape.size[1]);
            
        }
        else
            throw string("Undefiend Multiplication Method. ") + to_string(_multiplication_method);
        output->set_fill();
        output->Refresh();
        return output;
    }
}

Status Fully_Connected::do_float_mul_mul_batch(float* lhs, float* rhs, float* dst, 
                                               int lhs_rows, int lhs_columns, 
                                               int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;
    if (lhs_rows == 1)
        return Status::WrongMethod;

    float_mul_mul_batch(lhs, rhs, dst, lhs_rows, lhs_columns, rhs_columns);

    return Status::Success;
}

void Fully_Connected::float_mul_mul_batch(float* lhs, float* rhs, float* dst,
                                          int lhs_rows, int lhs_columns, int rhs_columns){
    if (lhs_rows % 4)
        throw string("number of batches must be multiply of 4");

    // Do matrix transpose instead!
    float *rhs_ptr = rhs;

    float *lhs_ptr_1 = lhs + 0 * lhs_columns;
    float *lhs_ptr_2 = lhs + 1 * lhs_columns;
    float *lhs_ptr_3 = lhs + 2 * lhs_columns;
    float *lhs_ptr_4 = lhs + 3 * lhs_columns;

    float *rhs_ptr_1 = rhs_ptr + 0 * lhs_columns;
    float *rhs_ptr_2 = rhs_ptr + 1 * lhs_columns;
    float *rhs_ptr_3 = rhs_ptr + 2 * lhs_columns;
    float *rhs_ptr_4 = rhs_ptr + 3 * lhs_columns;

    float *dst_ptr_1 = dst + 0 * rhs_columns;
    float *dst_ptr_2 = dst + 1 * rhs_columns;
    float *dst_ptr_3 = dst + 2 * rhs_columns;
    float *dst_ptr_4 = dst + 3 * rhs_columns;

    int i, j;

    for (j = 0 ; (j+4) <= lhs_rows ; j+=4){
        for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
            matrix_matrix_multiplication_accumulation_float32(
                lhs_ptr_1, lhs_ptr_2, lhs_ptr_3, lhs_ptr_4,
                rhs_ptr_1, rhs_ptr_2, rhs_ptr_3, rhs_ptr_4,
                dst_ptr_1, dst_ptr_2, dst_ptr_3, dst_ptr_4,
                lhs_columns);
            rhs_ptr_1 += 4 * lhs_columns;
            rhs_ptr_2 += 4 * lhs_columns;
            rhs_ptr_3 += 4 * lhs_columns;
            rhs_ptr_4 += 4 * lhs_columns;

            dst_ptr_1 += 4;
            dst_ptr_2 += 4;
            dst_ptr_3 += 4;
            dst_ptr_4 += 4;
        }
        i = rhs_columns - (i - 4);
        if (i == 1){
            rhs_ptr_2 = rhs_ptr_1;
            rhs_ptr_3 = rhs_ptr_1;
            rhs_ptr_4 = rhs_ptr_1;

            dst_ptr_2 = dst_ptr_1;
            dst_ptr_3 = dst_ptr_1;
            dst_ptr_4 = dst_ptr_1;
            matrix_matrix_multiplication_accumulation_float32(
                lhs_ptr_1, lhs_ptr_2, lhs_ptr_3, lhs_ptr_4,
                rhs_ptr_1, rhs_ptr_2, rhs_ptr_3, rhs_ptr_4,
                dst_ptr_1, dst_ptr_2, dst_ptr_3, dst_ptr_4,
                lhs_columns);
        }
        else if (i == 2){
            rhs_ptr_3 = rhs_ptr_2;
            rhs_ptr_4 = rhs_ptr_2;

            dst_ptr_3 = dst_ptr_1;
            dst_ptr_4 = dst_ptr_1;
            matrix_matrix_multiplication_accumulation_float32(
                lhs_ptr_1, lhs_ptr_2, lhs_ptr_3, lhs_ptr_4,
                rhs_ptr_1, rhs_ptr_2, rhs_ptr_3, rhs_ptr_4,
                dst_ptr_1, dst_ptr_2, dst_ptr_3, dst_ptr_4,
                lhs_columns);
        }
        else if (i == 3){
            rhs_ptr_4 = rhs_ptr_3;

            dst_ptr_4 = dst_ptr_1;
            matrix_matrix_multiplication_accumulation_float32(
                lhs_ptr_1, lhs_ptr_2, lhs_ptr_3, lhs_ptr_4,
                rhs_ptr_1, rhs_ptr_2, rhs_ptr_3, rhs_ptr_4,
                dst_ptr_1, dst_ptr_2, dst_ptr_3, dst_ptr_4,
                lhs_columns);
        }
        else if (i == 4){
            matrix_matrix_multiplication_accumulation_float32(
                lhs_ptr_1, lhs_ptr_2, lhs_ptr_3, lhs_ptr_4,
                rhs_ptr_1, rhs_ptr_2, rhs_ptr_3, rhs_ptr_4,
                dst_ptr_1, dst_ptr_2, dst_ptr_3, dst_ptr_4,
                lhs_columns);
        }
        lhs_ptr_1 += 4 * lhs_columns;
        lhs_ptr_2 += 4 * lhs_columns;
        lhs_ptr_3 += 4 * lhs_columns;
        lhs_ptr_4 += 4 * lhs_columns;

        dst_ptr_1 = dst + (j+4) * rhs_columns;
        dst_ptr_2 = dst + (j+4) * rhs_columns;
        dst_ptr_3 = dst + (j+4) * rhs_columns;
        dst_ptr_4 = dst + (j+4) * rhs_columns;

        rhs_ptr_1 = rhs_ptr + 0 * lhs_columns;
        rhs_ptr_2 = rhs_ptr + 1 * lhs_columns;
        rhs_ptr_3 = rhs_ptr + 2 * lhs_columns;
        rhs_ptr_4 = rhs_ptr + 3 * lhs_columns;
    }
}


void Fully_Connected::matrix_matrix_multiplication_accumulation_float32(
        float* src_ptr1_1, float* src_ptr1_2, float* src_ptr1_3, float* src_ptr1_4,
        float* src_ptr2_1, float* src_ptr2_2, float* src_ptr2_3, float* src_ptr2_4,
        float* dst_1     , float* dst_2     , float* dst_3     , float* dst_4     ,
    int size){
    // float_t c1, c2, c3, c4;
    size_t i, j;

    // these are the columns A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;
    
    // these are the columns C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    float* src_ptr1_1_ptr = src_ptr1_1;
    float* src_ptr1_2_ptr = src_ptr1_2;
    float* src_ptr1_3_ptr = src_ptr1_3;
    float* src_ptr1_4_ptr = src_ptr1_4;
    
    float* src_ptr2_1_ptr = src_ptr2_1;
    float* src_ptr2_2_ptr = src_ptr2_2;
    float* src_ptr2_3_ptr = src_ptr2_3;
    float* src_ptr2_4_ptr = src_ptr2_4;

    C0 = vmovq_n_f32(0);
    C1 = vmovq_n_f32(0);
    C2 = vmovq_n_f32(0);
    C3 = vmovq_n_f32(0);
    for (i = 0; (i+4) <= size; i+=4){
        A0 = vld1q_f32(src_ptr1_1_ptr); src_ptr1_1_ptr += 4;
        A1 = vld1q_f32(src_ptr1_2_ptr); src_ptr1_2_ptr += 4;
        A2 = vld1q_f32(src_ptr1_3_ptr); src_ptr1_3_ptr += 4;
        A3 = vld1q_f32(src_ptr1_4_ptr); src_ptr1_4_ptr += 4;

        B0 = vld1q_f32(src_ptr2_1_ptr); src_ptr2_1_ptr += 4;
        C0 = vfmaq_laneq_f32(C0 , A0 , B0 , 0);
        C0 = vfmaq_laneq_f32(C0 , A1 , B0 , 1);
        C0 = vfmaq_laneq_f32(C0 , A2 , B0 , 2);
        C0 = vfmaq_laneq_f32(C0 , A3 , B0 , 3);

        B1 = vld1q_f32(src_ptr2_2_ptr); src_ptr2_2_ptr += 4;
        C1 = vfmaq_laneq_f32(C1 , A0 , B1 , 0);
        C1 = vfmaq_laneq_f32(C1 , A1 , B1 , 1);
        C1 = vfmaq_laneq_f32(C1 , A2 , B1 , 2);
        C1 = vfmaq_laneq_f32(C1 , A3 , B1 , 3);

        B2 = vld1q_f32(src_ptr2_3_ptr); src_ptr2_3_ptr += 4;
        C2 = vfmaq_laneq_f32(C2 , A0 , B2 , 0);
        C2 = vfmaq_laneq_f32(C2 , A1 , B2 , 1);
        C2 = vfmaq_laneq_f32(C2 , A2 , B2 , 2);
        C2 = vfmaq_laneq_f32(C2 , A3 , B3 , 3);

        B3 = vld1q_f32(src_ptr2_4_ptr); src_ptr2_4_ptr += 4;
        C3 = vfmaq_laneq_f32(C3 , A0 , B3 , 0);
        C3 = vfmaq_laneq_f32(C3 , A1 , B3 , 1);
        C3 = vfmaq_laneq_f32(C3 , A2 , B3 , 2);
        C3 = vfmaq_laneq_f32(C3 , A3 , B3 , 3);

        #if DO_FLOAT_PREFETCHING
        __builtin_prefetch(src_ptr1_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_1_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_2_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_3_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_4_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        #endif
    }

    vst1q_f32(dst_1 , C0);
    vst1q_f32(dst_2 , C1);
    vst1q_f32(dst_3 , C2);
    vst1q_f32(dst_4 , C3);
}



