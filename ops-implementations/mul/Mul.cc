#include "Mul.h"

Mul::Mul(){}

// The Kernel Shape Must be Passed in the following shape (n_output, n_input)
Mul::Mul(Shape shape, Method multiplication_method, bool use_fused, void* context):
_shape(shape),
_multiplication_method(multiplication_method),
_use_fused(use_fused),
_shared_kernel(false),
_shared_kernels_copied(false){
    if (shape.number_dims != 2)
        throw std::string("Mul kernel shape dimensions invalid");
    if (shape.flatsize <= 0)
        throw std::string("Mul kernel shape empty");
    _kernel = new Tensor(shape);
    Status init = _kernel->Initialize();
    if (init != Status::Success)
        throw std::string("Mul kernel Initialization failed");
    if(_multiplication_method == Method::kLogMultiplication && !_use_fused){
        Shape temp_shape;
        temp_shape.number_dims = 1;
        temp_shape.size = new int[1];
        temp_shape.size[0] = _shape.size[1];
        temp_shape.flatsize = temp_shape.size[0];
        _temperories = new void*[1];
        Tensor* t = new Tensor(temp_shape);
        init = t->Allocate();
        if (init != Status::Success)
            throw std::string("Mul temperories Initialization failed");
        _temperories[0] = (void*) t;
    }
    else if(_multiplication_method == Method::kInt8Shift){
        int weight_shape;
        int activation_shape;
        // Shape = (packed weight shape) + (packed activation shape)
        #ifdef EXTEND_FOR_ACCURACY
        weight_shape       = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (2 + 2));
        activation_shape   = (_shape.size[0] * 2);
        #else
        weight_shape       = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (1 + 1));
        activation_shape   = (_shape.size[0] * 1);
        #endif
        int8_t *weight_packed, *activation_packed;
        weight_packed      = allocate<int8_t>(weight_shape);
        activation_packed  = allocate<int8_t>(activation_shape);
        int8_t* rhs        = _kernel->get_exponents();
        int8_t* rhs_sign   = _kernel->get_signs_8();
        _temperories = new void*[2];
        _temperories[0] = (void*) weight_packed;
        _temperories[1] = (void*) activation_packed;
        weight_matrix_pack_int8_signed(rhs, rhs_sign, weight_packed, _shape.size[0], _shape.size[1]);
    }
    else if(_multiplication_method == Method::kInt8Binary){
        Status preparation = Prepare_binary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kFloat32Binary){
        Status preparation = Prepare_binary(DataType::Float32);
        if (preparation == Status::NotImplemented)
            throw std::string("Mul binary multiplication preparation failed. Reason: Using Float32 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
        else if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kFloat16Binary){
        Status preparation = Prepare_binary(DataType::Float16);
        if (preparation == Status::NotSupported){
            _multiplication_method = Method::kFloat32Binary;
            preparation = Prepare_binary(DataType::Float32);
            if (preparation == Status::NotImplemented)
                throw std::string("Mul binary multiplication preparation failed. Reason: Using Float32 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
            else if (preparation == Status::DimensionsMisMatch)
                throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
            else if (preparation == Status::SizesMisMatch)
                throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
            else if (preparation != Status::Success)
                throw std::string("Mul binary multiplication preparation failed.");
        }
        else if (preparation == Status::NotImplemented)
            throw std::string("Mul binary multiplication preparation failed. Reason: Using Float16 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
        else if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8Ternary){
        Status preparation = Prepare_ternary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul Ternary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul Ternary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul Ternary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8QuaTernary){
        Status preparation = Prepare_quaternary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul QuaTernary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul QuaTernary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul QuaTernary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8Int4){
        Status preparation = Prepare_4Bit();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul 4-Bit multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul 4-Bit multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul 4-Bit multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8ShiftInt4){
        _temperories = new void*[1];
        Status preparation = Shift4Bit::Prepare(
            _kernel->get_exponents(), _kernel->get_shape(),
            _temperories, DataType::Int8, MemLayout::kRowMajor
        );
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul Shift4Bit multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul Shift4Bit multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul Shift4Bit multiplication preparation failed.");
    }
    else
        _temperories = nullptr;
    if (_multiplication_method == Method::kFloatRuyMultiplication ||
        _multiplication_method == Method::kInt8Multiplication)
        if (context)
            _ruy_context = get_pointer_as<ruy::Context>(context);
        else
            _ruy_context = new ruy::Context;
    else
        _ruy_context = nullptr;
}

// Kernel Must be Passed in the following shape (n_output, n_input)
Mul::Mul(Tensor* kernel, Method multiplication_method, bool use_fused, bool copy, void* context):
 _multiplication_method(multiplication_method),
 _use_fused(use_fused),
_shared_kernel(true),
_shared_kernels_copied(copy){
    Shape shape = kernel->get_shape();
    if (shape.number_dims != 2)
        throw std::string("Mul kernel shape dimensions invalid");
    if (shape.flatsize <= 0)
        throw std::string("Mul kernel empty");
    if (copy){
        _kernel = new Tensor(shape);
        *(_kernel) = *(kernel);
    }
    else
        _kernel = kernel;
    _shape = shape;
    if(_multiplication_method == Method::kLogMultiplication && !_use_fused){
        Shape temp_shape;
        temp_shape.number_dims = 1;
        temp_shape.size = new int[1];
        temp_shape.size[0] = _shape.size[1];
        temp_shape.flatsize = temp_shape.size[0];
        _temperories = new void*[1];
        Tensor* t = new Tensor(temp_shape);
        Status init = t->Allocate();
        if (init != Status::Success)
            throw std::string("Mul temperories Initialization failed");
        _temperories[0] = (void*) t;
    }
    else if(_multiplication_method == Method::kInt8Shift){
        int weight_shape;
        int activation_shape;
        // Shape = (packed weight shape) + (packed activation shape)
        #ifdef EXTEND_FOR_ACCURACY
        weight_shape       = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (2 + 2));
        activation_shape   = (_shape.size[0] * 2);
        #else
        weight_shape       = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (1 + 1));
        activation_shape   = (_shape.size[0] * 1);
        #endif
        int8_t *weight_packed, *activation_packed;
        weight_packed      = allocate<int8_t>(weight_shape);
        activation_packed  = allocate<int8_t>(activation_shape);
        int8_t* rhs        = _kernel->get_exponents();
        int8_t* rhs_sign   = _kernel->get_signs_8();
        _temperories = new void*[2];
        _temperories[0] = (void*) weight_packed;
        _temperories[1] = (void*) activation_packed;
        weight_matrix_pack_int8_signed(rhs, rhs_sign, weight_packed, _shape.size[0], _shape.size[1]);
    }
    else if(_multiplication_method == Method::kInt8Binary){
        Status preparation = Prepare_binary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kFloat32Binary){
        Status preparation = Prepare_binary(DataType::Float32);
        if (preparation == Status::NotImplemented)
            throw std::string("Mul binary multiplication preparation failed. Reason: Using Float32 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
        else if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kFloat16Binary){
        Status preparation = Prepare_binary(DataType::Float16);
        if (preparation == Status::NotSupported){
            _multiplication_method = Method::kFloat32Binary;
            preparation = Prepare_binary(DataType::Float32);
            if (preparation == Status::NotImplemented)
                throw std::string("Mul binary multiplication preparation failed. Reason: Using Float32 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
            else if (preparation == Status::DimensionsMisMatch)
                throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
            else if (preparation == Status::SizesMisMatch)
                throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
            else if (preparation != Status::Success)
                throw std::string("Mul binary multiplication preparation failed.");
        }
        else if (preparation == Status::NotImplemented)
            throw std::string("Mul binary multiplication preparation failed. Reason: Using Float16 I/O while Single Row Operations is activated is forbidden. Please recompile your binary with 'USE_SINGLE_ROW_BINARY_OP' off");
        else if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul binary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul binary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8Ternary){
        Status preparation = Prepare_ternary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul Ternary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul Ternary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul Ternary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8QuaTernary){
        Status preparation = Prepare_quaternary();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul QuaTernary multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul QuaTernary multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul QuaTernary multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8Int4){
        Status preparation = Prepare_4Bit();
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul 4-Bit multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul 4-Bit multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul 4-Bit multiplication preparation failed.");
    }
    else if(_multiplication_method == Method::kInt8ShiftInt4){
        _temperories = new void*[1];
        Status preparation = Shift4Bit::Prepare(
            _kernel->get_exponents(), _kernel->get_shape(),
            _temperories, DataType::Int8, MemLayout::kRowMajor
        );
        if (preparation == Status::DimensionsMisMatch)
            throw std::string("Mul Shift4Bit multiplication preparation failed. Reason: Dimensions MisMatch");
        else if (preparation == Status::SizesMisMatch)
            throw std::string("Mul Shift4Bit multiplication preparation failed. Reason: Sizes MisMatch");
        else if (preparation != Status::Success)
            throw std::string("Mul Shift4Bit multiplication preparation failed.");
    }
    else
        _temperories = nullptr;
    if (_multiplication_method == Method::kFloatRuyMultiplication ||
        _multiplication_method == Method::kInt8Multiplication)
        if (context)
            _ruy_context = get_pointer_as<ruy::Context>(context);
        else
            _ruy_context = new ruy::Context;
    else
        _ruy_context = nullptr;
}

Mul& Mul::operator=(const Mul& other){
    copy_shape(other._shape);
    this->_multiplication_method = other._multiplication_method;
    this->_use_fused             = other._use_fused;
    this->_kernel                = new Tensor();
    *(this->_kernel)             = *(other._kernel);
    if(_multiplication_method == Method::kLogMultiplication && !_use_fused){
        _temperories             = new void*[1];
        Tensor* temp             = new Tensor(get_pointer_as<Tensor>(other._temperories[0])->get_shape());
        *(temp)                  = *(get_pointer_as<Tensor>(other._temperories[0]));
        _temperories[0]          = (void*) temp;
    }
    else if(_multiplication_method == Method::kInt8Shift){
        _temperories             = new void*[2];
        int weight_shape, activation_shape;
        // Shape                 = (packed weight shape) + (packed activation shape)
        #ifdef EXTEND_FOR_ACCURACY
        weight_shape             = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (2 + 2));
        activation_shape         = (_shape.size[0] * 2);
        #else
        weight_shape             = (_shape.size[1] * (((int)(_shape.size[0]/4)) + 1) * (1 + 1));
        activation_shape         = (_shape.size[0] * 1);
        #endif
        int8_t *weight_packed, 
            *activation_packed;
        int8_t *o_weight_packed,
            *o_activation_packed;
        weight_packed            = allocate<int8_t>(weight_shape);
        activation_packed        = allocate<int8_t>(activation_shape);
        o_weight_packed          = get_pointer_as<int8_t>(other._temperories[0]);
        o_activation_packed      = get_pointer_as<int8_t>(other._temperories[1]);
        _temperories[0]          = (void*) weight_packed;
        _temperories[1]          = (void*) activation_packed;
        std::copy(o_weight_packed, o_weight_packed + weight_shape, weight_packed);
        std::copy(o_activation_packed, o_activation_packed + activation_shape, activation_packed);
    }
    else
        this->_temperories       = nullptr;
    if (_multiplication_method == Method::kFloatRuyMultiplication ||
        _multiplication_method == Method::kInt8Multiplication){
        if (other._ruy_context)
            _ruy_context         = other._ruy_context;
        else
            _ruy_context         = new ruy::Context;
    }
    else
        _ruy_context             = nullptr;
    return *this;
}

void Mul::copy_shape(const Shape& other){
    _shape = other;
}

Mul::~Mul(){
    if(!_shared_kernel || _shared_kernels_copied)
        deallocate(_kernel);
    if(_multiplication_method == Method::kLogMultiplication && !_use_fused){
        deallocate(get_pointer_as<Tensor>(_temperories[0]));
        deallocate(_temperories);
    }
    else if(_multiplication_method == Method::kInt8Shift){
        deallocate(get_pointer_as<int8_t>(_temperories[0]));
        deallocate(get_pointer_as<int8_t>(_temperories[1]));
        deallocate(_temperories);
    }
    else if(_multiplication_method == Method::kInt8Binary)
        Free_binary(DataType::Int8);
    else if(_multiplication_method == Method::kFloat16Binary)
        Free_binary(DataType::Float16);
    else if(_multiplication_method == Method::kFloat32Binary)
        Free_binary(DataType::Float32);
    else if(_multiplication_method == Method::kInt8Ternary)
        Free_ternary(DataType::Int8);
    else if(_multiplication_method == Method::kInt8QuaTernary)
        Free_quaternary(DataType::Int8);
    else if(_multiplication_method == Method::kInt8Int4)
        Free_quaternary(DataType::Int8);
    else if(_multiplication_method == Method::kInt8ShiftInt4)
        Shift4Bit::Free(DataType::Int8, _temperories);
}

Tensor* Mul::operator()(Tensor* input){
    Shape in_shape = input->get_shape();
    if (in_shape.number_dims != 1)
        throw string("Invalid number of dimensions in input. ") + to_string(in_shape.number_dims);
    if (in_shape.flatsize != _shape.size[0])
        throw string("Input size mismatch. Should be ") + to_string(_shape.size[0]) +
              string(" but the passed input has the size of ") + to_string(in_shape.flatsize);
    if(_kernel->get_shape().number_dims != 2)
        throw string("Kernel Shape in not 2D, is: ") + get_shape_string(_kernel->get_shape());
    if(in_shape.flatsize != _kernel->get_shape().size[0])
        throw string("Kernel Shape is diffrent from the input verified Shape. Kernel: ") + get_shape_string(_kernel->get_shape()) +
              string(" Input: ") + get_shape_string(input->get_shape());
    Shape output_shape;
    output_shape.number_dims = 1;
    output_shape.size = new int[1];
    output_shape.size[0] = _shape.size[1];
    output_shape.flatsize = output_shape.size[0];
    Tensor* output = new Tensor(output_shape);
    if(output_shape.flatsize != _kernel->get_shape().size[1])
        throw string("Kernel Shape is diffrent from the output verified Shape. Kernel: ") + get_shape_string(_kernel->get_shape()) +
              string(" Input: ") + get_shape_string(output_shape);
    output->Allocate();
    if (_multiplication_method == Method::kLogMultiplication ||
       _multiplication_method == Method::kHybridFusedLogMultiplication){
        Status in_extract = input->Extract();
        if (in_extract != Status::Success)
            throw string("Input Extraction Failed. ") + to_string(in_extract);
        int8_t *temperory_exponents = nullptr;
        if (_multiplication_method == kLogMultiplication && !_use_fused)
            temperory_exponents = get_pointer_as<Tensor>(_temperories[0])->get_exponents();
        do_int8_mul(input->get_exponents(), input->get_mantisas(),
                    temperory_exponents,
                    _kernel->get_exponents(), _kernel->get_signs(),
                    output->get_data(),
                    in_shape.flatsize, _shape.size[0], output_shape.flatsize);
        output->set_fill();
        output->Refresh();
    }
    else if (_multiplication_method == Method::kFloatMultiplication){
        output->Fill_Zero();
        do_float_mul(input->get_data(), _kernel->get_data(),
                     output->get_data(),
                     in_shape.flatsize, _shape.size[0], output_shape.flatsize);
        output->set_fill();
        
    }
    else if (_multiplication_method == Method::kFloatRuyMultiplication){
        do_ruy_float_mul(input, output);
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8Multiplication){
        Status downcast_s = input->DownCastInt32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        do_ruy_int8_mul(input, output);
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8Shift){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        do_int8_shift(
            input->get_int8_ptr(),
            output->get_int32_ptr(),
            in_shape.flatsize, _shape.size[0], output_shape.flatsize
        );
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8Binary){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        Eval_binary(
            input->get_int8_ptr(),
            output->get_int32_ptr(),
            in_shape.flatsize, _shape.size[0], output_shape.flatsize
        );
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else if (_multiplication_method == Method::kFloat32Binary){
        Eval_binary(input->get_data(),
                    output->get_data(),
                    in_shape.flatsize, _shape.size[0], output_shape.flatsize);
        output->set_fill();
    }
    else if (_multiplication_method == Method::kFloat16Binary){
        Eval_binary(input->get_data_float16(),
                    output->get_data_float16(),
                    in_shape.flatsize, _shape.size[0], output_shape.flatsize);
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8Ternary){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        Eval_ternary(
            input->get_int8_ptr(),
            output->get_int32_ptr(),
            in_shape.flatsize, _shape.size[0], output_shape.flatsize
        );
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8QuaTernary){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        Eval_quaternary(
            input->get_int8_ptr(),
            output->get_int32_ptr(),
            in_shape.flatsize, _shape.size[0], output_shape.flatsize
        );
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8Int4){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        Eval_4Bit(
            input->get_int8_ptr(),
            output->get_int32_ptr(),
            in_shape.flatsize, _shape.size[0], output_shape.flatsize
        );
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else if (_multiplication_method == Method::kInt8ShiftInt4){
        Status downcast_s = input->DownCastFloat32ToInt8();
        if (downcast_s == Status::NotAllocated)
            throw string("Input is not allocated");
        if (downcast_s == Status::NotFilled)
            throw string("Input is not filled");
        if (downcast_s != Status::Success)
            throw string("Input down casting failed. ") + to_string(downcast_s);
        Status eval_ret = Shift4Bit::Eval(
            input->get_int8_ptr(), input->get_shape(),
            get_pointer_as<int8_t>(_temperories[0]), _kernel->get_shape(),
            output->get_int32_ptr(), output->get_shape()
        );
        if (eval_ret != Status::Success)
            throw string("Running Main Layer failed. ") + to_string(eval_ret);
        output->set_f32_i8_parameters(input->get_f32_i8_parameters());
        output->UpCastInt32ToFloat32();
        output->set_fill();
    }
    else 
        throw string("Undefiend Multiplication Method. ") + to_string(_multiplication_method);
    return output;
}

Status Mul::do_int8_mul(int8_t* lhs, int32_t* lhs_mantises,
                        void* temperories, int8_t* rhs, Sign_DataType* rhs_sign, 
                        float* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;
    int8_mul(lhs, lhs_mantises,
             temperories, rhs, rhs_sign,
             dst, 
             lhs_columns, rhs_columns);
    
    return Status::Success;
}

Status Mul::do_int8_shift(int8_t* lhs,
                          int32_t* dst, 
                          int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int8_t* rhs_packed = get_pointer_as<int8_t>(_temperories[0]);
    int8_t* lhs_packed = get_pointer_as<int8_t>(_temperories[1]);
    
    activation_single_batched_vector_pack_int8(lhs, lhs_packed, lhs_columns);

    int i;

    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        vector_matrix_multiplication_accumulation_int8_shift(
            lhs_packed,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += 2 * 4 * lhs_columns;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        vector_matrix_multiplication_accumulation_int8_shift(
            lhs_packed,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        vector_matrix_multiplication_accumulation_int8_shift(
            lhs_packed,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        vector_matrix_multiplication_accumulation_int8_shift(
            lhs_packed,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    return Status::Success;
}

Status Mul::do_float_mul(float* lhs, float* rhs, float* dst, 
                         int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;
    float_mul(lhs, rhs, dst, lhs_columns, rhs_columns);

    return Status::Success;
}

Status Mul::do_ruy_float_mul(Tensor* input, Tensor* output){
    if (input->get_shape().number_dims != 1)
        throw string("Input Tensor MUST be 1D for float ruy multiplication. got ") +
              get_shape_string(input->get_shape());
    if (output->get_shape().number_dims != 1)
        throw string("Output Tensor MUST be 1D for float ruy multiplication. got ") +
              get_shape_string(output->get_shape());
    // This function ONLY works for 1D I/O
    ruy::Matrix<float> ruy_lhs;
    ruy::Matrix<float> ruy_rhs;
    ruy::Matrix<float> ruy_dst;

#if TENSORFLOW_LIKE_RUY_MATRIX_MAKE
    ruy::MakeSimpleLayout(
        _kernel->get_shape().size[0],
        _kernel->get_shape().size[1],
        ruy::Order::kRowMajor,
        ruy_lhs.mutable_layout()
    );
    ruy_lhs.set_cache_policy(ruy::CachePolicy::kCacheIfLargeSpeedup);
    ruy_lhs.set_data(_kernel->get_data());

    ruy::MakeSimpleLayout(
        input->get_shape().size[0],
        1,
        ruy::Order::kColMajor,
        ruy_rhs.mutable_layout()
    );
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kNeverCache);
    ruy_rhs.set_data(input->get_data());

    ruy::MakeSimpleLayout(
        output->get_shape().size[0],
        1,
        ruy::Order::kColMajor,
        ruy_dst.mutable_layout()
    );
    ruy_dst.set_cache_policy(ruy::CachePolicy::kNeverCache);
    ruy_dst.set_data(output->get_data());
#else
    input->MakeRuyMatrix(&ruy_lhs);
    _kernel->MakeRuyMatrix(&ruy_rhs, true);
    output->MakeRuyMatrix(&ruy_dst);
#endif

    ruy::MulParams<float, float> ruy_mul_params;

    if (_ruy_context == nullptr)
        _ruy_context = new ruy::Context;
    
    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

    return Status::Success;
}

Status Mul::do_ruy_int8_mul(Tensor* input, Tensor* output){
    if (input->get_shape().number_dims != 1)
        throw string("Input Tensor MUST be 1D for float ruy multiplication. got ") +
              get_shape_string(input->get_shape());
    if (output->get_shape().number_dims != 1)
        throw string("Output Tensor MUST be 1D for float ruy multiplication. got ") +
              get_shape_string(output->get_shape());
    // This function ONLY works for 1D I/O
    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int32_t> ruy_dst;

    input->MakeRuyMatrixInt8(&ruy_lhs);
    _kernel->MakeRuyMatrixInt8(&ruy_rhs, true);
    output->MakeRuyMatrixInt32(&ruy_dst);

    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    if (_ruy_context == nullptr)
        _ruy_context = new ruy::Context;
    
    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

    return Status::Success;
}

void Mul::int8_mul(int8_t* lhs_exponents, int32_t* lhs_mantises,
                   void* temperories, int8_t* rhs, Sign_DataType* rhs_sign,
                   float* dst,
                   int lhs_columns, int rhs_columns){
    int8_t  *src_ptr2_1 = rhs + 0 * lhs_columns;
    int8_t  *src_ptr2_2 = rhs + 1 * lhs_columns;
    int8_t  *src_ptr2_3 = rhs + 2 * lhs_columns;
    int8_t  *src_ptr2_4 = rhs + 3 * lhs_columns;

    Sign_DataType *src_ptr2_sign_1 = rhs_sign + 0 * lhs_columns;
    Sign_DataType *src_ptr2_sign_2 = rhs_sign + 1 * lhs_columns;
    Sign_DataType *src_ptr2_sign_3 = rhs_sign + 2 * lhs_columns;
    Sign_DataType *src_ptr2_sign_4 = rhs_sign + 3 * lhs_columns;

    int i;
    if (_multiplication_method == Method::kHybridFusedLogMultiplication){
        for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
            vector_matrix_multiplication_accumulation_int8_hybrid_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
                lhs_columns);
            src_ptr2_1 += 4 * lhs_columns;
            src_ptr2_2 += 4 * lhs_columns;
            src_ptr2_3 += 4 * lhs_columns;
            src_ptr2_4 += 4 * lhs_columns;

            src_ptr2_sign_1 += 4 * lhs_columns;
            src_ptr2_sign_2 += 4 * lhs_columns;
            src_ptr2_sign_3 += 4 * lhs_columns;
            src_ptr2_sign_4 += 4 * lhs_columns;
        }
        i = rhs_columns - (i - 4);
        if (i == 1){
            src_ptr2_2 = src_ptr2_1;
            src_ptr2_3 = src_ptr2_1;
            src_ptr2_4 = src_ptr2_1;

            src_ptr2_sign_2 = src_ptr2_sign_1;
            src_ptr2_sign_3 = src_ptr2_sign_1;
            src_ptr2_sign_4 = src_ptr2_sign_1;
            vector_matrix_multiplication_accumulation_int8_hybrid_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i], dst[i], dst[i],
                lhs_columns);
        }
        else if (i == 2){
            src_ptr2_3 = src_ptr2_2;
            src_ptr2_4 = src_ptr2_2;
            
            src_ptr2_sign_3 = src_ptr2_sign_2;
            src_ptr2_sign_4 = src_ptr2_sign_2;
            vector_matrix_multiplication_accumulation_int8_hybrid_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
                lhs_columns);
        }
        else if (i == 3){
            src_ptr2_4 = src_ptr2_3;
            src_ptr2_sign_4 = src_ptr2_sign_3;
            vector_matrix_multiplication_accumulation_int8_hybrid_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
                lhs_columns);
        }
    }
    else if (_use_fused){
        for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
            vector_matrix_multiplication_accumulation_int8_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
                lhs_columns);
            src_ptr2_1 += 4 * lhs_columns;
            src_ptr2_2 += 4 * lhs_columns;
            src_ptr2_3 += 4 * lhs_columns;
            src_ptr2_4 += 4 * lhs_columns;

            src_ptr2_sign_1 += 4 * lhs_columns;
            src_ptr2_sign_2 += 4 * lhs_columns;
            src_ptr2_sign_3 += 4 * lhs_columns;
            src_ptr2_sign_4 += 4 * lhs_columns;
        }
        i = rhs_columns - (i - 4);
        if (i == 1){
            src_ptr2_2 = src_ptr2_1;
            src_ptr2_3 = src_ptr2_1;
            src_ptr2_4 = src_ptr2_1;

            src_ptr2_sign_2 = src_ptr2_sign_1;
            src_ptr2_sign_3 = src_ptr2_sign_1;
            src_ptr2_sign_4 = src_ptr2_sign_1;
            vector_matrix_multiplication_accumulation_int8_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i], dst[i], dst[i],
                lhs_columns);
        }
        else if (i == 2){
            src_ptr2_3 = src_ptr2_2;
            src_ptr2_4 = src_ptr2_2;
            
            src_ptr2_sign_3 = src_ptr2_sign_2;
            src_ptr2_sign_4 = src_ptr2_sign_2;
            vector_matrix_multiplication_accumulation_int8_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
                lhs_columns);
        }
        else if (i == 3){
            src_ptr2_4 = src_ptr2_3;
            src_ptr2_sign_4 = src_ptr2_sign_3;
            vector_matrix_multiplication_accumulation_int8_fused(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
                lhs_columns);
        }
    }
    else{
        int8_t  *differ_from_max_1 = get_pointer_as<int8_t>(temperories);
        for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
            vector_matrix_multiplication_accumulation_int8(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                differ_from_max_1,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
                lhs_columns);
            src_ptr2_1 += 4 * lhs_columns;
            src_ptr2_2 += 4 * lhs_columns;
            src_ptr2_3 += 4 * lhs_columns;
            src_ptr2_4 += 4 * lhs_columns;

            src_ptr2_sign_1 += 4 * lhs_columns;
            src_ptr2_sign_2 += 4 * lhs_columns;
            src_ptr2_sign_3 += 4 * lhs_columns;
            src_ptr2_sign_4 += 4 * lhs_columns;
        }
        i = rhs_columns - (i - 4);
        if (i == 1){
            src_ptr2_2 = src_ptr2_1;
            src_ptr2_3 = src_ptr2_1;
            src_ptr2_4 = src_ptr2_1;

            src_ptr2_sign_2 = src_ptr2_sign_1;
            src_ptr2_sign_3 = src_ptr2_sign_1;
            src_ptr2_sign_4 = src_ptr2_sign_1;
            vector_matrix_multiplication_accumulation_int8(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                differ_from_max_1,
                dst[i], dst[i], dst[i], dst[i],
                lhs_columns);
        }
        else if (i == 2){
            src_ptr2_3 = src_ptr2_2;
            src_ptr2_4 = src_ptr2_2;
            
            src_ptr2_sign_3 = src_ptr2_sign_2;
            src_ptr2_sign_4 = src_ptr2_sign_2;
            vector_matrix_multiplication_accumulation_int8(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                differ_from_max_1,
                dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
                lhs_columns);
        }
        else if (i == 3){
            src_ptr2_4 = src_ptr2_3;
            src_ptr2_sign_4 = src_ptr2_sign_3;
            vector_matrix_multiplication_accumulation_int8(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                differ_from_max_1,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
                lhs_columns);
        }
        else if (i == 4){
            vector_matrix_multiplication_accumulation_int8(
                lhs_exponents, lhs_mantises,
                src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
                src_ptr2_sign_1, src_ptr2_sign_2,
                src_ptr2_sign_3, src_ptr2_sign_4,
                differ_from_max_1,
                dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
                lhs_columns);
        }
    }
    return;
}

void Mul::float_mul(float* lhs, float* rhs, float* dst,
                    int lhs_columns, int rhs_columns){
    float *src_ptr2_1 = rhs + 0 * lhs_columns;
    float *src_ptr2_2 = rhs + 1 * lhs_columns;
    float *src_ptr2_3 = rhs + 2 * lhs_columns;
    float *src_ptr2_4 = rhs + 3 * lhs_columns;

    int i;

    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        vector_matrix_multiplication_accumulation_float32(
            lhs, src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
            &dst[i], &dst[i + 1], &dst[i + 2], &dst[i + 3],
            lhs_columns);
        src_ptr2_1 += 4 * lhs_columns;
        src_ptr2_2 += 4 * lhs_columns;
        src_ptr2_3 += 4 * lhs_columns;
        src_ptr2_4 += 4 * lhs_columns;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        src_ptr2_2 = src_ptr2_1;
        src_ptr2_3 = src_ptr2_1;
        src_ptr2_4 = src_ptr2_1;
        vector_matrix_multiplication_accumulation_float32(
            lhs, src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
            &dst[i], &dst[i], &dst[i], &dst[i],
            lhs_columns);
    }
    else if (i == 2){
        src_ptr2_3 = src_ptr2_2;
        src_ptr2_4 = src_ptr2_2;
        vector_matrix_multiplication_accumulation_float32(
            lhs, src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
            &dst[i], &dst[i + 1], &dst[i + 1], &dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        src_ptr2_4 = src_ptr2_3;
        vector_matrix_multiplication_accumulation_float32(
            lhs, src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
            &dst[i], &dst[i + 1], &dst[i + 2], &dst[i + 2],
            lhs_columns);
    }
    else if (i == 4){
        vector_matrix_multiplication_accumulation_float32(
            lhs, src_ptr2_1, src_ptr2_2, src_ptr2_3, src_ptr2_4,
            &dst[i], &dst[i + 1], &dst[i + 2], &dst[i + 3],
            lhs_columns);
    }
}


void Mul::vector_matrix_multiplication_accumulation_int8(
    int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
    int8_t* src_ptr2_1, int8_t* src_ptr2_2, int8_t* src_ptr2_3, int8_t* src_ptr2_4, 
    Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2, Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4, 
    int8_t *differ_from_max_1_ref,
    float& dst_1, float& dst_2, float& dst_3, float& dst_4,
    int size){
    int8_t max_exponent_1 = -128;
    int8_t max_exponent_2 = -128;
    int8_t max_exponent_3 = -128;
    int8_t max_exponent_4 = -128;

    Sign_DataType *src_ptr2_s_1 = src_ptr2_s_ref_1;
    Sign_DataType *src_ptr2_s_2 = src_ptr2_s_ref_2;
    Sign_DataType *src_ptr2_s_3 = src_ptr2_s_ref_3;
    Sign_DataType *src_ptr2_s_4 = src_ptr2_s_ref_4;

    int8_t *differ_from_max_1 = differ_from_max_1_ref;

    int i;

    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        "1:\n"

        "ld1 {v0.16b}, [%[src_ptr1_exp]], #16\n"
        "ld1 {v1.16b}, [%[src_ptr2_1]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr2_2]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr2_3]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr2_4]], #16\n"

        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        "st1 {v1.D}[0], [%[differ_from_max_1]], #8\n"
        "st1 {v2.D}[0], [%[differ_from_max_1]], #8\n"
        "st1 {v3.D}[0], [%[differ_from_max_1]], #8\n"
        "st1 {v4.D}[0], [%[differ_from_max_1]], #8\n"

        "st1 {v1.D}[1], [%[differ_from_max_1]], #8\n"
        "st1 {v2.D}[1], [%[differ_from_max_1]], #8\n"
        "st1 {v3.D}[1], [%[differ_from_max_1]], #8\n"
        "st1 {v4.D}[1], [%[differ_from_max_1]], #8\n"

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"

        "smaxv B5, v1.16b\n"
        "smaxv B6, v2.16b\n"
        "smaxv B7, v3.16b\n"
        "smaxv B8, v4.16b\n"

        "mov w1, v5.s[0]\n"
        "mov w2, v6.s[0]\n"
        "mov w3, v7.s[0]\n"
        "mov w4, v8.s[0]\n"

        "tst w1, %w[max_exponent_1]\n"
        "tst w2, %w[max_exponent_2]\n"
        "tst w3, %w[max_exponent_3]\n"
        "tst w4, %w[max_exponent_4]\n"
        
        "csel %w[max_exponent_1], w1, %w[max_exponent_1], gt\n"
        "csel %w[max_exponent_2], w2, %w[max_exponent_2], gt\n"
        "csel %w[max_exponent_3], w3, %w[max_exponent_3], gt\n"
        "csel %w[max_exponent_4], w4, %w[max_exponent_4], gt\n"

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        LOAD_ONE_DATA(0)
        LOAD_ONE_DATA(1)
        LOAD_ONE_DATA(2)
        LOAD_ONE_DATA(3)
        LOAD_ONE_DATA(4)
        LOAD_ONE_DATA(5)
        LOAD_ONE_DATA(6)
        LOAD_ONE_DATA(7)
        LOAD_ONE_DATA(8)
        LOAD_ONE_DATA(9)
        LOAD_ONE_DATA(10)
        LOAD_ONE_DATA(11)
        LOAD_ONE_DATA(12)
        LOAD_ONE_DATA(13)
        LOAD_ONE_DATA(14)
        LOAD_ONE_DATA(15)

        "5:\n"
        
        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        STORE_ONE_DATA(0)
        STORE_ONE_DATA(1)
        STORE_ONE_DATA(2)
        STORE_ONE_DATA(3)
        STORE_ONE_DATA(4)
        STORE_ONE_DATA(5)
        STORE_ONE_DATA(6)
        STORE_ONE_DATA(7)
        STORE_ONE_DATA(8)
        STORE_ONE_DATA(9)
        STORE_ONE_DATA(10)
        STORE_ONE_DATA(11)
        STORE_ONE_DATA(12)
        STORE_ONE_DATA(13)
        STORE_ONE_DATA(14)
        STORE_ONE_DATA(15)

        "6:\n"

        "dup v0.4s, wzr\n"
        "smaxv B0, v1.16b\n"
        "mov w1, v0.s[0]\n"
        "tst w1, %w[max_exponent_1]\n"
        "csel %w[max_exponent_1], w1, %w[max_exponent_1], gt\n"

        "7:\n"

        "mov %[differ_from_max_1], %[differ_from_max_1_ref]\n"

        "mov %w[i], wzr\n"

        "cmp %w[size], #8\n"
        "blt 13f\n"

        "add %w[i], %w[i], #8\n"

        "add %w[max_exponent_1], %w[max_exponent_1], #1\n"
        "add %w[max_exponent_2], %w[max_exponent_2], #1\n"
        "add %w[max_exponent_3], %w[max_exponent_3], #1\n"
        "add %w[max_exponent_4], %w[max_exponent_4], #1\n"

        "dup v9.4s, %w[max_exponent_1]\n"
        "dup v10.4s, %w[max_exponent_2]\n"
        "dup v11.4s, %w[max_exponent_3]\n"
        "dup v12.4s, %w[max_exponent_4]\n"

        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"

        "11:\n"
        
        // Load 8 data from pointers
        "ld1 {v1.8b}, [%[differ_from_max_1]], #8\n"
        "ld1 {v2.8b}, [%[differ_from_max_1]], #8\n"
        "ld1 {v3.8b}, [%[differ_from_max_1]], #8\n"
        "ld1 {v4.8b}, [%[differ_from_max_1]], #8\n"
        
        // Extend each 1-byte data to 2-bytes, and then 4-bytes and places them to another vector
        // Beacause we have imported
        "sxtl v5.8h, v1.8b\n"
        "sxtl v6.8h, v2.8b\n"
        "sxtl v7.8h, v3.8b\n"
        "sxtl v8.8h, v4.8b\n"

        "sxtl v1.4s, v5.4h\n"
        "sxtl v2.4s, v6.4h\n"
        "sxtl v3.4s, v7.4h\n"
        "sxtl v4.4s, v8.4h\n"

        "sxtl2 v5.4s, v5.8h\n"
        "sxtl2 v6.4s, v6.8h\n"
        "sxtl2 v7.4s, v7.8h\n"
        "sxtl2 v8.4s, v8.8h\n"

        "12:\n"

        "sub v1.4s, v1.4s, v9.4s\n"
        "sub v2.4s, v2.4s, v10.4s\n"
        "sub v3.4s, v3.4s, v11.4s\n"
        "sub v4.4s, v4.4s, v12.4s\n"

        "sub v5.4s, v5.4s, v9.4s\n"
        "sub v6.4s, v6.4s, v10.4s\n"
        "sub v7.4s, v7.4s, v11.4s\n"
        "sub v8.4s, v8.4s, v12.4s\n"

        "ld1 {v17.4s}, [%[src_ptr1_mantise]], #16\n"
        "ld1 {v18.4s}, [%[src_ptr1_mantise]], #16\n"

#ifndef USE_32BIT_SIGN
        "ld1 {v19.8b}, [%[src_ptr2_s_1]], #8\n"
        "ld1 {v20.8b}, [%[src_ptr2_s_2]], #8\n"
        "ld1 {v21.8b}, [%[src_ptr2_s_3]], #8\n"
        "ld1 {v22.8b}, [%[src_ptr2_s_4]], #8\n"
        
        "sxtl v23.8h, v19.8b\n"
        "sxtl v24.8h, v20.8b\n"
        "sxtl v25.8h, v21.8b\n"
        "sxtl v26.8h, v22.8b\n"

        "sxtl v19.4s, v23.4h\n"
        "sxtl v20.4s, v24.4h\n"
        "sxtl v21.4s, v25.4h\n"
        "sxtl v22.4s, v26.4h\n"

        "sxtl2 v23.4s, v23.8h\n"
        "sxtl2 v24.4s, v24.8h\n"
        "sxtl2 v25.4s, v25.8h\n"
        "sxtl2 v26.4s, v26.8h\n"
#else
        "ld1 {v19.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v20.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v21.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v22.4s}, [%[src_ptr2_s_4]], #16\n"

        "ld1 {v23.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v24.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v25.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v26.4s}, [%[src_ptr2_s_4]], #16\n"
#endif

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v19.4s\n"
        "mul v2.4s, v2.4s, v20.4s\n"
        "mul v3.4s, v3.4s, v21.4s\n"
        "mul v4.4s, v4.4s, v22.4s\n"

        "mul v5.4s, v5.4s, v23.4s\n"
        "mul v6.4s, v6.4s, v24.4s\n"
        "mul v7.4s, v7.4s, v25.4s\n"
        "mul v8.4s, v8.4s, v26.4s\n"
#else
        "eor v1.16b, v1.16b, v19.16b\n"
        "eor v2.16b, v2.16b, v20.16b\n"
        "eor v3.16b, v3.16b, v21.16b\n"
        "eor v4.16b, v4.16b, v22.16b\n"

        "eor v5.16b, v5.16b, v23.16b\n"
        "eor v6.16b, v6.16b, v24.16b\n"
        "eor v7.16b, v7.16b, v25.16b\n"
        "eor v8.16b, v8.16b, v26.16b\n"
#endif

        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"

        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"
        
        "add %w[i], %w[i], #8\n"
        "cmp %w[i], %w[size]\n"
        "ble 11b\n"
        "sub %w[i], %w[i], #8\n"
        "sub %w[i], %w[size], %w[i]\n"

        "13:\n"
        
        "cmp %w[i], #0\n"
        "beq 17f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"

        LOAD_ONE_DATA_32_1(0)
        LOAD_ONE_DATA_32_1(1)
        LOAD_ONE_DATA_32_1(2)
        LOAD_ONE_DATA_32_1(3)
        LOAD_ONE_DATA_32_2(4, 0)
        LOAD_ONE_DATA_32_2(5, 1)
        LOAD_ONE_DATA_32_2(6, 2)
        LOAD_ONE_DATA_32_2(7, 3)

        "15:\n"

#ifndef USE_32BIT_SIGN
        "sxtl v23.8h, v19.8b\n"
        "sxtl v24.8h, v20.8b\n"
        "sxtl v25.8h, v21.8b\n"
        "sxtl v26.8h, v22.8b\n"

        "sxtl v19.4s, v23.4h\n"
        "sxtl v20.4s, v24.4h\n"
        "sxtl v21.4s, v25.4h\n"
        "sxtl v22.4s, v26.4h\n"

        "sxtl2 v23.4s, v23.8h\n"
        "sxtl2 v24.4s, v24.8h\n"
        "sxtl2 v25.4s, v25.8h\n"
        "sxtl2 v26.4s, v26.8h\n"
#endif

        "sub v1.4s, v1.4s, v9.4s\n"
        "sub v2.4s, v2.4s, v10.4s\n"
        "sub v3.4s, v3.4s, v11.4s\n"
        "sub v4.4s, v4.4s, v12.4s\n"

        "sub v5.4s, v5.4s, v9.4s\n"
        "sub v6.4s, v6.4s, v10.4s\n"
        "sub v7.4s, v7.4s, v11.4s\n"
        "sub v8.4s, v8.4s, v12.4s\n"

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v19.4s\n"
        "mul v2.4s, v2.4s, v20.4s\n"
        "mul v3.4s, v3.4s, v21.4s\n"
        "mul v4.4s, v4.4s, v22.4s\n"

        "mul v5.4s, v5.4s, v23.4s\n"
        "mul v6.4s, v6.4s, v24.4s\n"
        "mul v7.4s, v7.4s, v25.4s\n"
        "mul v8.4s, v8.4s, v26.4s\n"
#else
        "eor v1.16b, v1.16b, v19.16b\n"
        "eor v2.16b, v2.16b, v20.16b\n"
        "eor v3.16b, v3.16b, v21.16b\n"
        "eor v4.16b, v4.16b, v22.16b\n"

        "eor v5.16b, v5.16b, v23.16b\n"
        "eor v6.16b, v6.16b, v24.16b\n"
        "eor v7.16b, v7.16b, v25.16b\n"
        "eor v8.16b, v8.16b, v26.16b\n"
#endif

        SET_ZERO_IF_OUT_2(7, 3)
        SET_ZERO_IF_OUT_2(6, 2)
        SET_ZERO_IF_OUT_2(5, 1)
        SET_ZERO_IF_OUT_2(4, 0)
        SET_ZERO_IF_OUT_1(3)
        SET_ZERO_IF_OUT_1(2)
        SET_ZERO_IF_OUT_1(1)
        SET_ZERO_IF_OUT_1(0)

        "110:\n"

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"

        "111:\n"

        "addv s13, v13.4s\n"
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "addv s16, v16.4s\n"

        "mov v13.s[0], v13.s[0]\n"
        "mov v13.s[1], v14.s[0]\n"
        "mov v13.s[2], v15.s[0]\n"
        "mov v13.s[3], v16.s[0]\n"

        "mov v14.s[0], %w[max_exponent_1]\n"
        "mov v14.s[1], %w[max_exponent_2]\n"
        "mov v14.s[2], %w[max_exponent_3]\n"
        "mov v14.s[3], %w[max_exponent_4]\n"

        "mov w1, #0x007FFFFF\n" // maximum mantisa value 
        "mov w2, #0xFF800001\n" // minimum mantisa value 
        "mov w3, #0x000000FF\n" // exponent limiter
        "mov w4, #0x807FFFFF\n" // float exponent mask

        // Making sure that mantisa fits in its place
        "dup v15.4s, w1\n"
        "smin v13.4s, v13.4s, v15.4s\n"
        "dup v16.4s, w2\n"
        "smax v13.4s, v13.4s, v16.4s\n"
        
        // Making sure that exponent fits in its place
        "dup v15.4s, w3\n"
        "and v14.16b, v14.16b, v15.16b\n"
        "sqshl v14.4s, v14.4s, #23\n"

        // Clearing the place of exponent in mantisa
        "dup v16.4s, w4\n"
        "and v13.16b, v13.16b, v16.16b\n"

        // Placing exponent in the mantisa to create float number
        "eor v13.16b, v13.16b, v14.16b\n"

        "mov %w[dst_1], v13.s[0]\n"
        "mov %w[dst_2], v13.s[1]\n"
        "mov %w[dst_3], v13.s[2]\n"
        "mov %w[dst_4], v13.s[3]\n"

        "17:\n"

        : [ differ_from_max_1 ] "+r"(differ_from_max_1),
          [ max_exponent_1 ] "+r"(max_exponent_1), [ max_exponent_2 ] "+r"(max_exponent_2), 
          [ max_exponent_3 ] "+r"(max_exponent_3), [ max_exponent_4 ] "+r"(max_exponent_4),
          [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i)
        : [ src_ptr2_1 ] "r"(src_ptr2_1), [ src_ptr2_2 ] "r"(src_ptr2_2),
          [ src_ptr2_3 ] "r"(src_ptr2_3), [ src_ptr2_4 ] "r"(src_ptr2_4),
          [ src_ptr2_s_1 ] "r"(src_ptr2_s_1), [ src_ptr2_s_2 ] "r"(src_ptr2_s_2),
          [ src_ptr2_s_3 ] "r"(src_ptr2_s_3), [ src_ptr2_s_4 ] "r"(src_ptr2_s_4),
          [ src_ptr1_exp ] "r"(src_ptr1_exp), [ differ_from_max_1_ref ] "r"(differ_from_max_1_ref),
          [ src_ptr1_mantise ] "r"(src_ptr1_mantise), [ size ] "r"(size)
        : "w1", "w2", "w3", "w4",
          "v0", 
          "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16",
          "v17", "v18",
          "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26"
    );
}

void Mul::vector_matrix_multiplication_accumulation_int8_fused(
    int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
    int8_t* src_ptr2_1, int8_t* src_ptr2_2, int8_t* src_ptr2_3, int8_t* src_ptr2_4, 
    Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2, Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4, 
    float& dst_1, float& dst_2, float& dst_3, float& dst_4,
    int size){
    int8_t max_exponent_1 = -128;
    int8_t max_exponent_2 = -128;
    int8_t max_exponent_3 = -128;
    int8_t max_exponent_4 = -128;

    Sign_DataType *src_ptr2_s_1 = src_ptr2_s_ref_1;
    Sign_DataType *src_ptr2_s_2 = src_ptr2_s_ref_2;
    Sign_DataType *src_ptr2_s_3 = src_ptr2_s_ref_3;
    Sign_DataType *src_ptr2_s_4 = src_ptr2_s_ref_4;

    int i;

    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"

        "1:\n"

        "ld1 {v0.16b}, [%[src_ptr1_exp]], #16\n"
        "ld1 {v1.16b}, [%[src_ptr2_1]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr2_2]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr2_3]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr2_4]], #16\n"
#if DO_LOG_PREFETCHING
        PREFETCH_DATA("src_ptr1_mantise", 64)
        PREFETCH_DATA("src_ptr1_mantise", 80)

        PREFETCH_DATA("src_ptr1_exp", 64)
        PREFETCH_DATA("src_ptr2_1", 64)
        PREFETCH_DATA("src_ptr2_2", 64)
        PREFETCH_DATA("src_ptr2_3", 64)
        PREFETCH_DATA("src_ptr2_4", 64)
#endif

        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        /////////////////////////////////////////////
        "mov v19.16b, v1.16b\n"
        "mov v20.16b, v2.16b\n"
        "mov v21.16b, v3.16b\n"
        "mov v22.16b, v4.16b\n"

        // Start processing first half in 16 data
        "11:\n"

#ifndef USE_32BIT_SIGN
        "ld1 {v23.8b}, [%[src_ptr2_s_1]], #8\n"
        "ld1 {v24.8b}, [%[src_ptr2_s_2]], #8\n"
        "ld1 {v25.8b}, [%[src_ptr2_s_3]], #8\n"
        "ld1 {v26.8b}, [%[src_ptr2_s_4]], #8\n"
        
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#else
        "ld1 {v23.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v24.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v25.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v26.4s}, [%[src_ptr2_s_4]], #16\n"

        "ld1 {v27.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v28.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v29.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v30.4s}, [%[src_ptr2_s_4]], #16\n"
#endif

        // Extend each 1-byte data to 2-bytes, and then 4-bytes and places them to another vector
        // Beacause we have imported
        "sxtl v5.8h, v19.8b\n"
        "sxtl v6.8h, v20.8b\n"
        "sxtl v7.8h, v21.8b\n"
        "sxtl v8.8h, v22.8b\n"

        "sxtl v1.4s, v5.4h\n"
        "sxtl v2.4s, v6.4h\n"
        "sxtl v3.4s, v7.4h\n"
        "sxtl v4.4s, v8.4h\n"

        "sxtl2 v5.4s, v5.8h\n"
        "sxtl2 v6.4s, v6.8h\n"
        "sxtl2 v7.4s, v7.8h\n"
        "sxtl2 v8.4s, v8.8h\n"

        "ld1 {v17.4s}, [%[src_ptr1_mantise]], #16\n"
        "ld1 {v18.4s}, [%[src_ptr1_mantise]], #16\n"

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v23.4s\n"
        "mul v2.4s, v2.4s, v24.4s\n"
        "mul v3.4s, v3.4s, v25.4s\n"
        "mul v4.4s, v4.4s, v26.4s\n"

        "mul v5.4s, v5.4s, v27.4s\n"
        "mul v6.4s, v6.4s, v28.4s\n"
        "mul v7.4s, v7.4s, v29.4s\n"
        "mul v8.4s, v8.4s, v30.4s\n"
#else

        "eor v1.16b, v1.16b, v23.16b\n"
        "eor v2.16b, v2.16b, v24.16b\n"
        "eor v3.16b, v3.16b, v25.16b\n"
        "eor v4.16b, v4.16b, v26.16b\n"

        "eor v5.16b, v5.16b, v27.16b\n"
        "eor v6.16b, v6.16b, v28.16b\n"
        "eor v7.16b, v7.16b, v29.16b\n"
        "eor v8.16b, v8.16b, v30.16b\n"
#endif

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"

#ifndef USE_32BIT_SIGN
        "ld1 {v23.8b}, [%[src_ptr2_s_1]], #8\n"
        "ld1 {v24.8b}, [%[src_ptr2_s_2]], #8\n"
        "ld1 {v25.8b}, [%[src_ptr2_s_3]], #8\n"
        "ld1 {v26.8b}, [%[src_ptr2_s_4]], #8\n"
        
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#else
        "ld1 {v23.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v24.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v25.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v26.4s}, [%[src_ptr2_s_4]], #16\n"

        "ld1 {v27.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v28.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v29.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v30.4s}, [%[src_ptr2_s_4]], #16\n"
#endif
        
        // Second half of 16 data
        "sxtl2 v5.8h, v19.16b\n"
        "sxtl2 v6.8h, v20.16b\n"
        "sxtl2 v7.8h, v21.16b\n"
        "sxtl2 v8.8h, v22.16b\n"

        "sxtl v1.4s, v5.4h\n"
        "sxtl v2.4s, v6.4h\n"
        "sxtl v3.4s, v7.4h\n"
        "sxtl v4.4s, v8.4h\n"

        "sxtl2 v5.4s, v5.8h\n"
        "sxtl2 v6.4s, v6.8h\n"
        "sxtl2 v7.4s, v7.8h\n"
        "sxtl2 v8.4s, v8.8h\n"

        "ld1 {v17.4s}, [%[src_ptr1_mantise]], #16\n"
        "ld1 {v18.4s}, [%[src_ptr1_mantise]], #16\n"

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v23.4s\n"
        "mul v2.4s, v2.4s, v24.4s\n"
        "mul v3.4s, v3.4s, v25.4s\n"
        "mul v4.4s, v4.4s, v26.4s\n"

        "mul v5.4s, v5.4s, v27.4s\n"
        "mul v6.4s, v6.4s, v28.4s\n"
        "mul v7.4s, v7.4s, v29.4s\n"
        "mul v8.4s, v8.4s, v30.4s\n"
#else

        "eor v1.16b, v1.16b, v23.16b\n"
        "eor v2.16b, v2.16b, v24.16b\n"
        "eor v3.16b, v3.16b, v25.16b\n"
        "eor v4.16b, v4.16b, v26.16b\n"

        "eor v5.16b, v5.16b, v27.16b\n"
        "eor v6.16b, v6.16b, v28.16b\n"
        "eor v7.16b, v7.16b, v29.16b\n"
        "eor v8.16b, v8.16b, v30.16b\n"
#endif

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"
        /////////////////////////////////////////////

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"

        "smaxv B5, v19.16b\n"
        "smaxv B6, v20.16b\n"
        "smaxv B7, v21.16b\n"
        "smaxv B8, v22.16b\n"

        "mov w1, v5.s[0]\n"
        "mov w2, v6.s[0]\n"
        "mov w3, v7.s[0]\n"
        "mov w4, v8.s[0]\n"

        "tst w1, %w[max_exponent_1]\n"
        "tst w2, %w[max_exponent_2]\n"
        "tst w3, %w[max_exponent_3]\n"
        "tst w4, %w[max_exponent_4]\n"
        
        "csel %w[max_exponent_1], w1, %w[max_exponent_1], gt\n"
        "csel %w[max_exponent_2], w2, %w[max_exponent_2], gt\n"
        "csel %w[max_exponent_3], w3, %w[max_exponent_3], gt\n"
        "csel %w[max_exponent_4], w4, %w[max_exponent_4], gt\n"

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        // "mov w0, #0x80808080"

        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        LOAD_ONE_DATA(0)
        LOAD_ONE_DATA(1)
        LOAD_ONE_DATA(2)
        LOAD_ONE_DATA(3)
        LOAD_ONE_DATA(4)
        LOAD_ONE_DATA(5)
        LOAD_ONE_DATA(6)
        LOAD_ONE_DATA(7)
        LOAD_ONE_DATA(8)
        LOAD_ONE_DATA(9)
        LOAD_ONE_DATA(10)
        LOAD_ONE_DATA(11)
        LOAD_ONE_DATA(12)
        LOAD_ONE_DATA(13)
        LOAD_ONE_DATA(14)
        LOAD_ONE_DATA(15)

        "5:\n"
        
        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        /////////////////////////////////////////////
        "mov v19.16b, v1.16b\n"
        "mov v20.16b, v2.16b\n"
        "mov v21.16b, v3.16b\n"
        "mov v22.16b, v4.16b\n"

        // Expanding First Half
        "sxtl v5.8h, v19.8b\n"
        "sxtl v6.8h, v20.8b\n"
        "sxtl v7.8h, v21.8b\n"
        "sxtl v8.8h, v22.8b\n"

        "sxtl v1.4s, v5.4h\n"
        "sxtl v2.4s, v6.4h\n"
        "sxtl v3.4s, v7.4h\n"
        "sxtl v4.4s, v8.4h\n"

        "sxtl2 v5.4s, v5.8h\n"
        "sxtl2 v6.4s, v6.8h\n"
        "sxtl2 v7.4s, v7.8h\n"
        "sxtl2 v8.4s, v8.8h\n"

        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"

        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"

        LOAD_ONE_MANTISE_32_1(0, 0)
        LOAD_ONE_MANTISE_32_1(1, 1)
        LOAD_ONE_MANTISE_32_1(2, 2)
        LOAD_ONE_MANTISE_32_1(3, 3)
        LOAD_ONE_MANTISE_32_2(4, 0)
        LOAD_ONE_MANTISE_32_2(5, 1)
        LOAD_ONE_MANTISE_32_2(6, 2)
        LOAD_ONE_MANTISE_32_2(7, 3)

        "15:\n"

#ifndef USE_32BIT_SIGN
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#endif

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v23.4s\n"
        "mul v2.4s, v2.4s, v24.4s\n"
        "mul v3.4s, v3.4s, v25.4s\n"
        "mul v4.4s, v4.4s, v26.4s\n"

        "mul v5.4s, v5.4s, v27.4s\n"
        "mul v6.4s, v6.4s, v28.4s\n"
        "mul v7.4s, v7.4s, v29.4s\n"
        "mul v8.4s, v8.4s, v30.4s\n"
#else

        "eor v1.16b, v1.16b, v23.16b\n"
        "eor v2.16b, v2.16b, v24.16b\n"
        "eor v3.16b, v3.16b, v25.16b\n"
        "eor v4.16b, v4.16b, v26.16b\n"

        "eor v5.16b, v5.16b, v27.16b\n"
        "eor v6.16b, v6.16b, v28.16b\n"
        "eor v7.16b, v7.16b, v29.16b\n"
        "eor v8.16b, v8.16b, v30.16b\n"
#endif

        SET_ZERO_IF_OUT_2_2(7, 3)
        SET_ZERO_IF_OUT_2_2(6, 2)
        SET_ZERO_IF_OUT_2_2(5, 1)
        SET_ZERO_IF_OUT_2_2(4, 0)
        SET_ZERO_IF_OUT_2_1(3, 3)
        SET_ZERO_IF_OUT_2_1(2, 2)
        SET_ZERO_IF_OUT_2_1(1, 1)
        SET_ZERO_IF_OUT_2_1(0, 0)

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"

        // Expanding Second Half
        "sxtl2 v5.8h, v19.16b\n"
        "sxtl2 v6.8h, v20.16b\n"
        "sxtl2 v7.8h, v21.16b\n"
        "sxtl2 v8.8h, v22.16b\n"

        "sxtl v1.4s, v5.4h\n"
        "sxtl v2.4s, v6.4h\n"
        "sxtl v3.4s, v7.4h\n"
        "sxtl v4.4s, v8.4h\n"

        "sxtl2 v5.4s, v5.8h\n"
        "sxtl2 v6.4s, v6.8h\n"
        "sxtl2 v7.4s, v7.8h\n"
        "sxtl2 v8.4s, v8.8h\n"

        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"

        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"

        LOAD_ONE_MANTISE_32_1(8 , 0)
        LOAD_ONE_MANTISE_32_1(9 , 1)
        LOAD_ONE_MANTISE_32_1(10, 2)
        LOAD_ONE_MANTISE_32_1(11, 3)
        LOAD_ONE_MANTISE_32_2(12, 0)
        LOAD_ONE_MANTISE_32_2(13, 1)
        LOAD_ONE_MANTISE_32_2(14, 2)
        LOAD_ONE_MANTISE_32_2(15, 3)

        "15:\n"

#ifndef USE_32BIT_SIGN
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#endif

        "sshl v1.4s, v17.4s, v1.4s\n"
        "sshl v2.4s, v17.4s, v2.4s\n"
        "sshl v3.4s, v17.4s, v3.4s\n"
        "sshl v4.4s, v17.4s, v4.4s\n"

        "sshl v5.4s, v18.4s, v5.4s\n"
        "sshl v6.4s, v18.4s, v6.4s\n"
        "sshl v7.4s, v18.4s, v7.4s\n"
        "sshl v8.4s, v18.4s, v8.4s\n"

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v23.4s\n"
        "mul v2.4s, v2.4s, v24.4s\n"
        "mul v3.4s, v3.4s, v25.4s\n"
        "mul v4.4s, v4.4s, v26.4s\n"

        "mul v5.4s, v5.4s, v27.4s\n"
        "mul v6.4s, v6.4s, v28.4s\n"
        "mul v7.4s, v7.4s, v29.4s\n"
        "mul v8.4s, v8.4s, v30.4s\n"
#else

        "eor v1.16b, v1.16b, v23.16b\n"
        "eor v2.16b, v2.16b, v24.16b\n"
        "eor v3.16b, v3.16b, v25.16b\n"
        "eor v4.16b, v4.16b, v26.16b\n"

        "eor v5.16b, v5.16b, v27.16b\n"
        "eor v6.16b, v6.16b, v28.16b\n"
        "eor v7.16b, v7.16b, v29.16b\n"
        "eor v8.16b, v8.16b, v30.16b\n"
#endif

        SET_ZERO_IF_OUT_2_2(15, 3)
        SET_ZERO_IF_OUT_2_2(14, 2)
        SET_ZERO_IF_OUT_2_2(13, 1)
        SET_ZERO_IF_OUT_2_2(12, 0)
        SET_ZERO_IF_OUT_2_1(11, 3)
        SET_ZERO_IF_OUT_2_1(10, 2)
        SET_ZERO_IF_OUT_2_1(9 , 1)
        SET_ZERO_IF_OUT_2_1(8,  0)

        "110:\n"

        "addp v13.4s, v5.4s, v1.4s\n"
        "addp v14.4s, v6.4s, v2.4s\n"
        "addp v15.4s, v7.4s, v3.4s\n"
        "addp v16.4s, v8.4s, v4.4s\n"

        /////////////////////////////////////////////

        "6:\n"

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"

        "smaxv B5, v19.16b\n"
        "smaxv B6, v20.16b\n"
        "smaxv B7, v21.16b\n"
        "smaxv B8, v22.16b\n"

        "mov w1, v5.s[0]\n"
        "mov w2, v6.s[0]\n"
        "mov w3, v7.s[0]\n"
        "mov w4, v8.s[0]\n"

        "tst w1, %w[max_exponent_1]\n"
        "tst w2, %w[max_exponent_2]\n"
        "tst w3, %w[max_exponent_3]\n"
        "tst w4, %w[max_exponent_4]\n"
        
        "csel %w[max_exponent_1], w1, %w[max_exponent_1], gt\n"
        "csel %w[max_exponent_2], w2, %w[max_exponent_2], gt\n"
        "csel %w[max_exponent_3], w3, %w[max_exponent_3], gt\n"
        "csel %w[max_exponent_4], w4, %w[max_exponent_4], gt\n"

        "7:\n"

        /********************************************************************************/

        "111:\n"

        "addv s13, v13.4s\n"
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "addv s16, v16.4s\n"

        "mov v13.s[0], v13.s[0]\n"
        "mov v13.s[1], v14.s[0]\n"
        "mov v13.s[2], v15.s[0]\n"
        "mov v13.s[3], v16.s[0]\n"

        "mov v14.s[0], %w[max_exponent_1]\n"
        "mov v14.s[1], %w[max_exponent_2]\n"
        "mov v14.s[2], %w[max_exponent_3]\n"
        "mov v14.s[3], %w[max_exponent_4]\n"

        "neg v15.4s, v14.4s\n"

        "sshl v13.4s, v13.4s, v15.4s\n"

        "mov w1, #0x007FFFFF\n" // maximum mantisa value 
        "mov w2, #0xFF800001\n" // minimum mantisa value 
        "mov w3, #0x000000FF\n" // exponent limiter
        "mov w4, #0x807FFFFF\n" // float exponent mask

        // Making sure that mantisa fits in its place
        "dup v15.4s, w1\n"
        "smin v13.4s, v13.4s, v15.4s\n"
        "dup v16.4s, w2\n"
        "smax v13.4s, v13.4s, v16.4s\n"
        
        // Making sure that exponent fits in its place
        "dup v15.4s, w3\n"
        "and v14.16b, v14.16b, v15.16b\n"
        "sqshl v14.4s, v14.4s, #23\n"

        // Clearing the place of exponent in mantisa
        "dup v16.4s, w4\n"
        "and v13.16b, v13.16b, v16.16b\n"

        // Placing exponent in the mantisa to create float number
        "eor v13.16b, v13.16b, v14.16b\n"

        "mov %w[dst_1], v13.s[0]\n"
        "mov %w[dst_2], v13.s[1]\n"
        "mov %w[dst_3], v13.s[2]\n"
        "mov %w[dst_4], v13.s[3]\n"

        "17:\n"

        : [ max_exponent_1 ] "+r"(max_exponent_1), [ max_exponent_2 ] "+r"(max_exponent_2), 
          [ max_exponent_3 ] "+r"(max_exponent_3), [ max_exponent_4 ] "+r"(max_exponent_4),
          [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i)
        : [ src_ptr2_1 ] "r"(src_ptr2_1), [ src_ptr2_2 ] "r"(src_ptr2_2),
          [ src_ptr2_3 ] "r"(src_ptr2_3), [ src_ptr2_4 ] "r"(src_ptr2_4),
          [ src_ptr2_s_1 ] "r"(src_ptr2_s_1), [ src_ptr2_s_2 ] "r"(src_ptr2_s_2),
          [ src_ptr2_s_3 ] "r"(src_ptr2_s_3), [ src_ptr2_s_4 ] "r"(src_ptr2_s_4),
          [ src_ptr1_exp ] "r"(src_ptr1_exp),
          [ src_ptr1_mantise ] "r"(src_ptr1_mantise), [ size ] "r"(size)
        : "w1", "w2", "w3", "w4",
          "v0", 
          "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16",
          "v17", "v18",
          "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30"
    );
}

void Mul::vector_matrix_multiplication_accumulation_int8_hybrid_fused(
    int8_t* src_ptr1_exp, int32_t* src_ptr1_mantise,
    int8_t* src_ptr2_1, int8_t* src_ptr2_2,
    int8_t* src_ptr2_3, int8_t* src_ptr2_4, 
    Sign_DataType* src_ptr2_s_ref_1, Sign_DataType* src_ptr2_s_ref_2,
    Sign_DataType* src_ptr2_s_ref_3, Sign_DataType* src_ptr2_s_ref_4, 
    float& dst_1, float& dst_2, float& dst_3, float& dst_4,
    int size){
    int max = 0;

    auto *src_ptr2_s_1 = src_ptr2_s_ref_1;
    auto *src_ptr2_s_2 = src_ptr2_s_ref_2;
    auto *src_ptr2_s_3 = src_ptr2_s_ref_3;
    auto *src_ptr2_s_4 = src_ptr2_s_ref_4;

    int i;

    asm volatile(
        "mov %w[i], wzr\n"

        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        "1:\n"

        "ld1 {v0.16b}, [%[src_ptr1_exp]], #16\n"

        "ld1 {v1.16b}, [%[src_ptr2_1]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr2_2]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr2_3]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr2_4]], #16\n"

#if DO_LOG_PREFETCHING
        PREFETCH_DATA("src_ptr1_mantise", 64)
        PREFETCH_DATA("src_ptr1_mantise", 80)

        PREFETCH_DATA("src_ptr1_exp", 64)
        PREFETCH_DATA("src_ptr2_1", 64)
        PREFETCH_DATA("src_ptr2_2", 64)
        PREFETCH_DATA("src_ptr2_3", 64)
        PREFETCH_DATA("src_ptr2_4", 64)
#endif

        "ld1 {v17.4s}, [%[src_ptr1_mantise]], #16\n"
        "ld1 {v18.4s}, [%[src_ptr1_mantise]], #16\n"

        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        /////////////////////////////////////////////
        "mov v19.16b, v1.16b\n"
        "mov v20.16b, v2.16b\n"
        "mov v21.16b, v3.16b\n"
        "mov v22.16b, v4.16b\n"

        // Start processing first half in 16 data
        "11:\n"

#ifndef USE_32BIT_SIGN
        "ld1 {v23.8b}, [%[src_ptr2_s_1]], #8\n"
        "ld1 {v24.8b}, [%[src_ptr2_s_2]], #8\n"
        "ld1 {v25.8b}, [%[src_ptr2_s_3]], #8\n"
        "ld1 {v26.8b}, [%[src_ptr2_s_4]], #8\n"
        
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#else
        "ld1 {v23.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v24.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v25.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v26.4s}, [%[src_ptr2_s_4]], #16\n"

        "ld1 {v27.4s}, [%[src_ptr2_s_1]], #16\n"
        "ld1 {v28.4s}, [%[src_ptr2_s_2]], #16\n"
        "ld1 {v29.4s}, [%[src_ptr2_s_3]], #16\n"
        "ld1 {v30.4s}, [%[src_ptr2_s_4]], #16\n"
#endif

#ifndef USE_32BIT_SIGN
        "mul v23.4s, v17.4s, v23.4s\n"
        "mul v24.4s, v17.4s, v24.4s\n"
        "mul v25.4s, v17.4s, v25.4s\n"
        "mul v26.4s, v17.4s, v26.4s\n"

        "mul v27.4s, v18.4s, v27.4s\n"
        "mul v28.4s, v18.4s, v28.4s\n"
        "mul v29.4s, v18.4s, v29.4s\n"
        "mul v30.4s, v18.4s, v30.4s\n"
#else
        "eor v23.16b, v17.16b, v23.16b\n"
        "eor v24.16b, v17.16b, v24.16b\n"
        "eor v25.16b, v17.16b, v25.16b\n"
        "eor v26.16b, v17.16b, v26.16b\n"

        "eor v27.16b, v18.16b, v27.16b\n"
        "eor v28.16b, v18.16b, v28.16b\n"
        "eor v29.16b, v18.16b, v29.16b\n"
        "eor v30.16b, v18.16b, v30.16b\n"
#endif

        // Extend each 1-byte data to 2-bytes, and then 4-bytes and places them to another vector
        // First Half
        "uxtl v5.8h, v19.8b\n"
        "uxtl v6.8h, v20.8b\n"
        "uxtl v7.8h, v21.8b\n"
        "uxtl v8.8h, v22.8b\n"

        "uxtl v1.4s, v5.4h\n"
        "uxtl v2.4s, v6.4h\n"
        "uxtl v3.4s, v7.4h\n"
        "uxtl v4.4s, v8.4h\n"

        "uxtl2 v5.4s, v5.8h\n"
        "uxtl2 v6.4s, v6.8h\n"
        "uxtl2 v7.4s, v7.8h\n"
        "uxtl2 v8.4s, v8.8h\n"

        "shl v1.4s, v1.4s, #23\n"
        "shl v2.4s, v2.4s, #23\n"
        "shl v3.4s, v3.4s, #23\n"
        "shl v4.4s, v4.4s, #23\n"

        "shl v5.4s, v5.4s, #23\n"
        "shl v6.4s, v6.4s, #23\n"
        "shl v7.4s, v7.4s, #23\n"
        "shl v8.4s, v8.4s, #23\n"

        "eor v23.16b, v1.16b, v23.16b\n"
        "eor v24.16b, v2.16b, v24.16b\n"
        "eor v25.16b, v3.16b, v25.16b\n"
        "eor v26.16b, v4.16b, v26.16b\n"

        "eor v27.16b, v5.16b, v27.16b\n"
        "eor v28.16b, v6.16b, v28.16b\n"
        "eor v29.16b, v7.16b, v29.16b\n"
        "eor v30.16b, v8.16b, v30.16b\n"

        "faddp v13.4s, v13.4s, v23.4s\n"
        "faddp v14.4s, v14.4s, v24.4s\n"
        "faddp v15.4s, v15.4s, v25.4s\n"
        "faddp v16.4s, v16.4s, v26.4s\n"

        "faddp v13.4s, v13.4s, v27.4s\n"
        "faddp v14.4s, v14.4s, v28.4s\n"
        "faddp v15.4s, v15.4s, v29.4s\n"
        "faddp v16.4s, v16.4s, v30.4s\n"

        // Extend each 1-byte data to 2-bytes, and then 4-bytes and places them to another vector
        // Second Half
        "uxtl2 v5.8h, v19.16b\n"
        "uxtl2 v6.8h, v20.16b\n"
        "uxtl2 v7.8h, v21.16b\n"
        "uxtl2 v8.8h, v22.16b\n"

        "uxtl v1.4s, v5.4h\n"
        "uxtl v2.4s, v6.4h\n"
        "uxtl v3.4s, v7.4h\n"
        "uxtl v4.4s, v8.4h\n"

        "uxtl2 v5.4s, v5.8h\n"
        "uxtl2 v6.4s, v6.8h\n"
        "uxtl2 v7.4s, v7.8h\n"
        "uxtl2 v8.4s, v8.8h\n"

        "shl v1.4s, v1.4s, #23\n"
        "shl v2.4s, v2.4s, #23\n"
        "shl v3.4s, v3.4s, #23\n"
        "shl v4.4s, v4.4s, #23\n"

        "shl v5.4s, v5.4s, #23\n"
        "shl v6.4s, v6.4s, #23\n"
        "shl v7.4s, v7.4s, #23\n"
        "shl v8.4s, v8.4s, #23\n"

        "eor v23.16b, v1.16b, v23.16b\n"
        "eor v24.16b, v2.16b, v24.16b\n"
        "eor v25.16b, v3.16b, v25.16b\n"
        "eor v26.16b, v4.16b, v26.16b\n"

        "eor v27.16b, v5.16b, v27.16b\n"
        "eor v28.16b, v6.16b, v28.16b\n"
        "eor v29.16b, v7.16b, v29.16b\n"
        "eor v30.16b, v8.16b, v30.16b\n"

        "faddp v13.4s, v13.4s, v23.4s\n"
        "faddp v14.4s, v14.4s, v24.4s\n"
        "faddp v15.4s, v15.4s, v25.4s\n"
        "faddp v16.4s, v16.4s, v26.4s\n"

        "faddp v13.4s, v13.4s, v27.4s\n"
        "faddp v14.4s, v14.4s, v28.4s\n"
        "faddp v15.4s, v15.4s, v29.4s\n"
        "faddp v16.4s, v16.4s, v30.4s\n"

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"
        
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"

        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"

        LOAD_ONE_DATA(0)
        LOAD_ONE_DATA(1)
        LOAD_ONE_DATA(2)
        LOAD_ONE_DATA(3)
        LOAD_ONE_DATA(4)
        LOAD_ONE_DATA(5)
        LOAD_ONE_DATA(6)
        LOAD_ONE_DATA(7)
        LOAD_ONE_DATA(8)
        LOAD_ONE_DATA(9)
        LOAD_ONE_DATA(10)
        LOAD_ONE_DATA(11)
        LOAD_ONE_DATA(12)
        LOAD_ONE_DATA(13)
        LOAD_ONE_DATA(14)
        LOAD_ONE_DATA(15)

        "5:\n"

        "add v1.16b, v1.16b, v0.16b\n"
        "add v2.16b, v2.16b, v0.16b\n"
        "add v3.16b, v3.16b, v0.16b\n"
        "add v4.16b, v4.16b, v0.16b\n"

        LOAD_ONE_MANTISE_32_1(0, 0)
        LOAD_ONE_MANTISE_32_1(1, 1)
        LOAD_ONE_MANTISE_32_1(2, 2)
        LOAD_ONE_MANTISE_32_1(3, 3)
        LOAD_ONE_MANTISE_32_2(4, 0)
        LOAD_ONE_MANTISE_32_2(5, 1)
        LOAD_ONE_MANTISE_32_2(6, 2)
        LOAD_ONE_MANTISE_32_2(7, 3)

        "15:\n"

#ifndef USE_32BIT_SIGN
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#endif

        /////////////////////////////////////////////
        "mov v19.16b, v1.16b\n"
        "mov v20.16b, v2.16b\n"
        "mov v21.16b, v3.16b\n"
        "mov v22.16b, v4.16b\n"

#ifndef USE_32BIT_SIGN
        "mul v23.4s, v17.4s, v23.4s\n"
        "mul v24.4s, v17.4s, v24.4s\n"
        "mul v25.4s, v17.4s, v25.4s\n"
        "mul v26.4s, v17.4s, v26.4s\n"

        "mul v27.4s, v18.4s, v27.4s\n"
        "mul v28.4s, v18.4s, v28.4s\n"
        "mul v29.4s, v18.4s, v29.4s\n"
        "mul v30.4s, v18.4s, v30.4s\n"
#else
        "eor v23.16b, v17.16b, v23.16b\n"
        "eor v24.16b, v17.16b, v24.16b\n"
        "eor v25.16b, v17.16b, v25.16b\n"
        "eor v26.16b, v17.16b, v26.16b\n"

        "eor v27.16b, v18.16b, v27.16b\n"
        "eor v28.16b, v18.16b, v28.16b\n"
        "eor v29.16b, v18.16b, v29.16b\n"
        "eor v30.16b, v18.16b, v30.16b\n"
#endif

        // Expanding First Half
        "uxtl v5.8h, v19.8b\n"
        "uxtl v6.8h, v20.8b\n"
        "uxtl v7.8h, v21.8b\n"
        "uxtl v8.8h, v22.8b\n"

        "uxtl v1.4s, v5.4h\n"
        "uxtl v2.4s, v6.4h\n"
        "uxtl v3.4s, v7.4h\n"
        "uxtl v4.4s, v8.4h\n"

        "uxtl2 v5.4s, v5.8h\n"
        "uxtl2 v6.4s, v6.8h\n"
        "uxtl2 v7.4s, v7.8h\n"
        "uxtl2 v8.4s, v8.8h\n"

        "shl v1.4s, v1.4s, #23\n"
        "shl v2.4s, v2.4s, #23\n"
        "shl v3.4s, v3.4s, #23\n"
        "shl v4.4s, v4.4s, #23\n"

        "shl v5.4s, v5.4s, #23\n"
        "shl v6.4s, v6.4s, #23\n"
        "shl v7.4s, v7.4s, #23\n"
        "shl v8.4s, v8.4s, #23\n"

        "eor v23.16b, v1.16b, v23.16b\n"
        "eor v24.16b, v2.16b, v24.16b\n"
        "eor v25.16b, v3.16b, v25.16b\n"
        "eor v26.16b, v4.16b, v26.16b\n"

        "eor v27.16b, v5.16b, v27.16b\n"
        "eor v28.16b, v6.16b, v28.16b\n"
        "eor v29.16b, v7.16b, v29.16b\n"
        "eor v30.16b, v8.16b, v30.16b\n"

        SET_ZERO_IF_OUT_2_2(7, 3)
        SET_ZERO_IF_OUT_2_2(6, 2)
        SET_ZERO_IF_OUT_2_2(5, 1)
        SET_ZERO_IF_OUT_2_2(4, 0)
        SET_ZERO_IF_OUT_2_1(3, 3)
        SET_ZERO_IF_OUT_2_1(2, 2)
        SET_ZERO_IF_OUT_2_1(1, 1)
        SET_ZERO_IF_OUT_2_1(0, 0)

        "faddp v13.4s, v13.4s, v23.4s\n"
        "faddp v14.4s, v14.4s, v24.4s\n"
        "faddp v15.4s, v15.4s, v25.4s\n"
        "faddp v16.4s, v16.4s, v26.4s\n"

        "faddp v13.4s, v13.4s, v27.4s\n"
        "faddp v14.4s, v14.4s, v28.4s\n"
        "faddp v15.4s, v15.4s, v29.4s\n"
        "faddp v16.4s, v16.4s, v30.4s\n"

        // Expanding Second Half
        "uxtl2 v5.8h, v19.16b\n"
        "uxtl2 v6.8h, v20.16b\n"
        "uxtl2 v7.8h, v21.16b\n"
        "uxtl2 v8.8h, v22.16b\n"

        "uxtl v1.4s, v5.4h\n"
        "uxtl v2.4s, v6.4h\n"
        "uxtl v3.4s, v7.4h\n"
        "uxtl v4.4s, v8.4h\n"

        "uxtl2 v5.4s, v5.8h\n"
        "uxtl2 v6.4s, v6.8h\n"
        "uxtl2 v7.4s, v7.8h\n"
        "uxtl2 v8.4s, v8.8h\n"

        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"

        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"

        LOAD_ONE_MANTISE_32_1(8 , 0)
        LOAD_ONE_MANTISE_32_1(9 , 1)
        LOAD_ONE_MANTISE_32_1(10, 2)
        LOAD_ONE_MANTISE_32_1(11, 3)
        LOAD_ONE_MANTISE_32_2(12, 0)
        LOAD_ONE_MANTISE_32_2(13, 1)
        LOAD_ONE_MANTISE_32_2(14, 2)
        LOAD_ONE_MANTISE_32_2(15, 3)

        "15:\n"

#ifndef USE_32BIT_SIGN
        "sxtl v27.8h, v23.8b\n"
        "sxtl v28.8h, v24.8b\n"
        "sxtl v29.8h, v25.8b\n"
        "sxtl v30.8h, v26.8b\n"

        "sxtl v23.4s, v27.4h\n"
        "sxtl v24.4s, v28.4h\n"
        "sxtl v25.4s, v29.4h\n"
        "sxtl v26.4s, v30.4h\n"

        "sxtl2 v27.4s, v27.8h\n"
        "sxtl2 v28.4s, v28.8h\n"
        "sxtl2 v29.4s, v29.8h\n"
        "sxtl2 v30.4s, v30.8h\n"
#endif

#ifndef USE_32BIT_SIGN
        "mul v1.4s, v1.4s, v23.4s\n"
        "mul v2.4s, v2.4s, v24.4s\n"
        "mul v3.4s, v3.4s, v25.4s\n"
        "mul v4.4s, v4.4s, v26.4s\n"

        "mul v5.4s, v5.4s, v27.4s\n"
        "mul v6.4s, v6.4s, v28.4s\n"
        "mul v7.4s, v7.4s, v29.4s\n"
        "mul v8.4s, v8.4s, v30.4s\n"
#else

        "eor v1.16b, v1.16b, v23.16b\n"
        "eor v2.16b, v2.16b, v24.16b\n"
        "eor v3.16b, v3.16b, v25.16b\n"
        "eor v4.16b, v4.16b, v26.16b\n"

        "eor v5.16b, v5.16b, v27.16b\n"
        "eor v6.16b, v6.16b, v28.16b\n"
        "eor v7.16b, v7.16b, v29.16b\n"
        "eor v8.16b, v8.16b, v30.16b\n"
#endif

        "shl v1.4s, v1.4s, #23\n"
        "shl v2.4s, v2.4s, #23\n"
        "shl v3.4s, v3.4s, #23\n"
        "shl v4.4s, v4.4s, #23\n"

        "shl v5.4s, v5.4s, #23\n"
        "shl v6.4s, v6.4s, #23\n"
        "shl v7.4s, v7.4s, #23\n"
        "shl v8.4s, v8.4s, #23\n"

        "eor v23.16b, v1.16b, v23.16b\n"
        "eor v24.16b, v2.16b, v24.16b\n"
        "eor v25.16b, v3.16b, v25.16b\n"
        "eor v26.16b, v4.16b, v26.16b\n"

        "eor v27.16b, v5.16b, v27.16b\n"
        "eor v28.16b, v6.16b, v28.16b\n"
        "eor v29.16b, v7.16b, v29.16b\n"
        "eor v30.16b, v8.16b, v30.16b\n"

        SET_ZERO_IF_OUT_2_2(15, 3)
        SET_ZERO_IF_OUT_2_2(14, 2)
        SET_ZERO_IF_OUT_2_2(13, 1)
        SET_ZERO_IF_OUT_2_2(12, 0)
        SET_ZERO_IF_OUT_2_1(11, 3)
        SET_ZERO_IF_OUT_2_1(10, 2)
        SET_ZERO_IF_OUT_2_1(9 , 1)
        SET_ZERO_IF_OUT_2_1(8,  0)

        "110:\n"

        "faddp v13.4s, v13.4s, v23.4s\n"
        "faddp v14.4s, v14.4s, v24.4s\n"
        "faddp v15.4s, v15.4s, v25.4s\n"
        "faddp v16.4s, v16.4s, v26.4s\n"

        "faddp v13.4s, v13.4s, v27.4s\n"
        "faddp v14.4s, v14.4s, v28.4s\n"
        "faddp v15.4s, v15.4s, v29.4s\n"
        "faddp v16.4s, v16.4s, v30.4s\n"

        "7:\n"
        // Add 4 float in accumulation vetors together
        "faddp v13.4s, v13.4s, v13.4s\n"
        "faddp v14.4s, v14.4s, v14.4s\n"
        "faddp v15.4s, v15.4s, v15.4s\n"
        "faddp v16.4s, v16.4s, v16.4s\n"

        "faddp v13.4s, v13.4s, v13.4s\n"
        "faddp v14.4s, v14.4s, v14.4s\n"
        "faddp v15.4s, v15.4s, v15.4s\n"
        "faddp v16.4s, v16.4s, v16.4s\n"

        /////////////////////////////////////////////

        "mov %w[dst_1], v13.s[0]\n"
        "mov %w[dst_2], v14.s[0]\n"
        "mov %w[dst_3], v15.s[0]\n"
        "mov %w[dst_4], v16.s[0]\n"

        // "mov w1, #0x007FFFFF\n" // maximum mantisa value 
        // "mov w2, #0xFF800001\n" // minimum mantisa value 
        // "mov w3, #0x000000FF\n" // exponent limiter
        // "mov w4, #0x807FFFFF\n" // float exponent mask

        : [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i)
        : [ src_ptr2_1 ] "r"(src_ptr2_1), [ src_ptr2_2 ] "r"(src_ptr2_2),
          [ src_ptr2_3 ] "r"(src_ptr2_3), [ src_ptr2_4 ] "r"(src_ptr2_4),
          [ src_ptr2_s_1 ] "r"(src_ptr2_s_1), [ src_ptr2_s_2 ] "r"(src_ptr2_s_2),
          [ src_ptr2_s_3 ] "r"(src_ptr2_s_3), [ src_ptr2_s_4 ] "r"(src_ptr2_s_4),
          [ src_ptr1_exp ] "r"(src_ptr1_exp),
          [ src_ptr1_mantise ] "r"(src_ptr1_mantise), [ size ] "r"(size)
        : "v0", 
          "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8",
          "v13", "v14", "v15", "v16",
          "v17", "v18",
          "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30"
    );
}

void Mul::vector_matrix_multiplication_accumulation_int8_shift(
    int8_t* src_ptr1_exp,
    int8_t* src_ptr2_ref,
    int32_t& dst_1, int32_t& dst_2, int32_t& dst_3, int32_t& dst_4,
    int size){
    int8_t *src_ptr2 = src_ptr2_ref;

    int i;

    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        // Start of Main Loop
        "1:\n"
#ifdef EXTEND_FOR_ACCURACY
#ifdef IN_KERNEL_EXTEND
        "ld1 {v0.16b},  [%[src_ptr1_exp]], #16\n"

        "ld1 {v23.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v24.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v25.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v26.16b}, [%[src_ptr2]], #16\n"
#else
        "ld1 {v0.8h},  [%[src_ptr1_exp]], #16\n"
        "ld1 {v31.8h}, [%[src_ptr1_exp]], #16\n"

        "ld1 {v23.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v24.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v25.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v26.8h}, [%[src_ptr2]], #16\n"

        "ld1 {v27.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v28.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v29.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v30.8h}, [%[src_ptr2]], #16\n"
#endif
        "ld1 {v1.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v2.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v3.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v4.8h}, [%[src_ptr2]], #16\n"

        "ld1 {v5.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v6.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v7.8h}, [%[src_ptr2]], #16\n"
        "ld1 {v8.8h}, [%[src_ptr2]], #16\n"
#else
        "ld1 {v0.16b},  [%[src_ptr1_exp]], #16\n"

        "ld1 {v23.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v24.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v25.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v26.16b}, [%[src_ptr2]], #16\n"

        "ld1 {v1.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr2]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr2]], #16\n"
#endif

// #ifdef MULTIPLY_SIGN
//         "mul v23.16b, v0.16b, v23.16b\n"
//         "mul v24.16b, v0.16b, v24.16b\n"
//         "mul v25.16b, v0.16b, v25.16b\n"
//         "mul v26.16b, v0.16b, v26.16b\n"
// #else
        "eor v23.16b, v0.16b, v23.16b\n"
        "eor v24.16b, v0.16b, v24.16b\n"
        "eor v25.16b, v0.16b, v25.16b\n"
        "eor v26.16b, v0.16b, v26.16b\n"
#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
        "eor v27.16b, v31.16b, v27.16b\n"
        "eor v28.16b, v31.16b, v28.16b\n"
        "eor v29.16b, v31.16b, v29.16b\n"
        "eor v30.16b, v31.16b, v30.16b\n"
#endif
#endif
// #endif

#ifdef EXTEND_FOR_ACCURACY
#ifdef IN_KERNEL_EXTEND
        // Extend src2 from int8 to int16
        "sxtl2 v27.8h, v23.16b\n"
        "sxtl2 v28.8h, v24.16b\n"
        "sxtl2 v29.8h, v25.16b\n"
        "sxtl2 v30.8h, v26.16b\n"

        "sxtl v23.8h, v23.8b\n"
        "sxtl v24.8h, v24.8b\n"
        "sxtl v25.8h, v25.8b\n"
        "sxtl v26.8h, v26.8b\n"
#endif
        "sqshl v1.8h, v23.8h, v1.8h\n"
        "sqshl v2.8h, v24.8h, v2.8h\n"
        "sqshl v3.8h, v25.8h, v3.8h\n"
        "sqshl v4.8h, v26.8h, v4.8h\n"

        "sqshl v5.8h, v27.8h, v5.8h\n"
        "sqshl v6.8h, v28.8h, v6.8h\n"
        "sqshl v7.8h, v29.8h, v7.8h\n"
        "sqshl v8.8h, v30.8h, v8.8h\n"

        "sadalp v13.4s, v1.8h\n"
        "sadalp v14.4s, v2.8h\n"
        "sadalp v15.4s, v3.8h\n"
        "sadalp v16.4s, v4.8h\n"

        "sadalp v13.4s, v5.8h\n"
        "sadalp v14.4s, v6.8h\n"
        "sadalp v15.4s, v7.8h\n"
        "sadalp v16.4s, v8.8h\n"
#else
        "sqshl v1.16b, v23.16b, v1.16b\n"
        "sqshl v2.16b, v24.16b, v2.16b\n"
        "sqshl v3.16b, v25.16b, v3.16b\n"
        "sqshl v4.16b, v26.16b, v4.16b\n"

        "sadalp v13.8h, v1.16b\n"
        "sadalp v14.8h, v2.16b\n"
        "sadalp v15.8h, v3.16b\n"
        "sadalp v16.8h, v4.16b\n"
#endif

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"
        
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"

#ifdef EXTEND_FOR_ACCURACY
        LOAD_ONE_DATA_AND_SIGN_EXT_1(0, 0)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(1, 1)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(2, 2)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(3, 3)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(4, 4)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(5, 5)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(6, 6)
        LOAD_ONE_DATA_AND_SIGN_EXT_1(7, 7)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(8, 0)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(9, 1)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(10,2)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(11,3)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(12,4)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(13,5)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(14,6)
        LOAD_ONE_DATA_AND_SIGN_EXT_2(15,7)
#else
        LOAD_ONE_DATA_AND_SIGN(0)
        LOAD_ONE_DATA_AND_SIGN(1)
        LOAD_ONE_DATA_AND_SIGN(2)
        LOAD_ONE_DATA_AND_SIGN(3)
        LOAD_ONE_DATA_AND_SIGN(4)
        LOAD_ONE_DATA_AND_SIGN(5)
        LOAD_ONE_DATA_AND_SIGN(6)
        LOAD_ONE_DATA_AND_SIGN(7)
        LOAD_ONE_DATA_AND_SIGN(8)
        LOAD_ONE_DATA_AND_SIGN(9)
        LOAD_ONE_DATA_AND_SIGN(10)
        LOAD_ONE_DATA_AND_SIGN(11)
        LOAD_ONE_DATA_AND_SIGN(12)
        LOAD_ONE_DATA_AND_SIGN(13)
        LOAD_ONE_DATA_AND_SIGN(14)
        LOAD_ONE_DATA_AND_SIGN(15)
#endif

        "15:\n"

// #ifdef MULTIPLY_SIGN
//         "mul v23.16b, v1.16b, v23.16b\n"
//         "mul v24.16b, v2.16b, v24.16b\n"
//         "mul v25.16b, v3.16b, v25.16b\n"
//         "mul v26.16b, v4.16b, v26.16b\n"
// #else
        "eor v23.16b, v0.16b, v23.16b\n"
        "eor v24.16b, v0.16b, v24.16b\n"
        "eor v25.16b, v0.16b, v25.16b\n"
        "eor v26.16b, v0.16b, v26.16b\n"
#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
        "eor v27.16b, v31.16b, v27.16b\n"
        "eor v28.16b, v31.16b, v28.16b\n"
        "eor v29.16b, v31.16b, v29.16b\n"
        "eor v30.16b, v31.16b, v30.16b\n"
#endif
#endif
// #endif

#ifdef EXTEND_FOR_ACCURACY
#ifdef IN_KERNEL_EXTEND
        // Extend src2 from int8 to int16
        "sxtl2 v27.8h, v23.16b\n"
        "sxtl2 v28.8h, v24.16b\n"
        "sxtl2 v29.8h, v25.16b\n"
        "sxtl2 v30.8h, v26.16b\n"

        "sxtl v23.8h, v23.8b\n"
        "sxtl v24.8h, v24.8b\n"
        "sxtl v25.8h, v25.8b\n"
        "sxtl v26.8h, v26.8b\n"
#endif
        "sqshl v1.8h, v23.8h, v1.8h\n"
        "sqshl v2.8h, v24.8h, v2.8h\n"
        "sqshl v3.8h, v25.8h, v3.8h\n"
        "sqshl v4.8h, v26.8h, v4.8h\n"

        "sqshl v5.8h, v27.8h, v5.8h\n"
        "sqshl v6.8h, v28.8h, v6.8h\n"
        "sqshl v7.8h, v29.8h, v7.8h\n"
        "sqshl v8.8h, v30.8h, v8.8h\n"

        "sadalp v13.4s, v1.8h\n"
        "sadalp v14.4s, v2.8h\n"
        "sadalp v15.4s, v3.8h\n"
        "sadalp v16.4s, v4.8h\n"

        "sadalp v13.4s, v5.8h\n"
        "sadalp v14.4s, v6.8h\n"
        "sadalp v15.4s, v7.8h\n"
        "sadalp v16.4s, v8.8h\n"
#else
        "sqshl v1.16b, v23.16b, v1.16b\n"
        "sqshl v2.16b, v24.16b, v2.16b\n"
        "sqshl v3.16b, v25.16b, v3.16b\n"
        "sqshl v4.16b, v26.16b, v4.16b\n"

        "sadalp v13.8h, v1.16b\n"
        "sadalp v14.8h, v2.16b\n"
        "sadalp v15.8h, v3.16b\n"
        "sadalp v16.8h, v4.16b\n"
#endif

        "7:\n"
#ifdef EXTEND_FOR_ACCURACY
        // Add 4 outputs in accumulation vectors together
        "addv s13, v13.4s\n"
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "addv s16, v16.4s\n"
#else
        // Add 8 outputs in accumulation vectors together
        "addv h13, v13.8h\n"
        "addv h14, v14.8h\n"
        "addv h15, v15.8h\n"
        "addv h16, v16.8h\n"

        "mov v13.h[1], wzr\n"
        "mov v14.h[1], wzr\n"
        "mov v15.h[1], wzr\n"
        "mov v16.h[1], wzr\n"
#endif
        "mov %w[dst_1], v13.s[0]\n"
        "mov %w[dst_2], v14.s[0]\n"
        "mov %w[dst_3], v15.s[0]\n"
        "mov %w[dst_4], v16.s[0]\n"

        : [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i)
        : [ src_ptr2 ] "r"(src_ptr2),
          [ src_ptr1_exp ] "r"(src_ptr1_exp),
          [ size ] "r"(size)
        : "v0", 
          "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8",
          "v13", "v14", "v15", "v16",
          "v17", "v18",
          "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30"
    );
}

void Mul::vector_matrix_multiplication_accumulation_float32(
    float* src_ptr1,
    float* src_ptr2_1, float* src_ptr2_2, float* src_ptr2_3, float* src_ptr2_4,
    float *dst_1, float *dst_2, float *dst_3, float *dst_4,
    int size){
    float_t c1, c2, c3, c4;
    size_t i, j;

    // these are the columns A
    float32x4_t A0;

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

    float* src_ptr1_ptr = src_ptr1;
    float* src_ptr2_1_ptr = src_ptr2_1;
    float* src_ptr2_2_ptr = src_ptr2_2;
    float* src_ptr2_3_ptr = src_ptr2_3;
    float* src_ptr2_4_ptr = src_ptr2_4;

    C0 = vmovq_n_f32(0);
    C1 = vmovq_n_f32(0);
    C2 = vmovq_n_f32(0);
    C3 = vmovq_n_f32(0);
    for (i = 0; (i+4) <= size; i+=4){
        A0 = vld1q_f32(src_ptr1_ptr); src_ptr1_ptr += 4;

        B0 = vld1q_f32(src_ptr2_1_ptr); src_ptr2_1_ptr += 4;
        B1 = vld1q_f32(src_ptr2_2_ptr); src_ptr2_2_ptr += 4;
        B2 = vld1q_f32(src_ptr2_3_ptr); src_ptr2_3_ptr += 4;
        B3 = vld1q_f32(src_ptr2_4_ptr); src_ptr2_4_ptr += 4;

        #if DO_FLOAT_PREFETCHING
        __builtin_prefetch(src_ptr1_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_1_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_2_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_3_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        __builtin_prefetch(src_ptr2_4_ptr + PREFETCH_FLOAT_OFFSET, 0, 0);
        #endif

        C0 = vfmaq_f32(C0, A0, B0);
        C1 = vfmaq_f32(C1, A0, B1);
        C2 = vfmaq_f32(C2, A0, B2);
        C3 = vfmaq_f32(C3, A0, B3);
    }

    c1 = vaddvq_f32(C0);
    c2 = vaddvq_f32(C1);
    c3 = vaddvq_f32(C2);
    c4 = vaddvq_f32(C3);

    for (i; i < size; i++){
        c1 += src_ptr1_ptr[i] * src_ptr2_1_ptr[i];
        c2 += src_ptr1_ptr[i] * src_ptr2_2_ptr[i];
        c3 += src_ptr1_ptr[i] * src_ptr2_3_ptr[i];
        c4 += src_ptr1_ptr[i] * src_ptr2_4_ptr[i];
    }

    *dst_1 = c1;
    *dst_2 = c2;
    *dst_3 = c3;
    *dst_4 = c4;
}

void Mul::weight_matrix_pack_int8_signed(
        int8_t* src, int8_t* src_sign,
        int8_t* packed_matrix,
        int rows, int columns
    ){
    int i;

    int8_t *src_ptr_1 = src + 0 * columns;
    int8_t *src_ptr_2 = src + 1 * columns;
    int8_t *src_ptr_3 = src + 2 * columns;
    int8_t *src_ptr_4 = src + 3 * columns;

    int8_t *src_ptr_sign_1 = src_sign + 0 * columns;
    int8_t *src_ptr_sign_2 = src_sign + 1 * columns;
    int8_t *src_ptr_sign_3 = src_sign + 2 * columns;
    int8_t *src_ptr_sign_4 = src_sign + 3 * columns;

    int8_t *packed_ptr = packed_matrix;
    
    for (i = 0 ; (i+4) <= rows ; i+=4){
        matrix_pack_int8_signed_impl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            src_ptr_sign_1, src_ptr_sign_2,
            src_ptr_sign_3, src_ptr_sign_4,
            packed_ptr,
            columns);
        src_ptr_1 += 4 * columns;
        src_ptr_2 += 4 * columns;
        src_ptr_3 += 4 * columns;
        src_ptr_4 += 4 * columns;

        src_ptr_sign_1 += 4 * columns;
        src_ptr_sign_2 += 4 * columns;
        src_ptr_sign_3 += 4 * columns;
        src_ptr_sign_4 += 4 * columns;
    }
    i = rows - (i - 4);
    if (i == 1){
        src_ptr_2 = src_ptr_1;
        src_ptr_3 = src_ptr_1;
        src_ptr_4 = src_ptr_1;

        src_ptr_sign_2 = src_ptr_sign_1;
        src_ptr_sign_3 = src_ptr_sign_1;
        src_ptr_sign_4 = src_ptr_sign_1;
        matrix_pack_int8_signed_impl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            src_ptr_sign_1, src_ptr_sign_2,
            src_ptr_sign_3, src_ptr_sign_4,
            packed_ptr,
            columns);
    }
    else if (i == 2){
        src_ptr_3 = src_ptr_2;
        src_ptr_4 = src_ptr_2;
        
        src_ptr_sign_3 = src_ptr_sign_2;
        src_ptr_sign_4 = src_ptr_sign_2;
        matrix_pack_int8_signed_impl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            src_ptr_sign_1, src_ptr_sign_2,
            src_ptr_sign_3, src_ptr_sign_4,
            packed_ptr,
            columns);
    }
    else if (i == 3){
        src_ptr_4 = src_ptr_3;
        src_ptr_sign_4 = src_ptr_sign_3;
        matrix_pack_int8_signed_impl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            src_ptr_sign_1, src_ptr_sign_2,
            src_ptr_sign_3, src_ptr_sign_4,
            packed_ptr,
            columns);
    }
}

void Mul::activation_single_batched_vector_pack_int8(int8_t* src, int8_t* packed_matrix, int columns){
    int i;

    int8_t *src_ptr = src;
    int8_t *packed_ptr = packed_matrix;

    vector_pack_int8_impl(src_ptr, packed_ptr, columns);
}

void Mul::matrix_pack_int8_signed_impl(
        int8_t* src_ptr_ref_1, int8_t* src_ptr_ref_2,
        int8_t* src_ptr_ref_3, int8_t* src_ptr_ref_4,
        int8_t* src_ptr_s_ref_1, int8_t* src_ptr_s_ref_2,
        int8_t* src_ptr_s_ref_3, int8_t* src_ptr_s_ref_4,
        int8_t* dst_ptr,
        int size
    ){

    int8_t* src_ptr_1 = src_ptr_ref_1;
    int8_t* src_ptr_2 = src_ptr_ref_2;
    int8_t* src_ptr_3 = src_ptr_ref_3;
    int8_t* src_ptr_4 = src_ptr_ref_4;
    int8_t* src_ptr_s_1 = src_ptr_s_ref_1;
    int8_t* src_ptr_s_2 = src_ptr_s_ref_2;
    int8_t* src_ptr_s_3 = src_ptr_s_ref_3;
    int8_t* src_ptr_s_4 = src_ptr_s_ref_4;
    int8_t* dst_ptr_ref = dst_ptr;

    int i;

    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.16b}, [%[src_ptr_1]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr_2]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr_3]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr_4]], #16\n"

        "ld1 {v9.16b},  [%[src_ptr_s_1]], #16\n"
        "ld1 {v10.16b}, [%[src_ptr_s_2]], #16\n"
        "ld1 {v11.16b}, [%[src_ptr_s_3]], #16\n"
        "ld1 {v12.16b}, [%[src_ptr_s_4]], #16\n"

#ifdef EXTEND_FOR_ACCURACY
        // Extend src from int8 to int16
        "sxtl2 v5.8h, v1.16b\n"
        "sxtl2 v6.8h, v2.16b\n"
        "sxtl2 v7.8h, v3.16b\n"
        "sxtl2 v8.8h, v4.16b\n"

        "sxtl v1.8h, v1.8b\n"
        "sxtl v2.8h, v2.8b\n"
        "sxtl v3.8h, v3.8b\n"
        "sxtl v4.8h, v4.8b\n"
#ifndef IN_KERNEL_EXTEND
        // Extend sign from int8 to int16
        "sxtl2 v13.8h, v9.16b\n"
        "sxtl2 v14.8h, v10.16b\n"
        "sxtl2 v15.8h, v11.16b\n"
        "sxtl2 v16.8h, v12.16b\n"

        "sxtl v9.8h, v9.8b\n"
        "sxtl v10.8h, v10.8b\n"
        "sxtl v11.8h, v11.8b\n"
        "sxtl v12.8h, v12.8b\n"
#endif
#endif

        //////////////////////////////////////////////////////////////////
#ifndef EXTEND_FOR_ACCURACY
        "st1 {v9.16b},  [%[dst_ptr_ref]], #16\n"
        "st1 {v10.16b}, [%[dst_ptr_ref]], #16\n"
        "st1 {v11.16b}, [%[dst_ptr_ref]], #16\n"
        "st1 {v12.16b}, [%[dst_ptr_ref]], #16\n"

        "st1 {v1.16b},  [%[dst_ptr_ref]], #16\n"
        "st1 {v2.16b},  [%[dst_ptr_ref]], #16\n"
        "st1 {v3.16b},  [%[dst_ptr_ref]], #16\n"
        "st1 {v4.16b},  [%[dst_ptr_ref]], #16\n"
#else
#ifdef IN_KERNEL_EXTEND
        "st1 {v9.16b},  [%[dst_ptr_ref]], #16\n"
        "st1 {v10.16b}, [%[dst_ptr_ref]], #16\n"
        "st1 {v11.16b}, [%[dst_ptr_ref]], #16\n"
        "st1 {v12.16b}, [%[dst_ptr_ref]], #16\n"

        "st1 {v1.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v2.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v3.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v4.8h},   [%[dst_ptr_ref]], #16\n"

        "st1 {v5.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v6.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v7.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v8.8h},   [%[dst_ptr_ref]], #16\n"
#else
        "st1 {v9.8h},  [%[dst_ptr_ref]], #16\n"
        "st1 {v10.8h}, [%[dst_ptr_ref]], #16\n"
        "st1 {v11.8h}, [%[dst_ptr_ref]], #16\n"
        "st1 {v12.8h}, [%[dst_ptr_ref]], #16\n"

        "st1 {v13.8h}, [%[dst_ptr_ref]], #16\n"
        "st1 {v14.8h}, [%[dst_ptr_ref]], #16\n"
        "st1 {v15.8h}, [%[dst_ptr_ref]], #16\n"
        "st1 {v16.8h}, [%[dst_ptr_ref]], #16\n"

        "st1 {v1.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v2.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v3.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v4.8h},   [%[dst_ptr_ref]], #16\n"

        "st1 {v5.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v6.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v7.8h},   [%[dst_ptr_ref]], #16\n"
        "st1 {v8.8h},   [%[dst_ptr_ref]], #16\n"
#endif
#endif
        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"

        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"

        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"

        PACK_LOAD_ONE_DATA(0)
        PACK_LOAD_ONE_DATA(1)
        PACK_LOAD_ONE_DATA(2)
        PACK_LOAD_ONE_DATA(3)
        PACK_LOAD_ONE_DATA(4)
        PACK_LOAD_ONE_DATA(5)
        PACK_LOAD_ONE_DATA(6)
        PACK_LOAD_ONE_DATA(7)
        PACK_LOAD_ONE_DATA(8)
        PACK_LOAD_ONE_DATA(9)
        PACK_LOAD_ONE_DATA(10)
        PACK_LOAD_ONE_DATA(11)
        PACK_LOAD_ONE_DATA(12)
        PACK_LOAD_ONE_DATA(13)
        PACK_LOAD_ONE_DATA(14)
        PACK_LOAD_ONE_DATA(15)

        "5:\n"

#ifdef EXTEND_FOR_ACCURACY
        // Extend src from int8 to int16
        "sxtl2 v5.8h, v1.16b\n"
        "sxtl2 v6.8h, v2.16b\n"
        "sxtl2 v7.8h, v3.16b\n"
        "sxtl2 v8.8h, v4.16b\n"

        "sxtl v1.8h, v1.8b\n"
        "sxtl v2.8h, v2.8b\n"
        "sxtl v3.8h, v3.8b\n"
        "sxtl v4.8h, v4.8b\n"
#ifndef IN_KERNEL_EXTEND
        // Extend sign from int8 to int16
        "sxtl2 v13.8h, v9.16b\n"
        "sxtl2 v14.8h, v10.16b\n"
        "sxtl2 v15.8h, v11.16b\n"
        "sxtl2 v16.8h, v12.16b\n"

        "sxtl v9.8h, v9.8b\n"
        "sxtl v10.8h, v10.8b\n"
        "sxtl v11.8h, v11.8b\n"
        "sxtl v12.8h, v12.8b\n"
#endif
#endif

#ifdef EXTEND_FOR_ACCURACY
        PACK_STORE_ONE_DATA_16BIT_1(0, 0)
        PACK_STORE_ONE_DATA_16BIT_1(1, 1)
        PACK_STORE_ONE_DATA_16BIT_1(2, 2)
        PACK_STORE_ONE_DATA_16BIT_1(3, 3)
        PACK_STORE_ONE_DATA_16BIT_1(4, 4)
        PACK_STORE_ONE_DATA_16BIT_1(5, 5)
        PACK_STORE_ONE_DATA_16BIT_1(6, 6)
        PACK_STORE_ONE_DATA_16BIT_1(7, 7)
        PACK_STORE_ONE_DATA_16BIT_2(8, 0)
        PACK_STORE_ONE_DATA_16BIT_2(9, 1)
        PACK_STORE_ONE_DATA_16BIT_2(10, 2)
        PACK_STORE_ONE_DATA_16BIT_2(11, 3)
        PACK_STORE_ONE_DATA_16BIT_2(12, 4)
        PACK_STORE_ONE_DATA_16BIT_2(13, 5)
        PACK_STORE_ONE_DATA_16BIT_2(14, 6)
        PACK_STORE_ONE_DATA_16BIT_2(15, 7)
#else
        PACK_STORE_ONE_DATA(0)
        PACK_STORE_ONE_DATA(1)
        PACK_STORE_ONE_DATA(2)
        PACK_STORE_ONE_DATA(3)
        PACK_STORE_ONE_DATA(4)
        PACK_STORE_ONE_DATA(5)
        PACK_STORE_ONE_DATA(6)
        PACK_STORE_ONE_DATA(7)
        PACK_STORE_ONE_DATA(8)
        PACK_STORE_ONE_DATA(9)
        PACK_STORE_ONE_DATA(10)
        PACK_STORE_ONE_DATA(11)
        PACK_STORE_ONE_DATA(12)
        PACK_STORE_ONE_DATA(13)
        PACK_STORE_ONE_DATA(14)
        PACK_STORE_ONE_DATA(15)
#endif

        "6:\n"
        "15:\n"
        "7:\n"

        : [ dst_ptr_ref ] "+r"(dst_ptr_ref), [ i ] "+r"(i)
        : [ src_ptr_1 ] "r"(src_ptr_1), [ src_ptr_2 ] "r"(src_ptr_2),
          [ src_ptr_3 ] "r"(src_ptr_3), [ src_ptr_4 ] "r"(src_ptr_4),
          [ src_ptr_s_1 ] "r"(src_ptr_s_1), [ src_ptr_s_2 ] "r"(src_ptr_s_2),
          [ src_ptr_s_3 ] "r"(src_ptr_s_3), [ src_ptr_s_4 ] "r"(src_ptr_s_4),
          [ size ] "r"(size)
        : "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16"
    );
}

void Mul::vector_pack_int8_impl(int8_t* src_ptr_ref, int8_t* dst_ptr_ref, int size){// = 96

    int8_t* src_ptr = src_ptr_ref;
    int8_t* dst_ptr = dst_ptr_ref;

    int i = 0;
    int c = 0;
    int cc = 0;
    asm volatile(
        "mov %w[i], wzr\n"        // 0

        "cmp %w[size], #16\n"
        "blt 3f\n"
        "cmp %w[size], #64\n"
        "blt 2f\n"

        "add %w[i], %w[i], #64\n" // 64

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.16b}, [%[src_ptr]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr]], #16\n"
        "ld1 {v3.16b}, [%[src_ptr]], #16\n"
        "ld1 {v4.16b}, [%[src_ptr]], #16\n"

#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
        // Extend src from int8 to int16
        "sxtl2 v5.8h, v1.16b\n"
        "sxtl2 v6.8h, v2.16b\n"
        "sxtl2 v7.8h, v3.16b\n"
        "sxtl2 v8.8h, v4.16b\n"

        "sxtl v1.8h, v1.8b\n"
        "sxtl v2.8h, v2.8b\n"
        "sxtl v3.8h, v3.8b\n"
        "sxtl v4.8h, v4.8b\n"
#endif
#endif

        //////////////////////////////////////////////////////////////////
#ifndef EXTEND_FOR_ACCURACY
        "st1 {v1.16b},  [%[dst_ptr]], #16\n"
        "st1 {v2.16b},  [%[dst_ptr]], #16\n"
        "st1 {v3.16b},  [%[dst_ptr]], #16\n"
        "st1 {v4.16b},  [%[dst_ptr]], #16\n"
#else
#ifdef IN_KERNEL_EXTEND
        "st1 {v1.16b},  [%[dst_ptr]], #16\n"
        "st1 {v2.16b},  [%[dst_ptr]], #16\n"
        "st1 {v3.16b},  [%[dst_ptr]], #16\n"
        "st1 {v4.16b},  [%[dst_ptr]], #16\n"
#else
        "st1 {v1.8h},   [%[dst_ptr]], #16\n"
        "st1 {v5.8h},   [%[dst_ptr]], #16\n"

        "st1 {v2.8h},   [%[dst_ptr]], #16\n"
        "st1 {v6.8h},   [%[dst_ptr]], #16\n"
        
        "st1 {v3.8h},   [%[dst_ptr]], #16\n"
        "st1 {v7.8h},   [%[dst_ptr]], #16\n"
        
        "st1 {v4.8h},   [%[dst_ptr]], #16\n"
        "st1 {v8.8h},   [%[dst_ptr]], #16\n"
#endif
#endif
        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #64\n" // 128
        "cmp %w[i], %w[size]\n"   // 128 <?> 96
        "b.le 1b\n"               // goto 1: 128 <= 96
        "sub %w[i], %w[i], #64\n" // 64

        "add %w[i], %w[i], #16\n" // 80
        "cmp %w[i], %w[size]\n"   // 80 <?> 96
        "bgt 4f\n"                // goto 4: 80 > 96

        "2:\n"

        "ld1 {v1.16b}, [%[src_ptr]], #16\n"

#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
        // Extend src from int8 to int16
        "sxtl2 v5.8h, v1.16b\n"
        "sxtl v1.8h, v1.8b\n"
#endif
#endif

        //////////////////////////////////////////////////////////////////
#ifndef EXTEND_FOR_ACCURACY
        "st1 {v1.16b},  [%[dst_ptr]], #16\n"
#else
#ifndef IN_KERNEL_EXTEND
        "st1 {v1.16b},  [%[dst_ptr]], #16\n"
#else
        "st1 {v1.8h},   [%[dst_ptr]], #16\n"
        "st1 {v5.8h},   [%[dst_ptr]], #16\n"
#endif
#endif
        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n" // 96, 112
        "cmp %w[i], %w[size]\n"   // 96 <?> 96, 112 <?> 96
        "b.le 2b\n"               // goto 2: 96 <= 96, 112 <= 96
        "4:\n"
        "sub %w[i], %w[i], #16\n" // 96
        "sub %w[i], %w[size], %w[i]\n"// 0

        "3:\n"
        
        "cmp %w[i], #0\n"         // 0 <?> 0
        "beq 7f\n"                // goto 7: 0 == 0

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"

        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"

#define PACK_LOAD(R)\
        "cmp %w[i], #" #R "\n"                                \
        "beq 5f\n"                                            \
        "ld1 { v1.b }[" #R "], [%[src_ptr]], #1\n"

        PACK_LOAD(0)
        PACK_LOAD(1)
        PACK_LOAD(2)
        PACK_LOAD(3)
        PACK_LOAD(4)
        PACK_LOAD(5)
        PACK_LOAD(6)
        PACK_LOAD(7)
        PACK_LOAD(8)
        PACK_LOAD(9)
        PACK_LOAD(10)
        PACK_LOAD(11)
        PACK_LOAD(12)
        PACK_LOAD(13)
        PACK_LOAD(14)
        PACK_LOAD(15)
        "5:\n"

#undef PACK_LOAD

#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
        // Extend src from int8 to int16
        "sxtl2 v5.8h, v1.16b\n"
        "sxtl2 v6.8h, v2.16b\n"
        "sxtl2 v7.8h, v3.16b\n"
        "sxtl2 v8.8h, v4.16b\n"

        "sxtl v1.8h, v1.8b\n"
        "sxtl v2.8h, v2.8b\n"
        "sxtl v3.8h, v3.8b\n"
        "sxtl v4.8h, v4.8b\n"
#endif
#endif

#ifdef EXTEND_FOR_ACCURACY
#ifndef IN_KERNEL_EXTEND
#define PACK_STORE(R,R2)                                \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.h }[" #R2 "],  [%[dst_ptr]], #2\n"

        PACK_STORE(0, 0)
        PACK_STORE(1, 1)
        PACK_STORE(2, 2)
        PACK_STORE(3, 3)
        PACK_STORE(4, 4)
        PACK_STORE(5, 5)
        PACK_STORE(6, 6)
        PACK_STORE(7, 7)

#undef PACK_STORE
#define PACK_STORE(R,R2)                                \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v5.h }[" #R2 "],  [%[dst_ptr]], #2\n"

        PACK_STORE(8, 0)
        PACK_STORE(9, 1)
        PACK_STORE(10, 2)
        PACK_STORE(11, 3)
        PACK_STORE(12, 4)
        PACK_STORE(13, 5)
        PACK_STORE(14, 6)
        PACK_STORE(15, 7)
#else
#define PACK_STORE(R)                                   \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.b }[" #R "], [%[dst_ptr]], #1\n"

        PACK_STORE(0)
        PACK_STORE(1)
        PACK_STORE(2)
        PACK_STORE(3)
        PACK_STORE(4)
        PACK_STORE(5)
        PACK_STORE(6)
        PACK_STORE(7)
        PACK_STORE(8)
        PACK_STORE(9)
        PACK_STORE(10)
        PACK_STORE(11)
        PACK_STORE(12)
        PACK_STORE(13)
        PACK_STORE(14)
        PACK_STORE(15)
#undef PACK_STORE
#endif
#else
#define PACK_STORE(R)                                   \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.b }[" #R "], [%[dst_ptr]], #1\n"

        PACK_STORE(0)
        PACK_STORE(1)
        PACK_STORE(2)
        PACK_STORE(3)
        PACK_STORE(4)
        PACK_STORE(5)
        PACK_STORE(6)
        PACK_STORE(7)
        PACK_STORE(8)
        PACK_STORE(9)
        PACK_STORE(10)
        PACK_STORE(11)
        PACK_STORE(12)
        PACK_STORE(13)
        PACK_STORE(14)
        PACK_STORE(15)
#undef PACK_STORE
#endif

        "6:\n"
        "15:\n"
        "7:\n"

        : [ dst_ptr ] "+r"(dst_ptr), [ i ] "+r"(i),
          [ c ] "+r"(c), [ cc ] "+r"(cc)
        : [ src_ptr ] "r"(src_ptr), [ size ] "r"(size)
        : "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8"
    );
}




