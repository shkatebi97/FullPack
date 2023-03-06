#include "Tensor.h"

using std::string;
using std::to_string;

unsigned long Tensor::last_id = 0;

Tensor::Tensor(){
    _initialized = false; 
    _filled = false; 
    _extracted = false; 
    id = last_id; 
    last_id++;
    _data = nullptr;
    _exponents = nullptr;
    _mantisas = nullptr;
    _signs = nullptr;
    _transposed = false;
    _sliced_with_dont_copy = false;
    int sizes[] = {0,0};
    _shape = create_shape(sizes, 2);
}

Tensor::Tensor(Shape shape):Tensor(){_shape = shape;}

Tensor& Tensor::operator=(const Tensor& other){
    // std::cout << "[Tensor::operator=] " << std::to_string(id) << std::endl;
    copy_shape(other._shape);
    if (other._initialized){
        _initialized = false;
        _filled = false;
        _extracted = false;
        Status alloc_s = this->Allocate();
        if(alloc_s != Status::Success && alloc_s != Status::AlreadyInitialized)
            throw string("Allocate Failed while assigning: ") + to_string(alloc_s);
        if(other._filled){
            for (int i = 0 ; i < other._shape.flatsize ; i++){
                this->_data[i]          = other._data[i];
                this->_data_T[i]        = other._data_T[i];
                if(other._extracted){
                    this->_exponents[i] = other._exponents[i];
                    this->_mantisas[i]  = other._mantisas[i];
                    this->_signs[i]     = other._signs[i];
                }
            }
        }
    }
    this->_initialized                  = other._initialized;
    this->_filled                       = other._filled;
    this->_extracted                    = other._extracted;
    this->_transposed                   = other._transposed;
    this->_sliced_with_dont_copy        = false;
    id = last_id; 
    last_id++;
    return *this;
}

void Tensor::copy_shape(const Shape& other){
    _shape = other;
}

Tensor::~Tensor(){
    // std::cout << "[Tensor::~Tensor] " << std::to_string(id) << std::endl;
    if (_initialized && !_sliced_with_dont_copy){
        deallocate(_data);
        deallocate(_data_T);
        deallocate(_exponents, true);
        deallocate(_mantisas, true);
        deallocate(_signs, true);
        deallocate(_signs_8, true);
        deallocate(_data_float16, true);
        _initialized = false;
    }
}

Status Tensor::Fill_Random(){
    srand(time(NULL));
    if(!_transposed)
        for(int i = 0; i < _shape.flatsize; i++)
            _data[i] = rand() % RAND_MAX;
    else
        for(int i = 0; i < _shape.flatsize; i++)
            _data_T[i] = rand() % RAND_MAX;
    Status ref_s = Refresh();
    if(ref_s != Status::Success)
        return ref_s;
    _filled = true;
    return Status::Success;
}

Status Tensor::Fill_Zero(){
    zero_vector(_data, _shape.flatsize);
    zero_vector(_data_T, _shape.flatsize);
    _filled = true;
    return Status::Success;
}

Status Tensor::Extract(){
    long i;
    Float* src;
    if(_transposed)
        src = get_pointer_as<Float>(_data_T);
    else
        src = get_pointer_as<Float>(_data);
    for( i = 0; i < _shape.flatsize ; i++ ){
        _exponents[i] = src[i].parts.exponent;
        _mantisas[i] = src[i].parts.sign?src[i].parts.mantisa:-src[i].parts.mantisa;
        _mantisas[i] &= 0x807FFFFF;
#ifdef USE_32BIT_SIGN
        _signs[i]   = (src[i].parts.sign)?(0x00000000):(0x80000000);
#else
        _signs[i]   = (src[i].parts.sign)?(0x00):(0x80);
#endif
        _signs_8[i] = (src[i].parts.sign)?(0x00):(0x80);
        _data_float16[i] = src[i].f;
    }
    
    _extracted = true;
    return Status::Success;
}

Status Tensor::Allocate(){
    if(_initialized){
        return Status::AlreadyInitialized;
    }
    _data         = _allocate<float>(_shape.flatsize);
    _data_T       = _allocate<float>(_shape.flatsize);
    _exponents    = _allocate<int8_t>(_shape.flatsize);
    _signs        = _allocate<Sign_DataType>(_shape.flatsize);
    _signs_8      = _allocate<int8_t>(_shape.flatsize);
    _mantisas     = _allocate<int32_t>(_shape.flatsize);
    _data_float16 = _allocate<float16>(_shape.flatsize);
    _initialized  = true;
    return Status::Success;
}

Status Tensor::Initialize(){
    if(_initialized)
        return Status::AlreadyInitialized;
    Status _allocate_return = Allocate();
    if(_allocate_return != Status::Success)
        return _allocate_return;
    Status _fill_return = Fill_Random();
    if(_fill_return != Status::Success)
        return _fill_return;
    Status _extract_return = Extract();
    if(_extract_return != Status::Success)
        return _extract_return;
    return Status::Success;
}

Status Tensor::Refresh(){
    if(_shape.number_dims > 2 || _shape.number_dims < 1) return Status::NotImplemented;
    int t;
    if(_transposed)
        if(_shape.number_dims == 1)
            for (int i = 0 ; i < _shape.flatsize ; i++)
                _data[i] = _data_T[i];
        else
            for(int j = 0 ; j < _shape.size[1] ; j++)
                for(int i = 0 ; i < _shape.size[0] ; i++){
                    _data[i * _shape.size[1] + j] = _data_T[j * _shape.size[1] + i];
                }
    else
        if(_shape.number_dims == 1)
            for (int i = 0 ; i < _shape.flatsize ; i++)
                _data_T[i] = _data[i];
        else
            for(int i = 0 ; i < _shape.size[0] ; i++)
                for(int j = 0 ; j < _shape.size[1] ; j++){
                    t = _data[i * _shape.size[1] + j];
                    _data_T[j * _shape.size[0] + i] = t;
                }

    return Status::Success;
}

Status Tensor::DownCastInt32ToInt8(){
    if (!_initialized) return Status::NotAllocated;
    if (!_filled) return Status::NotFilled;
    for (int i = 0 ; i < _shape.flatsize ; i++)
        _exponents[i] = (_mantisas[i] < 0)?((_mantisas[i] % 127) * -1):(_mantisas[i] % 127);
    return Status::Success;
}

Status Tensor::DownCastFloat32ToInt8(){
    if (!_initialized) return Status::NotAllocated;
    if (!_filled) return Status::NotFilled;
    float max_abs, min_abs;
    float* data;
    if (_transposed) data = _data_T;
    else data = _data;
    get_absed_max_min(data, _shape.flatsize, max_abs, min_abs);
    _float_to_int_max = max_abs;
    _float_to_int_min = min_abs;
    _float_to_int_coeff = INT8_MAX / (_float_to_int_max - _float_to_int_min);
    for (int i = 0 ; i < _shape.flatsize ; i++)
        _exponents[i] = (int8_t)((data[i] - _float_to_int_min) * _float_to_int_coeff);
    return Status::Success;
}

Status Tensor::UpCastInt32ToFloat32(){
    if (!_initialized) return Status::NotAllocated;
    float* data;
    if (_transposed) data = _data_T;
    else data = _data;
    for (int i = 0 ; i < _shape.flatsize ; i++)
        data[i] = (_mantisas[i] / _float_to_int_coeff) + _float_to_int_min;
    return Status::Success;
}

void Tensor::set_f32_i8_parameters(const Float32ToInt8QunatizationParameters_t& i){
    _float_to_int_coeff = i.coeff;
    _float_to_int_max = i.max;
    _float_to_int_min = i.min;
}

Float32ToInt8QunatizationParameters_t Tensor::get_f32_i8_parameters(){
    Float32ToInt8QunatizationParameters_t o;
    o.coeff = _float_to_int_coeff;
    o.max = _float_to_int_max;
    o.min = _float_to_int_min;
    return o;
}

Status Tensor::Transpose(bool extract){
    if(_shape.number_dims == 1) return Status::Success;
    if(_shape.number_dims != 2) return Status::NotImplemented;
    _transposed = !_transposed;
    return Status::Success;
}

Status Tensor::TransposeData(){
    float* t = _data; 
    _data = _data_T; 
    _data_T = t;
    int tt;
    if(_shape.number_dims == 2){
        tt = _shape.size[0];
        _shape.size[0] = _shape.size[1];
        _shape.size[1] = tt;
    }
    return Status::Success;
}

Tensor* Tensor::SliceOnRows(int start, int length, bool dont_copy){
    if (_shape.number_dims > 2) throw string("slice expects 1D or 2D tensor, but got ") + 
                                       to_string(_shape.number_dims) + 
                                       string("D\n");
    if (dont_copy) throw string("dont copy is disabled for the sake of memory leakage problems");
    /*
    TODO: we should find another way to speed up the copy
          of both _data and _data_T together
    */
    Shape new_shape;
    int new_size[2];
    int n;
    if(_shape.number_dims == 1){
        new_size[0] = length;
        n = 1;
    }
    else if(_shape.number_dims == 2){
        if(length == 1){
            new_size[0] = _shape.size[1];
            n = 1;
        }
        else{
            new_size[0] = length;
            new_size[1] = _shape.size[1];
            n = 2;
        }
    }
    else
        throw string("Wrong number of dimensions\n");
    new_shape = create_shape(new_size, n);
    Tensor* output = new Tensor(new_shape);
    output->Allocate();
    unsigned long start_idx;
    if(_shape.number_dims == 1)
        start_idx = start;
    else
        start_idx = start * _shape.size[1];
    output->_transposed = _transposed;
    if (!dont_copy)
        if(output->_transposed)
            for (size_t i = 0; i < new_shape.flatsize; i++)
                output->_data_T[i] = this->_data_T[(start_idx + i) % _shape.flatsize];
        else
            for (size_t i = 0; i < new_shape.flatsize; i++)
                output->_data[i] = this->_data[(start_idx + i) % _shape.flatsize];
    else
        if(output->_transposed)
            output->_data_T = this->_data_T + start_idx;
        else
            output->_data = this->_data + start_idx;
    output->set_fill();
    output->Refresh();
    output->_sliced_with_dont_copy = dont_copy;
    return output;
}

Tensor* Tensor::ConcatenateOnRows(Tensor* other){
    if (other->get_shape().number_dims > 2) throw string("concate expects for the other tensor to be 1D or 2D tensor, but got ") + 
                                                  to_string(other->get_shape().number_dims) + 
                                                  string("D\n");
    if (_shape.number_dims > 2) throw string("concate expects for the current tensor to be 1D or 2D tensor, but got ") + 
                                      to_string(_shape.number_dims) + 
                                      string("D\n");
    if (_shape.number_dims != other->get_shape().number_dims)
        throw string("current concat and other tensor number of dimensions mismatch. ") +
              string("current: ") + to_string(_shape.number_dims) + string("D, ") +
              string("other: ") + to_string(other->get_shape().number_dims) + string("D");
    if(_shape.number_dims == 2)
        if (_shape.size[1] != other->get_shape().size[1])
            throw string("current concat and other tensor size of second dim mismatch. ") +
                  string("current: ") + to_string(_shape.size[1]) + string("D, ") +
                  string("other: ") + to_string(other->get_shape().size[1]) + string("D");
    if(_transposed)
        throw string("we don't support transposed matrix as the current matrix in concatenate yet");
    if(other->_transposed)
        throw string("we don't support transposed matrix as the other matrix in concatenate yet");
    Shape new_shape;
    int new_size[2];
    int n;
    if(_shape.number_dims == 1){
        new_size[0] = _shape.size[0] + other->get_shape().size[0];
        n = 1;
    }
    else if(_shape.number_dims == 2){
        new_size[0] = _shape.size[0] + other->get_shape().size[0];
        new_size[1] = _shape.size[1];
        n = 2;
    }
    else
        throw string("Wrong number of dimensions\n");
    new_shape = create_shape(new_size, n);
    Tensor* output = new Tensor(new_shape);
    output->Allocate();
    for (size_t i = 0; i < _shape.flatsize; i++)
        output->_data[i] = _data[i];
    output->_data += _shape.flatsize;
    for (size_t i = 0; i < other->get_shape().flatsize; i++)
        output->_data[i] = other->_data[i];
    output->_data -= _shape.flatsize;
    output->set_fill();
    output->Refresh();
    return output;
}

void    Tensor::SliceOnRowsInPlace(int start, int length){
    if (_shape.number_dims > 2) throw string("slice expects 1D or 2D tensor, but got ") + 
                                       to_string(_shape.number_dims) + 
                                       string("D\n");
    int new_size[2];
    int n;
    if(_shape.number_dims == 1){
        new_size[0] = length;
        n = 1;
    }
    else if(_shape.number_dims == 2){
        if(length == 1){
            new_size[0] = _shape.size[1];
            n = 1;
        }
        else{
            new_size[0] = length;
            new_size[1] = _shape.size[1];
            n = 2;
        }
    }
    else
        throw string("Wrong number of dimensions\n");
    Shape old_shape = _shape;
    Shape new_shape = create_shape(new_size, n);
    float* old_data = _data;
    float* old_data_T = _data_T;
    _shape = new_shape;
    if (_initialized && !_sliced_with_dont_copy){
        deallocate(_exponents, true);
        deallocate(_mantisas, true);
        deallocate(_signs, true);
    }
    bool T_initialized = _initialized;
    bool T_sliced_with_dont_copy = _sliced_with_dont_copy;
    _initialized = false;
    _filled = false;
    _extracted = false;
    Allocate();
    unsigned long start_idx;
    if(old_shape.number_dims == 1)
        start_idx = start;
    else
        start_idx = start * old_shape.size[1];
    if(_transposed)
        for (size_t i = 0; i < _shape.flatsize; i++)
            _data_T[i] = old_data_T[(start_idx + i) % old_shape.flatsize];
    else
        for (size_t i = 0; i < _shape.flatsize; i++)
            _data[i] = old_data[(start_idx + i) % old_shape.flatsize];
    if (T_initialized && !T_sliced_with_dont_copy){
        deallocate(old_data);
        deallocate(old_data_T);
    }
    set_fill();
    Refresh();
}

void    Tensor::ConcatenateOnRowsInPlace(Tensor* other){
    if (other->get_shape().number_dims > 2) throw string("concate expects for the other tensor to be 1D or 2D tensor, but got ") + 
                                                  to_string(other->get_shape().number_dims) + 
                                                  string("D\n");
    if (_shape.number_dims > 2) throw string("concate expects for the current tensor to be 1D or 2D tensor, but got ") + 
                                      to_string(_shape.number_dims) + 
                                      string("D\n");
    if (_shape.number_dims != other->get_shape().number_dims)
        throw string("current concat and other tensor number of dimensions mismatch. ") +
              string("current: ") + to_string(_shape.number_dims) + string("D, ") +
              string("other: ") + to_string(other->get_shape().number_dims) + string("D");
    if(_shape.number_dims == 2)
        if (_shape.size[1] != other->get_shape().size[1] && _shape.flatsize != 0)
            throw string("current concat and other tensor size of second dim mismatch. ") +
                  string("current: ") + to_string(_shape.size[1]) + string("D, ") +
                  string("other: ") + to_string(other->get_shape().size[1]) + string("D");
    if(_transposed)
        throw string("we don't support transposed matrix as the current matrix in concatenate yet");
    if(other->_transposed)
        throw string("we don't support transposed matrix as the other matrix in concatenate yet");
    Shape new_shape;
    int new_size[2];
    int n;
    if(_shape.number_dims == 1){
        new_size[0] = _shape.size[0] + other->get_shape().size[0];
        n = 1;
    }
    else if(_shape.number_dims == 2){
        new_size[0] = _shape.size[0] + other->get_shape().size[0];
        new_size[1] = other->get_shape().size[1];
        n = 2;
    }
    else
        throw string("Wrong number of dimensions\n");
    int prev_length = _shape.flatsize;
    _shape = create_shape(new_size, n);
    bool T_initialized = _initialized;
    bool T_sliced_with_dont_copy = _sliced_with_dont_copy;
    float* T_data = _data;
    float* T_data_T = _data_T;
    int8_t* T_exponents = _exponents;
    int32_t* T_mantisas = _mantisas;
    Sign_DataType* T_signs = _signs;
    int8_t* T_signs_8 = _signs_8;
    float16* T_data_float16 = _data_float16;
    _initialized = false;
    _filled = false;
    _extracted = false;
    Allocate();
    for (size_t i = 0; i < prev_length; i++)
        _data[i] = T_data[i];
    _data += prev_length;
    if (T_initialized && !T_sliced_with_dont_copy){
        deallocate(T_data);
        deallocate(T_data_T);
        deallocate(T_exponents);
        deallocate(T_mantisas);
        deallocate(T_signs);
        deallocate(T_signs_8);
        deallocate(T_data_float16);
    }
    for (size_t i = 0; i < other->get_shape().flatsize; i++)
        _data[i] = other->_data[i];
    _data -= prev_length;
    set_fill();
    Refresh();
}

void    Tensor::Flatten(){
    _shape.number_dims = 1;
    _shape.size[0] = _shape.flatsize;
}

void    Tensor::ExtendDimensions(){
    if (_shape.number_dims > 1)
        throw string("can not extend tensor with number of dimensions more than 1, got: ") +
              to_string(_shape.number_dims);
    if (_shape.number_dims == 1){
        _shape.number_dims = 2;
        delete[] _shape.size;
        _shape.size = new int[_shape.number_dims];
        _shape.size[0] = 1;
        _shape.size[1] = _shape.flatsize;
    }
    else if (_shape.number_dims == 0){
        _shape.number_dims = 1;
        _shape.size = new int[1] {0};
        _shape.flatsize = 0;
    }
    else
        throw string("undefined number of dimensions in");
}

void    Tensor::AccumulateOnRows(Tensor* other, unsigned int on_row){
    if (_shape.number_dims != 2) 
        throw string("accumulation base matrix must be 2D, got ") + to_string(_shape.number_dims);
    if (other->_shape.number_dims != 1)
        throw string("accumulation other matrix must be 1D, got ") + to_string(other->_shape.number_dims);
    if (other->_shape.size[0] != _shape.size[1])
        throw string("other matrix size must be equal to the size of second dimension of accumulation matrix.") +
              string(" accumulation matrix size: ") + get_shape_string(_shape) +
              string(", other matrix size: ") + get_shape_string(other->_shape);
    if (_transposed)
        throw string("we don't support transposed matrix as the accumulation matrix in AccumulateOnRows yet");
    if (other->_transposed)
        throw string("we don't support transposed matrix as the other matrix in AccumulateOnRows yet");
    if (on_row >= _shape.size[0])
        throw string("accumulation row invalid. Must be in the range of 0 to ") +
              to_string(_shape.size[0] - 1) + string(", but got: ") +
              to_string(on_row);
    int columns_length = _shape.size[1];
    float* offset_data = _data + on_row * columns_length;

    for (size_t i = 0; i < columns_length; i++)
        offset_data[i] += other->_data[i];
    set_fill();
    Refresh();
}

void    Tensor::MakeRuyMatrix(ruy::Matrix<float>* dst, bool use_caching) {
    if (_transposed)
        throw string("Can not Make Ruy Matrix with transoposed matrix");
    ruy::Order ruy_order = ruy::Order::kColMajor;
    if (_shape.number_dims == 2)
        ruy::MakeSimpleLayout(_shape.size[0], _shape.size[1], ruy_order,
                              dst->mutable_layout());
    else if (_shape.number_dims == 1)
        ruy::MakeSimpleLayout(1, _shape.size[0], ruy_order,
                              dst->mutable_layout());
    else
        throw string("Number of dimensions not 1 or 2 for ruy multiplication");
    dst->set_data(_data);
    if (use_caching)
        dst->set_cache_policy(ruy::CachePolicy::kAlwaysCache);
}

void    Tensor::MakeRuyMatrixInt8(ruy::Matrix<int8_t>* dst, bool use_caching) {
    if (_transposed)
        throw string("Can not Make Ruy Matrix with transoposed matrix");
    ruy::Order ruy_order = ruy::Order::kColMajor;
    if (_shape.number_dims == 2)
        ruy::MakeSimpleLayout(_shape.size[0], _shape.size[1], ruy_order,
                              dst->mutable_layout());
    else if (_shape.number_dims == 1)
        ruy::MakeSimpleLayout(1, _shape.size[0], ruy_order,
                              dst->mutable_layout());
    else
        throw string("Number of dimensions not 1 or 2 for ruy multiplication");
    dst->set_data(_exponents);
    if (use_caching)
        dst->set_cache_policy(ruy::CachePolicy::kAlwaysCache);
}

void    Tensor::MakeRuyMatrixInt32(ruy::Matrix<int32_t>* dst, bool use_caching) {
    if (_transposed)
        throw string("Can not Make Ruy Matrix with transoposed matrix");
    ruy::Order ruy_order = ruy::Order::kColMajor;
    if (_shape.number_dims == 2)
        ruy::MakeSimpleLayout(_shape.size[0], _shape.size[1], ruy_order,
                              dst->mutable_layout());
    else if (_shape.number_dims == 1)
        ruy::MakeSimpleLayout(1, _shape.size[0], ruy_order,
                              dst->mutable_layout());
    else
        throw string("Number of dimensions not 1 or 2 for ruy multiplication");
    dst->set_data(_mantisas);
    if (use_caching)
        dst->set_cache_policy(ruy::CachePolicy::kAlwaysCache);
}

template <typename T>
T* Tensor::_allocate(int size){
    return static_cast<T*>(new T[size]);
}
