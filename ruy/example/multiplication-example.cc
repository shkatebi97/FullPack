#include "ruy/ruy.h"
#include "ruy/context.h"
#include "ruy/path.h"
#include <iostream>
#include <string>

using namespace std;

class Shape
{
    long int id;
public:
    static long int last_id;
    int number_dims;
    int* size;
    int flatsize;

    Shape():number_dims(0),size(nullptr),flatsize(0){id = last_id; last_id++;}
    ~Shape(){}
    long int get_id(){return id;}
    void extend_dims(bool as_highest_dim=true){
        number_dims++;
        int* old_size = size;
        size = new int[number_dims];
        if(as_highest_dim){
            size[0] = 1;
            for (int i = 1; i < number_dims; i++)
                size[i] = old_size[i - 1];
        }
        else{
            size[number_dims - 1] = 1;
            for (int i = 0; i < number_dims - 1; i++)
                size[i] = old_size[i];
        }
    }
    Shape& operator=(const Shape& other){
        this->id = last_id;
        last_id++;
        this->number_dims = other.number_dims;
        this->flatsize = other.flatsize;
        this->size = new int[other.number_dims];
        for (int i = 0 ; i < other.number_dims ; i++)
            this->size[i] = other.size[i];
        return *this;
    }
    bool operator==(const Shape& other){
        if (this->number_dims != other.number_dims)
            return false;
        bool isEqual = true;
        for (int i = 0 ; i < number_dims ; i++)
            isEqual &= this->size[i] == other.size[i];
        return isEqual;
    }
    bool operator!=(const Shape& other){
        if (this->number_dims != other.number_dims)
            return true;
        bool isEqual = true;
        for (int i = 0 ; i < number_dims ; i++)
            isEqual &= this->size[i] == other.size[i];
        return !isEqual;
    }
};

inline Shape get_shape(int* in_shape, int n){
    Shape shape;
    shape.number_dims = n;
    shape.flatsize = 1;
    shape.size = new int[n];
    for (int i = 0; i < n; i++){
        shape.size[i] = in_shape[i];
        shape.flatsize *= shape.size[i];
    }
    return shape;
}

long int Shape::last_id = 0;

template <typename T>
T* allocate(int size){
    return new T[size];
}

int main(){
    int num_spaces = 40 - string("I8-I8").length();
    vector<char> spaces_vec(num_spaces, ' ');
    string spaces(spaces_vec.begin(), spaces_vec.end());
    // Setting Constant Values
    int num_batch               = 512,
        num_inputs              = 2048,
        num_output              = 2048;

    // Creating Size Arrays
    int _input_shape[2]         = {     1     , num_inputs },
        _activation_shape[2]    = {     1     , num_inputs },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs},
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs},
        _output_shape[2]        = {     1     , num_output },
        _output_shape_MB[2]     = { num_batch , num_output };

    // Creating Shapes
    Shape input_shape           = get_shape(_input_shape,           2),
          activation_shape      = get_shape(_activation_shape,      2),
          input_shape_MB        = get_shape(_input_shape_MB,        2),
          activation_shape_MB   = get_shape(_activation_MB_shape,   2),
          kernel_shape          = get_shape(_kernel_shape,          2),
          filter_shape          = get_shape(_filter_shape,          2),
          output_shape          = get_shape(_output_shape,          2),
          output_shape_MB       = get_shape(_output_shape_MB,       2);

    // Allocating Matrices
    int8_t*  input_data         = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = allocate<int8_t>(activation_shape_MB.flatsize);
    int8_t*  kernel_data        = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = allocate<int32_t>(output_shape_MB.flatsize);
    // Creating Context and Parameters
    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;
    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int32_t> ruy_dst;
    ruy::Matrix<int8_t> ruy_rhs_MB;
    ruy::Matrix<int32_t> ruy_dst_MB;

    // Creating Filter Matrix
    ruy::MakeSimpleLayout(
        kernel_shape.size[0], 
        kernel_shape.size[1], 
        ruy::Order::kRowMajor,
        ruy_lhs.mutable_layout()
    );
    ruy_lhs.set_data(kernel_data);
    ruy_lhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ///////////////////////////////////////////////////////////////
    //////////////////   Single Batch API Test   //////////////////
    ///////////////////////////////////////////////////////////////

    // Creating Output Matrix
    ruy::MakeSimpleLayout(
        output_shape.size[1],
        output_shape.size[0],
        ruy::Order::kColMajor,
        ruy_dst.mutable_layout()
    );
    ruy_dst.set_data(output_data);

    // Creating Input Matrix
    ruy::MakeSimpleLayout(
        activation_shape.size[1],
        activation_shape.size[0],
        ruy::Order::kColMajor,
        ruy_rhs.mutable_layout()
    );
    ruy_rhs.set_data(activation_data);

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);
    
    cout << "I8-I8" << " Mul API Single-Batch Test" << spaces << "=> \033[1m\033[32mPASSED\033[0m" << endl;

    ///////////////////////////////////////////////////////////////
    /////////////////    Multi Batch API Test    //////////////////
    ///////////////////////////////////////////////////////////////

    // Creating MultiBatch Output Matrix
    ruy::MakeSimpleLayout(
        output_shape_MB.size[1],
        output_shape_MB.size[0],
        ruy::Order::kColMajor,
        ruy_dst_MB.mutable_layout()
    );
    ruy_dst_MB.set_data(output_data_MB);

    // Creating MultiBatch Input Matrix
    ruy::MakeSimpleLayout(
        activation_shape_MB.size[1],
        activation_shape_MB.size[0],
        ruy::Order::kColMajor,
        ruy_rhs_MB.mutable_layout()
    );
    ruy_rhs_MB.set_data(activation_data_MB);

    ruy::Mul/*<ruy::Path::kNeon>*/(ruy_lhs, ruy_rhs_MB, ruy_mul_params, _ruy_context, &ruy_dst_MB);

    cout << "I8-I8" << " Mul API Multi-Batch Test" << spaces << "=> \033[1m\033[32mPASSED\033[0m" << endl;
}