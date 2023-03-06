#ifndef TENSOR_H
#define TENSOR_H
#include <string>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cstdlib>
#include "../../common/types.h"
#include "../../common/flags.h"
#include "../../profiler/profiler.h"
#include "ruy/ruy.h"

using std::string;
using std::to_string;

#ifdef USE_32BIT_SIGN
typedef uint32_t Sign_DataType;
#else
typedef uint8_t Sign_DataType;
#endif

typedef struct{
    float coeff;
    float max;
    float min;
} Float32ToInt8QunatizationParameters_t;


class Tensor
{
private:
    Shape                _shape;
    float*               _data;
    float*               _data_T;
    int8_t*              _exponents;
    int32_t*             _mantisas;
    Sign_DataType*       _signs;
    int8_t*              _signs_8;
    float16*             _data_float16;
    bool                 _initialized;
    bool                 _filled;
    bool                 _extracted;
    bool                 _transposed;
    unsigned long        id;
    bool                 _sliced_with_dont_copy;
    float                _float_to_int_coeff;
    float                _float_to_int_max;
    float                _float_to_int_min;
public:
    Tensor();
    Tensor(Shape);
    Tensor& operator=(const Tensor&);
    ~Tensor();
    Status Allocate();
    Status Fill_Random();
    Status Fill_Zero();
    Status Extract();
    Status Initialize();
    Status Transpose(bool extract = false);
    Status TransposeData();
    Status Refresh();
    Status DownCastInt32ToInt8();
    Status DownCastFloat32ToInt8();
    Status UpCastInt32ToFloat32();
    
    Float32ToInt8QunatizationParameters_t get_f32_i8_parameters();
    void set_f32_i8_parameters(const Float32ToInt8QunatizationParameters_t& i);

    // Utility operations
    Tensor* SliceOnRows(int start, int length, bool dont_copy=false);
    Tensor* ConcatenateOnRows(Tensor* other);
    void    SliceOnRowsInPlace(int start, int length);
    void    ConcatenateOnRowsInPlace(Tensor* other);
    void    AccumulateOnRows(Tensor* other, unsigned int on_row);
    void    Flatten();
    void    ExtendDimensions();
    void    MakeRuyMatrix(ruy::Matrix<float>* dst, bool use_caching = false);
    void    MakeRuyMatrixInt8(ruy::Matrix<int8_t>* dst, bool use_caching = false);
    void    MakeRuyMatrixInt32(ruy::Matrix<int32_t>* dst, bool use_caching = false);

    float*               get_data(){ if (_transposed) return _data_T; else return _data; }
    Float*               get_Float_data(){ if (_transposed) return get_pointer_as<Float>(_data_T); else return get_pointer_as<Float>(_data); }
    int8_t*              get_exponents(){ return _exponents; }
    int32_t*             get_mantisas(){ return _mantisas; }
    Sign_DataType*       get_signs(){ return _signs; }
    int8_t*              get_signs_8(){ return _signs_8; }
    Shape                get_shape(){ return _shape; }
    unsigned long        get_id(){ return id; }
    bool                 get_initialized(){return _initialized; }
    bool                 get_filled(){return _filled; }
    bool                 get_extracted(){return _extracted; }
    bool                 get_transpose(){return _transposed; }
    int8_t*              get_int8_ptr(){return _exponents; }
    int32_t*             get_int32_ptr(){return _mantisas; }
    float16*             get_data_float16(){ return _data_float16; }

    void                 set_fill(bool fill=true){ _filled = fill; }
    void                 set_extracted(bool extracted=true){ _extracted = extracted; }

    static unsigned long last_id;

    operator string() {
        string str = "";
        if (_shape.flatsize > 10){
            str += "Tensor(shape=";
            str += get_shape_string(_shape);
            str += ", data=[";
            if(_transposed){
                str += to_string(_data_T[0]) + ", ";
                str += to_string(_data_T[1]) + ", ";
                str += to_string(_data_T[2]) + ", ";
                str += to_string(_data_T[3]) + ", ";
                str += to_string(_data_T[4]) + ", ";
                str += "... , ";
                str += to_string(_data_T[_shape.flatsize - 5]) + ", ";
                str += to_string(_data_T[_shape.flatsize - 4]) + ", ";
                str += to_string(_data_T[_shape.flatsize - 3]) + ", ";
                str += to_string(_data_T[_shape.flatsize - 2]) + ", ";
                str += to_string(_data_T[_shape.flatsize - 1]) + ", ";
            }
            else{
                str += to_string(_data[0]) + ", ";
                str += to_string(_data[1]) + ", ";
                str += to_string(_data[2]) + ", ";
                str += to_string(_data[3]) + ", ";
                str += to_string(_data[4]) + ", ";
                str += "... , ";
                str += to_string(_data[_shape.flatsize - 5]) + ", ";
                str += to_string(_data[_shape.flatsize - 4]) + ", ";
                str += to_string(_data[_shape.flatsize - 3]) + ", ";
                str += to_string(_data[_shape.flatsize - 2]) + ", ";
                str += to_string(_data[_shape.flatsize - 1]) + ", ";
            }
            str += "])";
        }
        else if (_shape.flatsize != 0){
            str += "Tensor(shape=";
            str += get_shape_string(_shape);
            str += ", data=[";
            if(_transposed){
                for (int i = 0; i < _shape.flatsize; i++)
                    str += to_string(_data_T[i]) + ", ";
            }
            else{
                for (int i = 0; i < _shape.flatsize; i++)
                    str += to_string(_data[i]) + ", ";
            }
            str.pop_back();
            str.pop_back();
            str += "])";
        }
        else
            str += "Tensor(shape=(0), data=[])";
        return str;
    }

private:
    template <typename T> T* _allocate(int size);
    void copy_shape(const Shape& other);
};

#endif