#ifndef LSTM_H
#define LSTM_H

#include "mul/Tensor.h"
#include "fully-connected.h"
#include "../common/types.h"
#include <string>
#include <iostream>
#include <vector>

using std::vector;

class LSTM
{
private:
    int _num_batches;
    Shape _shape;
    bool _fc_used_shared_kernels;
    bool _fc_used_shared_kernels_copied;

    Method _fc_multiplication_method;
    bool _fc_use_fused;

    bool _use_spilt_fc_matrix;

    vector<Fully_Connected*> single_batched_fully_connecteds;
    Fully_Connected*         multi_batched_fully_connected;
    Tensor*                  state_h;
public:
    LSTM(int num_batches,
         Shape shape, 
         bool use_spilt_matrix = true,
         Method multiplication_method = Method::kFloatMultiplication, 
         bool use_fused = false);
    LSTM(int num_batches,
         Tensor* kernel, 
         bool use_spilt_matrix = true,
         Method multiplication_method = Method::kFloatMultiplication, 
         bool use_fused = false, 
         bool copy = false);
    ~LSTM();

    Tensor* operator()(Tensor* input);
private:
    Tensor* ApplyActivations(Tensor* gates);
};

#endif