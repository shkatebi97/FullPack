#include "LSTM.h"

LSTM::LSTM(
    int num_batches, 
    Shape shape, 
    bool use_spilt_matrix,
    Method multiplication_method, 
    bool use_fused
){
    if (_shape.number_dims != 2) throw string("LSTM shape must be 2D. got: ") +
                                       to_string(_shape.number_dims);
    if (num_batches % 4) throw string("LSTM number of batches must be dividable to 4. got: ") +
                                       to_string(num_batches);
    _num_batches = num_batches;
    _shape = shape;
    _fc_multiplication_method = multiplication_method;
    _fc_use_fused = use_fused;
    single_batched_fully_connecteds.reserve(num_batches);
    _use_spilt_fc_matrix = use_spilt_matrix;
    _fc_used_shared_kernels = false;
    _fc_used_shared_kernels_copied = false;

    int num_output, num_input;
    num_output = _shape.size[1];
    num_input  = _shape.size[0];

    Shape state_h_shape;

    state_h_shape.number_dims = 1;
    state_h_shape.size = new int[state_h_shape.number_dims];
    state_h_shape.size[0] = num_output;
    state_h_shape.flatsize = state_h_shape.size[0];

    state_h = new Tensor(state_h_shape);

    state_h->Allocate();
    state_h->Fill_Random();
    state_h->Flatten();

    Shape single_batch_shape;

    if(!_use_spilt_fc_matrix){
        single_batch_shape.number_dims = 2;
        single_batch_shape.size = new int[single_batch_shape.number_dims];
        single_batch_shape.size[0] = num_input  * num_output;
        single_batch_shape.size[1] = num_output * 4;
        single_batch_shape.flatsize = single_batch_shape.size[0] * single_batch_shape.size[1];
        multi_batched_fully_connected = nullptr;
    }
    else{
        single_batch_shape.number_dims = 2;
        single_batch_shape.size = new int[single_batch_shape.number_dims];
        single_batch_shape.size[0] = num_output;
        single_batch_shape.size[1] = num_output * 4;
        single_batch_shape.flatsize = single_batch_shape.size[0] * single_batch_shape.size[1];

        Shape multi_batch_shape;
        multi_batch_shape.number_dims = 2;
        multi_batch_shape.size = new int[multi_batch_shape.number_dims];
        multi_batch_shape.size[0] = num_input;
        multi_batch_shape.size[1] = num_output * 4;
        multi_batch_shape.flatsize = multi_batch_shape.size[0] * multi_batch_shape.size[1];

        multi_batched_fully_connected = new Fully_Connected(
            multi_batch_shape,
            _fc_multiplication_method,
            _fc_use_fused
        );
        multi_batched_fully_connected->set_multi_batch();
    }
    for (int i = 0 ; i < _num_batches ; i++)
        single_batched_fully_connecteds.push_back(
            new Fully_Connected(
                single_batch_shape,
                _fc_multiplication_method,
                _fc_use_fused
            )
        );
}

LSTM::LSTM(
    int num_batches, 
    Tensor* kernel, 
    bool use_spilt_matrix,
    Method multiplication_method, 
    bool use_fused, 
    bool copy
){
    _shape = kernel->get_shape();
    if (_shape.number_dims != 2) throw string("LSTM shape must be 2D. got: ") +
                                       to_string(_shape.number_dims);
    if (num_batches % 4) throw string("LSTM number of batches must be dividable to 4. got: ") +
                               to_string(num_batches);
    if (_shape.size[1] % 4) throw string("LSTM's shared kernel's number of columns must be dividable to 4. got: ") +
                                  get_shape_string(_shape);
    if (_shape.size[0] > _shape.size[1] / 4) 
        throw string("LSTM shared kernels MUST have more rows than columns. got: ") +
              get_shape_string(_shape);
    _shape.size[1] /= 4;
    _shape.size[0] -= _shape.size[1];

    _num_batches = num_batches;
    _fc_multiplication_method = multiplication_method;
    _fc_use_fused = use_fused;
    single_batched_fully_connecteds.reserve(num_batches);
    _use_spilt_fc_matrix = use_spilt_matrix;
    _fc_used_shared_kernels = true;
    _fc_used_shared_kernels_copied = copy;

    int num_output, num_input;
    num_output = _shape.size[1];
    num_input  = _shape.size[0];

    Tensor *single_batch_kernel, *multi_batch_kernel;

    Shape state_h_shape;

    state_h_shape.number_dims = 1;
    state_h_shape.size = new int[state_h_shape.number_dims];
    state_h_shape.size[0] = num_output;
    state_h_shape.flatsize = state_h_shape.size[0];

    state_h = new Tensor(state_h_shape);
    state_h->Allocate();
    state_h->Fill_Random();
    // Just to make sure!
    state_h->Flatten();

    Shape single_batch_shape;

    if(!_use_spilt_fc_matrix){
        single_batch_kernel = kernel;
        multi_batched_fully_connected = nullptr;
    }
    else{
        single_batch_kernel = kernel->SliceOnRows(num_input, num_output);
        multi_batch_kernel = kernel->SliceOnRows(0, num_input);
        
        multi_batched_fully_connected = new Fully_Connected(
            multi_batch_kernel,
            _fc_multiplication_method,
            _fc_use_fused,
            copy
        );
        multi_batched_fully_connected->set_multi_batch();
    }
    for (int i = 0 ; i < _num_batches ; i++)
        single_batched_fully_connecteds.push_back(
            new Fully_Connected(
                kernel,
                _fc_multiplication_method,
                _fc_use_fused,
                copy
            )
        );
}

LSTM::~LSTM(){
    Tensor *single_batch_shared_kernels, *multi_batch_shared_kernels;
    single_batch_shared_kernels = nullptr;
    multi_batch_shared_kernels = nullptr;
    if (
        _fc_used_shared_kernels && 
        _use_spilt_fc_matrix && 
        !_fc_used_shared_kernels_copied
    ){
        single_batch_shared_kernels = multi_batched_fully_connected->_kernel;
        multi_batch_shared_kernels = single_batched_fully_connecteds[0]->_kernel;
    }
    if(_use_spilt_fc_matrix)
        delete multi_batched_fully_connected;
    while (!single_batched_fully_connecteds.empty()){
        delete single_batched_fully_connecteds.back();
        single_batched_fully_connecteds.pop_back();
    }
    if (
        _fc_used_shared_kernels && 
        _use_spilt_fc_matrix && 
        !_fc_used_shared_kernels_copied
    ){
        deallocate(single_batch_shared_kernels);
        deallocate(multi_batch_shared_kernels);
    }
}

Tensor* LSTM::operator()(Tensor* input){
    Shape input_shape = input->get_shape();
    if (input_shape.number_dims != 2 || 
        input_shape.size[0] != _num_batches || 
        input_shape.size[1] != _shape.size[0])
        throw string("Invalid input size for LSTM, expected (") +
              to_string(_num_batches) + " , " + to_string(_shape.size[0]) +
              string(") but got ") + get_shape_string(input_shape);
    Tensor* output = nullptr;
    throw string("Not Implemented!");
    if (_use_spilt_fc_matrix){
        Tensor* tempMid = nullptr;
        tempMid = (*multi_batched_fully_connected)(input);
        Tensor *tempOutput, *tempInput;
        for (int i = 0 ; i < single_batched_fully_connecteds.size() ; i++){
            tempInput = (*single_batched_fully_connecteds[i])(state_h);
            tempMid->AccumulateOnRows(tempInput, i);
            tempOutput = tempMid->SliceOnRows(i, 1);
            ApplyActivations(tempOutput);
            output->ConcatenateOnRowsInPlace(state_h);
            state_h->Flatten();
            delete tempInput;
            delete tempOutput;
        }
    }
    else{
        output = new Tensor;
        Tensor *tempInput, *tempOutput;
        for (int i = 0 ; i < single_batched_fully_connecteds.size() ; i++){
            tempInput  = input->SliceOnRows(i, 1);
            tempInput->Flatten();
            tempInput->ConcatenateOnRowsInPlace(state_h);
            tempOutput = (*single_batched_fully_connecteds[i])(tempInput);
            ApplyActivations(tempOutput);
            output->ConcatenateOnRowsInPlace(state_h);
            state_h->Flatten();
            delete tempInput;
            delete tempOutput;
        }
    }
    return output;
}

Tensor* LSTM::ApplyActivations(Tensor* gates){
    // The input shape is (4 * output_size)
    // The output must be in the shape of (1, output_size)
    // Also the output maybe written in state_h
    if (_shape.size[1] * 4 != gates->get_shape().flatsize)
        throw string("Activation size mismatch. expected (1, ") +
              to_string(_shape.size[1] * 4) + string(") but got ") +
              get_shape_string(gates->get_shape());
    state_h->ExtendDimensions();
    float *old_data = state_h->get_data();
    float *new_data = gates->get_data();
    for(int i = 0 ; i < _shape.size[1] ; i++)
        old_data[i] = new_data[i * 4];
    state_h->set_fill();
    state_h->Refresh();
    return state_h;
}

