#ifndef _4_BIT_SHIFT_H_
#include "../../common/types.h"
#include "LowPrecisionPacking.h"
#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>

#ifdef IS_ARM
#include <arm_neon.h>
#endif
// #define PRINT_VALUES true
namespace Shift4Bit {
    Status Prepare(const int8_t* weight, Shape k_shape, void** &output, DataType data_type, MemLayout layout = MemLayout::kRowMajor);
    Status Init();
    Status Free(DataType data_type, void** temperories);
    Status Eval(const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape);
    void kernel1Col(const int8_t* activation, 
                    int8_t* weights, 
                    int32_t* dst, 
                    int size);
    uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
}
#endif