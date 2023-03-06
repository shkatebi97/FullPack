#include "Mul.h"
#include "LowPrecisionPacking.h"
#include <fstream>
#include <streambuf>


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                     Declaration                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////
///////////     Int8         /////////
//////////////////////////////////////

void doBinaryMultiplication1Row(int8_t* activation, 
                                int8_t* weights, 
                                int32_t& dst, int size);

void doBinaryMultiplication(int8_t* activation, 
                            int8_t* weights, 
                            int32_t& dst_1, int32_t& dst_2,
                            int32_t& dst_3, int32_t& dst_4, 
                            int size);

//////////////////////////////////////
/////////       Float32       ////////
//////////////////////////////////////

void doBinaryMultiplication(float* activation, 
                            int32_t* weights, 
                            float& dst_1, float& dst_2,
                            float& dst_3, float& dst_4, 
                            int size);

//////////////////////////////////////
/////////       Float16       ////////
//////////////////////////////////////

void doBinaryMultiplication(float16* activation, 
                            int16_t* weights, 
                            float16& dst_1, float16& dst_2,
                            float16& dst_3, float16& dst_4, 
                            int size);


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                     Definitions                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/////////////////////////////////////
/////////////// Common ///////////////
//////////////////////////////////////

Status Mul::Prepare_binary(DataType data_type){
    Shape k_shape = get_kernel_shape();
    if (k_shape.number_dims != 2)
        return Status::DimensionsMisMatch;
    if (k_shape.size[0] % 128)
        return Status::SizesMisMatch;
    if (data_type != DataType::Int8 && data_type != DataType::Float32 && data_type != DataType::Float16)
        return Status::WrongDataType;
    if (data_type == DataType::Int8){
        int new_weights_length = k_shape.size[1] * (k_shape.size[0] / 8);
        int8_t* weights_reference = _kernel->get_signs_8();
        int8_t* weights_packed = allocate<int8_t>(new_weights_length);
        for (int i = 0 ; i < k_shape.size[1] ; i++){
            for (int j = 0 ; j < k_shape.size[0] ; j++){
                int cluster_id = j / 128, 
                    container_id = (j % 128) % 16,
                    shift_amount = (128 / 16 - 1) - ((j % 128) / 16);
                weights_packed[i * (k_shape.size[0]/8) + ((cluster_id * 16) + container_id)] |= 
                    (weights_reference[i * k_shape.size[0] + j])?(1):(0) << shift_amount;
            }
        }
        #ifdef USE_SINGLE_ROW_BINARY_OP
        _temperories = new void*[1];
        _temperories[0] = get_pointer_as<void>(weights_packed);
        #else
        int8_t* tmp = allocate<int8_t>(new_weights_length);
        doLowPrecisionWeightPack(weights_packed, tmp, k_shape.size[1], k_shape.size[0]/8);
        _temperories = new void*[2];
        _temperories[0] = get_pointer_as<void>(tmp);
        _temperories[1] = get_pointer_as<void>(weights_packed);
        #endif
    }
    else if (data_type == DataType::Float32){
        #ifdef USE_SINGLE_ROW_BINARY_OP
        return Status::NotImplemented;
        #endif
        int new_weights_length = k_shape.size[1] * (k_shape.size[0] / 32);
        int8_t* weights_reference = _kernel->get_signs_8();
        int32_t* weights_packed = allocate<int32_t>(new_weights_length);
        for (int i = 0 ; i < k_shape.size[1] ; i++){
            for (int j = 0 ; j < k_shape.size[0] ; j++){
                int cluster_id = j / 128, 
                    container_id = (j % 128) % 4,
                    shift_amount = (128 / 4 - 1) - ((j % 128) / 4);
                weights_packed[i * (k_shape.size[0]/32) + ((cluster_id * 4) + container_id)] |= 
                    (weights_reference[i * k_shape.size[0] + j])?(1):(0) << shift_amount;
            }
        }
        int32_t* tmp = allocate<int32_t>(new_weights_length);
        doLowPrecisionWeightPack(weights_packed, tmp, k_shape.size[1], k_shape.size[0]/32);
        _temperories = new void*[2];
        _temperories[0] = get_pointer_as<void>(tmp);
        _temperories[1] = get_pointer_as<void>(weights_packed);
    }
    else if (data_type == DataType::Float16){
        #ifdef USE_SINGLE_ROW_BINARY_OP
        return Status::NotImplemented;
        #endif
        if (!check_for_fp16_support())
            return Status::NotSupported;
        int new_weights_length = k_shape.size[1] * (k_shape.size[0] / 16);
        int8_t* weights_reference = _kernel->get_signs_8();
        int16_t* weights_packed = allocate<int16_t>(new_weights_length);
        for (int i = 0 ; i < k_shape.size[1] ; i++){
            for (int j = 0 ; j < k_shape.size[0] ; j++){
                int cluster_id = j / 128, 
                    container_id = (j % 128) % 8,
                    shift_amount = (128 / 8 - 1) - ((j % 128) / 8);
                weights_packed[i * (k_shape.size[0]/16) + ((cluster_id * 8) + container_id)] |= 
                    (weights_reference[i * k_shape.size[0] + j])?(1):(0) << shift_amount;
            }
        }
        int16_t* tmp = allocate<int16_t>(new_weights_length);
        doLowPrecisionWeightPack(weights_packed, tmp, k_shape.size[1], k_shape.size[0]/16);
        _temperories = new void*[2];
        _temperories[0] = get_pointer_as<void>(tmp);
        _temperories[1] = get_pointer_as<void>(weights_packed);
    }
    else
        throw;
    return Status::Success;
}

Status Mul::Init_binary(){return Status::Success;}

Status Mul::Free_binary(DataType data_type){
    if (data_type == DataType::Int8){
        deallocate(get_pointer_as<int8_t>(_temperories[0]));
        #ifndef USE_SINGLE_ROW_BINARY_OP
        deallocate(get_pointer_as<int8_t>(_temperories[1]));
        #endif
    }
    else if (data_type == DataType::Float16){
        deallocate(get_pointer_as<int16_t>(_temperories[0]));
        deallocate(get_pointer_as<int16_t>(_temperories[1]));
    }
    else if (data_type == DataType::Float32){
        deallocate(get_pointer_as<int32_t>(_temperories[0]));
        deallocate(get_pointer_as<int32_t>(_temperories[1]));
    }
    else
        return Status::WrongDataType;
    deallocate(_temperories);
    return Status::Success;
}

//////////////////////////////////////
///////////      Int8       //////////
//////////////////////////////////////

Status Mul::Eval_binary(int8_t* lhs,
                        int32_t* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int8_t* rhs_packed = get_pointer_as<int8_t>(_temperories[0]);

    int i;
    #ifdef USE_SINGLE_ROW_BINARY_OP
    for (i = 0 ; (i+0) <= rhs_columns ; i+=1){
        doBinaryMultiplication1Row(lhs, rhs_packed, dst[i], lhs_columns);
        rhs_packed += lhs_columns / 8;
    }
    #else
    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        doBinaryMultiplication(
            lhs,
            rhs_packed, 
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += lhs_columns / 2;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    #endif
    return Status::Success;
}

void doBinaryMultiplication1Row(int8_t* activation, 
                                int8_t* weights, 
                                int32_t& dst, int size){
    int i, end;
    int8_t mask = 0x80;
    /* Vector assignments:
        * V_r  = v31,
        * V_rr = v30,
        * V_M  = v29,
        * V_W  = V0,
        * V_A  = V4,
        * V_MW = V8,
        * V_t  = V9
     */
    asm volatile(
        "dup v30.4s, wzr\n"
        "dup v29.16b, %w[mask]\n"
        "mov %w[i], wzr\n"
        
        "cmp %w[size], #0\n"
        "beq 3f\n"

        // Start of Outer Loop Over Weights
        "1:\n"
        "ld1 {v0.16b},  [%[weights]], #16\n"
        "ld1 {v4.16b},  [%[activation]], #16\n"

        "add %w[end], %w[i], #128\n"
        "tst %w[size], %w[end]\n"
        "csel %w[end], %w[end], %w[size], hs\n"

        "dup v31.8h, wzr\n"
        
        // Start of Inner Loop Over Activations
        "2:\n"

        "and v8.16b, v0.16b, v29.16b\n"
        "eor v9.16b, v4.16b, v8.16b\n"
        "ld1 {v4.16b},  [%[activation]], #16\n"
        "shl v0.16b, v0.16b, #1\n"
        "sadalp v31.8h, v9.16b\n"
        
        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[end]\n"
        "b.ls 2b\n"

        "sadalp v30.4s, v31.8h\n"

        "cmp %w[i], %w[size]\n"
        "b.ls 1b\n"

        "addv s30, v30.4s\n"
        "mov %w[dst], v30.s[0]\n"

        "3:\n"

        : [ dst ] "=g"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
        : [ activation ] "r"(activation), [ weights ] "r"(weights),
          [ size ] "r"(size), [ mask ] "r"(mask)
        : "v0",  "v1",  "v2",  "v3",
          "v4",  "v5",  "v6",  "v7",
          "v8",  "v9",  "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31"
    );
}

void doBinaryMultiplication(int8_t* activation, 
                            int8_t* weights, 
                            int32_t& dst_1, int32_t& dst_2,
                            int32_t& dst_3, int32_t& dst_4, 
                            int size){
    int i, end;
    int8_t mask = 0x80;
    /* Vector assignments:
        * V_r  = v26-28,31,
        * V_rr = v23-25,30,
        * V_M  = v29,
        * V_W  = V0-3,
        * V_A  = V4,
        * V_AP = V5,
        * V_MW = V8,10-12,
        * V_t  = V9,13-15
     */
    asm volatile(
        "dup v30.4s, wzr\n"
        "dup v29.16b, %w[mask]\n"
        "mov %w[i], wzr\n"
        
        "cmp %w[size], #0\n"
        "beq 3f\n"

        // Start of Outer Loop Over Weights
        "1:\n"
        "ld1 {v0.16b},  [%[weights]], #16\n"
        "ld1 {v1.16b},  [%[weights]], #16\n"
        "ld1 {v2.16b},  [%[weights]], #16\n"
        "ld1 {v3.16b},  [%[weights]], #16\n"

        "ld1 {v4.16b},  [%[activation]], #16\n"
        // Generate negate of activations
        "sqneg v5.16b,  v4.16b\n"

        "add %w[end], %w[i], #128\n"
        "tst %w[size], %w[end]\n"
        "csel %w[end], %w[end], %w[size], hs\n"

        "dup v31.8h, wzr\n"
        
        // Start of Inner Loop Over Activations
        "2:\n"

        // CMGE T1, W, #0
        "cmge v8.16b,  v0.16b, #0\n"
        "cmge v10.16b, v1.16b, #0\n"
        "cmge v11.16b, v2.16b, #0\n"
        "cmge v12.16b, v3.16b, #0\n"

        // AND At, T1, A
        "and v9.16b,  v8.16b,  v4.16b\n"
        "and v13.16b, v10.16b, v4.16b\n"
        "and v14.16b, v11.16b, v4.16b\n"
        "and v15.16b, v12.16b, v4.16b\n"

        "ld1 {v4.16b},  [%[activation]], #16\n"

        // NEG T1, T1
        "neg v8.16b,  v8.16b\n"
        "neg v10.16b, v10.16b\n"
        "neg v11.16b, v11.16b\n"
        "neg v12.16b, v12.16b\n"

        // AND APt, T1, AP
        "and v8.16b,  v8.16b,  v5.16b\n"
        "and v10.16b, v10.16b, v5.16b\n"
        "and v11.16b, v11.16b, v5.16b\n"
        "and v12.16b, v12.16b, v5.16b\n"

        // Generate negate of activations
        "sqneg v5.16b,  v4.16b\n"

        // ORR AT, At, APt
        "orr v9.16b,  v8.16b,  v9.16b\n"
        "orr v13.16b, v10.16b, v13.16b\n"
        "orr v14.16b, v11.16b, v14.16b\n"
        "orr v15.16b, v12.16b, v15.16b\n"

        "shl v0.16b, v0.16b, #1\n"
        "shl v1.16b, v1.16b, #1\n"
        "shl v2.16b, v2.16b, #1\n"
        "shl v3.16b, v3.16b, #1\n"

        "sadalp v26.8h, v9.16b\n"
        "sadalp v27.8h, v13.16b\n"
        "sadalp v28.8h, v14.16b\n"
        "sadalp v31.8h, v15.16b\n"
        
        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[end]\n"
        "b.ls 2b\n"

        "sadalp v23.4s, v26.8h\n"
        "sadalp v24.4s, v27.8h\n"
        "sadalp v25.4s, v28.8h\n"
        "sadalp v30.4s, v31.8h\n"

        "cmp %w[i], %w[size]\n"
        "b.ls 1b\n"

        "addv s23, v23.4s\n"
        "addv s24, v24.4s\n"
        "addv s25, v25.4s\n"
        "addv s30, v30.4s\n"

        "mov %w[dst_1], v23.s[0]\n"
        "mov %w[dst_2], v24.s[0]\n"
        "mov %w[dst_3], v25.s[0]\n"
        "mov %w[dst_4], v30.s[0]\n"

        "3:\n"

        : [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i), [ end ] "+r"(end)
        : [ activation ] "r"(activation), [ weights ] "r"(weights),
          [ size ] "r"(size), [ mask ] "r"(mask)
        : "v0",  "v1",  "v2",  "v3",
          "v4",  "v5",  "v6",  "v7",
          "v8",  "v9",  "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31"
    );
}

//////////////////////////////////////
/////////      Float32       /////////
//////////////////////////////////////

Status Mul::Eval_binary(float* lhs,
                        float* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int32_t* rhs_packed = get_pointer_as<int32_t>(_temperories[0]);
    
    // auto timer = SelfProfiler::get_main_profiler()->NewTimer(
    //     string("Eval_binary_float: ") + 
    //     to_string(lhs_columns) + "x" +
    //     to_string(rhs_rows) + "x" +
    //     to_string(rhs_columns)
    // );
    // timer->start();

    int i;
    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        doBinaryMultiplication(
            lhs,
            rhs_packed, 
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += lhs_columns / 8;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    else if (i == 4){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
    }
    // timer->end();
    return Status::Success;
}

void doBinaryMultiplication(float* activation, 
                            int32_t* weights, 
                            float& dst_1, float& dst_2,
                            float& dst_3, float& dst_4, 
                            int size){
    int i, end;
    int32_t mask = 0x80000000;
    /* Vector assignments:
        * V_rr = v23-25,30,
        * V_M  = v29,
        * V_W  = V0-3,
        * V_A  = V4,
        * V_MW = V8,10-12,
        * V_t  = V9,13-15
     */
    asm volatile(
        "dup v30.4s, wzr\n"
        "dup v29.4s, %w[mask]\n"
        "mov %w[i], wzr\n"
        
        "cmp %w[size], #0\n"
        "beq 3f\n"

        // Start of Outer Loop Over Weights
        "1:\n"
        "ld1 {v0.4s},  [%[weights]], #16\n"
        "ld1 {v1.4s},  [%[weights]], #16\n"
        "ld1 {v2.4s},  [%[weights]], #16\n"
        "ld1 {v3.4s},  [%[weights]], #16\n"

        "ld1 {v4.4s},  [%[activation]], #16\n"

        "add %w[end], %w[i], #128\n"
        "tst %w[size], %w[end]\n"
        "csel %w[end], %w[end], %w[size], hs\n"

        "dup v31.4s, wzr\n"
        
        // Start of Inner Loop Over Activations
        "2:\n"

        "and v8.16b,  v0.16b, v29.16b\n"
        "and v10.16b, v1.16b, v29.16b\n"
        "and v11.16b, v2.16b, v29.16b\n"
        "and v12.16b, v3.16b, v29.16b\n"

        "eor v9.16b,  v4.16b, v8.16b\n"
        "eor v13.16b, v4.16b, v10.16b\n"
        "eor v14.16b, v4.16b, v11.16b\n"
        "eor v15.16b, v4.16b, v12.16b\n"

        "ld1 {v4.4s},  [%[activation]], #16\n"

        "shl v0.4s, v0.4s, #1\n"
        "shl v1.4s, v1.4s, #1\n"
        "shl v2.4s, v2.4s, #1\n"
        "shl v3.4s, v3.4s, #1\n"

        "fadd v23.4s, v23.4s, v9.4s\n"
        "fadd v24.4s, v24.4s, v13.4s\n"
        "fadd v25.4s, v25.4s, v14.4s\n"
        "fadd v30.4s, v30.4s, v15.4s\n"
        
        "add %w[i], %w[i], #4\n"
        "cmp %w[i], %w[end]\n"
        "b.ls 2b\n"

        "cmp %w[i], %w[size]\n"
        "b.ls 1b\n"

        "fadd v23.4s, v23.4s, v23.4s\n"
        "fadd v24.4s, v24.4s, v24.4s\n"
        "fadd v25.4s, v25.4s, v25.4s\n"
        "fadd v30.4s, v30.4s, v30.4s\n"

        "fadd v23.4s, v23.4s, v23.4s\n"
        "fadd v24.4s, v24.4s, v24.4s\n"
        "fadd v25.4s, v25.4s, v25.4s\n"
        "fadd v30.4s, v30.4s, v30.4s\n"

        "mov %w[dst_1], v23.s[0]\n"
        "mov %w[dst_2], v24.s[0]\n"
        "mov %w[dst_3], v25.s[0]\n"
        "mov %w[dst_4], v30.s[0]\n"

        "3:\n"

        : [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i), [ end ] "+r"(end)
        : [ activation ] "r"(activation), [ weights ] "r"(weights),
          [ size ] "r"(size), [ mask ] "r"(mask)
        : "v0",  "v1",  "v2",  "v3",
          "v4",  "v5",  "v6",  "v7",
          "v8",  "v9",  "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31"
    );
}

//////////////////////////////////////
/////////      Float16       /////////
//////////////////////////////////////

Status Mul::Eval_binary(float16* lhs,
                        float16* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int16_t* rhs_packed = get_pointer_as<int16_t>(_temperories[0]);
    
    // auto timer = SelfProfiler::get_main_profiler()->NewTimer(
    //     string("Eval_binary_float16: ") + 
    //     to_string(lhs_columns) + "x" +
    //     to_string(rhs_rows) + "x" +
    //     to_string(rhs_columns)
    // );
    // timer->start();

    int i;
    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        doBinaryMultiplication(
            lhs,
            rhs_packed, 
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += lhs_columns / 8;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    else if (i == 4){
        doBinaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
    }
    // timer->end();
    return Status::Success;
}

void doBinaryMultiplication(float16* activation, 
                            int16_t* weights, 
                            float16& dst_1, float16& dst_2,
                            float16& dst_3, float16& dst_4, 
                            int size){
    int i, end;
    int16_t mask = 0x8000;
    /* Vector assignments:
        * V_rr = v23-25,30,
        * V_M  = v29,
        * V_W  = V0-3,
        * V_A  = V4,
        * V_MW = V8,10-12,
        * V_t  = V9,13-15
     */
    asm volatile(
        "dup v30.8h, wzr\n"
        "dup v29.8h, %w[mask]\n"
        "mov %w[i], wzr\n"
        
        "cmp %w[size], #0\n"
        "beq 3f\n"

        // Start of Outer Loop Over Weights
        "1:\n"
        "ld1 {v0.8h},  [%[weights]], #16\n"
        "ld1 {v1.8h},  [%[weights]], #16\n"
        "ld1 {v2.8h},  [%[weights]], #16\n"
        "ld1 {v3.8h},  [%[weights]], #16\n"

        "ld1 {v4.8h},  [%[activation]], #16\n"

        "add %w[end], %w[i], #128\n"
        "tst %w[size], %w[end]\n"
        "csel %w[end], %w[end], %w[size], hs\n"

        "dup v31.8h, wzr\n"
        
        // Start of Inner Loop Over Activations
        "2:\n"

        "and v8.16b,  v0.16b, v29.16b\n"
        "and v10.16b, v1.16b, v29.16b\n"
        "and v11.16b, v2.16b, v29.16b\n"
        "and v12.16b, v3.16b, v29.16b\n"

        "eor v9.16b,  v4.16b, v8.16b\n"
        "eor v13.16b, v4.16b, v10.16b\n"
        "eor v14.16b, v4.16b, v11.16b\n"
        "eor v15.16b, v4.16b, v12.16b\n"

        "ld1 {v4.8h},  [%[activation]], #16\n"

        "shl v0.8h, v0.8h, #1\n"
        "shl v1.8h, v1.8h, #1\n"
        "shl v2.8h, v2.8h, #1\n"
        "shl v3.8h, v3.8h, #1\n"

        "fadd v23.8h, v23.8h, v9.8h\n"
        "fadd v24.8h, v24.8h, v13.8h\n"
        "fadd v25.8h, v25.8h, v14.8h\n"
        "fadd v30.8h, v30.8h, v15.8h\n"
        
        "add %w[i], %w[i], #8\n"
        "cmp %w[i], %w[end]\n"
        "b.ls 2b\n"

        "cmp %w[i], %w[size]\n"
        "b.ls 1b\n"

        "fadd v23.8h, v23.8h, v23.8h\n"
        "fadd v24.8h, v24.8h, v24.8h\n"
        "fadd v25.8h, v25.8h, v25.8h\n"
        "fadd v30.8h, v30.8h, v30.8h\n"

        "fadd v23.8h, v23.8h, v23.8h\n"
        "fadd v24.8h, v24.8h, v24.8h\n"
        "fadd v25.8h, v25.8h, v25.8h\n"
        "fadd v30.8h, v30.8h, v30.8h\n"

        "fadd v23.8h, v23.8h, v23.8h\n"
        "fadd v24.8h, v24.8h, v24.8h\n"
        "fadd v25.8h, v25.8h, v25.8h\n"
        "fadd v30.8h, v30.8h, v30.8h\n"

        "mov v23.h[1], wzr\n"
        "mov v24.h[1], wzr\n"
        "mov v25.h[1], wzr\n"
        "mov v30.h[1], wzr\n"

        "mov %w[dst_1], v23.s[0]\n"
        "mov %w[dst_2], v24.s[0]\n"
        "mov %w[dst_3], v25.s[0]\n"
        "mov %w[dst_4], v30.s[0]\n"

        "3:\n"

        : [ dst_1 ] "=g"(dst_1), [ dst_2 ] "=g"(dst_2),
          [ dst_3 ] "=g"(dst_3), [ dst_4 ] "=g"(dst_4),
          [ i ] "+r"(i), [ end ] "+r"(end)
        : [ activation ] "r"(activation), [ weights ] "r"(weights),
          [ size ] "r"(size), [ mask ] "r"(mask)
        : "v0",  "v1",  "v2",  "v3",
          "v4",  "v5",  "v6",  "v7",
          "v8",  "v9",  "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31"
    );
}
