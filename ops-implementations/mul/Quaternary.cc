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

void doQuaTernaryMultiplication(int8_t* activation, 
                            int8_t* weights, 
                            int32_t& dst_1, int32_t& dst_2,
                            int32_t& dst_3, int32_t& dst_4, 
                            int size);

//////////////////////////////////////
/////////       Float32       ////////
//////////////////////////////////////

void doQuaTernaryMultiplication(float* activation, 
                            int32_t* weights, 
                            float& dst_1, float& dst_2,
                            float& dst_3, float& dst_4, 
                            int size);

//////////////////////////////////////
/////////       Float16       ////////
//////////////////////////////////////

void doQuaTernaryMultiplication(float16* activation, 
                            int16_t* weights, 
                            float16& dst_1, float16& dst_2,
                            float16& dst_3, float16& dst_4, 
                            int size);


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                     Definitions                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////
/////////////// Common ///////////////
//////////////////////////////////////

Status Mul::Prepare_quaternary(DataType data_type){
    Shape k_shape = get_kernel_shape();
    if (k_shape.number_dims != 2)
        return Status::DimensionsMisMatch;
    if (k_shape.size[0] % 64)
        return Status::SizesMisMatch;
    if (data_type != DataType::Int8)
        return Status::WrongDataType;
    if (data_type == DataType::Int8){
        int new_weights_length = k_shape.size[1] * (k_shape.size[0] / 4);
        int8_t* weights_reference = _kernel->get_int8_ptr();
        int8_t* weights_packed = allocate<int8_t>(new_weights_length);
        for (int i = 0 ; i < k_shape.size[1] ; i++){
            for (int j = 0 ; j < k_shape.size[0] ; j++){
                int cluster_id = j / 64, 
                    container_id = (j % 64) % 16,
                    shift_amount = (64 / 16 - 1) - ((j % 64) / 16);
                weights_packed[i * (k_shape.size[0]/8) + ((cluster_id * 16) + container_id)] |= 
                    (weights_reference[i * k_shape.size[0] + j] < 0)?
                    (
                        (weights_reference[i * k_shape.size[0] + j] < -1)?(0b11):(0b10)
                    ):(
                        (weights_reference[i * k_shape.size[0] + j] >  1)?(0b01):(0b00)
                    )
                    << (shift_amount * 2);
            }
        }
        int8_t* tmp = allocate<int8_t>(new_weights_length);
        doLowPrecisionWeightPack(weights_packed, tmp, k_shape.size[1], k_shape.size[0] / 4);
        _temperories = new void*[2];
        _temperories[0] = get_pointer_as<void>(tmp);
        _temperories[1] = get_pointer_as<void>(weights_packed);
    }
    else
        throw;
    return Status::Success;
}

Status Mul::Init_quaternary(){return Status::Success;}

Status Mul::Free_quaternary(DataType data_type){
    if (data_type == DataType::Int8){
        deallocate(get_pointer_as<int8_t>(_temperories[0]));
        deallocate(get_pointer_as<int8_t>(_temperories[1]));
    }
    else if (data_type == DataType::Float16){
        return Status::NotSupported;
        deallocate(get_pointer_as<int16_t>(_temperories[0]));
        deallocate(get_pointer_as<int16_t>(_temperories[1]));
    }
    else if (data_type == DataType::Float32){
        return Status::NotSupported;
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

Status Mul::Eval_quaternary(int8_t* lhs,
                        int32_t* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int8_t* rhs_packed = get_pointer_as<int8_t>(_temperories[0]);

    int i;
    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        doQuaTernaryMultiplication(
            lhs,
            rhs_packed, 
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += lhs_columns / 2;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        doQuaTernaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        doQuaTernaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        doQuaTernaryMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    return Status::Success;
}

void doQuaTernaryMultiplication(int8_t* activation, 
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

        "add %w[end], %w[i], #64\n"
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

        // NEG T2, T1
        "neg v8.16b,  v8.16b\n"
        "neg v10.16b, v10.16b\n"
        "neg v11.16b, v11.16b\n"
        "neg v12.16b, v12.16b\n"

        // AND APt, T2, AP
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

        // SHL W, W, #1
        "shl v0.16b, v0.16b, #1\n"
        "shl v1.16b, v1.16b, #1\n"
        "shl v2.16b, v2.16b, #1\n"
        "shl v3.16b, v3.16b, #1\n"

        // USHR T3, W, #7
        "ushr v8.16b,  v0.16b, #7\n"
        "ushr v10.16b, v1.16b, #7\n"
        "ushr v11.16b, v2.16b, #7\n"
        "ushr v12.16b, v3.16b, #7\n"

        // SHL W, W, #1
        "shl v0.16b, v0.16b, #1\n"
        "shl v1.16b, v1.16b, #1\n"
        "shl v2.16b, v2.16b, #1\n"
        "shl v3.16b, v3.16b, #1\n"

        // SSHL AT, AT, T3
        "sshl v9.16b,  v9.16b,  v8.16b\n"
        "sshl v13.16b, v13.16b, v10.16b\n"
        "sshl v14.16b, v14.16b, v11.16b\n"
        "sshl v15.16b, v15.16b, v12.16b\n"
        
        // ACCUMULATE ACC, AT 
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


//////////////////////////////////////
/////////      Float16       /////////
//////////////////////////////////////



