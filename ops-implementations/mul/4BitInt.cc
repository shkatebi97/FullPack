#ifdef IS_ARM
#ifdef TFLITE_BUILD
#include <arm_neon.h>
#else
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

void do4BitMultiplication(int8_t* activation, 
                            int8_t* weights, 
                            int32_t& dst_1, int32_t& dst_2,
                            int32_t& dst_3, int32_t& dst_4, 
                            int size);


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                     Definitions                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////
/////////////// Common ///////////////
//////////////////////////////////////
Status Mul::Prepare_4Bit(DataType data_type){
    Shape k_shape = get_kernel_shape();
    if (k_shape.number_dims != 2)
        return Status::DimensionsMisMatch;
    if (k_shape.size[0] % 32)
        return Status::SizesMisMatch;
    if (data_type != DataType::Int8)
        return Status::WrongDataType;
    if (data_type == DataType::Int8){
        int new_weights_length = k_shape.size[1] * (k_shape.size[0] / 2);
        int8_t* weights_reference = _kernel->get_int8_ptr();
        int8_t* weights_packed = allocate<int8_t>(new_weights_length);
        for (int i = 0 ; i < k_shape.size[1] ; i++){
            for (int j = 0 ; j < k_shape.size[0] ; j++){
                int cluster_id = j / 32,
                    container_id = (j % 32) % 16,
                    shift_amount = (32 / 16 - 1) - ((j % 32) / 16);
                weights_packed[i * (k_shape.size[0]/8) + ((cluster_id * 16) + container_id)] |= 
                    (weights_reference[i * k_shape.size[0] + j] < 0)?
                    (
                        (weights_reference[i * k_shape.size[0] + j] < -8)?(
                            -8
                        ):(
                            weights_reference[i * k_shape.size[0] + j]
                        )
                    ):(
                        (weights_reference[i * k_shape.size[0] + j] >  7)?(
                            7
                        ):(
                            weights_reference[i * k_shape.size[0] + j]
                        )
                    )
                    << (shift_amount * 4);
            }
        }
        int8_t* tmp = allocate<int8_t>(new_weights_length);
        doLowPrecisionWeightPack(weights_packed, tmp, k_shape.size[1], k_shape.size[0] / 2);
        _temperories = new void*[2];
        _temperories[0] = get_pointer_as<void>(tmp);
        _temperories[1] = get_pointer_as<void>(weights_packed);
    }
    else
        throw;
    return Status::Success;
}

Status Mul::Init_4Bit(){return Status::Success;}

Status Mul::Free_4Bit(DataType data_type){
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


Status Mul::Eval_4Bit(int8_t* lhs,
                      int32_t* dst, 
                      int lhs_columns, int rhs_rows, int rhs_columns){
    if (lhs_columns != rhs_rows)
        return Status::SizesMisMatch;

    int8_t* rhs_packed = get_pointer_as<int8_t>(_temperories[0]);

    int i;
    for (i = 0 ; (i+4) <= rhs_columns ; i+=4){
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        dst[i + 3] = 0;
        do4BitMultiplication(
            lhs,
            rhs_packed, 
            dst[i], dst[i + 1], dst[i + 2], dst[i + 3],
            lhs_columns);
        rhs_packed += lhs_columns / 2;
    }
    i = rhs_columns - (i - 4);
    if (i == 1){
        dst[i] = 0;
        do4BitMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i], dst[i], dst[i],
            lhs_columns);
    }
    else if (i == 2){
        dst[i] = 0;
        dst[i + 1] = 0;
        do4BitMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 1], dst[i + 1],
            lhs_columns);
    }
    else if (i == 3){
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        do4BitMultiplication(
            lhs,
            rhs_packed,
            dst[i], dst[i + 1], dst[i + 2], dst[i + 2],
            lhs_columns);
    }
    return Status::Success;
}
#endif

void do4BitMultiplication(int8_t* activation, 
                          int8_t* weights, 
                          int32_t& dst_1, int32_t& dst_2,
                          int32_t& dst_3, int32_t& dst_4, 
                          int size){
    int i, end;
    int8_t mask = 0x0F;
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
        "dup v29.16b, %w[mask]\n\t"
        "dup v23.8h, wzr\n\t"
        "dup v24.8h, wzr\n\t"
        "dup v25.8h, wzr\n\t"
        "dup v30.8h, wzr\n\t"
        "mov %w[i], wzr\n\t"
        
        "cmp %w[size], #0\n\t"
        "beq 3f\n\t"

        // Start of Outer Loop Over Weights
        "1:\n\t"
        "ld1 {v0.16b},  [%[weights]], #16\n\t"
        "ld1 {v1.16b},  [%[weights]], #16\n\t"
        "ld1 {v2.16b},  [%[weights]], #16\n\t"
        "ld1 {v3.16b},  [%[weights]], #16\n\t"

        "ld1 {v4.16b},  [%[activation]], #16\n\t"
        "ld1 {v5.16b},  [%[activation]], #16\n\t"

        // AND WL, W, M
        "and v8.16b,  v0.16b, v29.16b\n\t"
        "and v10.16b, v1.16b, v29.16b\n\t"
        "and v11.16b, v2.16b, v29.16b\n\t"
        "and v12.16b, v3.16b, v29.16b\n\t"

        // SHL WL, WL, #4
        "shl v8.16b,  v8.16b,  #4\n\t"
        "shl v10.16b, v10.16b, #4\n\t"
        "shl v11.16b, v11.16b, #4\n\t"
        "shl v12.16b, v12.16b, #4\n\t"

        // SSHR WH, W, #4
        "sshr v0.16b, v0.16b, #4\n\t"
        "sshr v1.16b, v1.16b, #4\n\t"
        "sshr v2.16b, v2.16b, #4\n\t"
        "sshr v3.16b, v3.16b, #4\n\t"

        // SSHR WL, WL, #4
        "sshr v8.16b,  v8.16b,  #4\n\t"
        "sshr v10.16b, v10.16b, #4\n\t"
        "sshr v11.16b, v11.16b, #4\n\t"
        "sshr v12.16b, v12.16b, #4\n\t"

        // SMULL T1.8h, WL.8b, A1.8b
        "smull v13.8h, v8.8b,  v4.8b\n\t"
        "smull v14.8h, v10.8b, v4.8b\n\t"
        "smull v15.8h, v11.8b, v4.8b\n\t"
        "smull v16.8h, v12.8b, v4.8b\n\t"

        // SMLAL2 T1.8h, WL.16b, A1.16b
        "smlal2 v13.8h, v8.16b,  v4.16b\n\t"
        "smlal2 v14.8h, v10.16b, v4.16b\n\t"
        "smlal2 v15.8h, v11.16b, v4.16b\n\t"
        "smlal2 v16.8h, v12.16b, v4.16b\n\t"

        // SMULL T2.8h, WH.8b, A2.8b
        "smull v8.8h,  v0.8b, v5.8b\n\t"
        "smull v10.8h, v1.8b, v5.8b\n\t"
        "smull v11.8h, v2.8b, v5.8b\n\t"
        "smull v12.8h, v3.8b, v5.8b\n\t"

        // SMLAL2 T2.8h, WH.16b, A2.16b
        "smlal2 v8.8h,  v0.16b, v5.16b\n\t"
        "smlal2 v10.8h, v1.16b, v5.16b\n\t"
        "smlal2 v11.8h, v2.16b, v5.16b\n\t"
        "smlal2 v12.8h, v3.16b, v5.16b\n\t"

        // ACCUMULATE ACC, T1
        "sadalp v23.4s, v13.8h\n\t"
        "sadalp v24.4s, v14.8h\n\t"
        "sadalp v25.4s, v15.8h\n\t"
        "sadalp v30.4s, v16.8h\n\t"

        // ACCUMULATE ACC, T2
        "sadalp v23.4s, v8.8h\n\t"
        "sadalp v24.4s, v10.8h\n\t"
        "sadalp v25.4s, v11.8h\n\t"
        "sadalp v30.4s, v12.8h\n\t"

        "add %w[i], %w[i], #32\n\t"
        "cmp %w[i], %w[size]\n\t"
        "b.lt 1b\n\t"

        "addv s23, v23.4s\n\t"
        "addv s24, v24.4s\n\t"
        "addv s25, v25.4s\n\t"
        "addv s30, v30.4s\n\t"

        "mov w3, v23.s[0]\n\t"
        "mov w4, v24.s[0]\n\t"
        "mov w5, v25.s[0]\n\t"
        "mov w6, v30.s[0]\n\t"

        "add %w[dst_1], %w[dst_1], w3\n\t"
        "add %w[dst_2], %w[dst_2], w4\n\t"
        "add %w[dst_3], %w[dst_3], w5\n\t"
        "add %w[dst_4], %w[dst_4], w6\n\t"

        "3:\n\t"

        : [ dst_1 ] "=r"(dst_1), [ dst_2 ] "=r"(dst_2),
          [ dst_3 ] "=r"(dst_3), [ dst_4 ] "=r"(dst_4),
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
          "v28", "v29", "v30", "v31",
          "w3",  "w4",  "w5",  "w6"
    );
}

#else
Status Mul::Prepare_4Bit(DataType data_type){
    return Status::NotImplemented;
}
Status Mul::Init_4Bit(){
    return Status::NotImplemented;
}
Status Mul::Free_4Bit(DataType data_type){
    return Status::NotImplemented;
}
Status Mul::Eval_4Bit(int8_t* lhs,
                      int32_t* dst, 
                      int lhs_columns, int rhs_rows, int rhs_columns){
    return Status::NotImplemented;
}
void do4BitMultiplication(int8_t* activation, 
                          int8_t* weights, 
                          int32_t& dst_1, int32_t& dst_2,
                          int32_t& dst_3, int32_t& dst_4, 
                          int size){ }
#endif





