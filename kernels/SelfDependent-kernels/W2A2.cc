#include "../../low_precision_fully_connected.h"

#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            namespace W2A2{ // TODO: Implement this.
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
                    if (k_shape.number_dims != 2)
                        return Status::DimensionsMisMatch;
                    if (k_shape.size[0] % 4)
                        return Status::SizesMisMatch;
                    if (layout != MemLayout::kRowMajor)
                        return Status::WrongMemLayout;
                    if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                        doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0] / 4, k_shape.size[1]);
                    }
                    else {
                        int new_weights_length = (k_shape.size[0] / 4) * k_shape.size[1];
                        int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                        uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                        int i , size = k_shape.flatsize;
                        const uint8_t* input_u = get_pointer_as<uint8_t>(input);
                        
                        #if SelfDependent_Type == SelfDependent_Offset_Vector_Size
                        return Status::NotImplemented;
                        #elif SelfDependent_Type == SelfDependent_Continious
                        size_t z = 0;
                        for (size_t i = 4; i < k_shape.size[0]; i += 4){
                            for (size_t j = 0; j < k_shape.size[1]; j++){
                                temp_u[j * k_shape.size[0] / 4 + z] = (input_u[(i - 4) * k_shape.size[1] + j] & 0x03) | 
                                                                     ((input_u[(i - 3) * k_shape.size[1] + j] & 0x03) << 2) |
                                                                     ((input_u[(i - 2) * k_shape.size[1] + j] & 0x03) << 4) | 
                                                                     ((input_u[(i - 1) * k_shape.size[1] + j] & 0x03) << 6);
                            }
                            z++;
                        }
                        if (k_shape.size[0] % 4 == 1)
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[j * k_shape.size[0] / 4 + z] = input_u[(k_shape.size[0] - 1) * k_shape.size[1] + j] & 0x03;
                        else if (k_shape.size[0] % 4 == 2)
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[j * k_shape.size[0] / 4 + z] = (input_u[(k_shape.size[0] - 2) * k_shape.size[1] + j] & 0x03) |
                                                                     ((input_u[(k_shape.size[0] - 1) * k_shape.size[1] + j] & 0x03) << 2);
                        else if (k_shape.size[0] % 4 == 3)
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[j * k_shape.size[0] / 4 + z] = (input_u[(k_shape.size[0] - 3) * k_shape.size[1] + j] & 0x03) |
                                                                     ((input_u[(k_shape.size[0] - 2) * k_shape.size[1] + j] & 0x03) << 2) |
                                                                     ((input_u[(k_shape.size[0] - 1) * k_shape.size[1] + j] & 0x03) << 4);
                        else
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[j * k_shape.size[0] / 4 + z] = (input_u[(k_shape.size[0] - 4) * k_shape.size[1] + j] & 0x03) |
                                                                     ((input_u[(k_shape.size[0] - 3) * k_shape.size[1] + j] & 0x03) << 2) |
                                                                     ((input_u[(k_shape.size[0] - 2) * k_shape.size[1] + j] & 0x03) << 4) |
                                                                     ((input_u[(k_shape.size[0] - 1) * k_shape.size[1] + j] & 0x03) << 6);
                        #endif
                        Shape k_shape_T;
                        k_shape_T = k_shape.T();
                        doLowPrecisionWeightPack(temp, output, k_shape_T.size[0], k_shape_T.size[1] / 4);
                        LowPrecision::deallocate(temp);
                    }
                    return Status::Success;
                }
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){
                    return LowPrecision::Status::NotUpdated;
                    if (k_shape.number_dims != 2)
                        return Status::DimensionsMisMatch;
                    if (k_shape.size[0] % 4)
                        return Status::SizesMisMatch;
                    if (layout != MemLayout::kRowMajor)
                        return Status::WrongMemLayout;
                    if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                        // doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 2);
                    }
                    else {
                        // TODO: `FilterPackingStep` is not implemeneted without cvector, which is deprecated.
                        return LowPrecision::Status::NotImplemented;
                        uint8_t* temp = output;
                        uint8_t* input_u = const_cast<uint8_t*>(input);
                        int i , K = k_shape.size[0], N = k_shape.size[1];
                        for (int i = 0 ; i < N ; i += 8){
                            FilterPackingStep(input_u + i, output + (i * (K / 2)), K, N);
                        }
                    }
                    return Status::Success;
                }
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){
                    #if SelfDependent_LHS_Packing != SelfDependent_Simple_Packing
                    if (shape.size[1] % 4)
                        return Status::SizesMisMatch;
                    #endif 
                    if (layout != MemLayout::kRowMajor)
                        return Status::WrongMemLayout;
                    bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                    if (is_multibatched && shape.size[0] % 4)
                        return Status::SizesMisMatch; 
                    if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                        GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                        std::copy(input, input + (shape.flatsize / 4), output);
                    }
                    else {
                        #if SelfDependent_LHS_Packing != SelfDependent_ASM_TLB_Packing

                        return Status::NotImplemented;
                        int8_t* temp = output;
                        if (is_multibatched){
                            int new_weights_length = ((int)shape.flatsize / 2);
                            temp = LowPrecision::allocate<int8_t>(new_weights_length);
                        }
                        int i, j , size = shape.flatsize;
                        uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                        const uint8_t* input_u = get_pointer_as<uint8_t>(input);

                        #endif
                        #if SelfDependent_Type == SelfDependent_Offset_Vector_Size
                        return Status::NotImplemented;
                        asm volatile(
                            "mov %w[i], wzr\n\t"
                            "movi v31.16b, #15\n\t"
                            "movi v30.16b, #7\n\t"
                            "movi v29.16b, #248\n\t"
                            
                            "cmp %w[size], #0\n\t"
                            "beq 3f\n\t"

                            // Start of Outer Loop Over Weights
                            "1:\n\t"
                            "ld1 {v0.16b},  [%[input]], #16\n\t"
                            "ld1 {v1.16b},  [%[input]], #16\n\t"

                            // Saturate to 7
                            // CMGE AT, A, 7
                            "cmge v2.16b, v0.16b, v30.16b\n\t"
                            "cmge v3.16b, v1.16b, v30.16b\n\t"
                            // NOT AT, AT
                            "not v2.16b, v2.16b\n\t"
                            "not v3.16b, v3.16b\n\t"
                            // AND A, A, AT
                            "and v0.16b, v0.16b, v2.16b\n\t"
                            "and v1.16b, v1.16b, v3.16b\n\t"
                            // NOT ATT, AT
                            "not v2.16b, v2.16b\n\t"
                            "not v3.16b, v3.16b\n\t"
                            // AND ATTT, A, ATT
                            "and v2.16b, v30.16b, v2.16b\n\t"
                            "and v3.16b, v30.16b, v3.16b\n\t"
                            // ORR A, A, ATT
                            "orr v0.16b, v0.16b, v2.16b\n\t"
                            "orr v1.16b, v1.16b, v3.16b\n\t"
                            
                            // Saturate to -8
                            // CMGE AT, A, -8
                            "cmge v2.16b, v0.16b, v29.16b\n\t"
                            "cmge v3.16b, v1.16b, v29.16b\n\t"
                            // AND A, A, AT
                            "and v0.16b, v0.16b, v2.16b\n\t"
                            "and v1.16b, v1.16b, v3.16b\n\t"
                            // NOT ATT, AT
                            "not v2.16b, v2.16b\n\t"
                            "not v3.16b, v3.16b\n\t"
                            // AND ATTT, A, ATT
                            "and v2.16b, v29.16b, v2.16b\n\t"
                            "and v3.16b, v29.16b, v3.16b\n\t"
                            // ORR A, A, ATTT
                            "orr v0.16b, v0.16b, v2.16b\n\t"
                            "orr v1.16b, v1.16b, v3.16b\n\t"

                            // Pack 2 Saturated Int4 in 1 Int8
                            // AND AL, AL, 0x0F
                            "and v0.16b, v0.16b, v31.16b\n\t"
                            // SHL AH, AH, #4
                            "shl v1.16b, v1.16b, #4\n\t"
                            // ORR AM, AL, Ah
                            "orr v0.16b, v0.16b, v1.16b\n\t"
                            // ST1 AM, output
                            "st1 {v0.16b},  [%[output]], #16\n\t"

                            "add %w[i], %w[i], #32\n\t"
                            "cmp %w[i], %w[size]\n\t"
                            "b.lt 1b\n\t"

                            "sub %[input], %[input], %[size]\n\t"
                            "sub %[output], %[output], %[size], asr #1\n\t"

                            "3:\n\t"

                            : [ i ] "+r"(i)
                            : [ input ]  "r" (input), [ size ] "r"(size), [ output ] "r"(temp)
                            : "v0",  "v1",  "v2",  "v3",
                            "v28", "v29", "v30", "v31",
                            "w3",  "w4",  "w5",  "w6"
                        );
                        #elif SelfDependent_Type == SelfDependent_Continious
                        #if SelfDependent_LHS_Packing == SelfDependent_Simple_Packing
                        
                        return Status::NotImplemented;
                        if (shape.size[1] % 2)
                            for (size_t i = 0; i < shape.size[0]; i++){
                                size_t z = 0;
                                for (size_t j = 0; j < shape.size[1] - 1; j += 2){
                                    temp_u[i * (shape.size[1] / 2) + z++] = (input_u[i * shape.size[1] + (j    )] & 0x0F) | 
                                                                           ((input_u[i * shape.size[1] + (j + 1)] & 0x0F) << 4);
                                }
                                temp_u[i * (shape.size[1] / 2) + z++] =      input_u[(i + 1) * shape.size[1] - 1] & 0x0F;
                            }
                        else
                            for (size_t i = 0; i < shape.size[0]; i++){
                                size_t z = 0;
                                for (size_t j = 0; j < shape.size[1]; j += 2){
                                    temp_u[i * (shape.size[1] / 2) + z++] = (input_u[i * shape.size[1] + (j    )] & 0x0F) | 
                                                                           ((input_u[i * shape.size[1] + (j + 1)] & 0x0F) << 4);
                                }
                            }
                        if (is_multibatched){
                            doLowPrecisionWeightPack(temp, output, shape.size[0], shape.size[1] / 2);
                            LowPrecision::deallocate(temp);
                        }

                        #elif SelfDependent_LHS_Packing == SelfDependent_ASM_Packing
                        
                        return Status::NotImplemented;
                        /*  x0  <-  i
                            x1  <-  j       (Not using currently)
                            x2  <-  K/2     (Not using currently)
                            X3  <-  M * K
                            X4  <-  input
                            X5  <-  output
                            x6  <-  In_data_1
                            x7  <-  In_data_2
                            x8  <-  Out_data
                            x9  <-  Mask
                            x10 <-  Temporary_1
                        */
                        #define SelfDependent_InputPacking_PackSingleElement_ShiftRight(num_shift)  \
                            "and x10, x6, x9\n\t"                                                   \
                            "lsr x10, x10, #" #num_shift "\n\t"                                     \
                            "orr x8,  x8,  x10\n\t"                                                 \
                            "lsl x9,  x9,  #8\n\t"
                        #define SelfDependent_InputPacking_PackSingleElement_ShiftLeft(num_shift)   \
                            "and x10, x7, x9\n\t"                                                   \
                            "lsl x10, x10, #" #num_shift "\n\t"                                     \
                            "orr x8,  x8,  x10\n\t"                                                 \
                            "lsl x9,  x9,  #8\n\t"
                        asm volatile(
                            "mov x0, xzr\n\t"
                            "mov x1, xzr\n\t"
                            "mov x2, xzr\n\t"
                            "add x2, x2, %x[K], asr #1\n\t"
                            "mul x3, %x[K], %x[M]\n\t"
                            "mov x4, %x[input]\n\t"
                            "mov x5, %x[output]\n\t"

                            // Load first 64-bit data
                            "ldr x6, [x4], #8\n\t"
                            
                            // Main Loop
                            "0:\n\t"

                            // Reset Mask
                            "mov x9, #0x0F\n\t"

                            // O[0] |= (I[7:0] & 0x000000000000000F) >> (0 * 4 = 0)
                            "and x8, x6, x9\n\t"
                            "lsl x9,  x9,  #8\n\t"

                            // O[0] |= (I[7:0] & 0x0000000000000F00) >> (1 * 4 = 4)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(4)

                            // O[1] |= (I[7:0] & 0x00000000000F0000) >> (2 * 4 = 8)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(8)

                            // O[1] |= (I[7:0] & 0x000000000F000000) >> (3 * 4 = 12)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(12)

                            // Load second 64-bit data
                            "ldr x7, [x4], #8\n\t"

                            // O[2] |= (I[7:0] & 0x0000000F00000000) >> (4 * 4 = 16)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(16)

                            // O[2] |= (I[7:0] & 0x00000F0000000000) >> (5 * 4 = 20)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(20)

                            // O[3] |= (I[7:0] & 0x000F000000000000) >> (6 * 4 = 24)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(24)

                            // O[3] |= (I[7:0] & 0x0F00000000000000) >> (7 * 4 = 28)
                            SelfDependent_InputPacking_PackSingleElement_ShiftRight(28)
                            
                            // Second 64-bit Data
                            // Reset Mask
                            "mov x9, #0x0F\n\t"

                            // O[4] |= (I[7:0] & 0x000000000000000F) << (8 * 4 = 32)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(32)

                            // O[4] |= (I[7:0] & 0x0000000000000F00) >> (7 * 4 = 28)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(28)

                            // O[5] |= (I[7:0] & 0x00000000000F0000) >> (6 * 4 = 24)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(24)

                            // O[5] |= (I[7:0] & 0x000000000F000000) >> (5 * 4 = 20)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(20)

                            // Load second 64-bit data
                            "ldr x6, [x4], #8\n\t"

                            // O[6] |= (I[7:0] & 0x0000000F00000000) >> (4 * 4 = 16)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(16)

                            // O[6] |= (I[7:0] & 0x00000F0000000000) >> (3 * 4 = 12)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(12)

                            // O[7] |= (I[7:0] & 0x000F000000000000) >> (2 * 4 = 8)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(8)

                            // O[7] |= (I[7:0] & 0x0F00000000000000) >> (1 * 4 = 4)
                            SelfDependent_InputPacking_PackSingleElement_ShiftLeft(4)

                            // Store 64-bit data
                            "str x8, [x5], #8\n\t"

                            "add x0, x0, #16\n\t"
                            "cmp x0, x3\n\t"
                            "b.lt 0b\n\t"

                            : 
                            : [ input ]  "r" (input), [ K ] "r"(shape.size[1]), [ M ] "r"(shape.size[0]), [ output ] "r"(temp)
                            : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "v0", "v1", "v2"
                        );
                        if (is_multibatched){
                            doLowPrecisionWeightPack(temp, output, shape.size[0], shape.size[1] / 2);
                            LowPrecision::deallocate(temp);
                        }
                        #undef SelfDependent_InputPacking_PackSingleElement_ShiftRight
                        #undef SelfDependent_InputPacking_PackSingleElement_ShiftLeft

                        #elif SelfDependent_LHS_Packing == SelfDependent_ASM_TLB_Packing
                        
                        uint64_t mask_half_1 = 0x1C1814100C080400,
                                 mask_half_2 = 0x3C3834302C282420,
                                 mask_add    = 0x0101010101010101;
                        /*  
                            x0  <-  i
                            x1  <-  j
                            x2  <-  K/2     (Not using currently)
                            X3  <-  M * K   (Not using currently)
                            X4  <-  input
                            X5  <-  output
                            x6  <-  Mask_Lower_Half  Register
                            x7  <-  Mask_Higher_Half Register
                            x8  <-  Input Pointer #1
                            x9  <-  Input Pointer #2
                            x10 <-  Input Pointer #3
                            x11 <-  Input Pointer #4
                            -------------------------------------------------------------
                            v0  <-  Input Row #1 First 16-bytes Vector
                            v4  <-  Input Row #2 First 16-bytes Vector
                            v8  <-  Input Row #3 First 16-bytes Vector
                            v12 <-  Input Row #4 First 16-bytes Vector
                            v1  <-  Input Row #1 Second 16-bytes Vector
                            v5  <-  Input Row #2 Second 16-bytes Vector
                            v9  <-  Input Row #3 Second 16-bytes Vector
                            v13 <-  Input Row #4 Second 16-bytes Vector
                            v2  <-  Input Row #1 Third 16-bytes Vector
                            v6  <-  Input Row #2 Third 16-bytes Vector
                            v10 <-  Input Row #3 Third 16-bytes Vector
                            v14 <-  Input Row #4 Third 16-bytes Vector
                            v3  <-  Input Row #1 Fourth 16-bytes Vector
                            v7  <-  Input Row #2 Fourth 16-bytes Vector
                            v11 <-  Input Row #3 Fourth 16-bytes Vector
                            v15 <-  Input Row #4 Fourth 16-bytes Vector
                            v16 <-  Output Row #1 Vector
                            v17 <-  Output Row #2 Vector
                            v18 <-  Output Row #3 Vector
                            v19 <-  Output Row #4 Vector
                            v20 <-  Extracted Row #1 Temporary Values
                            v21 <-  Extracted Row #2 Temporary Values
                            v22 <-  Extracted Row #3 Temporary Values
                            v23 <-  Extracted Row #4 Temporary Values
                            v24 <-  Mask 0 Vector (0,4,...,60)
                            v25 <-  Mask 1 Vector (1,5,...,61)
                            v26 <-  Mask 2 Vector (2,6,...,62)
                            v27 <-  Mask 3 Vector (3,7,...,63)
                            v28 <-  AND Mask
                        */
                        asm volatile(
                            "mov x0, xzr\n\t"
                            "mov x1, xzr\n\t"
                            "mov x2, xzr\n\t"
                            "add x2, x2, %x[K], asr #2\n\t"
                            "mul x3, %x[K], %x[M]\n\t"
                            "mov x4, %x[input]\n\t"
                            "mov x5, %x[output]\n\t"

                            // Preparing Input Pointers
                            "mov x8,  %x[input]\n\t"
                            "add x9,  x8,  %x[K]\n\t"
                            "add x10, x9,  %x[K]\n\t"
                            "add x11, x10, %x[K]\n\t"

                            // Preparing AND Mask
                            "movi v28.16b, #0x03\n\t"
                            
                            // Preparing Masks
                            // Preparing 0 Values Mask
                            "mov x6, %[MH1]\n\t"
                            "mov x7, %[MH2]\n\t"

                            // 0 Values Mask Vector
                            "mov v24.d[0], x6\n\t"
                            "mov v24.d[1], x7\n\t"

                            // Preparing 1 Values Mask
                            "add x6, x6, %[MA]\n\t"
                            "add x7, x7, %[MA]\n\t"

                            // 1 Values Mask Vector
                            "mov v25.d[0], x6\n\t"
                            "mov v25.d[1], x7\n\t"

                            // Preparing 2 Values Mask
                            "add x6, x6, %[MA]\n\t"
                            "add x7, x7, %[MA]\n\t"

                            // 2 Values Mask Vector
                            "mov v26.d[0], x6\n\t"
                            "mov v26.d[1], x7\n\t"

                            // Preparing 3 Values Mask
                            "add x6, x6, %[MA]\n\t"
                            "add x7, x7, %[MA]\n\t"

                            // 3 Values Mask Vector
                            "mov v27.d[0], x6\n\t"
                            "mov v27.d[1], x7\n\t"

                            // Outer Loop
                            "0:\n\t"

                            // Load first 512-bit data
                            "ld1 {v0.16b,  v1.16b,  v2.16b,  v3.16b},  [x8],  #64\n\t"
                            "ld1 {v4.16b,  v5.16b,  v6.16b,  v7.16b},  [x9],  #64\n\t"
                            "ld1 {v8.16b,  v9.16b,  v10.16b, v11.16b}, [x10], #64\n\t"
                            "ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x11], #64\n\t"
                            
                            // Main Loop (Inner)
                            "1:\n\t"

                            // Extract 0 Values
                            "tbl v16.16b, {v0.16b,  v1.16b,  v2.16b,  v3.16b},  v24.16b\n\t"
                            "tbl v17.16b, {v4.16b,  v5.16b,  v6.16b,  v7.16b},  v24.16b\n\t"
                            "tbl v18.16b, {v8.16b,  v9.16b,  v10.16b, v11.16b}, v24.16b\n\t"
                            "tbl v19.16b, {v12.16b, v13.16b, v14.16b, v15.16b}, v24.16b\n\t"

                            // Mask 0 Values
                            "and v16.16b, v16.16b, v28.16b\n\t"
                            "and v17.16b, v17.16b, v28.16b\n\t"
                            "and v18.16b, v18.16b, v28.16b\n\t"
                            "and v19.16b, v19.16b, v28.16b\n\t"

                            // Extract 1 Values
                            "tbl v20.16b, {v0.16b,  v1.16b,  v2.16b,  v3.16b},  v25.16b\n\t"
                            "tbl v21.16b, {v4.16b,  v5.16b,  v6.16b,  v7.16b},  v25.16b\n\t"
                            "tbl v22.16b, {v8.16b,  v9.16b,  v10.16b, v11.16b}, v25.16b\n\t"
                            "tbl v23.16b, {v12.16b, v13.16b, v14.16b, v15.16b}, v25.16b\n\t"

                            // Mask 1 Values
                            "and v20.16b, v20.16b, v28.16b\n\t"
                            "and v21.16b, v21.16b, v28.16b\n\t"
                            "and v22.16b, v22.16b, v28.16b\n\t"
                            "and v23.16b, v23.16b, v28.16b\n\t"

                            // Shift 1 Values
                            "shl v20.16b, v20.16b, #2\n\t"
                            "shl v21.16b, v21.16b, #2\n\t"
                            "shl v22.16b, v22.16b, #2\n\t"
                            "shl v23.16b, v23.16b, #2\n\t"

                            // Merge 1 Values into the Output
                            "orr v16.16b, v16.16b, v20.16b\n\t"
                            "orr v17.16b, v17.16b, v21.16b\n\t"
                            "orr v18.16b, v18.16b, v22.16b\n\t"
                            "orr v19.16b, v19.16b, v23.16b\n\t"

                            // Extract 2 Values
                            "tbl v20.16b, {v0.16b,  v1.16b,  v2.16b,  v3.16b},  v26.16b\n\t"
                            "tbl v21.16b, {v4.16b,  v5.16b,  v6.16b,  v7.16b},  v26.16b\n\t"
                            "tbl v22.16b, {v8.16b,  v9.16b,  v10.16b, v11.16b}, v26.16b\n\t"
                            "tbl v23.16b, {v12.16b, v13.16b, v14.16b, v15.16b}, v26.16b\n\t"

                            // Mask 2 Values
                            "and v20.16b, v20.16b, v28.16b\n\t"
                            "and v21.16b, v21.16b, v28.16b\n\t"
                            "and v22.16b, v22.16b, v28.16b\n\t"
                            "and v23.16b, v23.16b, v28.16b\n\t"

                            // Shift 2 Values
                            "shl v20.16b, v20.16b, #4\n\t"
                            "shl v21.16b, v21.16b, #4\n\t"
                            "shl v22.16b, v22.16b, #4\n\t"
                            "shl v23.16b, v23.16b, #4\n\t"

                            // Merge 2 Values into the Output
                            "orr v16.16b, v16.16b, v20.16b\n\t"
                            "orr v17.16b, v17.16b, v21.16b\n\t"
                            "orr v18.16b, v18.16b, v22.16b\n\t"
                            "orr v19.16b, v19.16b, v23.16b\n\t"

                            // Extract 3 Values
                            "tbl v20.16b, {v0.16b,  v1.16b,  v2.16b,  v3.16b},  v27.16b\n\t"
                            "tbl v21.16b, {v4.16b,  v5.16b,  v6.16b,  v7.16b},  v27.16b\n\t"
                            "tbl v22.16b, {v8.16b,  v9.16b,  v10.16b, v11.16b}, v27.16b\n\t"
                            "tbl v23.16b, {v12.16b, v13.16b, v14.16b, v15.16b}, v27.16b\n\t"

                            // Load first 512-bit data
                            "ld1 {v0.16b,  v1.16b,  v2.16b,  v3.16b},  [x8],  #64\n\t"
                            "ld1 {v4.16b,  v5.16b,  v6.16b,  v7.16b},  [x9],  #64\n\t"
                            "ld1 {v8.16b,  v9.16b,  v10.16b, v11.16b}, [x10], #64\n\t"
                            "ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x11], #64\n\t"

                            // Mask 3 Values
                            "and v20.16b, v20.16b, v28.16b\n\t"
                            "and v21.16b, v21.16b, v28.16b\n\t"
                            "and v22.16b, v22.16b, v28.16b\n\t"
                            "and v23.16b, v23.16b, v28.16b\n\t"

                            // Shift 3 Values
                            "shl v20.16b, v20.16b, #6\n\t"
                            "shl v21.16b, v21.16b, #6\n\t"
                            "shl v22.16b, v22.16b, #6\n\t"
                            "shl v23.16b, v23.16b, #6\n\t"

                            // Merge 3 Values into the Output
                            "orr v16.16b, v16.16b, v20.16b\n\t"
                            "orr v17.16b, v17.16b, v21.16b\n\t"
                            "orr v18.16b, v18.16b, v22.16b\n\t"
                            "orr v19.16b, v19.16b, v23.16b\n\t"

                            // Store 128-bit data
                            "st1 {v16.16b}, [x5], #16\n\t"
                            "st1 {v17.16b}, [x5], #16\n\t"
                            "st1 {v18.16b}, [x5], #16\n\t"
                            "st1 {v19.16b}, [x5], #16\n\t"
                            
                            // Increase the loop counter by 64
                            "add x0, x0, #64\n\t"
                            // Check if we already iterated enough
                            "cmp x0, %x[K]\n\t"
                            // Branch to the start of the loop if not finished
                            "b.lt 1b\n\t"

                            // Preparing Next Input Pointers
                            "sub x11,  x11, #64\n\t"
                            "mov x8,  x11\n\t"
                            "add x9,  x8,  %x[K]\n\t"
                            "add x10, x9,  %x[K]\n\t"
                            "add x11, x10, %x[K]\n\t"
                            
                            // Increase the loop counter by 4
                            "add x1, x1, #4\n\t"
                            // Check if we already iterated enough
                            "cmp x1, %x[M]\n\t"
                            // Branch to the start of the loop if not finished
                            "b.lt 0b\n\t"

                            : 
                            : [ input ]  "r" (input), [ K ] "r"(shape.size[1]), [ M ] "r"(shape.size[0]), [ output ] "r"(output),
                              [ MH1 ] "r"(mask_half_1), [ MH2 ] "r"(mask_half_2), [ MA ] "r"(mask_add)
                            : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"
                        );

                        #endif
                        #endif
                    }
                    return Status::Success;
                }
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){
                    return LowPrecision::Status::NotUpdated;
                    if (shape.size[shape.number_dims - 1] % 32)
                        return Status::SizesMisMatch; 
                    if (layout != MemLayout::kRowMajor)
                        return Status::WrongMemLayout;
                    bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                    if (is_multibatched && shape.size[0] % 4)
                        return Status::SizesMisMatch; 
                    if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                        GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                        std::copy(input, input + (shape.flatsize / 2), output);
                    }
                    else {
                        // TODO: `InputPackingStep` is not implemeneted without cvector, which is deprecated.
                        return LowPrecision::Status::NotImplemented;
                        uint8_t* temp = output;
                        uint8_t* input_u = const_cast<uint8_t*>(input);
                        int M = shape.size[0], K = shape.size[1];
                        #ifdef W4A4_UNSIGNED_PROCESS_8x8
                        for (int i = 0 ; i < M ; i += 8)
                            InputPackingStep(input_u + (i * K), output + ((i / 2) * K), K, K);
                        #else
                        InputPackingStep(input_u, output, M * K, K);
                        #endif
                    }
                    return Status::Success;
                }
                Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                ){
                    // TODO: This has not been tested yet, it might not work.
                    return LowPrecision::Status::NotImplemented;
                    int lhs_columns = input_shape.size[input_shape.number_dims - 1] ,
                        rhs_rows = kernel_shape.size[0] ,
                        rhs_columns = kernel_shape.size[1];
                    if (lhs_columns != rhs_columns)
                        return Status::SizesMisMatch;
                    if(lhs_columns == 0 || rhs_rows == 0)
                        return Status::Success;
                    
                    int8_t*         _kernel = const_cast<int8_t*>(kernel);
                    const int8_t*   _input  = input;
                    int32_t*        _output = output;
                    int i, j;

                    asm volatile(
                        "mov %w[j], wzr\n\t"

                        // "cmp %w[rows], #0\n\t"
                        // "beq 5f\n\t"

                        "0:\n\t"
                        "mov %w[i], wzr\n\t"
                        "dup v23.4s, wzr\n\t"
                        "dup v24.4s, wzr\n\t"
                        "dup v25.4s, wzr\n\t"
                        "dup v30.4s, wzr\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.16b},  [%[weights]], #16\n\t"
                        "ld1 {v1.16b},  [%[weights]], #16\n\t"
                        "ld1 {v2.16b},  [%[weights]], #16\n\t"
                        "ld1 {v3.16b},  [%[weights]], #16\n\t"

                        "ld1 {v4.16b},  [%[activation]], #16\n\t"

                        // SHL AL, A, #4
                        "shl v5.16b,  v4.16b,  #4\n\t"
                        // SSHR AH, A, #4
                        "sshr v4.16b, v4.16b, #4\n\t"
                        // SSHR AL, AL, #4
                        "sshr v5.16b,  v5.16b,  #4\n\t"

                        // SHL WL, W, #4
                        "shl v8.16b,  v0.16b, #4\n\t"
                        "shl v10.16b, v1.16b, #4\n\t"
                        "shl v11.16b, v2.16b, #4\n\t"
                        "shl v12.16b, v3.16b, #4\n\t"
                        
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

                        // SMULL T1.8h, WH.8b, A2.8b
                        "smull v13.8h, v0.8b, v5.8b\n\t"
                        "smull v14.8h, v1.8b, v5.8b\n\t"
                        "smull v15.8h, v2.8b, v5.8b\n\t"
                        "smull v16.8h, v3.8b, v5.8b\n\t"

                        // SMLAL2 T1.8h, WH.16b, A2.16b
                        "smlal2 v13.8h, v0.16b, v5.16b\n\t"
                        "smlal2 v14.8h, v1.16b, v5.16b\n\t"
                        "smlal2 v15.8h, v2.16b, v5.16b\n\t"
                        "smlal2 v16.8h, v3.16b, v5.16b\n\t"

                        // SMLAL T1.8h, WL.8b, A1.8b
                        "smlal v13.8h, v8.8b,  v4.8b\n\t"
                        "smlal v14.8h, v10.8b, v4.8b\n\t"
                        "smlal v15.8h, v11.8b, v4.8b\n\t"
                        "smlal v16.8h, v12.8b, v4.8b\n\t"

                        // SMLAL2 T1.8h, WL.16b, A1.16b
                        "smlal2 v13.8h, v8.16b,  v4.16b\n\t"
                        "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                        "smlal2 v15.8h, v11.16b, v4.16b\n\t"
                        "smlal2 v16.8h, v12.16b, v4.16b\n\t"

                        // ACCUMULATE ACC, T1
                        "sadalp v23.4s, v13.8h\n\t"
                        "sadalp v24.4s, v14.8h\n\t"
                        "sadalp v25.4s, v15.8h\n\t"
                        "sadalp v30.4s, v16.8h\n\t"

                        "add %w[i], %w[i], #32\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "addv s23, v23.4s\n\t"
                        "addv s24, v24.4s\n\t"
                        "addv s25, v25.4s\n\t"
                        "addv s30, v30.4s\n\t"

                        "mov v23.s[1], v24.s[0]\n\t"
                        "mov v23.s[2], v25.s[0]\n\t"
                        "mov v23.s[3], v30.s[0]\n\t"

                        // "ld1 {v30.4s},  [%[dst]]\n\t"
                        // "add v30.4s, v30.4s, v23.4s\n\t"
                        // "st1 {v30.4s},  [%[dst]]\n\t"
                        "st1 {v23.4s},  [%[dst]], #16\n\t"

                        "mov %[activation], %[lhs_base]\n\t"

                        "3:\n\t"

                        "add %w[j], %w[j], #4\n\t"
                        "cmp %w[j], %w[rows]\n\t"
                        "b.lt 0b\n\t"


                        : [ dst ]        "+r" (_output),     [ i ]           "+r" (i),
                        [ j ]          "+r" (j)

                        : [ activation ] "r"  (_input),      [ lhs_base ]    "r"  (input),
                        [ weights ]    "r"  (_kernel),     [ size ]        "r"  (lhs_columns),
                        [ rows ]       "r"  (rhs_rows)

                        : "v0",  "v1",  "v2",  "v3",
                        "v4",  "v5",  "v6",  "v7",
                        "v8",  "v9",  "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31"
                    );

                    return Status::Success;
                }
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params
                ){
                    int lhs_batches = input_shape.size[0],
                        lhs_columns = input_shape.size[1],
                        rhs_rows    = kernel_shape.size[0],
                        rhs_columns = kernel_shape.size[1];
                    
                    size_t M        = lhs_batches,
                           K        = lhs_columns,
                           N        = rhs_columns;
                    
                    int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                    
                    // This might cause issues; had changed to work with `test-gemm-api` test
                    // if (lhs_columns != rhs_columns)
                    //     return Status::SizesMisMatch;
                    if (K != rhs_rows)
                        return Status::SizesMisMatch;
                    if(K == 0 || N == 0 || M == 0)
                        return Status::Success;
                    if (M % 4)
                        return Status::NotSupported;
                    
                    int8_t*         _kernel     = const_cast<int8_t*>(kernel);
                    int8_t*         _kernel_base= const_cast<int8_t*>(kernel);
                    int8_t*         _input      = const_cast<int8_t*>(input);
                    int8_t*         _input_base = const_cast<int8_t*>(input);
                    int i, j, k, end;
                    int32_t*        _output_1   = output + 0 * N;
                    int32_t*        _output_2   = output + 1 * N;
                    int32_t*        _output_3   = output + 2 * N;
                    int32_t*        _output_4   = output + 3 * N;
                    /* Vector assignments:
                        * W         -> v0-3      (Weights)
                        * WC        -> v4-7      (Weights Current)
                        * A         -> v8-11     (Activations)
                        * AC        -> v12-15    (Activations Current)
                        * ACC1      -> v24-27    (Accumulators input row #1)
                        * ACC2      -> v28-31    (Accumulators input row #2)
                        * ACC3      -> v16-19    (Accumulators input row #3)
                        * ACC4      -> v20-23    (Accumulators input row #4)
                    */
                    asm volatile(
                        "mov x1, %[activation]\n\t"
                        "mov x2, %[weights]\n\t"
                        "mov %w[k], wzr\n\t"

                        // Start of The Loop Over Batches
                        "5:\n\t"
                        "mov %w[j], wzr\n\t"

                        // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]]\n\t"
#else
                        "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]], #64\n\t"
#endif

                        "0:\n\t"

                        // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "ld1 {v8.16b, v9.16b, v10.16b, v11.16b},  [%[activation]]\n\t"
#else
                        "ld1 {v8.16b, v9.16b, v10.16b, v11.16b},  [%[activation]], #64\n\t"
#endif

                        "mov %w[i], wzr\n\t"
                        "movi v16.4s, #0\n\t"
                        "movi v17.4s, #0\n\t"
                        "movi v18.4s, #0\n\t"
                        "movi v19.4s, #0\n\t"
                        "movi v20.4s, #0\n\t"
                        "movi v21.4s, #0\n\t"
                        "movi v22.4s, #0\n\t"
                        "movi v23.4s, #0\n\t"
                        "movi v24.4s, #0\n\t"
                        "movi v25.4s, #0\n\t"
                        "movi v26.4s, #0\n\t"
                        "movi v27.4s, #0\n\t"
                        "movi v28.4s, #0\n\t"
                        "movi v29.4s, #0\n\t"
                        "movi v30.4s, #0\n\t"
                        "movi v31.4s, #0\n\t"

                        // Start of Outer Loop Over Weights

                        "1:\n\t"

                        //
                        // First Quarter of weights and activations vectors
                        //

                        // Activation Row #1
                        // SMLAL ACC1, WC, AC[0]
                        "smlal v24.8h, v0.8b, v8.8b\n\t"
                        "smlal v25.8h, v1.8b, v8.8b\n\t"
                        "smlal v26.8h, v2.8b, v8.8b\n\t"
                        "smlal v27.8h, v3.8b, v8.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WC, AC[1]
                        "smlal v28.8h, v0.8b, v9.8b\n\t"
                        "smlal v29.8h, v1.8b, v9.8b\n\t"
                        "smlal v30.8h, v2.8b, v9.8b\n\t"
                        "smlal v31.8h, v3.8b, v9.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WC, AC[2]
                        "smlal v16.8h, v0.8b, v10.8b\n\t"
                        "smlal v17.8h, v1.8b, v10.8b\n\t"
                        "smlal v18.8h, v2.8b, v10.8b\n\t"
                        "smlal v19.8h, v3.8b, v10.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WC, AC[3]
                        "smlal v20.8h, v0.8b, v11.8b\n\t"
                        "smlal v21.8h, v1.8b, v11.8b\n\t"
                        "smlal v22.8h, v2.8b, v11.8b\n\t"
                        "smlal v23.8h, v3.8b, v11.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WC, AC[0]
                        "smlal2 v24.8h, v0.16b, v8.16b\n\t"
                        "smlal2 v25.8h, v1.16b, v8.16b\n\t"
                        "smlal2 v26.8h, v2.16b, v8.16b\n\t"
                        "smlal2 v27.8h, v3.16b, v8.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WC, AC[1]
                        "smlal2 v28.8h, v0.16b, v9.16b\n\t"
                        "smlal2 v29.8h, v1.16b, v9.16b\n\t"
                        "smlal2 v30.8h, v2.16b, v9.16b\n\t"
                        "smlal2 v31.8h, v3.16b, v9.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WC, AC[2]
                        "smlal2 v16.8h, v0.16b, v10.16b\n\t"
                        "smlal2 v17.8h, v1.16b, v10.16b\n\t"
                        "smlal2 v18.8h, v2.16b, v10.16b\n\t"
                        "smlal2 v19.8h, v3.16b, v10.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WC, AC[3]
                        "smlal2 v20.8h, v0.16b, v11.16b\n\t"
                        "smlal2 v21.8h, v1.16b, v11.16b\n\t"
                        "smlal2 v22.8h, v2.16b, v11.16b\n\t"
                        "smlal2 v23.8h, v3.16b, v11.16b\n\t"

                        // 
                        // Second Quarter of weights and activations vectors
                        // 

                        // SSHR WC, W, #2
                        "sshr v4.16b, v0.16b, #2\n\t"
                        "sshr v5.16b, v1.16b, #2\n\t"
                        "sshr v6.16b, v2.16b, #2\n\t"
                        "sshr v7.16b, v3.16b, #2\n\t"

                        // SSHR AC, A, #2
                        "sshr v12.16b, v8.16b,  #2\n\t"
                        "sshr v13.16b, v9.16b,  #2\n\t"
                        "sshr v14.16b, v10.16b, #2\n\t"
                        "sshr v15.16b, v11.16b, #2\n\t"

                        // Activation Row #1
                        // SMLAL ACC1, WC, AC[0]
                        "smlal v24.8h, v4.8b, v12.8b\n\t"
                        "smlal v25.8h, v5.8b, v12.8b\n\t"
                        "smlal v26.8h, v6.8b, v12.8b\n\t"
                        "smlal v27.8h, v7.8b, v12.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WC, AC[1]
                        "smlal v28.8h, v4.8b, v13.8b\n\t"
                        "smlal v29.8h, v5.8b, v13.8b\n\t"
                        "smlal v30.8h, v6.8b, v13.8b\n\t"
                        "smlal v31.8h, v7.8b, v13.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WC, AC[2]
                        "smlal v16.8h, v4.8b, v14.8b\n\t"
                        "smlal v17.8h, v5.8b, v14.8b\n\t"
                        "smlal v18.8h, v6.8b, v14.8b\n\t"
                        "smlal v19.8h, v7.8b, v14.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WC, AC[3]
                        "smlal v20.8h, v4.8b, v15.8b\n\t"
                        "smlal v21.8h, v5.8b, v15.8b\n\t"
                        "smlal v22.8h, v6.8b, v15.8b\n\t"
                        "smlal v23.8h, v7.8b, v15.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WC, AC[0]
                        "smlal2 v24.8h, v4.16b, v12.16b\n\t"
                        "smlal2 v25.8h, v5.16b, v12.16b\n\t"
                        "smlal2 v26.8h, v6.16b, v12.16b\n\t"
                        "smlal2 v27.8h, v7.16b, v12.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WC, AC[1]
                        "smlal2 v28.8h, v4.16b, v13.16b\n\t"
                        "smlal2 v29.8h, v5.16b, v13.16b\n\t"
                        "smlal2 v30.8h, v6.16b, v13.16b\n\t"
                        "smlal2 v31.8h, v7.16b, v13.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WC, AC[2]
                        "smlal2 v16.8h, v4.16b, v14.16b\n\t"
                        "smlal2 v17.8h, v5.16b, v14.16b\n\t"
                        "smlal2 v18.8h, v6.16b, v14.16b\n\t"
                        "smlal2 v19.8h, v7.16b, v14.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WC, AC[3]
                        "smlal2 v20.8h, v4.16b, v15.16b\n\t"
                        "smlal2 v21.8h, v5.16b, v15.16b\n\t"
                        "smlal2 v22.8h, v6.16b, v15.16b\n\t"
                        "smlal2 v23.8h, v7.16b, v15.16b\n\t"

                        // 
                        // Third Quarter of weights and activations vectors
                        // 

                        // SSHR WC, W, #4
                        "sshr v4.16b, v0.16b, #4\n\t"
                        "sshr v5.16b, v1.16b, #4\n\t"
                        "sshr v6.16b, v2.16b, #4\n\t"
                        "sshr v7.16b, v3.16b, #4\n\t"

                        // SSHR AC, A, #4
                        "sshr v12.16b, v8.16b,  #4\n\t"
                        "sshr v13.16b, v9.16b,  #4\n\t"
                        "sshr v14.16b, v10.16b, #4\n\t"
                        "sshr v15.16b, v11.16b, #4\n\t"

                        // Activation Row #1
                        // SMLAL ACC1, WC, AC[0]
                        "smlal v24.8h, v4.8b, v12.8b\n\t"
                        "smlal v25.8h, v5.8b, v12.8b\n\t"
                        "smlal v26.8h, v6.8b, v12.8b\n\t"
                        "smlal v27.8h, v7.8b, v12.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WC, AC[1]
                        "smlal v28.8h, v4.8b, v13.8b\n\t"
                        "smlal v29.8h, v5.8b, v13.8b\n\t"
                        "smlal v30.8h, v6.8b, v13.8b\n\t"
                        "smlal v31.8h, v7.8b, v13.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WC, AC[2]
                        "smlal v16.8h, v4.8b, v14.8b\n\t"
                        "smlal v17.8h, v5.8b, v14.8b\n\t"
                        "smlal v18.8h, v6.8b, v14.8b\n\t"
                        "smlal v19.8h, v7.8b, v14.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WC, AC[3]
                        "smlal v20.8h, v4.8b, v15.8b\n\t"
                        "smlal v21.8h, v5.8b, v15.8b\n\t"
                        "smlal v22.8h, v6.8b, v15.8b\n\t"
                        "smlal v23.8h, v7.8b, v15.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WC, AC[0]
                        "smlal2 v24.8h, v4.16b, v12.16b\n\t"
                        "smlal2 v25.8h, v5.16b, v12.16b\n\t"
                        "smlal2 v26.8h, v6.16b, v12.16b\n\t"
                        "smlal2 v27.8h, v7.16b, v12.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WC, AC[1]
                        "smlal2 v28.8h, v4.16b, v13.16b\n\t"
                        "smlal2 v29.8h, v5.16b, v13.16b\n\t"
                        "smlal2 v30.8h, v6.16b, v13.16b\n\t"
                        "smlal2 v31.8h, v7.16b, v13.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WC, AC[2]
                        "smlal2 v16.8h, v4.16b, v14.16b\n\t"
                        "smlal2 v17.8h, v5.16b, v14.16b\n\t"
                        "smlal2 v18.8h, v6.16b, v14.16b\n\t"
                        "smlal2 v19.8h, v7.16b, v14.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WC, AC[3]
                        "smlal2 v20.8h, v4.16b, v15.16b\n\t"
                        "smlal2 v21.8h, v5.16b, v15.16b\n\t"
                        "smlal2 v22.8h, v6.16b, v15.16b\n\t"
                        "smlal2 v23.8h, v7.16b, v15.16b\n\t"

                        // 
                        // Fourth Quarter of weights and activations vectors
                        // 

                        // SSHR WC, W, #6
                        "sshr v4.16b, v0.16b, #6\n\t"
                        "sshr v5.16b, v1.16b, #6\n\t"
                        "sshr v6.16b, v2.16b, #6\n\t"
                        "sshr v7.16b, v3.16b, #6\n\t"

                        // SSHR AC, A, #6
                        "sshr v12.16b, v8.16b,  #6\n\t"
                        "sshr v13.16b, v9.16b,  #6\n\t"
                        "sshr v14.16b, v10.16b, #6\n\t"
                        "sshr v15.16b, v11.16b, #6\n\t"

                        // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]]\n\t"
#else
                        "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]], #64\n\t"
#endif

                        // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "ld1 {v8.16b, v9.16b, v10.16b, v11.16b},  [%[activation]]\n\t"
#else
                        "ld1 {v8.16b, v9.16b, v10.16b, v11.16b},  [%[activation]], #64\n\t"
#endif

                        // Activation Row #1
                        // SMLAL ACC1, WC, AC[0]
                        "smlal v24.8h, v4.8b, v12.8b\n\t"
                        "smlal v25.8h, v5.8b, v12.8b\n\t"
                        "smlal v26.8h, v6.8b, v12.8b\n\t"
                        "smlal v27.8h, v7.8b, v12.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WC, AC[1]
                        "smlal v28.8h, v4.8b, v13.8b\n\t"
                        "smlal v29.8h, v5.8b, v13.8b\n\t"
                        "smlal v30.8h, v6.8b, v13.8b\n\t"
                        "smlal v31.8h, v7.8b, v13.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WC, AC[2]
                        "smlal v16.8h, v4.8b, v14.8b\n\t"
                        "smlal v17.8h, v5.8b, v14.8b\n\t"
                        "smlal v18.8h, v6.8b, v14.8b\n\t"
                        "smlal v19.8h, v7.8b, v14.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WC, AC[3]
                        "smlal v20.8h, v4.8b, v15.8b\n\t"
                        "smlal v21.8h, v5.8b, v15.8b\n\t"
                        "smlal v22.8h, v6.8b, v15.8b\n\t"
                        "smlal v23.8h, v7.8b, v15.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WC, AC[0]
                        "smlal2 v24.8h, v4.16b, v12.16b\n\t"
                        "smlal2 v25.8h, v5.16b, v12.16b\n\t"
                        "smlal2 v26.8h, v6.16b, v12.16b\n\t"
                        "smlal2 v27.8h, v7.16b, v12.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WC, AC[1]
                        "smlal2 v28.8h, v4.16b, v13.16b\n\t"
                        "smlal2 v29.8h, v5.16b, v13.16b\n\t"
                        "smlal2 v30.8h, v6.16b, v13.16b\n\t"
                        "smlal2 v31.8h, v7.16b, v13.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WC, AC[2]
                        "smlal2 v16.8h, v4.16b, v14.16b\n\t"
                        "smlal2 v17.8h, v5.16b, v14.16b\n\t"
                        "smlal2 v18.8h, v6.16b, v14.16b\n\t"
                        "smlal2 v19.8h, v7.16b, v14.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WC, AC[3]
                        "smlal2 v20.8h, v4.16b, v15.16b\n\t"
                        "smlal2 v21.8h, v5.16b, v15.16b\n\t"
                        "smlal2 v22.8h, v6.16b, v15.16b\n\t"
                        "smlal2 v23.8h, v7.16b, v15.16b\n\t"

                        // Check if the loop over rows of weight matrix is done
                        "add %w[i], %w[i], #64\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        // SADDLP ACC1, ACC1
                        "saddlp v24.4s, v24.8h\n\t"
                        "saddlp v25.4s, v25.8h\n\t"
                        "saddlp v26.4s, v26.8h\n\t"
                        "saddlp v27.4s, v27.8h\n\t"
                        
                        // SADDLP ACC2, ACC2
                        "saddlp v28.4s, v28.8h\n\t"
                        "saddlp v29.4s, v29.8h\n\t"
                        "saddlp v30.4s, v30.8h\n\t"
                        "saddlp v31.4s, v31.8h\n\t"

                        // SADDLP ACC3, ACC3
                        "saddlp v16.4s, v16.8h\n\t"
                        "saddlp v17.4s, v17.8h\n\t"
                        "saddlp v18.4s, v18.8h\n\t"
                        "saddlp v19.4s, v19.8h\n\t"
                        
                        // SADDLP ACC4, ACC4
                        "saddlp v20.4s, v20.8h\n\t"
                        "saddlp v21.4s, v21.8h\n\t"
                        "saddlp v22.4s, v22.8h\n\t"
                        "saddlp v23.4s, v23.8h\n\t"

                        // Accumulate the ACC1 to one int32
                        "addv s24, v24.4s\n\t"
                        "addv s25, v25.4s\n\t"
                        "addv s26, v26.4s\n\t"
                        "addv s27, v27.4s\n\t"

                        // Accumulate the ACC2 to one int32
                        "addv s28, v28.4s\n\t"
                        "addv s29, v29.4s\n\t"
                        "addv s30, v30.4s\n\t"
                        "addv s31, v31.4s\n\t"

                        // Accumulate the ACC3 to one int32
                        "addv s16, v16.4s\n\t"
                        "addv s17, v17.4s\n\t"
                        "addv s18, v18.4s\n\t"
                        "addv s19, v19.4s\n\t"

                        // Accumulate the ACC4 to one int32
                        "addv s20, v20.4s\n\t"
                        "addv s21, v21.4s\n\t"
                        "addv s22, v22.4s\n\t"
                        "addv s23, v23.4s\n\t"

                        // Reorder ACC3 to store
                        "mov v16.s[1], v17.s[0]\n\t"
                        "mov v16.s[2], v18.s[0]\n\t"
                        "mov v16.s[3], v19.s[0]\n\t"

                        // Reorder ACC4 to store
                        "mov v20.s[1], v21.s[0]\n\t"
                        "mov v20.s[2], v22.s[0]\n\t"
                        "mov v20.s[3], v23.s[0]\n\t"

                        // Reorder ACC1 to store
                        "mov v24.s[1], v25.s[0]\n\t"
                        "mov v24.s[2], v26.s[0]\n\t"
                        "mov v24.s[3], v27.s[0]\n\t"

                        // Reorder ACC2 to store
                        "mov v28.s[1], v29.s[0]\n\t"
                        "mov v28.s[2], v30.s[0]\n\t"
                        "mov v28.s[3], v31.s[0]\n\t"

                        // Check if the output need downcasting to int8
                        "cmp %w[downcast], 0xff\n\t"
                        "beq 6f\n\t"

                        // Output is needed in 32-bit; no need to downcast
                        // Store the 4 int32 results
                        "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                        "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                        "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                        "st1 {v20.4s},  [%[dst_2]], #16\n\t"

                        // Jump to after dowcasting since we dont need to downcast
                        "b 7f\n\t"

                        // Need to Downcast to Int8
                        "6:\n\t"

                        // Cast 32-bit to 16-bit
                        "sqxtn v24.4h, v24.4s\n\t"
                        "sqxtn v28.4h, v28.4s\n\t"
                        "sqxtn v16.4h, v16.4s\n\t"
                        "sqxtn v20.4h, v20.4s\n\t"

                        // Cast 16-bit to 8-bit
                        "sqxtn v24.8b, v24.8h\n\t"
                        "sqxtn v28.8b, v28.8h\n\t"
                        "sqxtn v16.8b, v16.8h\n\t"
                        "sqxtn v20.8b, v20.8h\n\t"
                        
                        // Store the 4 int8 results
                        "st1 {v24.s}[0],  [%[dst_3]], #4\n\t"
                        "st1 {v28.s}[0],  [%[dst_4]], #4\n\t"
                        "st1 {v16.s}[0],  [%[dst_1]], #4\n\t"
                        "st1 {v20.s}[0],  [%[dst_2]], #4\n\t"

                        "7:\n\t"

                        // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "mov %[activation], %[activation]\n\t"
#else
                        "mov %[activation], x1\n\t"
#endif

                        // Check if the all the columns of weight matrix are processed
                        "add %w[j], %w[j], #4\n\t"
                        "cmp %w[j], %w[rows]\n\t"
                        "b.lt 0b\n\t"

                        // Prepare the activation base for next 4 batches
                        "add x1, x1, %[size]\n\t"
                        
                        // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "mov %[activation], %[activation]\n\t"
#else
                        "mov %[activation], x1\n\t"
#endif

                        // Reset the weights to the start
#ifdef DISABLE_KERNELS_MEM_ACCESS
                        "mov %[weights], %[weights]\n\t"
#else
                        "mov %[weights], x2\n\t"
#endif

                        // Check if the output needed downcasting to int8
                        "cmp %w[downcast], 0xff\n\t"
                        "beq 8f\n\t"

                        // Prepare the destination base for next 4 batches
                        "mov %[dst_1], %[dst_4]\n\t"
                        "add %[dst_2], %[dst_1], %[rows], lsl #2\n\t"
                        "add %[dst_3], %[dst_2], %[rows], lsl #2\n\t"
                        "add %[dst_4], %[dst_3], %[rows], lsl #2\n\t"

                        // Jump to after the case for downcasting
                        "b 9f\n\t"

                        // Needed to Downcast to Int8
                        "8:\n\t"

                        // Prepare the destination base for next 4 batches
                        "mov %[dst_1], %[dst_3]\n\t"
                        "add %[dst_2], %[dst_1], %[rows]\n\t"
                        "add %[dst_3], %[dst_2], %[rows]\n\t"
                        "add %[dst_4], %[dst_3], %[rows]\n\t"

                        "9:\n\t"

                        // Check if the all the columns of weight matrix are processed
                        "add %w[k], %w[k], #4\n\t"
                        "cmp %w[k], %w[batches]\n\t"
                        "b.lt 5b\n\t"


                        : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                        [ dst_3 ]      "+r" (_output_3),   [ dst_4 ]       "+r" (_output_4),
                        [ i ]          "+r" (i),           [ end ]         "+r" (end),
                        [ j ]          "+r" (j),           [ k ]           "+r" (k)

                        : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                        [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                        [ size ]       "r"  (K),           [ rows ]        "r"  (N),
                        [ batches ]    "r"  (M),           [ downcast ]    "r"  (need_downcasting)

                        : "v0",  "v1",  "v2",  "v3",
                        "v4",  "v5",  "v6",  "v7",
                        "v8",  "v9",  "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                        "x0" , "x1" , "x2"
                    );
                    return Status::Success;
                }
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params
                ){
                    return LowPrecision::Status::NotUpdated;
                    int lhs_batches = input_shape.size[0],
                        lhs_columns = input_shape.size[1],
                        rhs_rows    = kernel_shape.size[0],
                        rhs_columns = kernel_shape.size[1];
                    
                    int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                    
                    if (lhs_columns != rhs_rows)
                        return Status::SizesMisMatch;
                    if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                        return Status::Success;
                    if (lhs_batches % 4)
                        return Status::NotSupported;

                    const int M        = lhs_batches,
                            K        = lhs_columns,
                            N        = rhs_columns,
                            a_stride = lhs_columns / 2,
                            c_stride = N;
                    int mr_block_size = 8, nr_block_size = 8;
                    for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                        for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                            const uint8_t* w = kernel + nr_block_start * (K / 2);
                            const uint8_t* a = input  + mr_block_start * (K / 2);
                            int32_t*       c = output + mr_block_start * N + nr_block_start;
                            int k = K;

                            const uint8_t* a0 = a;
                            const uint8_t* a1 = (a0 + a_stride);
                            const uint8_t* a2 = (a1 + a_stride);
                            const uint8_t* a3 = (a2 + a_stride);

                            uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
                            uint32x4_t vacc0x4567_=vacc0x0123_; 
                            uint32x4_t vacc1x0123_=vacc0x0123_; 
                            uint32x4_t vacc1x4567_=vacc0x0123_; 
                            uint32x4_t vacc2x0123_=vacc0x0123_; 
                            uint32x4_t vacc2x4567_=vacc0x0123_; 
                            uint32x4_t vacc3x0123_=vacc0x0123_; 
                            uint32x4_t vacc3x4567_=vacc0x0123_;

                            //4,4 : 24-12-0
                            uint16x4_t mask = vdup_n_u16(0x000f);
                            for (; k >= 32; k -= 32) {
                                uint16x4_t vxa0_, vxa1_, vxa2_, vxa3_;

                                uint16x4_t vxb01234567c0l_;
                                uint16x4_t vxb01234567c0h_;
                                uint16x4_t vxa0, vxa1, vxa2, vxa3;

                                uint16x4_t vxb01234567c0l, vxb01234567c0h;

                                vxa0_ = vld1_u16((uint16_t*)a0); a0 += 8;
                                vxa1_ = vld1_u16((uint16_t*)a1); a1 += 8;
                                vxa2_ = vld1_u16((uint16_t*)a2); a2 += 8;
                                vxa3_ = vld1_u16((uint16_t*)a3); a3 += 8;

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);

                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);

                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0_ = vld1_u16((uint16_t*)a0); a0 += 8;
                                vxa1_ = vld1_u16((uint16_t*)a1); a1 += 8;
                                vxa2_ = vld1_u16((uint16_t*)a2); a2 += 8;
                                vxa3_ = vld1_u16((uint16_t*)a3); a3 += 8;

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);

                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);

                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 4);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 4);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 4);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 4);
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 4);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 4);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);
                            }

                            int32_t* c0 = c;
                            int32_t* c1 = c0 + c_stride;
                            int32_t* c2 = c1 + c_stride;
                            int32_t* c3 = c2 + c_stride;
                            
                            vst1q_s32(c0,vacc0x0123_);
                            vst1q_s32(c0+4, vacc0x4567_);
                            vst1q_s32(c1, vacc1x0123_);
                            vst1q_s32(c1+4, vacc1x4567_);
                            vst1q_s32(c2, vacc2x0123_);
                            vst1q_s32(c2+4, vacc2x4567_);
                            vst1q_s32(c3, vacc3x0123_);
                            vst1q_s32(c3+4, vacc3x4567_);
                        }
                    }
                    
                    return Status::Success;
                }
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const Params params
                ){
                    return LowPrecision::Status::NotImplemented;
                }
                void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride){
                    // #ifdef W4A4_UNSIGNED_PROCESS_8x8
                    // v_uint16x8_t mask;
                    // mask = 0x000f;
                    // uint8_t  *i = nullptr;
                    // uint16_t *o = nullptr;

                    // v_uint8x8_t vi01_, vi23_;
                    // v_uint16x8_t vi01, vi23;
                    // v_uint16x4_t vi0, vi1, vi2, vi3, vo0123;

                    // for (int j = 0 ; j < size; j += 16){
                    //     i = input_u + j;
                    //     o = get_pointer_as<uint16_t>(output + (j * 4));

                    //     for (int z = 0 ; z < 8 ; z++) {
                    //         const uint8_t* in = i + (z * stride);
                    //         uint16_t* out = o + (z * 4);

                    //         vi01_ = in;
                    //         vi23_ = in + 8;

                    //         vi01 = vi01_;
                    //         vi23 = vi23_;

                    //         vi01 &= mask;
                    //         vi23 &= mask;
                            
                    //         vi0 = vi01.low();
                    //         vi1 = vi01.high();
                            
                    //         vi2 = vi23.low();
                    //         vi3 = vi23.high();

                    //         vi1 <<= 4;
                    //         vi2 <<= 8;
                    //         vi3 <<= 12;

                    //         vo0123 = vi0 | vi1;
                    //         vo0123 = vo0123 | vi2;
                    //         vo0123 = vo0123 | vi3;
                            
                    //         vo0123(out);
                    //     }
                    // }
                    // #else
                    // v_uint16x8_t mask;
                    // mask = 0x000f;
                    // uint8_t  *i = nullptr;
                    // uint16_t *o = nullptr;

                    // v_uint8x8_t vi01_, vi23_;
                    // v_uint16x8_t vi01, vi23;
                    // v_uint16x4_t vi0, vi1, vi2, vi3, vo0123;

                    // for (long long int j = 0 ; j < size; j += 16){
                    //     i = input_u + j;
                    //     o = get_pointer_as<uint16_t>(output + (j / 2));
                    //     vi01_ = i;
                    //     vi23_ = i + 8;

                    //     vi01 = vi01_;
                    //     vi23 = vi23_;

                    //     vi01 &= mask;
                    //     vi23 &= mask;
                        
                    //     vi0 = vi01.low();
                    //     vi1 = vi01.high();
                        
                    //     vi2 = vi23.low();
                    //     vi3 = vi23.high();

                    //     vi1 <<= 4;
                    //     vi2 <<= 8;
                    //     vi3 <<= 12;

                    //     vo0123 = vi0 | vi1;
                    //     vo0123 = vo0123 | vi2;
                    //     vo0123 = vo0123 | vi3;
                        
                    //     vo0123(o);
                    // }
                    // #endif
                }
                void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride){
                    // v_uint16x8_t mask;
                    // mask = 0x000f;
                    // uint8_t  *i = nullptr;
                    // uint16_t *o = nullptr;

                    // v_uint8x8_t vir0c0t7_,  vir1c0t7_,  vir2c0t7_,  vir3c0t7_;
                    // v_uint16x8_t vir0c0t7,  vir1c0t7,  vir2c0t7,  vir3c0t7;
                    // v_uint16x4_t virc0t3, virc4t7, virc0t3T, virc4t7T;

                    // for (long long int row = 0 ; row < size; row += 4){
                    //     i = input_u + row * stride;
                    //     o = get_pointer_as<uint16_t>(output) + ((row / 4) * 8);
                    //     vir0c0t7_  = i + (0 * stride);
                    //     vir1c0t7_  = i + (1 * stride);
                    //     vir2c0t7_  = i + (2 * stride);
                    //     vir3c0t7_  = i + (3 * stride);

                    //     vir0c0t7  = vir0c0t7_;
                    //     vir1c0t7  = vir1c0t7_;
                    //     vir2c0t7  = vir2c0t7_;
                    //     vir3c0t7  = vir3c0t7_;

                    //     vir0c0t7  &= mask;
                    //     vir1c0t7  &= mask;
                    //     vir2c0t7  &= mask;
                    //     vir3c0t7  &= mask;

                    //     virc0t3   = vir0c0t7.low();
                    //     virc4t7   = vir0c0t7.high();
                        
                    //     virc0t3T = vir1c0t7.low();
                    //     virc4t7T = vir1c0t7.high();

                    //     virc0t3T <<= 4;
                    //     virc4t7T <<= 4;
                        
                    //     virc0t3 |= virc0t3T;
                    //     virc4t7 |= virc4t7T;

                    //     virc0t3T = vir2c0t7.low();
                    //     virc4t7T = vir2c0t7.high();

                    //     virc0t3T <<= 8;
                    //     virc4t7T <<= 8;
                        
                    //     virc0t3 |= virc0t3T;
                    //     virc4t7 |= virc4t7T;

                    //     virc0t3T = vir3c0t7.low();
                    //     virc4t7T = vir3c0t7.high();

                    //     virc0t3T <<= 12;
                    //     virc4t7T <<= 12;
                        
                    //     virc0t3 |= virc0t3T;
                    //     virc4t7 |= virc4t7T;

                    //     virc0t3(o);
                    //     virc4t7(o + 4);
                    // }
                }
                LowPrecision::PreprocessType InputPreProcess()  { return LowPrecision::PreprocessType::PaddingAndPacking; }
                LowPrecision::PreprocessType FilterPreProcess() { return LowPrecision::PreprocessType::PaddingAndPacking; }
                LowPrecision::PreprocessType OutputPreProcess() { return OutputPostProcess(); }
                LowPrecision::PreprocessType OutputPostProcess(){ return LowPrecision::PreprocessType::PaddingIfNeccessery;}
                LowPrecision::GEMMType GEMMSupport(){ return LowPrecision::GEMMType::SupportsGEMM; }
            }
        }
    }
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            namespace W2A2{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status MultiplyInt8SingleBatch(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape
                ){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const int8_t* input, LowPrecision::Shape input_shape,
                    const int8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                ){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status MultiplyInt8MultiBatched(
                    const uint8_t* input, LowPrecision::Shape input_shape,
                    const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                    int32_t* output, LowPrecision::Shape output_shape,
                    LowPrecision::MulParams params = LowPrecision::MulParams()
                ){ return LowPrecision::Status::NotImplemented; }
                LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                    const int8_t* input, const int8_t* kernel,
                    int32_t* output, const Params params){ return LowPrecision::Status::NotImplemented; }
            }
        }
    }
}
#endif