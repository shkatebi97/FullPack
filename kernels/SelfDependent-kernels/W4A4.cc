#include "../../low_precision_fully_connected.h"

#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            namespace W4A4{
                LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
                    if (k_shape.number_dims != 2)
                        return Status::DimensionsMisMatch;
                    if (k_shape.size[0] % 4)
                        return Status::SizesMisMatch;
                    if (layout != MemLayout::kRowMajor)
                        return Status::WrongMemLayout;
                    if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                        doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0] / 2, k_shape.size[1]);
                    }
                    else {
                        int new_weights_length = (k_shape.size[0] / 2) * k_shape.size[1];
                        int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                        uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                        int i , size = k_shape.flatsize;
                        const uint8_t* input_u = get_pointer_as<uint8_t>(input);
                        
                        #if SelfDependent_Type == SelfDependent_Offset_Vector_Size
                        for (size_t i = 0; i < k_shape.size[0]; i += 32)
                            for (size_t j = 0; j < 16; j++)
                                for (size_t z = 0; z < k_shape.size[1]; z++)
                                    temp_u[((i / 2) + j) * k_shape.size[1] + z] = (input_u[z * k_shape.size[0] + (i + j     )] & 0x0F) | 
                                                                                ((input_u[z * k_shape.size[0] + (i + j + 16)] & 0x0F) << 4);
                        #elif SelfDependent_Type == SelfDependent_Continious
                        size_t z = 0;
                        for (size_t i = 2; i < k_shape.size[0]; i += 2){
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[z * k_shape.size[1] + j] = (input_u[j * k_shape.size[0] + (i - 2)] & 0x0F) | 
                                                                 ((input_u[j * k_shape.size[0] + (i - 1)] & 0x0F) << 4);
                            z++;
                        }
                        if (k_shape.size[0] % 2)
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[z * k_shape.size[1] + j] = input_u[j * k_shape.size[0] + (k_shape.size[0] - 1)] & 0x0F;
                        else
                            for (size_t j = 0; j < k_shape.size[1]; j++)
                                temp_u[z * k_shape.size[1] + j] = (input_u[j * k_shape.size[0] + (k_shape.size[0] - 2)] & 0x0F) |
                                                                 ((input_u[j * k_shape.size[0] + (k_shape.size[0] - 1)] & 0x0F) << 4);
                        #endif
                        doLowPrecisionWeightPack(temp, output, k_shape.size[0] / 2, k_shape.size[1]);
                        LowPrecision::deallocate(temp);
                    }
                    return Status::Success;
                }
                LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){
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
                        int8_t* temp = output;
                        if (is_multibatched){
                            int new_weights_length = ((int)shape.flatsize / 2);
                            temp = LowPrecision::allocate<int8_t>(new_weights_length);
                        }
                        int i , size = shape.flatsize;
                        uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                        const uint8_t* input_u = get_pointer_as<uint8_t>(input);
                        #if SelfDependent_Type == SelfDependent_Offset_Vector_Size
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
                        for (size_t i = 0; i < shape.size[0]; i++){
                            size_t z = 0;
                            for (size_t j = 2; j < shape.size[1]; j += 2)
                                temp_u[i * (shape.size[1] / 2) + z++] = (input_u[i * shape.size[0] + (j - 2)] & 0x0F) | 
                                                                       ((input_u[i * shape.size[0] + (j - 1)] & 0x0F) << 4);
                            if (shape.size[1] % 2)
                                temp_u[i * (shape.size[1] / 2) + z++] = input_u[i * shape.size[0] + (shape.size[1] - 1)] & 0x0F;
                            else
                                temp_u[i * (shape.size[1] / 2) + z++] = (input_u[i * shape.size[0] + (shape.size[1] - 2)] & 0x0F) | 
                                                                       ((input_u[i * shape.size[0] + (shape.size[1] - 1)] & 0x0F) << 4);
                        }
                        #endif
                        if (is_multibatched){
                            doLowPrecisionWeightPack(temp, output, shape.size[0], shape.size[1] / 2);
                            LowPrecision::deallocate(temp);
                        }
                    }
                    return Status::Success;
                }
                LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){
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
                        * W, WL     -> v0-3      (Weights, Weights Low)
                        * WH        -> v4-7      (Weights High)
                        * A, AL     -> v8-11     (Activations, Activations Low)
                        * AH        -> v12-15    (Activations High)
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

                        // // SHL WL, W, #4
                        // "shl v8.16b, v0.16b,  #4\n\t"
                        // "shl v9.16b, v1.16b,  #4\n\t"
                        // "shl v10.16b, v2.16b, #4\n\t"
                        // "shl v11.16b, v3.16b, #4\n\t"
                        
                        // // SHL AL, A, #4
                        // "shl v12.16b, v4.16b,  #4\n\t"
                        // "shl v13.16b, v5.16b,  #4\n\t"
                        // "shl v14.16b, v6.16b,  #4\n\t"
                        // "shl v15.16b, v7.16b,  #4\n\t"

                        // SSHR WH, W, #4
                        "sshr v4.16b, v0.16b, #4\n\t"
                        "sshr v5.16b, v1.16b, #4\n\t"
                        "sshr v6.16b, v2.16b, #4\n\t"
                        "sshr v7.16b, v3.16b, #4\n\t"

                        // SSHR AH, A, #4
                        "sshr v12.16b, v8.16b,  #4\n\t"
                        "sshr v13.16b, v9.16b,  #4\n\t"
                        "sshr v14.16b, v10.16b, #4\n\t"
                        "sshr v15.16b, v11.16b, #4\n\t"
                        
                        // // SSHR WL, WL, #4
                        // "sshr v8.16b, v8.16b, #4\n\t"
                        // "sshr v9.16b, v9.16b, #4\n\t"
                        // "sshr v10.16b, v10.16b, #4\n\t"
                        // "sshr v11.16b, v11.16b, #4\n\t"

                        // // SSHR AL, AL, #4
                        // "sshr v12.16b, v12.16b,  #4\n\t"
                        // "sshr v13.16b, v13.16b,  #4\n\t"
                        // "sshr v14.16b, v14.16b,  #4\n\t"
                        // "sshr v15.16b, v15.16b,  #4\n\t"

                        // Activation Row #1
                        // SMLAL ACC1, WL, AL[0]
                        "smlal v24.8h, v0.8b, v8.8b\n\t"
                        "smlal v25.8h, v1.8b, v8.8b\n\t"
                        "smlal v26.8h, v2.8b, v8.8b\n\t"
                        "smlal v27.8h, v3.8b, v8.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WL, AL[1]
                        "smlal v28.8h, v0.8b, v9.8b\n\t"
                        "smlal v29.8h, v1.8b, v9.8b\n\t"
                        "smlal v30.8h, v2.8b, v9.8b\n\t"
                        "smlal v31.8h, v3.8b, v9.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WL, AL[2]
                        "smlal v16.8h, v0.8b, v10.8b\n\t"
                        "smlal v17.8h, v1.8b, v10.8b\n\t"
                        "smlal v18.8h, v2.8b, v10.8b\n\t"
                        "smlal v19.8h, v3.8b, v10.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WL, AL[3]
                        "smlal v20.8h, v0.8b, v11.8b\n\t"
                        "smlal v21.8h, v1.8b, v11.8b\n\t"
                        "smlal v22.8h, v2.8b, v11.8b\n\t"
                        "smlal v23.8h, v3.8b, v11.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WL, AL[0]
                        "smlal2 v24.8h, v0.16b, v8.16b\n\t"
                        "smlal2 v25.8h, v1.16b, v8.16b\n\t"
                        "smlal2 v26.8h, v2.16b, v8.16b\n\t"
                        "smlal2 v27.8h, v3.16b, v8.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WL, AL[1]
                        "smlal2 v28.8h, v0.16b, v9.16b\n\t"
                        "smlal2 v29.8h, v1.16b, v9.16b\n\t"
                        "smlal2 v30.8h, v2.16b, v9.16b\n\t"
                        "smlal2 v31.8h, v3.16b, v9.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WL, AL[2]
                        "smlal2 v16.8h, v0.16b, v10.16b\n\t"
                        "smlal2 v17.8h, v1.16b, v10.16b\n\t"
                        "smlal2 v18.8h, v2.16b, v10.16b\n\t"
                        "smlal2 v19.8h, v3.16b, v10.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WL, AL[3]
                        "smlal2 v20.8h, v0.16b, v11.16b\n\t"
                        "smlal2 v21.8h, v1.16b, v11.16b\n\t"
                        "smlal2 v22.8h, v2.16b, v11.16b\n\t"
                        "smlal2 v23.8h, v3.16b, v11.16b\n\t"

                        // 
                        // Higher half of weight and activations vectors
                        // 

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
                        // SMLAL ACC1, WH, AH[0]
                        "smlal v24.8h, v4.8b, v12.8b\n\t"
                        "smlal v25.8h, v5.8b, v12.8b\n\t"
                        "smlal v26.8h, v6.8b, v12.8b\n\t"
                        "smlal v27.8h, v7.8b, v12.8b\n\t"

                        // Activation Row #2
                        // SMLAL ACC2, WH, AH[1]
                        "smlal v28.8h, v4.8b, v13.8b\n\t"
                        "smlal v29.8h, v5.8b, v13.8b\n\t"
                        "smlal v30.8h, v6.8b, v13.8b\n\t"
                        "smlal v31.8h, v7.8b, v13.8b\n\t"

                        // Activation Row #3
                        // SMLAL ACC3, WH, AH[2]
                        "smlal v16.8h, v4.8b, v14.8b\n\t"
                        "smlal v17.8h, v5.8b, v14.8b\n\t"
                        "smlal v18.8h, v6.8b, v14.8b\n\t"
                        "smlal v19.8h, v7.8b, v14.8b\n\t"

                        // Activation Row #4
                        // SMLAL ACC4, WH, AH[3]
                        "smlal v20.8h, v4.8b, v15.8b\n\t"
                        "smlal v21.8h, v5.8b, v15.8b\n\t"
                        "smlal v22.8h, v6.8b, v15.8b\n\t"
                        "smlal v23.8h, v7.8b, v15.8b\n\t"

                        // Activation Row #1
                        // SMLAL2 ACC1, WH, AH[0]
                        "smlal2 v24.8h, v4.16b, v12.16b\n\t"
                        "smlal2 v25.8h, v5.16b, v12.16b\n\t"
                        "smlal2 v26.8h, v6.16b, v12.16b\n\t"
                        "smlal2 v27.8h, v7.16b, v12.16b\n\t"

                        // Activation Row #2
                        // SMLAL2 ACC2, WH, AH[1]
                        "smlal2 v28.8h, v4.16b, v13.16b\n\t"
                        "smlal2 v29.8h, v5.16b, v13.16b\n\t"
                        "smlal2 v30.8h, v6.16b, v13.16b\n\t"
                        "smlal2 v31.8h, v7.16b, v13.16b\n\t"

                        // Activation Row #3
                        // SMLAL2 ACC3, WH, AH[2]
                        "smlal2 v16.8h, v4.16b, v14.16b\n\t"
                        "smlal2 v17.8h, v5.16b, v14.16b\n\t"
                        "smlal2 v18.8h, v6.16b, v14.16b\n\t"
                        "smlal2 v19.8h, v7.16b, v14.16b\n\t"

                        // Activation Row #4
                        // SMLAL2 ACC4, WH, AH[3]
                        "smlal2 v20.8h, v4.16b, v15.16b\n\t"
                        "smlal2 v21.8h, v5.16b, v15.16b\n\t"
                        "smlal2 v22.8h, v6.16b, v15.16b\n\t"
                        "smlal2 v23.8h, v7.16b, v15.16b\n\t"

                        // Check if the loop over rows of weight matrix is done
                        "add %w[i], %w[i], #32\n\t"
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
                        "add x1, x1, %[size], asr #2\n\t"
                        
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
            }
        }
    }
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        namespace SelfDependent {
            namespace W4A4{
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