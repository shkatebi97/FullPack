#ifdef BAZEL_BUILD
#include "../low_precision_fully_connected.h"
#else
#include "../low_precision_fully_connected.h"
#endif
#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        namespace Int8InputsInt4PowerWeights {
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 4);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 32)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                if (is_multibatched && shape.size[0] % 4)
                    return Status::SizesMisMatch; 
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    if (is_multibatched)
                        std::copy(input, input + shape.flatsize, output);
                    else
                        return Status::NotNeeded;
                }
                else {
                    if (is_multibatched){
                        int8_t* input_casted = const_cast<int8_t*>(input);
                        doLowPrecisionWeightPack(input_casted, output, shape.size[0], shape.size[1]);
                    }
                    else
                        return Status::NotNeeded;
                    // std::copy(input, input + shape.flatsize, output);
                }
                return Status::Success;
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 2);
                }
                else {
                    int new_weights_length = k_shape.size[0] * (k_shape.size[1] / 2);
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    zero_vector(temp, new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    int i , size = k_shape.flatsize;
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
                        "st1 {v0.4s},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #32\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"

                        "3:\n\t"

                        : [ output ] "+r"(temp_u), [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size)
                        : "v0",  "v1",  "v2",  "v3",
                            "v28", "v29", "v30", "v31",
                            "w3",  "w4",  "w5",  "w6"
                    );
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 2);
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
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

                    "0:\n\t"
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

                    // SMLAL T1.8h, WH.8b, A2.8b
                    "smlal v13.8h, v0.8b, v5.8b\n\t"
                    "smlal v14.8h, v1.8b, v5.8b\n\t"
                    "smlal v15.8h, v2.8b, v5.8b\n\t"
                    "smlal v16.8h, v3.8b, v5.8b\n\t"

                    // SMLAL2 T1.8h, WH.16b, A2.16b
                    "smlal2 v13.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v14.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v16.8h, v3.16b, v5.16b\n\t"

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
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 4)
                    return Status::NotSupported;
                int8_t*         _kernel     = const_cast<int8_t*>(kernel);
                int8_t*         _kernel_base= const_cast<int8_t*>(kernel);
                int8_t*         _input      = get_pointer_as<int8_t>(const_cast<int8_t*>(input));
                int8_t*         _input_base = get_pointer_as<int8_t>(const_cast<int8_t*>(input));
                int i, j, k, end;
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W, WH     -> v0-3      (Weights, Weights High)
                    * A         -> v4-7      (Activations)
                    * WL        -> v8-11     (Weights Low)
                    * MiniACC   -> v12-15    (Mini Accumulator)
                    * ACC1      -> v16-19    (Accumulators input row #1)
                    * ACC2      -> v20-23    (Accumulators input row #2)
                    * ACC3      -> v24-27    (Accumulators input row #3)
                    * ACC4      -> v28-31    (Accumulators input row #4)
                */
                asm volatile(
                    "mov w0, #12\n\t"
                    "mul w0, %w[rows], w0\n\t"
                    "mov %w[k], wzr\n\t"
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
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

#ifdef DISABLE_KERNELS_MEM_ACCESS
                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]]\n\t"
                    "ld1 {v5.16b},  [%[activation]]\n\t"
                    "ld1 {v6.16b},  [%[activation]]\n\t"
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif
                    // Start of Outer Loop Over Weights
                    "1:\n\t"

#ifdef DISABLE_KERNELS_MEM_ACCESS
                    // Load Weights
                    "ld1 {v0.16b},  [%[weights]]\n\t"
                    "ld1 {v1.16b},  [%[weights]]\n\t"
                    "ld1 {v2.16b},  [%[weights]]\n\t"
                    "ld1 {v3.16b},  [%[weights]]\n\t"
#else
                    // Load Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"
#endif
                    
                    // SHL WL, WL, #4
                    "shl v8.16b,  v0.16b,  #4\n\t"
                    "shl v9.16b, v1.16b,  #4\n\t"
                    "shl v10.16b, v2.16b, #4\n\t"
                    "shl v11.16b, v3.16b, #4\n\t"

                    // SSHR WH, W, #4
                    "sshr v0.16b, v0.16b, #4\n\t"
                    "sshr v1.16b, v1.16b, #4\n\t"
                    "sshr v2.16b, v2.16b, #4\n\t"
                    "sshr v3.16b, v3.16b, #4\n\t"

                    // SSHR WL, WL, #4
                    "sshr v8.16b,  v8.16b,  #4\n\t"
                    "sshr v9.16b,  v9.16b,  #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // Activation Row #1
                    // SMULL MiniACC, WL, A
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, WL, A
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC1, MiniACC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WL, A
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, WL, A
                    "smlal2 v12.8h, v8.16b,  v5.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v5.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v5.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v5.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC2, MiniACC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #3
                    // SMULL MiniACC, WL, A
                    "smull v12.8h, v8.8b,  v6.8b\n\t"
                    "smull v13.8h, v9.8b,  v6.8b\n\t"
                    "smull v14.8h, v10.8b, v6.8b\n\t"
                    "smull v15.8h, v11.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, WL, A
                    "smlal2 v12.8h, v8.16b,  v6.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v6.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v6.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v6.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC3, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #4
                    // SMULL MiniACC, WL, A
                    "smull v12.8h, v8.8b,  v7.8b\n\t"
                    "smull v13.8h, v9.8b,  v7.8b\n\t"
                    "smull v14.8h, v10.8b, v7.8b\n\t"
                    "smull v15.8h, v11.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, WL, A
                    "smlal2 v12.8h, v8.16b,  v7.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v7.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v7.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC4, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // 
                    // Higher half of weight vectors
                    // 
                    // Activation Row #1
                    // SMULL MiniACC, WH, A
                    "smull v12.8h, v0.8b, v4.8b\n\t"
                    "smull v13.8h, v1.8b, v4.8b\n\t"
                    "smull v14.8h, v2.8b, v4.8b\n\t"
                    "smull v15.8h, v3.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, WH, A
                    "smlal2 v12.8h, v0.16b, v4.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v4.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v4.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC1, MiniACC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WH, A
                    "smull v12.8h, v0.8b, v5.8b\n\t"
                    "smull v13.8h, v1.8b, v5.8b\n\t"
                    "smull v14.8h, v2.8b, v5.8b\n\t"
                    "smull v15.8h, v3.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, WH, A
                    "smlal2 v12.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v5.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v5.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC2, MiniACC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #3
                    // SMULL MiniACC, WH, A
                    "smull v12.8h, v0.8b, v6.8b\n\t"
                    "smull v13.8h, v1.8b, v6.8b\n\t"
                    "smull v14.8h, v2.8b, v6.8b\n\t"
                    "smull v15.8h, v3.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, WH, A
                    "smlal2 v12.8h, v0.16b, v6.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v6.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v6.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v6.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC3, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #4
                    // SMULL MiniACC, WH, A
                    "smull v12.8h, v0.8b, v7.8b\n\t"
                    "smull v13.8h, v1.8b, v7.8b\n\t"
                    "smull v14.8h, v2.8b, v7.8b\n\t"
                    "smull v15.8h, v3.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, WH, A
                    "smlal2 v12.8h, v0.16b, v7.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v7.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v7.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC4, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // Check if the loop over rows of weight matrix is done
                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Accumulate the ACC1 to one int32
                    "addv s16, v16.4s\n\t"
                    "addv s17, v17.4s\n\t"
                    "addv s18, v18.4s\n\t"
                    "addv s19, v19.4s\n\t"

                    // Accumulate the ACC2 to one int32
                    "addv s20, v20.4s\n\t"
                    "addv s21, v21.4s\n\t"
                    "addv s22, v22.4s\n\t"
                    "addv s23, v23.4s\n\t"

                    // Accumulate the ACC3 to one int32
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s26, v26.4s\n\t"
                    "addv s27, v27.4s\n\t"

                    // Accumulate the ACC4 to one int32
                    "addv s28, v28.4s\n\t"
                    "addv s29, v29.4s\n\t"
                    "addv s30, v30.4s\n\t"
                    "addv s31, v31.4s\n\t"

                    // Reorder ACC1 to store
                    "mov v16.s[1], v17.s[0]\n\t"
                    "mov v16.s[2], v18.s[0]\n\t"
                    "mov v16.s[3], v19.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v20.s[1], v21.s[0]\n\t"
                    "mov v20.s[2], v22.s[0]\n\t"
                    "mov v20.s[3], v23.s[0]\n\t"

                    // Reorder ACC3 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC4 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"
                    
                    // Store the 4 int32 results
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                    
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

                    // Prepare the destination base for next 4 batches
                    "add %[dst_1], %[dst_1], x0\n\t"
                    "add %[dst_2], %[dst_2], x0\n\t"
                    "add %[dst_3], %[dst_3], x0\n\t"
                    "add %[dst_4], %[dst_4], x0\n\t"

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
                        [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                        [ batches ]    "r"  (lhs_batches)

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
        }
    }
}
#endif
