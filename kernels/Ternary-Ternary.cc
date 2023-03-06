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
        using ::LowPrecision::MulParams;
        namespace TernaryInputsTernaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 16)?(16 - (shape.size[1] % 16)):(0);
                Shape new_shape;
                new_shape.number_dims = shape.number_dims;
                new_shape.size = new int[new_shape.number_dims];
                new_shape.size[0] = shape.size[0];
                new_shape.size[1] = shape.size[1] + padding_size;
                new_shape.flatsize = new_shape.size[0] * new_shape.size[1];
                int8_t* new_weight = allocate<int8_t>(new_shape.flatsize);
                for(int i = 0 ; i < shape.size[0] ; i++){
                    for(int j = 0 ; j < shape.size[1] ; j++)
                        new_weight[i * new_shape.size[1] + j] = weight[i * shape.size[1] + j];
                    for (int j = shape.size[1]; j < shape.size[1] + padding_size; j++)
                        new_weight[i * new_shape.size[1] + j] = 0;
                }
                return new_weight;
            }
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 2);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 2);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 4 || k_shape.size[k_shape.number_dims - 1] % 64)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 4);
                }
                else {
                    int new_weights_length = (k_shape.size[1] / 4) * k_shape.size[0];
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    int i , size = k_shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #255\n\t"
                        "movi v30.16b, #6\n\t"
                        "movi v29.16b, #4\n\t"
                        "movi v28.16b, #2\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of The Main Loop
                        "1:\n\t"
                        // Load A
                        "ld1 {v0.16b},  [%[input]], #16\n\t"
                        "ld1 {v1.16b},  [%[input]], #16\n\t"
                        "ld1 {v2.16b},  [%[input]], #16\n\t"
                        "ld1 {v3.16b},  [%[input]], #16\n\t"

                        // Saturate and set zero

                        // USHR AT, A, 6
                        "ushr v4.16b, v0.16b, #6\n\t"
                        "ushr v5.16b, v1.16b, #6\n\t"
                        "ushr v6.16b, v2.16b, #6\n\t"
                        "ushr v7.16b, v3.16b, #6\n\t"

                        // CMTST ATT, A, 6
                        "cmtst v0.16b, v0.16b, v31.16b\n\t"
                        "cmtst v1.16b, v1.16b, v31.16b\n\t"
                        "cmtst v2.16b, v2.16b, v31.16b\n\t"
                        "cmtst v3.16b, v3.16b, v31.16b\n\t"

                        // USHR ATT, ATT, 7
                        "ushr v0.16b, v0.16b, #7\n\t"
                        "ushr v1.16b, v1.16b, #7\n\t"
                        "ushr v2.16b, v2.16b, #7\n\t"
                        "ushr v3.16b, v3.16b, #7\n\t"

                        // ORR AS, ATT, AT
                        "orr v0.16b, v0.16b, v4.16b\n\t"
                        "orr v1.16b, v1.16b, v5.16b\n\t"
                        "orr v2.16b, v2.16b, v6.16b\n\t"
                        "orr v3.16b, v3.16b, v7.16b\n\t"

                        // USHL AS, AS, x
                        "ushl v0.16b, v0.16b, v30.16b\n\t"
                        "ushl v1.16b, v1.16b, v29.16b\n\t"
                        "ushl v2.16b, v2.16b, v28.16b\n\t"

                        // USHL AS, AS, x
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v0.16b, v0.16b, v3.16b\n\t"
                        
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #64\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"

                        "3:\n\t"

                        : [ output ] "+r"(temp_u), [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size)
                        : "v0",  "v1",  "v2",  "v3",
                          "v4",  "v5",  "v6",  "v7",
                          "v28", "v29", "v30", "v31",
                          "w3",  "w4",  "w5",  "w6"
                    );
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 4);
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 64)
                    return Status::SizesMisMatch; 
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
                    int8_t* temp = output;
                    if (is_multibatched){
                        int new_weights_length = ((int)shape.flatsize / 4);
                        temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    }
                    int i , size = shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #255\n\t"
                        "movi v30.16b, #6\n\t"
                        "movi v29.16b, #4\n\t"
                        "movi v28.16b, #2\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        // Load A
                        "ld1 {v0.16b},  [%[input]], #16\n\t"
                        "ld1 {v1.16b},  [%[input]], #16\n\t"
                        "ld1 {v2.16b},  [%[input]], #16\n\t"
                        "ld1 {v3.16b},  [%[input]], #16\n\t"

                        // Saturate and set zero

                        // USHR AT, A, 6
                        "ushr v4.16b, v0.16b, #6\n\t"
                        "ushr v5.16b, v1.16b, #6\n\t"
                        "ushr v6.16b, v2.16b, #6\n\t"
                        "ushr v7.16b, v3.16b, #6\n\t"

                        // CMTST ATT, A, 6
                        "cmtst v0.16b, v0.16b, v31.16b\n\t"
                        "cmtst v1.16b, v1.16b, v31.16b\n\t"
                        "cmtst v2.16b, v2.16b, v31.16b\n\t"
                        "cmtst v3.16b, v3.16b, v31.16b\n\t"

                        // USHR ATT, ATT, 7
                        "ushr v0.16b, v0.16b, #7\n\t"
                        "ushr v1.16b, v1.16b, #7\n\t"
                        "ushr v2.16b, v2.16b, #7\n\t"
                        "ushr v3.16b, v3.16b, #7\n\t"

                        // ORR AS, ATT, AT
                        "orr v0.16b, v0.16b, v4.16b\n\t"
                        "orr v1.16b, v1.16b, v5.16b\n\t"
                        "orr v2.16b, v2.16b, v6.16b\n\t"
                        "orr v3.16b, v3.16b, v7.16b\n\t"

                        // USHL AS, AS, x
                        "ushl v0.16b, v0.16b, v30.16b\n\t"
                        "ushl v1.16b, v1.16b, v29.16b\n\t"
                        "ushl v2.16b, v2.16b, v28.16b\n\t"

                        // USHL AS, AS, x
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v0.16b, v0.16b, v3.16b\n\t"
                        
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #64\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input],  %[input],  %[size]\n\t"
                        "sub %[output], %[output], %[size], asr #2\n\t"

                        "3:\n\t"

                        : [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size), [ output ] "r"(temp)
                        : "v0",  "v1",  "v2",  "v3",
                          "v4",  "v5",  "v6",  "v7",
                          "v28", "v29", "v30", "v31",
                          "w3",  "w4",  "w5",  "w6"
                    );
                    if (is_multibatched){
                        doLowPrecisionWeightPack(temp, output, shape.size[0], shape.size[1] / 4);
                        LowPrecision::deallocate(temp);
                    }
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
                int i, j, end;

                asm volatile(
                    "mov %w[j], wzr\n\t"

                    // "cmp %w[rows], #0\n\t"
                    // "beq 5f\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    
                    // Reseting AC
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    // If size is zero, discard
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    
                    // Reseting MiniAC
                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"

                    // Loading Next Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"
                    
                    // Start of Inner Loop Over Activations
                    // Unfolding Inner Loop
                    // Iteration #1 Unfolded
                    // SSHR AT, A, #6
                    "sshr v5.16b,  v4.16b, #6\n\t"

                    // SSHR WT, W, #6
                    "sshr v6.16b,  v0.16b, #6\n\t"
                    "sshr v7.16b,  v1.16b, #6\n\t"
                    "sshr v8.16b,  v2.16b, #6\n\t"
                    "sshr v9.16b,  v3.16b, #6\n\t"
                    
                    // SMLAL2 MiniAC.8h, W.8b, AT.8b
                    "smlal v26.8h, v6.8b, v5.8b\n\t"
                    "smlal v27.8h, v7.8b, v5.8b\n\t"
                    "smlal v28.8h, v8.8b, v5.8b\n\t"
                    "smlal v31.8h, v9.8b, v5.8b\n\t"

                    // SMLAL2 MiniAC.8h, W.16b, AT.16b
                    "smlal2 v26.8h, v6.16b,  v5.16b\n\t"
                    "smlal2 v27.8h, v7.16b, v5.16b\n\t"
                    "smlal2 v28.8h, v8.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v9.16b, v5.16b\n\t"
                    
                    // SHL A, A , #2
                    "shl v4.16b, v4.16b, #2\n"
                    
                    // SHL W, W , #2
                    "shl v0.16b, v0.16b, #2\n"
                    "shl v1.16b, v1.16b, #2\n"
                    "shl v2.16b, v2.16b, #2\n"
                    "shl v3.16b, v3.16b, #2\n"

                    // Iteration #2 Unfolded
                    // SSHR AT, A, #6
                    "sshr v5.16b,  v4.16b, #6\n\t"

                    // SSHR WT, W, #6
                    "sshr v6.16b,  v0.16b, #6\n\t"
                    "sshr v7.16b,  v1.16b, #6\n\t"
                    "sshr v8.16b,  v2.16b, #6\n\t"
                    "sshr v9.16b,  v3.16b, #6\n\t"
                    
                    // SMLAL2 MiniAC.8h, W.8b, AT.8b
                    "smlal v26.8h, v6.8b, v5.8b\n\t"
                    "smlal v27.8h, v7.8b, v5.8b\n\t"
                    "smlal v28.8h, v8.8b, v5.8b\n\t"
                    "smlal v31.8h, v9.8b, v5.8b\n\t"

                    // SMLAL2 MiniAC.8h, W.16b, AT.16b
                    "smlal2 v26.8h, v6.16b,  v5.16b\n\t"
                    "smlal2 v27.8h, v7.16b, v5.16b\n\t"
                    "smlal2 v28.8h, v8.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v9.16b, v5.16b\n\t"
                    
                    // SHL A, A , #2
                    "shl v4.16b, v4.16b, #2\n"
                    
                    // SHL W, W , #2
                    "shl v0.16b, v0.16b, #2\n"
                    "shl v1.16b, v1.16b, #2\n"
                    "shl v2.16b, v2.16b, #2\n"
                    "shl v3.16b, v3.16b, #2\n"

                    // Iteration #3 Unfolded
                    // SSHR AT, A, #6
                    "sshr v5.16b,  v4.16b, #6\n\t"

                    // SSHR WT, W, #6
                    "sshr v6.16b,  v0.16b, #6\n\t"
                    "sshr v7.16b,  v1.16b, #6\n\t"
                    "sshr v8.16b,  v2.16b, #6\n\t"
                    "sshr v9.16b,  v3.16b, #6\n\t"
                    
                    // SMLAL2 MiniAC.8h, W.8b, AT.8b
                    "smlal v26.8h, v6.8b, v5.8b\n\t"
                    "smlal v27.8h, v7.8b, v5.8b\n\t"
                    "smlal v28.8h, v8.8b, v5.8b\n\t"
                    "smlal v31.8h, v9.8b, v5.8b\n\t"

                    // SMLAL2 MiniAC.8h, W.16b, AT.16b
                    "smlal2 v26.8h, v6.16b,  v5.16b\n\t"
                    "smlal2 v27.8h, v7.16b, v5.16b\n\t"
                    "smlal2 v28.8h, v8.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v9.16b, v5.16b\n\t"
                    
                    // SHL A, A , #2
                    "shl v4.16b, v4.16b, #2\n"
                    
                    // SHL W, W , #2
                    "shl v0.16b, v0.16b, #2\n"
                    "shl v1.16b, v1.16b, #2\n"
                    "shl v2.16b, v2.16b, #2\n"
                    "shl v3.16b, v3.16b, #2\n"

                    // Iteration #4 Unfolded
                    // SSHR AT, A, #6
                    "sshr v5.16b,  v4.16b, #6\n\t"

                    // SSHR WT, W, #6
                    "sshr v6.16b,  v0.16b, #6\n\t"
                    "sshr v7.16b,  v1.16b, #6\n\t"
                    "sshr v8.16b,  v2.16b, #6\n\t"
                    "sshr v9.16b,  v3.16b, #6\n\t"
                    
                    // SMLAL2 MiniAC.8h, W.8b, AT.8b
                    "smlal v26.8h, v6.8b, v5.8b\n\t"
                    "smlal v27.8h, v7.8b, v5.8b\n\t"
                    "smlal v28.8h, v8.8b, v5.8b\n\t"
                    "smlal v31.8h, v9.8b, v5.8b\n\t"

                    // SMLAL2 MiniAC.8h, W.16b, AT.16b
                    "smlal2 v26.8h, v6.16b,  v5.16b\n\t"
                    "smlal2 v27.8h, v7.16b, v5.16b\n\t"
                    "smlal2 v28.8h, v8.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v9.16b, v5.16b\n\t"
                    // End of Unfolded Inner Loop

                    // ACCUMULATE ACC, MiniAC
                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

                    // Increment the outer loop counter and check if the whole row is processed
                    "add %w[i], %w[i], #64\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Add the 4 values inisde each vector to each other
                    "addv s23, v23.4s\n\t"
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s30, v30.4s\n\t"

                    // Put each rows end result, inside a vector
                    "mov v23.s[1], v24.s[0]\n\t"
                    "mov v23.s[2], v25.s[0]\n\t"
                    "mov v23.s[3], v30.s[0]\n\t"

                    // Save the end result for 4 rows
                    "st1 {v23.4s},  [%[dst]], #16\n\t"

                    // Reseting activation to start
                    "mov %[activation], %[lhs_base]\n\t"

                    "3:\n\t"

                    // Check if whole matrix is processed and we are done 
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"


                    : [ dst ]        "+r" (_output),     [ i ]           "+r" (i),
                      [ j ]          "+r" (j),           [ end ]         "+r" (end)

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
                int32_t* output, Shape output_shape,
                MulParams params
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
                int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 4)
                    return Status::NotSupported;
                
                int8_t*         _kernel     = const_cast<int8_t*>(kernel);
                int8_t*         _kernel_base= const_cast<int8_t*>(kernel);
                int8_t*         _input      = const_cast<int8_t*>(input);
                int8_t*         _input_base = const_cast<int8_t*>(input);
                int8_t*         _input_next = const_cast<int8_t*>(input);
                int i, j, k, end;
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W         -> v0-3      (Weights)
                    * A         -> v4-7      (Activations)
                    * WM        -> v8-11     (Masked Weights)
                    * MA        -> v12-15    (Masked Activations)
                    * ACC3      -> v16-19    (Accumulators input row #3)
                    * ACC4      -> v20-23    (Accumulators input row #4)
                    * ACC1      -> v24-27    (Accumulators input row #1)
                    * ACC2      -> v28-31    (Accumulators input row #2)
                */
                asm volatile(
                    "mov w0, #12\n\t"
                    "mul w0, %w[rows], w0\n\t"
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    
                    // Reset ACC1
                    "movi v24.8h, #0\n\t"
                    "movi v25.8h, #0\n\t"
                    "movi v26.8h, #0\n\t"
                    "movi v27.8h, #0\n\t"
                    
                    // Reset ACC2
                    "movi v28.8h, #0\n\t"
                    "movi v29.8h, #0\n\t"
                    "movi v30.8h, #0\n\t"
                    "movi v31.8h, #0\n\t"
                    
                    // Reset ACC3
                    "movi v16.8h, #0\n\t"
                    "movi v17.8h, #0\n\t"
                    "movi v18.8h, #0\n\t"
                    "movi v19.8h, #0\n\t"
                    
                    // Reset ACC4
                    "movi v20.8h, #0\n\t"
                    "movi v21.8h, #0\n\t"
                    "movi v22.8h, #0\n\t"
                    "movi v23.8h, #0\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"

                    // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v0.16b},  [%[weights]]\n\t"
                    "ld1 {v1.16b},  [%[weights]]\n\t"
                    "ld1 {v2.16b},  [%[weights]]\n\t"
                    "ld1 {v3.16b},  [%[weights]]\n\t"
#else
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"
#endif

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b},  [%[activation]]\n\t"
                    "ld1 {v5.16b},  [%[activation]]\n\t"
                    "ld1 {v6.16b},  [%[activation]]\n\t"
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif

                    "add  %w[end],  %w[i], #64\n\t"
                    "tst  %w[size], %w[end]\n\t"
                    "csel %w[end],  %w[end], %w[size], lo\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR MA, A, #6
                    "sshr v12.16b, v4.16b, #6\n\t"
                    "sshr v13.16b, v5.16b, #6\n\t"
                    "sshr v14.16b, v6.16b, #6\n\t"
                    "sshr v15.16b, v7.16b, #6\n\t"

                    // SSHR MW, W, #6
                    "sshr v8.16b,  v0.16b, #6\n\t"
                    "sshr v9.16b,  v1.16b, #6\n\t"
                    "sshr v10.16b, v2.16b, #6\n\t"
                    "sshr v11.16b, v3.16b, #6\n\t"

                    // Activation Row #1
                    // SMLAL ACC1, MW, MA[0]
                    "smlal v24.8h, v8.8b,  v12.8b\n\t"
                    "smlal v25.8h, v9.8b,  v12.8b\n\t"
                    "smlal v26.8h, v10.8b, v12.8b\n\t"
                    "smlal v27.8h, v11.8b, v12.8b\n\t"

                    // SMLAL2 ACC1, MW, MA[0]
                    "smlal2 v24.8h, v8.16b,  v12.16b\n\t"
                    "smlal2 v25.8h, v9.16b,  v12.16b\n\t"
                    "smlal2 v26.8h, v10.16b, v12.16b\n\t"
                    "smlal2 v27.8h, v11.16b, v12.16b\n\t"

                    // Activation Row #2
                    // SMLAL ACC2, MW, MA[1]
                    "smlal v28.8h, v8.8b,  v13.8b\n\t"
                    "smlal v29.8h, v9.8b,  v13.8b\n\t"
                    "smlal v30.8h, v10.8b, v13.8b\n\t"
                    "smlal v31.8h, v11.8b, v13.8b\n\t"

                    // SMLAL2 ACC2, MW, MA[1]
                    "smlal2 v28.8h, v8.16b,  v13.16b\n\t"
                    "smlal2 v29.8h, v9.16b,  v13.16b\n\t"
                    "smlal2 v30.8h, v10.16b, v13.16b\n\t"
                    "smlal2 v31.8h, v11.16b, v13.16b\n\t"

                    // Activation Row #3
                    // SMLAL ACC3, MW, MA[2]
                    "smlal v16.8h, v8.8b,  v14.8b\n\t"
                    "smlal v17.8h, v9.8b,  v14.8b\n\t"
                    "smlal v18.8h, v10.8b, v14.8b\n\t"
                    "smlal v19.8h, v11.8b, v14.8b\n\t"

                    // SMLAL2 ACC3, MW, MA[2]
                    "smlal2 v16.8h, v8.16b,  v14.16b\n\t"
                    "smlal2 v17.8h, v9.16b,  v14.16b\n\t"
                    "smlal2 v18.8h, v10.16b, v14.16b\n\t"
                    "smlal2 v19.8h, v11.16b, v14.16b\n\t"

                    // Activation Row #4
                    // SMLAL ACC4, MW, MA[3]
                    "smlal v20.8h, v8.8b,  v15.8b\n\t"
                    "smlal v21.8h, v9.8b,  v15.8b\n\t"
                    "smlal v22.8h, v10.8b, v15.8b\n\t"
                    "smlal v23.8h, v11.8b, v15.8b\n\t"

                    // SMLAL2 ACC4, MW, MA[3]
                    "smlal2 v20.8h, v8.16b,  v15.16b\n\t"
                    "smlal2 v21.8h, v9.16b,  v15.16b\n\t"
                    "smlal2 v22.8h, v10.16b, v15.16b\n\t"
                    "smlal2 v23.8h, v11.16b, v15.16b\n\t"

                    // SHL A, A, #2
                    "shl v4.16b, v4.16b, #2\n\t"
                    "shl v5.16b, v5.16b, #2\n\t"
                    "shl v6.16b, v6.16b, #2\n\t"
                    "shl v7.16b, v7.16b, #2\n\t"

                    // SHL W, W, #2
                    "shl v0.16b, v0.16b, #2\n\t"
                    "shl v1.16b, v1.16b, #2\n\t"
                    "shl v2.16b, v2.16b, #2\n\t"
                    "shl v3.16b, v3.16b, #2\n\t"

                    // Check if the 4 iterations of inner loop are done
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // SADDLP ACC1, ACC1
                    "sadalp v24.4s, v24.8h\n\t"
                    "sadalp v25.4s, v25.8h\n\t"
                    "sadalp v26.4s, v26.8h\n\t"
                    "sadalp v27.4s, v27.8h\n\t"
                    
                    // SADDLP ACC2, ACC2
                    "sadalp v28.4s, v28.8h\n\t"
                    "sadalp v29.4s, v29.8h\n\t"
                    "sadalp v30.4s, v30.8h\n\t"
                    "sadalp v31.4s, v31.8h\n\t"

                    // SADDLP ACC3, ACC3
                    "sadalp v16.4s, v16.8h\n\t"
                    "sadalp v17.4s, v17.8h\n\t"
                    "sadalp v18.4s, v18.8h\n\t"
                    "sadalp v19.4s, v19.8h\n\t"
                    
                    // SADDLP ACC4, ACC4
                    "sadalp v20.4s, v20.8h\n\t"
                    "sadalp v21.4s, v21.8h\n\t"
                    "sadalp v22.4s, v22.8h\n\t"
                    "sadalp v23.4s, v23.8h\n\t"

                    // Check if the loop over rows of activation matrix (outer loop) is done
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

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

                    // Reorder ACC1 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"

                    // Reorder ACC3 to store
                    "mov v16.s[1], v17.s[0]\n\t"
                    "mov v16.s[2], v18.s[0]\n\t"
                    "mov v16.s[3], v19.s[0]\n\t"

                    // Reorder ACC4 to store
                    "mov v20.s[1], v21.s[0]\n\t"
                    "mov v20.s[2], v22.s[0]\n\t"
                    "mov v20.s[3], v23.s[0]\n\t"
                    
                    // Check if the output need downcasting to int8
                    "cmp %w[downcast], 0xff\n\t"
                    "beq 6f\n\t"

                    // Output is needed in 32-bit; no need to downcast
                    // Store the 4 int32 results
                    "st1 {v24.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v16.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_4]], #16\n\t"

                    // Jump to after dowcasting since we dont need to downcast
                    "b 7f\n\t"

                    // Need to Downcast to Int8
                    "6:\n\t"

                    // Cast 32-bit to 16-bit
                    "sqxtn v16.4h, v16.4s\n\t"
                    "sqxtn v20.4h, v20.4s\n\t"
                    "sqxtn v24.4h, v24.4s\n\t"
                    "sqxtn v28.4h, v28.4s\n\t"

                    // Cast 16-bit to 8-bit
                    "sqxtn v16.8b, v16.8h\n\t"
                    "sqxtn v20.8b, v20.8h\n\t"
                    "sqxtn v24.8b, v24.8h\n\t"
                    "sqxtn v28.8b, v28.8h\n\t"
                    
                    // Store the 4 int8 results
                    "st1 {v24.s}[0],  [%[dst_1]], #4\n\t"
                    "st1 {v28.s}[0],  [%[dst_2]], #4\n\t"
                    "st1 {v16.s}[0],  [%[dst_3]], #4\n\t"
                    "st1 {v20.s}[0],  [%[dst_4]], #4\n\t"

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
                    "mov %[dst_1], %[dst_3]\n\t"
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
                      [ j ]          "+r" (j),           [ k ]           "+r" (k),
                      [ activation ] "+r"  (_input),     [ act_base ]    "+r"  (_input_base),
                      [ act_next ]   "+r"  (_input_next)

                    : [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                      [ batches ]    "r"  (lhs_batches), [ downcast ]    "r"  (need_downcasting)

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
            Status MultiplyInt8MultiBatched(
                const uint8_t* input, Shape input_shape,
                const uint8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                MulParams params
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
                int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 4)
                    return Status::NotSupported;

                uint8_t* _kernel = const_cast<uint8_t*>(kernel);
                uint8_t* _input  = const_cast<uint8_t*>(input);

                const int M        = lhs_batches,
                          K        = lhs_columns,
                          N        = rhs_columns,
                          a_stride = lhs_columns / 4,
                          c_stride = 4 * N;
                
                int mr_block_size = 8, nr_block_size = 8;
                for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
                    for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
                        const uint8_t* w = kernel + nr_block_start * (K / 4);
                        const uint8_t* a = input  + mr_block_start * (K / 4);
                        int32_t*       c = output + mr_block_start * N + nr_block_start;
                        int k = K;

                        const uint8_t* a0 = a;
                        const uint8_t* a1 = (a0 + a_stride);
                        const uint8_t* a2 = (a1 + a_stride);
                        const uint8_t* a3 = (a2 + a_stride);

                        // Assumes that kernel_zero_points is an array padded with necessary elements
                        // in order to make it multiple of 8.
                        uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
                        uint32x4_t vacc0x4567_=vacc0x0123_; 
                        uint32x4_t vacc1x0123_=vacc0x0123_; 
                        uint32x4_t vacc1x4567_=vacc0x0123_; 
                        uint32x4_t vacc2x0123_=vacc0x0123_; 
                        uint32x4_t vacc2x4567_=vacc0x0123_; 
                        uint32x4_t vacc3x0123_=vacc0x0123_; 
                        uint32x4_t vacc3x4567_=vacc0x0123_;

                        //4,4 : 24-12-0
                        uint16x4_t mask = vdup_n_u16(0x0003);
                        for (; k >= 64; k -= 64) {
                            uint16x4_t vxa0_ = vld1_u16((uint16_t*)a0); a0 += 8;
                            uint16x4_t vxa1_ = vld1_u16((uint16_t*)a1); a1 += 8;
                            uint16x4_t vxa2_ = vld1_u16((uint16_t*)a2); a2 += 8;
                            uint16x4_t vxa3_ = vld1_u16((uint16_t*)a3); a3 += 8;

                            uint16x4_t vxb01234567c0l_;
                            uint16x4_t vxb01234567c0h_;
                            for (int i = 0; i < 4; i++){
                                uint16x4_t vxa0, vxa1, vxa2, vxa3;

                                uint16x4_t vxb01234567c0l, vxb01234567c0h;

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 2);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 2);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 2);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 2);
                                
                                vxb01234567c0l_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h_ = vld1_u16((uint16_t*)w); w += 8;
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);

                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);

                                vxa0 = vand_u16(vxa0_, mask);
                                vxa0_ = vshr_n_u16(vxa0_, 2);
                                vxa1 = vand_u16(vxa1_, mask);
                                vxa1_ = vshr_n_u16(vxa1_, 2);
                                vxa2 = vand_u16(vxa2_, mask);
                                vxa2_ = vshr_n_u16(vxa2_, 2);
                                vxa3 = vand_u16(vxa3_, mask);
                                vxa3_ = vshr_n_u16(vxa3_, 2);
                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 0);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 0);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 0);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 0);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 0);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 0);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 0);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 0);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 1);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 1);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 1);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 1);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 1);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 1);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 1);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 1);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 2);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 2);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 2);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 2);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 2);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 2);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 2);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 2);

                                vxb01234567c0l = vand_u16(vxb01234567c0l_, mask);
                                vxb01234567c0l_ = vshr_n_u16(vxb01234567c0l_, 2);
                                vxb01234567c0h = vand_u16(vxb01234567c0h_, mask);
                                vxb01234567c0h_ = vshr_n_u16(vxb01234567c0h_, 2);
                                vacc0x0123_ = vmlal_lane_u16(vacc0x0123_, vxb01234567c0l, vxa0, 3);
                                vacc0x4567_ = vmlal_lane_u16(vacc0x4567_, vxb01234567c0h, vxa0, 3);
                                vacc1x0123_ = vmlal_lane_u16(vacc1x0123_, vxb01234567c0l, vxa1, 3);
                                vacc1x4567_ = vmlal_lane_u16(vacc1x4567_, vxb01234567c0h, vxa1, 3);
                                vacc2x0123_ = vmlal_lane_u16(vacc2x0123_, vxb01234567c0l, vxa2, 3);
                                vacc2x4567_ = vmlal_lane_u16(vacc2x4567_, vxb01234567c0h, vxa2, 3);
                                vacc3x0123_ = vmlal_lane_u16(vacc3x0123_, vxb01234567c0l, vxa3, 3);
                                vacc3x4567_ = vmlal_lane_u16(vacc3x4567_, vxb01234567c0h, vxa3, 3);
                            }
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
            Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){}
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount){
                if (input < 0)
                    if (input < -8)
                        return (get_as<uint8_t>(-8) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                    else
                        return (get_as<uint8_t>(input) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                else
                    if(input >  7)
                        return (get_as<uint8_t>(7) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                    else
                        return (get_as<uint8_t>(input) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
            }
        }
    }
}
#endif
