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
        namespace Int3InputsInt3Weights {
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 8.0) * 8 / floor(16 / 3) * 2;
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] /= ceil(shape[n_dims - 1] / 8.0) * 8 / floor(16 / 3) * 2;
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 4 || k_shape.size[k_shape.number_dims - 1] % 40)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], (k_shape.size[1] / 5) * 2);
                }
                else {
                    int new_weights_length = k_shape.size[0] * ((k_shape.size[1] / 5) * 2);
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    zero_vector(temp, new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    int i , size = k_shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #7  \n\t"
                        "movi v30.16b, #3  \n\t"
                        "movi v29.16b, #252\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.8b},  [%[input]], #8\n\t"
                        "ld1 {v1.8b},  [%[input]], #8\n\t"
                        "ld1 {v2.8b},  [%[input]], #8\n\t"
                        "ld1 {v3.8b},  [%[input]], #8\n\t"
                        "ld1 {v4.8b},  [%[input]], #8\n\t"

                        // Saturate to 7
                        // CMGE AT, A, 7
                        "cmge v5.8b, v0.8b, v30.8b\n\t"
                        "cmge v6.8b, v1.8b, v30.8b\n\t"
                        "cmge v7.8b, v2.8b, v30.8b\n\t"
                        "cmge v8.8b, v3.8b, v30.8b\n\t"
                        "cmge v9.8b, v4.8b, v30.8b\n\t"
                        // NOT AT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND A, A, AT
                        "and v0.8b, v0.8b, v5.8b\n\t"
                        "and v1.8b, v1.8b, v6.8b\n\t"
                        "and v2.8b, v2.8b, v7.8b\n\t"
                        "and v3.8b, v3.8b, v8.8b\n\t"
                        "and v4.8b, v4.8b, v9.8b\n\t"
                        // NOT ATT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND ATTT, A, ATT
                        "and v5.8b, v30.8b, v5.8b\n\t"
                        "and v6.8b, v30.8b, v6.8b\n\t"
                        "and v7.8b, v30.8b, v7.8b\n\t"
                        "and v8.8b, v30.8b, v8.8b\n\t"
                        "and v9.8b, v30.8b, v9.8b\n\t"
                        // ORR A, A, ATT
                        "orr v0.8b, v0.8b, v5.8b\n\t"
                        "orr v1.8b, v1.8b, v6.8b\n\t"
                        "orr v2.8b, v2.8b, v7.8b\n\t"
                        "orr v3.8b, v3.8b, v8.8b\n\t"
                        "orr v4.8b, v4.8b, v9.8b\n\t"
                        
                        // Saturate to -8
                        // CMGE AT, A, -8
                        "cmge v5.8b, v0.8b, v29.8b\n\t"
                        "cmge v6.8b, v1.8b, v29.8b\n\t"
                        "cmge v7.8b, v2.8b, v29.8b\n\t"
                        "cmge v8.8b, v3.8b, v29.8b\n\t"
                        "cmge v9.8b, v4.8b, v29.8b\n\t"
                        // AND A, A, AT
                        "and v0.8b, v0.8b, v5.8b\n\t"
                        "and v1.8b, v1.8b, v6.8b\n\t"
                        "and v2.8b, v2.8b, v7.8b\n\t"
                        "and v3.8b, v3.8b, v8.8b\n\t"
                        "and v4.8b, v4.8b, v9.8b\n\t"
                        // NOT ATT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND ATTT, A, ATT
                        "and v5.8b, v29.8b, v5.8b\n\t"
                        "and v6.8b, v29.8b, v6.8b\n\t"
                        "and v7.8b, v29.8b, v7.8b\n\t"
                        "and v8.8b, v29.8b, v8.8b\n\t"
                        "and v9.8b, v29.8b, v9.8b\n\t"
                        // ORR A, A, ATTT
                        "orr v0.8b, v0.8b, v5.8b\n\t"
                        "orr v1.8b, v1.8b, v6.8b\n\t"
                        "orr v2.8b, v2.8b, v7.8b\n\t"
                        "orr v3.8b, v3.8b, v8.8b\n\t"
                        "orr v4.8b, v4.8b, v9.8b\n\t"

                        // Pack 5 Saturated Int3 in 1 Int16
                        // AND AL, AL, 0x07
                        "and v0.8b, v0.8b, v31.8b\n\t"
                        "and v1.8b, v1.8b, v31.8b\n\t"
                        "and v2.8b, v2.8b, v31.8b\n\t"
                        "and v3.8b, v3.8b, v31.8b\n\t"
                        "and v4.8b, v4.8b, v31.8b\n\t"
                        // UXTL A, A
                        "uxtl v0.8h, v0.8b\n\t"
                        "uxtl v1.8h, v1.8b\n\t"
                        "uxtl v2.8h, v2.8b\n\t"
                        "uxtl v3.8h, v3.8b\n\t"
                        "uxtl v4.8h, v4.8b\n\t"
                        // SHL AH, AH, #x
                        "shl v0.8h, v0.8h, #13\n\t"
                        "shl v1.8h, v1.8h, #10\n\t"
                        "shl v2.8h, v2.8h, #7 \n\t"
                        "shl v3.8h, v3.8h, #4 \n\t"
                        "shl v4.8h, v4.8h, #1 \n\t"
                        // ORR AM, AL, Ah
                        "orr v0.16b, v0.16b, v0.16b\n\t"
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v0.16b, v0.16b, v3.16b\n\t"
                        "orr v0.16b, v0.16b, v4.16b\n\t"
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #40\n\t"
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
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], (k_shape.size[1] / 5) * 2);
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (!((shape.number_dims == 2 && shape.size[0] == 1) || shape.number_dims == 1))
                    return Status::DimensionsMisMatch;
                if (shape.size[shape.number_dims - 1] % 40)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    std::copy(input, input + ((shape.flatsize / 5) * 2), output);
                }
                else {
                    int i , size = shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #7  \n\t"
                        "movi v30.16b, #3  \n\t"
                        "movi v29.16b, #252\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.8b},  [%[input]], #8\n\t"
                        "ld1 {v1.8b},  [%[input]], #8\n\t"
                        "ld1 {v2.8b},  [%[input]], #8\n\t"
                        "ld1 {v3.8b},  [%[input]], #8\n\t"
                        "ld1 {v4.8b},  [%[input]], #8\n\t"

                        // Saturate to 7
                        // CMGE AT, A, 7
                        "cmge v5.8b, v0.8b, v30.8b\n\t"
                        "cmge v6.8b, v1.8b, v30.8b\n\t"
                        "cmge v7.8b, v2.8b, v30.8b\n\t"
                        "cmge v8.8b, v3.8b, v30.8b\n\t"
                        "cmge v9.8b, v4.8b, v30.8b\n\t"
                        // NOT AT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND A, A, AT
                        "and v0.8b, v0.8b, v5.8b\n\t"
                        "and v1.8b, v1.8b, v6.8b\n\t"
                        "and v2.8b, v2.8b, v7.8b\n\t"
                        "and v3.8b, v3.8b, v8.8b\n\t"
                        "and v4.8b, v4.8b, v9.8b\n\t"
                        // NOT ATT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND ATTT, A, ATT
                        "and v5.8b, v30.8b, v5.8b\n\t"
                        "and v6.8b, v30.8b, v6.8b\n\t"
                        "and v7.8b, v30.8b, v7.8b\n\t"
                        "and v8.8b, v30.8b, v8.8b\n\t"
                        "and v9.8b, v30.8b, v9.8b\n\t"
                        // ORR A, A, ATT
                        "orr v0.8b, v0.8b, v5.8b\n\t"
                        "orr v1.8b, v1.8b, v6.8b\n\t"
                        "orr v2.8b, v2.8b, v7.8b\n\t"
                        "orr v3.8b, v3.8b, v8.8b\n\t"
                        "orr v4.8b, v4.8b, v9.8b\n\t"
                        
                        // Saturate to -8
                        // CMGE AT, A, -8
                        "cmge v5.8b, v0.8b, v29.8b\n\t"
                        "cmge v6.8b, v1.8b, v29.8b\n\t"
                        "cmge v7.8b, v2.8b, v29.8b\n\t"
                        "cmge v8.8b, v3.8b, v29.8b\n\t"
                        "cmge v9.8b, v4.8b, v29.8b\n\t"
                        // AND A, A, AT
                        "and v0.8b, v0.8b, v5.8b\n\t"
                        "and v1.8b, v1.8b, v6.8b\n\t"
                        "and v2.8b, v2.8b, v7.8b\n\t"
                        "and v3.8b, v3.8b, v8.8b\n\t"
                        "and v4.8b, v4.8b, v9.8b\n\t"
                        // NOT ATT, AT
                        "not v5.8b, v5.8b\n\t"
                        "not v6.8b, v6.8b\n\t"
                        "not v7.8b, v7.8b\n\t"
                        "not v8.8b, v8.8b\n\t"
                        "not v9.8b, v9.8b\n\t"
                        // AND ATTT, A, ATT
                        "and v5.8b, v29.8b, v5.8b\n\t"
                        "and v6.8b, v29.8b, v6.8b\n\t"
                        "and v7.8b, v29.8b, v7.8b\n\t"
                        "and v8.8b, v29.8b, v8.8b\n\t"
                        "and v9.8b, v29.8b, v9.8b\n\t"
                        // ORR A, A, ATTT
                        "orr v0.8b, v0.8b, v5.8b\n\t"
                        "orr v1.8b, v1.8b, v6.8b\n\t"
                        "orr v2.8b, v2.8b, v7.8b\n\t"
                        "orr v3.8b, v3.8b, v8.8b\n\t"
                        "orr v4.8b, v4.8b, v9.8b\n\t"

                        // Pack 5 Saturated Int3 in 1 Int16
                        // AND AL, AL, 0x07
                        "and v0.8b, v0.8b, v31.8b\n\t"
                        "and v1.8b, v1.8b, v31.8b\n\t"
                        "and v2.8b, v2.8b, v31.8b\n\t"
                        "and v3.8b, v3.8b, v31.8b\n\t"
                        "and v4.8b, v4.8b, v31.8b\n\t"
                        // UXTL A, A
                        "uxtl v0.8h, v0.8b\n\t"
                        "uxtl v1.8h, v1.8b\n\t"
                        "uxtl v2.8h, v2.8b\n\t"
                        "uxtl v3.8h, v3.8b\n\t"
                        "uxtl v4.8h, v4.8b\n\t"
                        // SHL AH, AH, #x
                        "shl v0.8h, v0.8h, #13\n\t"
                        "shl v1.8h, v1.8h, #10\n\t"
                        "shl v2.8h, v2.8h, #7 \n\t"
                        "shl v3.8h, v3.8h, #4 \n\t"
                        "shl v4.8h, v4.8h, #1 \n\t"
                        // ORR AM, AL, Ah
                        "orr v0.16b, v0.16b, v0.16b\n\t"
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v0.16b, v0.16b, v3.16b\n\t"
                        "orr v0.16b, v0.16b, v4.16b\n\t"
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #40\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"

                        "3:\n\t"

                        : [ output ] "+r"(output), [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size)
                        : "v0",  "v1",  "v2",  "v3", "v4",
                          "v5",  "v6",  "v7",  "v8", "v9",
                          "v29", "v30", "v31"
                    );
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
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    
                    // Load Weights
                    "ld1 {v0.8h},  [%[weights]], #16\n\t"
                    "ld1 {v1.8h},  [%[weights]], #16\n\t"
                    "ld1 {v2.8h},  [%[weights]], #16\n\t"
                    "ld1 {v3.8h},  [%[weights]], #16\n\t"
                    // Load Activation
                    "ld1 {v4.8h},  [%[activation]], #16\n\t"

                    // Setting the iterations of inner loop
                    "add %w[end], %w[i], #40\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    "2:\n\t"
                    // SSHR AT, A, #13
                    "sshr v5.8h, v4.8h, #13\n\t"
                    // SHL A, A, #3
                    "shl v4.8h, v4.8h,  #3\n\t"
                    
                    // SSHR WT, W, #13
                    "sshr v6.8h, v0.8h, #13\n\t"
                    "sshr v7.8h, v1.8h, #13\n\t"
                    "sshr v8.8h, v2.8h, #13\n\t"
                    "sshr v9.8h, v3.8h, #13\n\t"
                    // SHL W, W, #3
                    "shl  v0.8h, v0.8h, #3\n\t"
                    "shl  v1.8h, v1.8h, #3\n\t"
                    "shl  v2.8h, v2.8h, #3\n\t"
                    "shl  v3.8h, v3.8h, #3\n\t"

                    // SMLAL ACC, W, A
                    "smlal  v23.4s, v6.4h, v5.4h\n\t"
                    "smlal  v24.4s, v7.4h, v5.4h\n\t"
                    "smlal  v25.4s, v8.4h, v5.4h\n\t"
                    "smlal  v30.4s, v9.4h, v5.4h\n\t"
                    "smlal2 v23.4s, v6.8h, v5.8h\n\t"
                    "smlal2 v24.4s, v7.8h, v5.8h\n\t"
                    "smlal2 v25.4s, v8.8h, v5.8h\n\t"
                    "smlal2 v30.4s, v9.8h, v5.8h\n\t"

                    // Increment the loop counter with 16 and compare with end
                    "add %w[i], %w[i], #8\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

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
