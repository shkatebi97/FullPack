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
        
        namespace Float32 {
            
            // not implimented for m and k != 4*x
            void doInputPackImpl(float32_t* src_ptr_1, float32_t* src_ptr_2,
                                        float32_t* src_ptr_3, float32_t* src_ptr_4,
                                        float32_t* dst_ptr_r, int rows, int size){
                                        
                
                int i, j;
                
                auto dst_ptr = dst_ptr_r;
                asm volatile(
                    "mov x0, %[src_ptr_1]\n\t"
                    
                    "0:\n\t"

                    "mov %w[i], wzr\n\t"

                    "1:\n\t"

                    "ld1 {v1.4s}, [%[src_ptr_1]], #16\n"
                    "ld1 {v2.4s}, [%[src_ptr_2]], #16\n"
                    "ld1 {v3.4s}, [%[src_ptr_3]], #16\n"
                    "ld1 {v4.4s}, [%[src_ptr_4]], #16\n"

                    //////////////////////////////////////////////////////////////////

                    "st1 {v1.4s},  [%[dst_ptr]], #16\n"
                    "st1 {v2.4s},  [%[dst_ptr]], #16\n"
                    "st1 {v3.4s},  [%[dst_ptr]], #16\n"
                    "st1 {v4.4s},  [%[dst_ptr]], #16\n"

                    //////////////////////////////////////////////////////////////////

                    "add %w[i], %w[i], #4\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"


                    // Prepare the destination base for next 4 batches
                    "mov %[src_ptr_1], %[src_ptr_4]\n\t"
                    "add %[src_ptr_2], %[src_ptr_1], %[size], lsl #2\n\t"
                    "add %[src_ptr_3], %[src_ptr_2], %[size], lsl #2\n\t"
                    "add %[src_ptr_4], %[src_ptr_3], %[size], lsl #2\n\t"

                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"


                    
                    "mov %[src_ptr_1], x0\n\t"

                    :   [ dst_ptr ]   "+r"(dst_ptr)
                    :   [ src_ptr_1 ]   "r" (src_ptr_1), [ src_ptr_2 ] "r" (src_ptr_2),
                        [ src_ptr_3 ]   "r" (src_ptr_3), [ src_ptr_4 ] "r" (src_ptr_4),
                        [i]             "r" (i),         [ size ]      "r" (size),
                        [j]             "r" (j),         [ rows ]       "r" (rows)
                    : "v1", "v2", "v3", "v4", "x0"
                );
                
                

            }
            
            void doInputPack(float32_t* src, float32_t* packed, int rows, int columns){
                float32_t* src_ptr = src;
                float32_t* src_ptr_1 = src_ptr;
                float32_t* src_ptr_2 = src + 1 * columns;
                float32_t* src_ptr_3 = src + 2 * columns;
                float32_t* src_ptr_4 = src + 3 * columns;
                float32_t* packed_ptr = packed;
                doInputPackImpl(src_ptr_1, src_ptr_2, src_ptr_3, src_ptr_4, packed_ptr, rows, columns);
            }

            Status QuantizeInput(const float32_t* input, Shape shape, float32_t* output, MemLayout layout){

                float32_t* input_casted = const_cast<float32_t*>(input);

                // Maybe can use doLowPrecisionPack() on LowPrecisionPacking.cc
                doInputPack(input_casted, output, shape.size[0], shape.size[1]);
                return Status::Success;
            }

            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
        
                int8_t* src = const_cast<int8_t*>(input);

                int8_t* dst = output;
                int i,j;
                
                asm volatile(
                    "mov x0, %x[src]\n\t"
                    "mov %w[i], wzr\n\t"

                    "0:\n\t"
                    "add x1, x0, %x[i]\n\t"

                    "mov %w[j], wzr\n\t"

                    "1:\n\t"
                    "add x2, x1, %x[columns]\n\t"
                    "add x3, x2, %x[columns]\n\t"
                    "add x4, x3, %x[columns]\n\t"
                    "add x5, x4, %x[columns]\n\t"
                    "add x6, x5, %x[columns]\n\t"
                    "add x7, x6, %x[columns]\n\t"
                    "add x8, x7, %x[columns]\n\t"

                    // load 4 int8 for each row
                    "ldr x9, [x1]\n\t"
                    "ldr x10, [x2]\n\t"
                    "ldr x11, [x3]\n\t"
                    "ldr x12, [x4]\n\t"
                    "ldr x13, [x5]\n\t"
                    "ldr x14, [x6]\n\t"
                    "ldr x15, [x7]\n\t"
                    "ldr x16, [x8]\n\t"
                    
                    // move to vectors
                    "mov v0.s[0], w9\n\t"   // v0 = 0|0|0|r0
                    "mov v1.s[0], w10\n\t"   // v1 = 0|0|0|r1
                    "mov v2.s[0], w11\n\t"   // v2 = 0|0|0|r2
                    "mov v3.s[0], w12\n\t"   // v3 = 0|0|0|r3

                    "mov v0.s[1], w13\n\t"   // v0 = 0|0|r4|r0 = | 53,52,51,50 | 13,12,11,10
                    "mov v1.s[1], w14\n\t"   // v1 = 0|0|r5|r1 = | 63,62,61,60 | 23,22,21,20
                    "mov v2.s[1], w15\n\t"   // v2 = 0|0|r6|r2 = | 73,72,71,70 | 33,32,31,30
                    "mov v3.s[1], w16\n\t"   // v3 = 0|0|r7|r3 = | 83,82,81,80 | 43,42,41,40

                    // A = | 53,52,51,50 | 13,12,11,10
                    // B = | 63,62,61,60 | 23,22,21,20
                    // C = | 62,52,60,50 | 22,12,20,10
                    "trn1 v4.8b, v0.8b, v1.8b\n\t"  // v4 = | 62,52,60,50 | 22,12,20,10
                    "trn1 v5.8b, v2.8b, v3.8b\n\t"  // v5 = | 82,72,80,70 | 42,32,40,30

                    // A = | 53,52,51,50 | 13,12,11,10
                    // B = | 63,62,61,60 | 23,22,21,20
                    // C = | 63,53,61,51 | 23,13,21,11
                    "trn2 v6.8b, v0.8b, v1.8b\n\t"  // v6 = | 63,53,61,51 | 23,13,21,11
                    "trn2 v7.8b, v2.8b, v3.8b\n\t"  // v7 = | 83,73,81,71 | 43,33,41,31

                    // A = | 51,50 | 11,10
                    // B = | 61,60 | 21,20
                    // C = | 60,50 | 20,10
                    "trn1 v0.8h, v4.8h, v5.8h\n\t"  // v0 = | 80,70,60,50 | 40,30,20,10
                    "trn1 v1.8h, v6.8h, v7.8h\n\t"  // v1 = | 81,71,61,51 | 41,31,21,11

                    // A = | 51,50 | 11,10
                    // B = | 61,60 | 21,20
                    // C = | 61,51 | 21,11
                    "trn2 v2.8h, v4.8h, v5.8h\n\t"  // v2 = | 82,72,62,52 | 42,32,22,12
                    "trn2 v3.8h, v6.8h, v7.8h\n\t"  // v3 = | 83,73,63,53 | 43,33,23,13

                    "trn1 v4.4s, v0.4s, v1.4s\n\t"  // v4 = | 41,31,21,11 | 40,30,20,10
                    "trn1 v5.4s, v2.4s, v3.4s\n\t"  // v5 = | 43,33,23,13 | 42,32,22,12

                    "trn2 v6.4s, v0.4s, v1.4s\n\t"  // v6 = | 81,71,61,51 | 80,70,60,50
                    "trn2 v7.4s, v2.4s, v3.4s\n\t"  // v7 = | 82,72,62,52 | 82,72,62,52

                    "st1 {v4.8b}, [%[output]], #8\n\t"
                    "st1 {v5.8b}, [%[output]], #8\n\t"
                    "st1 {v6.8b}, [%[output]], #8\n\t"
                    "st1 {v7.8b}, [%[output]], #8\n\t"

                    "add x1, x8, %[columns]\n\t"
                    
                    "add %w[j], %w[j], #8\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 1b\n\t"

                    "add %w[i], %w[i], #4\n\t"
                    "cmp %w[i], %w[columns]\n\t"
                    "b.lt 0b\n\t"

                    :   [ output ] "+r" (dst)
                    :   [ src ] "r" ( src ),
                        [ j    ] "r" ( j    ), [ i ] "r" (i),
                        [ rows ] "r" ( k_shape.size[0] ), [ columns ] "r" ( k_shape.size[1] )
                    :   "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
                        "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16"
                );
                return Status::Success;
            }
            
            Status MultiplyFloat32MultiBatched(
                const float32_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                float32_t* output, Shape output_shape,
                MulParams params
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
                float32_t*      _input      = const_cast<float32_t*>(input);
                float32_t*      _input_base = const_cast<float32_t*>(input);
                int i, j, k, end;
                float32_t*        _output_1   = output;
                float32_t*        _output_2   = output + 1 * rhs_rows;
                float32_t*        _output_3   = output + 2 * rhs_rows;
                float32_t*        _output_4   = output + 3 * rhs_rows;
                
                /* Vector assignments:
                    * A         -> v0-3      (Activations)
                    * A         -> v4-7      (Weights)
                    * MiniACC   -> v8-11     (Mini Accumulator)
                    * ACC1      -> v12-23    (tmp)
                    * ACC       -> v24-27    (Accumulators rows #1-4)
                    * max(exp)  -> v28       (max(exp) A0-A3)
                    * ACC       -> v29-31    (Constants)
                */
                asm volatile (

                    "mov w1, #0x7F800000\n\t"
                    "mov w2, #0x007fffff\n\t"
                    "mov w0, #0x00800000\t\n"
                    "dup v29.4s, w1\n\t"                 // v0 = 7F80 0000
                    "dup v30.4s, w2\n\t"                 // v1 = 007f ffff
                    "dup v31.4s, w0\n\t"                 // v2 = 0080 0000

                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"


                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "movi v24.4s, #0\n\t"
                    "movi v25.4s, #0\n\t"
                    "movi v26.4s, #0\n\t"
                    "movi v27.4s, #0\n\t"


                    // Load Activations
                    "ld1 {v0.4s},  [%[activation]], #16\n\t"
                    "ld1 {v1.4s},  [%[activation]], #16\n\t"
                    "ld1 {v2.4s},  [%[activation]], #16\n\t"
                    "ld1 {v3.4s},  [%[activation]], #16\n\t"
                    

                    "1:\n\t"
                    "movi v8.4s, #0\n\t"
                    "movi v9.4s, #0\n\t"
                    "movi v10.4s, #0\n\t"
                    "movi v11.4s, #0\n\t"
                    "movi v12.4s, #0\n\t"
                    "movi v13.4s, #0\n\t"
                    "movi v14.4s, #0\n\t"
                    "movi v15.4s, #0\n\t"
                    "movi v16.4s, #0\n\t"
                    "movi v17.4s, #0\n\t"
                    "movi v18.4s, #0\n\t"
                    "movi v19.4s, #0\n\t"
                    "movi v20.4s, #0\n\t"
                    "movi v21.4s, #0\n\t"
                    "movi v22.4s, #0\n\t"
                    "movi v23.4s, #0\n\t"
                    "movi v28.4s, #0\n\t"

                    // Load Weights
                    "ld1 {v4.4s}, [%[weights]], #16\n\t"

                    "sxtl2 v6.8h, v4.16b\n\t"
                    "sxtl2 v7.4s, v6.8h\n\t"
                    "sxtl v6.4s, v6.4h\n\t"
                    "sxtl v4.8h, v4.8b\n\t"
                    "sxtl2 v5.4s, v4.8h\n\t"
                    "sxtl v4.4s, v4.4h\n\t"

                    // v7 = 0eee eeee e000 0000 (e = exp)
                    "and v8.16B,    v0.16B, v29.16B\t\n"    
                    "and v9.16B,    v1.16B, v29.16B\t\n"    
                    "and v10.16B,   v2.16B, v29.16B\t\n"    
                    "and v11.16B,   v3.16B, v29.16B\t\n"
                    
                    // get max exponent (m = max(exp))
                    "umaxv s12, v8.4s\n\t"
                    "umaxv s13, v9.4s\n\t"
                    "umaxv s14, v10.4s\n\t"
                    "umaxv s15, v11.4s\n\t"               

                    // Duplicate max to other vectors
                    // v12 = 0mmm mmmm m000 0000
                    "dup v12.4s, v12.s[0]\n\t"
                    "dup v13.4s, v13.s[0]\n\t"
                    "dup v14.4s, v14.s[0]\n\t"
                    "dup v15.4s, v15.s[0]\n\t"

                    // v12 = MM00 0000
                    "shl v12.4s, v12.4s, #1\t\n"
                    "shl v13.4s, v13.4s, #1\t\n"
                    "shl v14.4s, v14.4s, #1\t\n"
                    "shl v15.4s, v15.4s, #1\t\n"

                    // v12 = max(exp) SSSS SSMM (s = sign)
                    "ushr v12.4s, v12.4s, #24\t\n"
                    "ushr v13.4s, v13.4s, #24\t\n"
                    "ushr v14.4s, v14.4s, #24\t\n"
                    "ushr v15.4s, v15.4s, #24\t\n"

                    // v8 = EE00 0000
                    "shl v8.4s,     v8.4s,  #1\t\n"
                    "shl v9.4s,     v9.4s,  #1\t\n"
                    "shl v10.4s,    v10.4s, #1\t\n"
                    "shl v11.4s,    v11.4s, #1\t\n"

                    // v8 = SSSS SSEE
                    "ushr v8.4s,     v8.4s,  #24\t\n"
                    "ushr v9.4s,     v9.4s,  #24\t\n"
                    "ushr v10.4s,    v10.4s, #24\t\n"
                    "ushr v11.4s,    v11.4s, #24\t\n"

                    // v8 = max - exp
                    "sub v8.4s,     v8.4s,  v12.4s\t\n"
                    "sub v9.4s,     v9.4s,  v13.4s\t\n"
                    "sub v10.4s,    v10.4s, v14.4s\t\n"
                    "sub v11.4s,    v11.4s, v15.4s\t\n"
                    

                    // save maxs to v28
                    "mov v28.s[0], v12.s[0]\n\t"
                    "mov v28.s[1], v13.s[0]\n\t"
                    "mov v28.s[2], v14.s[0]\n\t"
                    "mov v28.s[3], v15.s[0]\n\t"

                    
                    // Extract fractions
                    // v9 = 0000 0000 0aaa aaaa AAAA
                    "and v12.16B, v30.16B, v0.16B\t\n"
                    "and v13.16B, v30.16B, v1.16B\t\n"
                    "and v14.16B, v30.16B, v2.16B\t\n"
                    "and v15.16B, v30.16B, v3.16B\t\n"

                    // add 1 left of fractions... 23 to 24 bit
                    // v9 = 00AA AAAA (A = fraction)
                    "eor v12.16B, v12.16B, v31.16B\t\n"
                    "eor v13.16B, v13.16B, v31.16B\t\n"
                    "eor v14.16B, v14.16B, v31.16B\t\n"
                    "eor v15.16B, v15.16B, v31.16B\t\n"


                    // v31 = 0080 0000 ==> 8000 0000
                    "shl v31.4s, v31.4s, #8\n\t"

                    // v12 = shift right mantisa
                    "sshl v16.4s, v12.4s, v8.4s\t\n"
                    "sshl v17.4s, v13.4s, v9.4s\t\n"
                    "sshl v18.4s, v14.4s, v10.4s\t\n"
                    "sshl v19.4s, v15.4s, v11.4s\t\n"

                    // 2's compelete of fraction
                    // v8 = sign extend floats
                    "sshr v8.4s,    v0.4s, #31\t\n"
                    "sshr v9.4s,    v1.4s, #31\t\n"
                    "sshr v10.4s,   v2.4s, #31\t\n"
                    "sshr v11.4s,   v3.4s, #31\t\n"
            
                    // v20 = not fraction if sign is 1
                    "eor v20.16B, v16.16B, v8.16B\t\n"
                    "eor v21.16B, v17.16B, v9.16B\t\n"
                    "eor v22.16B, v18.16B, v10.16B\t\n"
                    "eor v23.16B, v19.16B, v11.16B\t\n"
                    
                    // v10 = 0 or 1
                    "ushr v8.4s, v8.4s, #31\t\n"
                    "ushr v9.4s, v9.4s, #31\t\n"
                    "ushr v10.4s, v10.4s, #31\t\n"
                    "ushr v11.4s, v11.4s, #31\t\n"

                    // v0 = not(fraction)+1 if sign is 1 === 2's complement of fractions
                    "add v0.4s, v20.4s, v8.4s\t\n"
                    "add v1.4s, v21.4s, v9.4s\t\n"
                    "add v2.4s, v22.4s, v10.4s\t\n"
                    "add v3.4s, v23.4s, v11.4s\t\n"

                    // Multiplication  
                    // Rows #1
                    "mul v8.4s, v0.4s, v4.4s\n\t"
                    "mul v9.4s, v0.4s, v5.4s\n\t"
                    "mul v10.4s, v0.4s, v6.4s\n\t"
                    "mul v11.4s, v0.4s, v7.4s\n\t"
                
                    "ld1 {v0.4s},  [%[activation]], #16\n\t"

                    // Rows #2
                    "mul v12.4s, v1.4s, v4.4s\n\t"
                    "mul v13.4s, v1.4s, v5.4s\n\t"
                    "mul v14.4s, v1.4s, v6.4s\n\t"
                    "mul v15.4s, v1.4s, v7.4s\n\t"

                    "ld1 {v1.4s},  [%[activation]], #16\n\t"

                    // Rows #3
                    "mul v16.4s, v2.4s, v4.4s\n\t"
                    "mul v17.4s, v2.4s, v5.4s\n\t"
                    "mul v18.4s, v2.4s, v6.4s\n\t"
                    "mul v19.4s, v2.4s, v7.4s\n\t"
                    
                    "ld1 {v2.4s},  [%[activation]], #16\n\t"

                    // Rows #4
                    "mul v20.4s, v3.4s, v4.4s\n\t"
                    "mul v21.4s, v3.4s, v5.4s\n\t"
                    "mul v22.4s, v3.4s, v6.4s\n\t"
                    "mul v23.4s, v3.4s, v7.4s\n\t"

                    "ld1 {v3.4s},  [%[activation]], #16\n\t"

                    // compress Resuls
                    // Rows #1
                    "addv s8,   v8.4s\n\t"
                    "addv s9,   v9.4s\n\t"
                    "addv s10,  v10.4s\n\t"
                    "addv s11,  v11.4s\n\t"

                    // Rows #2
                    "addv s12,  v12.4s\n\t"
                    "addv s13,  v13.4s\n\t"
                    "addv s14,  v14.4s\n\t"
                    "addv s15,  v15.4s\n\t"
                    
                    // Rows #3
                    "addv s16,  v16.4s\n\t"
                    "addv s17,  v17.4s\n\t"
                    "addv s18,  v18.4s\n\t"
                    "addv s19,  v19.4s\n\t"
                    
                    // Rows #4
                    "addv s20,  v20.4s\n\t"
                    "addv s21,  v21.4s\n\t"
                    "addv s22,  v22.4s\n\t"
                    "addv s23,  v23.4s\n\t"

                    // mov Resuls to v12-v15
                    
                    "mov v13.s[3], v15.s[0]\n\t"
                    "mov v13.s[2], v14.s[0]\n\t"
                    "mov v13.s[1], v13.s[0]\n\t"
                    "mov v13.s[0], v12.s[0]\n\t"

                    "mov v12.s[0], v8.s[0]\n\t"
                    "mov v12.s[1], v9.s[0]\n\t"
                    "mov v12.s[2], v10.s[0]\n\t"
                    "mov v12.s[3], v11.s[0]\n\t"
                    
                    "mov v14.s[0], v16.s[0]\n\t"
                    "mov v14.s[1], v17.s[0]\n\t"
                    "mov v14.s[2], v18.s[0]\n\t"
                    "mov v14.s[3], v19.s[0]\n\t"
                    
                    "mov v15.s[0], v20.s[0]\n\t"
                    "mov v15.s[1], v21.s[0]\n\t"
                    "mov v15.s[2], v22.s[0]\n\t"
                    "mov v15.s[3], v23.s[0]\n\t"

                    // extract result sign
                    "and v8.16B, v12.16B, v31.16B\t\n" 
                    "and v9.16B, v13.16B, v31.16B\t\n" 
                    "and v10.16B, v14.16B, v31.16B\t\n" 
                    "and v11.16B, v14.16B, v31.16B\t\n" 

                    // v31 = 80000000 ==> 00000008
                    "ushr v31.4s, v31.4s, #28\n\t"

                    // 2's compelete of fraction
                    // sign extend
                    "sshr v16.4s, v12.4s, #31\t\n"
                    "sshr v17.4s, v13.4s, #31\t\n"
                    "sshr v18.4s, v14.4s, #31\t\n"
                    "sshr v19.4s, v15.4s, #31\t\n"

                    // not fraction if sign is 1
                    "eor v12.16B, v12.16B, v16.16B\t\n"
                    "eor v13.16B, v13.16B, v17.16B\t\n"
                    "eor v14.16B, v14.16B, v18.16B\t\n"
                    "eor v15.16B, v15.16B, v19.16B\t\n"

                    "ushr v16.4s, v16.4s, #31\t\n"
                    "ushr v17.4s, v17.4s, #31\t\n"
                    "ushr v18.4s, v18.4s, #31\t\n"
                    "ushr v19.4s, v19.4s, #31\t\n"

                    // not(fraction)+1 if sign is 1
                    "add v12.4s, v12.4s, v16.4s\t\n"
                    "add v13.4s, v13.4s, v17.4s\t\n"
                    "add v14.4s, v14.4s, v18.4s\t\n"
                    "add v15.4s, v15.4s, v19.4s\t\n"

                    // extend maxs to vectors
                    "dup v16.4s, v28.s[0]\n\t"
                    "dup v17.4s, v28.s[1]\n\t"
                    "dup v18.4s, v28.s[2]\n\t"
                    "dup v19.4s, v28.s[3]\n\t"

                    // Find bitwidth of result of each register?
                    "clz v20.4s, v12.4s\n\t"
                    "clz v21.4s, v13.4s\n\t"
                    "clz v22.4s, v14.4s\n\t"
                    "clz v23.4s, v15.4s\n\t"
                    
                    // v16 = count zero - 8
                    "sub v20.4s, v20.4s, v31.4s\n\t"
                    "sub v21.4s, v21.4s, v31.4s\n\t"
                    "sub v22.4s, v22.4s, v31.4s\n\t"
                    "sub v23.4s, v23.4s, v31.4s\n\t"

                    // v0 = v0 << v16
                    "sshl v12.4s, v12.4s, v20.4s\n\t"
                    "sshl v13.4s, v13.4s, v21.4s\n\t"
                    "sshl v14.4s, v14.4s, v22.4s\n\t"
                    "sshl v15.4s, v15.4s, v23.4s\n\t"
                    
                    // v8 = right exp
                    "sub v16.4s, v16.4s, v20.4s\n\t"
                    "sub v17.4s, v17.4s, v21.4s\n\t"
                    "sub v18.4s, v18.4s, v22.4s\n\t"
                    "sub v19.4s, v19.4s, v23.4s\n\t"

                    // v31 = 00000008 ==> 00800000
                    "shl v31.4s, v31.4s, #20\n\t"

                    // extract mantisa
                    "eor v12.16b, v12.16b, v31.16b\n\t"
                    "eor v13.16b, v13.16b, v31.16b\n\t"
                    "eor v14.16b, v14.16b, v31.16b\n\t"
                    "eor v15.16b, v15.16b, v31.16b\n\t"

                    "eor v8.16B,    v8.16B,  v12.16B\n\t"
                    "eor v9.16B,    v9.16B,  v13.16B\n\t"
                    "eor v10.16B,   v10.16B, v14.16B\n\t"
                    "eor v11.16B,   v11.16B, v15.16B\n\t"

                    // v12 = 000000mm ==> 7m800000
                    "shl v16.4s, v16.4s, #23\n\t"
                    "shl v17.4s, v17.4s, #23\n\t"
                    "shl v18.4s, v18.4s, #23\n\t"
                    "shl v19.4s, v19.4s, #23\n\t"
            
                    "eor v8.16B,    v8.16B,  v16.16B\n\t"
                    "eor v9.16B,    v9.16B,  v17.16B\n\t"
                    "eor v10.16B,   v10.16B, v18.16B\n\t"
                    "eor v11.16B,   v11.16B, v19.16B\n\t"

                    "fadd v24.4s, v24.4s, v8.4s\n\t"
                    "fadd v25.4s, v25.4s, v9.4s\n\t"
                    "fadd v26.4s, v26.4s, v10.4s\n\t"
                    "fadd v27.4s, v27.4s, v11.4s\n\t"

                    
                    "add %w[i], %w[i], #4\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    "st1 {v24.4s}, [%[dst_1]], #16\n\t"
                    "st1 {v25.4s}, [%[dst_2]], #16\n\t"
                    "st1 {v26.4s}, [%[dst_3]], #16\n\t"
                    "st1 {v27.4s}, [%[dst_4]], #16\n\t"

                    // Reset the activations to the start of the row
                    "mov %[activation], x1\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add x1, x1, %[size], lsl #4\n\t"

                    "mov %[activation], x1\n\t"

                    
                    "mov %[weights], x2\n\t"

                    

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_4]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], lsl #2\n\t"
                    "add %[dst_3], %[dst_2], %[rows], lsl #2\n\t"
                    "add %[dst_4], %[dst_3], %[rows], lsl #2\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #4\n\t"
                    "cmp %w[k], %w[batches]\n\t"
                    "b.lt 5b\n\t"

                    :   [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                        [ dst_3 ]      "+r" (_output_3),   [ dst_4 ]       "+r" (_output_4),
                        [ i ]          "+r" (i),           [ end ]         "+r" (end),
                        [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    :   [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                        [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                        [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                        [ batches ]    "r"  (lhs_batches)

                    :   "v0",  "v1",  "v2",  "v3",
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
