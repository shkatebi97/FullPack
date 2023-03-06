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
        namespace Binary{
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 128)?(128 - (shape.size[1] % 128)):(0);
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
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 1);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                // if (k_shape.size[1] % 128)
                //     return Status::SizesMisMatch; 
                if (layout == MemLayout::kColumnMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 8);
                }
                else {
                    // std::cerr << "Filter Qantizer Step #1 with shape of " << get_shape_string(k_shape) << std::endl;
                    int new_weights_length = (k_shape.size[1] / 8) * k_shape.size[0];
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    // std::cerr << "Filter Qantizer Step #2\n";
                    #ifdef PRINT_VALUES_DETAILED
                    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int>> memory_accesses;
                    std::cout << get_shape_string(k_shape) << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #3\n";
                    if (layout == MemLayout::kColumnMajor)
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 128,
                                    container_id = (j % 128) % 16,
                                    shift_amount = (8 - 1) - ((j % 128) / 16);
                                temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = j * k_shape.size[0] + i;
                                std::get<7>(access) = input[j * k_shape.size[0] + i];
                                std::get<8>(access) = 0x01 << (shift_amount);
                                std::get<9>(access) = (input[j * k_shape.size[0] + i] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)];
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    else
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 128,
                                    container_id = (j % 128) % 16,
                                    shift_amount = (8 - 1) - ((j % 128) / 16);
                                temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = i * k_shape.size[1] + j;
                                std::get<7>(access) = input[i * k_shape.size[1] + j];
                                std::get<8>(access) = 0x01 << shift_amount;
                                std::get<9>(access) = (input[i * k_shape.size[1] + j] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)];
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    #ifdef PRINT_VALUES_DETAILED
                    for (int i = 0 ; i < memory_accesses.size() ; i++)
                        std::cout << "\t( " 
                                << ((std::get<0>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<0>(memory_accesses[i]) < 10 && std::get<0>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<0>(memory_accesses[i]) << ", "
                                << ((std::get<1>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<1>(memory_accesses[i]) < 10 && std::get<1>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<1>(memory_accesses[i]) << ", "
                                << ((std::get<2>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<2>(memory_accesses[i]) < 10 && std::get<2>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<2>(memory_accesses[i]) << ", "
                                << ((std::get<3>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<3>(memory_accesses[i]) < 10 && std::get<3>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<3>(memory_accesses[i]) << ", ("
                                << ((std::get<4>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<4>(memory_accesses[i]) < 10 && std::get<4>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<4>(memory_accesses[i]) << ", "
                                << ((std::get<5>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<5>(memory_accesses[i]) < 10 && std::get<5>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<5>(memory_accesses[i]) << "), "
                                << ((std::get<6>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<6>(memory_accesses[i]) < 10 && std::get<6>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<6>(memory_accesses[i]) << ", "
                                << ((std::get<7>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<7>(memory_accesses[i]) < 10 && std::get<7>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<7>(memory_accesses[i]) << ", "
                                << std::hex
                                << ((std::get<8>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<8>(memory_accesses[i]) << ", "
                                << ((std::get<9>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<9>(memory_accesses[i]) << ", "
                                << ((std::get<10>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<10>(memory_accesses[i]) << ", "
                                << ((std::get<11>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<11>(memory_accesses[i]) << ", "
                                << std::dec
                                << ")\n";
                    #endif
                    #ifdef PRINT_VALUES
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < k_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < k_shape.size[1] / 8; j++)
                            std::cout << ((int)temp[i * (k_shape.size[1] / 8) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #4 with shape of (" << k_shape.size[0] << ", " << k_shape.size[1] / 8 << ")" << std::endl;
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 8);
                    #ifdef PRINT_VALUES
                    std::cout << "After Packing:" << std::endl;
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < k_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < k_shape.size[1] / 8; j++)
                            std::cout << ((int)output[i * (k_shape.size[1] / 8) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #5\n";
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 128)
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
                    
                    // "cmp %w[size], #0\n\t"
                    // "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
#ifdef LEGACY_BINARY
                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"
#else
#endif

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR T1, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v10.16b, v1.16b, #7\n\t"
                    "sshr v11.16b, v2.16b, #7\n\t"
                    "sshr v12.16b, v3.16b, #7\n\t"
                    
#if LEGACY_BINARY
                    // AND APt, T1, AP
                    "and v9.16b,  v8.16b,  v5.16b\n\t"
                    "and v13.16b, v10.16b, v5.16b\n\t"
                    "and v14.16b, v11.16b, v5.16b\n\t"
                    "and v15.16b, v12.16b, v5.16b\n\t"

                    // NOT T1, T1
                    "not v8.16b,  v8.16b\n\t"
                    "not v10.16b, v10.16b\n\t"
                    "not v11.16b, v11.16b\n\t"
                    "not v12.16b, v12.16b\n\t"

                    // AND At, T1, A
                    "and v8.16b,  v8.16b,  v4.16b\n\t"
                    "and v10.16b, v10.16b, v4.16b\n\t"
                    "and v11.16b, v11.16b, v4.16b\n\t"
                    "and v12.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // ORR AT, At, APt
                    "orr v9.16b,  v8.16b,  v9.16b\n\t"
                    "orr v13.16b, v10.16b, v13.16b\n\t"
                    "orr v14.16b, v11.16b, v14.16b\n\t"
                    "orr v15.16b, v12.16b, v15.16b\n\t"

                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"
#else
                    // XOR APt, T1, A
                    "eor v9.16b,  v8.16b,  v4.16b\n\t"
                    "eor v13.16b, v10.16b, v4.16b\n\t"
                    "eor v14.16b, v11.16b, v4.16b\n\t"
                    "eor v15.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // USHR T1, W, #7
                    "ushr v8.16b,  v8.16b,  #7\n\t"
                    "ushr v10.16b, v10.16b, #7\n\t"
                    "ushr v11.16b, v11.16b, #7\n\t"
                    "ushr v12.16b, v12.16b, #7\n\t"

                    // add APt, T1, AP
                    "add v9.16b,  v8.16b,  v9.16b\n\t"
                    "add v13.16b, v10.16b, v13.16b\n\t"
                    "add v14.16b, v11.16b, v14.16b\n\t"
                    "add v15.16b, v12.16b, v15.16b\n\t"
#endif

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"
                    
                    // SADALP MiniAC, AT
                    "sadalp v26.8h, v9.16b\n\t"
                    "sadalp v27.8h, v13.16b\n\t"
                    "sadalp v28.8h, v14.16b\n\t"
                    "sadalp v31.8h, v15.16b\n\t"

                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

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
                int i, j, k, end;
// #define W1A8_DECREASED_CONCURRENT_BATCH
#define W1A8_MULL_MULTIBATCH
#ifdef W1A8_MULL_MULTIBATCH
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W         -> v0-3      (Weights)
                    * A         -> v4-7      (Activations)
                    * MW        -> v8-11     (Masked Weights)
                    * MiniACC   -> v12-15    (Mini Accumulator)
                    * ACC1      -> v16-19    (Accumulators input row #1)
                    * ACC2      -> v20-23    (Accumulators input row #2)
                    * ACC3      -> v24-27    (Accumulators input row #3)
                    * ACC4      -> v28-31    (Accumulators input row #4)
                */
                asm volatile(
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"
                    "mov %w[k], wzr\n\t"

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

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"

                    // ORR MW, MW, #1
                    "movi v12.16b, 0x01\n\t"
                    "orr v8.16b,  v8.16b,  v12.16b\n\t"
                    "orr v9.16b,  v9.16b,  v12.16b\n\t"
                    "orr v10.16b, v10.16b, v12.16b\n\t"
                    "orr v11.16b, v11.16b, v12.16b\n\t"
                    
                    // Activation Row #1
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
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
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
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
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v6.8b\n\t"
                    "smull v13.8h, v9.8b,  v6.8b\n\t"
                    "smull v14.8h, v10.8b, v6.8b\n\t"
                    "smull v15.8h, v11.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
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
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v7.8b\n\t"
                    "smull v13.8h, v9.8b,  v7.8b\n\t"
                    "smull v14.8h, v10.8b, v7.8b\n\t"
                    "smull v15.8h, v11.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
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

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // Check if the 8 iterations of inner loop are done
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // Check if the loop over rows of weight matrix (outer loop) is done
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
                    
                    // Check if the output need downcasting to int8
                    "cmp %w[downcast], 0xff\n\t"
                    "beq 6f\n\t"

                    // Output is needed in 32-bit; no need to downcast
                    // Store the 4 int32 results
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"

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
                    "st1 {v16.s}[0],  [%[dst_1]], #4\n\t"
                    "st1 {v20.s}[0],  [%[dst_2]], #4\n\t"
                    "st1 {v24.s}[0],  [%[dst_3]], #4\n\t"
                    "st1 {v28.s}[0],  [%[dst_4]], #4\n\t"

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
                      [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
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
#else
#ifndef W1A8_DECREASED_CONCURRENT_BATCH
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W    -> v0-3      (Weights)
                    * A    -> v4-7      (Activations)
                    * MW   -> v8-11     (Masked Weights)
                    * T    -> v12-15    (Temporary Values)
                    * ACC1 -> v16-19    (Accumulators input row #1)
                    * ACC2 -> v20-23    (Accumulators input row #2)
                    * ACC3 -> v24-27    (Accumulators input row #3)
                    * ACC4 -> v28-31    (Accumulators input row #4)
                */
                asm volatile(
                    "mov %w[k], wzr\n\t"

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

                    // Start of Outer Loop Over Weights
                    "1:\n\t"

                    // Load Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // Activation Row #1
                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"

                    // XOR T, MW, A
                    "eor v12.16b, v8.16b,  v4.16b\n\t"
                    "eor v13.16b, v9.16b,  v4.16b\n\t"
                    "eor v14.16b, v10.16b, v4.16b\n\t"
                    "eor v15.16b, v11.16b, v4.16b\n\t"

                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // USHR MW, W, #7
                    "ushr v8.16b,  v0.16b, #7\n\t"
                    "ushr v9.16b,  v1.16b,  #7\n\t"
                    "ushr v10.16b, v2.16b, #7\n\t"
                    "ushr v11.16b, v3.16b, #7\n\t"

                    // ADD T, MW, T
                    "add v12.16b, v8.16b,  v12.16b\n\t"
                    "add v13.16b, v9.16b,  v13.16b\n\t"
                    "add v14.16b, v10.16b, v14.16b\n\t"
                    "add v15.16b, v11.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC1, MiniAC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"
                    
                    // XOR T, MW, A
                    "eor v12.16b, v8.16b,  v5.16b\n\t"
                    "eor v13.16b, v9.16b,  v5.16b\n\t"
                    "eor v14.16b, v10.16b, v5.16b\n\t"
                    "eor v15.16b, v11.16b, v5.16b\n\t"

                    // Load Activations
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    // USHR MW, W, #7
                    "ushr v8.16b,  v0.16b, #7\n\t"
                    "ushr v9.16b,  v1.16b,  #7\n\t"
                    "ushr v10.16b, v2.16b, #7\n\t"
                    "ushr v11.16b, v3.16b, #7\n\t"

                    // ADD T, MW, T
                    "add v12.16b, v8.16b,  v12.16b\n\t"
                    "add v13.16b, v9.16b,  v13.16b\n\t"
                    "add v14.16b, v10.16b, v14.16b\n\t"
                    "add v15.16b, v11.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC2, MiniAC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #3
                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"
                    
                    // XOR T, MW, A
                    "eor v12.16b, v8.16b,  v6.16b\n\t"
                    "eor v13.16b, v9.16b,  v6.16b\n\t"
                    "eor v14.16b, v10.16b, v6.16b\n\t"
                    "eor v15.16b, v11.16b, v6.16b\n\t"

                    // Load Activations
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"

                    // USHR MW, W, #7
                    "ushr v8.16b,  v0.16b, #7\n\t"
                    "ushr v9.16b,  v1.16b,  #7\n\t"
                    "ushr v10.16b, v2.16b, #7\n\t"
                    "ushr v11.16b, v3.16b, #7\n\t"

                    // ADD T, MW, T
                    "add v12.16b, v8.16b,  v12.16b\n\t"
                    "add v13.16b, v9.16b,  v13.16b\n\t"
                    "add v14.16b, v10.16b, v14.16b\n\t"
                    "add v15.16b, v11.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC3, MiniAC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #4
                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"
                    
                    // XOR T, MW, A
                    "eor v12.16b, v8.16b,  v7.16b\n\t"
                    "eor v13.16b, v9.16b,  v7.16b\n\t"
                    "eor v14.16b, v10.16b, v7.16b\n\t"
                    "eor v15.16b, v11.16b, v7.16b\n\t"

                    // Load Activations
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    // USHR MW, W, #7
                    "ushr v8.16b,  v0.16b, #7\n\t"
                    "ushr v9.16b,  v1.16b,  #7\n\t"
                    "ushr v10.16b, v2.16b, #7\n\t"
                    "ushr v11.16b, v3.16b, #7\n\t"

                    // ADD T, MW, T
                    "add v12.16b, v8.16b,  v12.16b\n\t"
                    "add v13.16b, v9.16b,  v13.16b\n\t"
                    "add v14.16b, v10.16b, v14.16b\n\t"
                    "add v15.16b, v11.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC4, MiniAC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // Check if the 8 iterations of inner loop are done
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // Check if the loop over rows of weight matrix (outer loop) is done
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
                    "mov %[activation], %[act_base]\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add %[act_base], %[act_base], %[size], asr #2\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Reset the weights to the start
                    "mov %[weights], %[wts_base]\n\t"

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_3]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], asr #2\n\t"
                    "add %[dst_3], %[dst_2], %[rows], asr #2\n\t"
                    "add %[dst_4], %[dst_3], %[rows], asr #2\n\t"

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
                      "v28", "v29", "v30", "v31"
                );
#else
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                /* Vector assignments:
                    * W    -> v0-3      (Weights)
                    * A    -> v4-7      (Activations)
                    * --   -> v8-11     (Not Used)
                    * T    -> v12-15    (Temporary Values)
                    * WM1  -> v16-19    (Mask Weight #1)
                    * WM2  -> v20-23    (Mask Weight #2)
                    * ACC1 -> v24-27    (Accumulators input row #3)
                    * ACC2 -> v28-31    (Accumulators input row #4)
                */
               asm volatile(
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
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

                    // Load Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR MW1, W, #7
                    "sshr v16.16b, v0.16b, #7\n\t"
                    "sshr v17.16b, v1.16b, #7\n\t"
                    "sshr v18.16b, v2.16b, #7\n\t"
                    "sshr v19.16b, v3.16b, #7\n\t"

                    // USHR MW2, W, #7
                    "ushr v20.16b, v0.16b, #7\n\t"
                    "ushr v21.16b, v1.16b, #7\n\t"
                    "ushr v22.16b, v2.16b, #7\n\t"
                    "ushr v23.16b, v3.16b, #7\n\t"

                    // Activation Row #1                    
                    // XOR T, MW1, A
                    "eor v12.16b, v16.16b, v4.16b\n\t"
                    "eor v13.16b, v17.16b, v4.16b\n\t"
                    "eor v14.16b, v18.16b, v4.16b\n\t"
                    "eor v15.16b, v19.16b, v4.16b\n\t"

                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // ADD T, MW2, T
                    "add v12.16b, v20.16b, v12.16b\n\t"
                    "add v13.16b, v21.16b, v13.16b\n\t"
                    "add v14.16b, v22.16b, v14.16b\n\t"
                    "add v15.16b, v23.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC1, MiniAC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #2
                    // XOR T, MW1, A
                    "eor v12.16b, v16.16b, v5.16b\n\t"
                    "eor v13.16b, v17.16b, v5.16b\n\t"
                    "eor v14.16b, v18.16b, v5.16b\n\t"
                    "eor v15.16b, v19.16b, v5.16b\n\t"

                    // Load Activations
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    // ADD T, MW2, T
                    "add v12.16b, v20.16b, v12.16b\n\t"
                    "add v13.16b, v21.16b, v13.16b\n\t"
                    "add v14.16b, v22.16b, v14.16b\n\t"
                    "add v15.16b, v23.16b, v15.16b\n\t"
                    
                    // SADDLP MiniAC, T
                    "saddlp v12.8h, v12.16b\n\t"
                    "saddlp v13.8h, v13.16b\n\t"
                    "saddlp v14.8h, v14.16b\n\t"
                    "saddlp v15.8h, v15.16b\n\t"

                    // SADALP ACC2, MiniAC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // Check if the 8 iterations of inner loop are done
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // Check if the loop over rows of weight matrix (outer loop) is done
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

                    // Reorder ACC1 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"
                    
                    // Store the 4 int32 results
                    "st1 {v24.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_2]], #16\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 2 batches
                    "add %[act_base], %[act_base], %[size], asr #1\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Reset the weights to the start
                    "mov %[weights], %[wts_base]\n\t"

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_1]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], asr #2\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #2\n\t"
                    "cmp %w[k], %w[rows]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                      [ j ]          "+r" (j),           [ k ]           "+r" (k),
                      [ end ]        "+r" (end),         [ i ]           "+r" (i)

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
                      "v28", "v29", "v30", "v31"
                );
#endif
#endif

                return Status::Success;
            }
            Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){}
            Status MultiplyInt8(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                // In case of multi-batched input
                // std::cout << "Inside MultiplyInt8Binary with multibatch set to " 
                //           << ((input_shape.number_dims == 2 && input_shape.size[0] != 1)?("ON"):("OFF"))
                //           << std::endl;
                if (input_shape.number_dims == 2 && input_shape.size[0] != 1)
                    return MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape, MulParams());
                
                int lhs_columns = input_shape.size[input_shape.number_dims - 1] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                int8_t*  rhs = const_cast<int8_t*>(kernel);
                const int8_t*  lhs = input;

                // std::cout << "MultiplyInt8Binary ";
                // std::cout << lhs_columns << ", ";
                // std::cout << rhs_rows << ", ";
                // std::cout << rhs_columns << ", ";
                // std::cout << std::endl;

                int i;
                for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                    rhs += 4 * (lhs_columns / 8);
                }
                if (rhs_rows - i == 1){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 2){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 3){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 4){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                return Status::Success;
            }
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst, int size){
                const int8_t* _activation = activation;
                int i, end;
                asm volatile(
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
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
#ifdef LEGACY_BINARY
                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"
#else
#endif

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR T1, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v10.16b, v1.16b, #7\n\t"
                    "sshr v11.16b, v2.16b, #7\n\t"
                    "sshr v12.16b, v3.16b, #7\n\t"
                    
#if LEGACY_BINARY
                    // AND APt, T1, AP
                    "and v9.16b,  v8.16b,  v5.16b\n\t"
                    "and v13.16b, v10.16b, v5.16b\n\t"
                    "and v14.16b, v11.16b, v5.16b\n\t"
                    "and v15.16b, v12.16b, v5.16b\n\t"

                    // NOT T1, T1
                    "not v8.16b,  v8.16b\n\t"
                    "not v10.16b, v10.16b\n\t"
                    "not v11.16b, v11.16b\n\t"
                    "not v12.16b, v12.16b\n\t"

                    // AND At, T1, A
                    "and v8.16b,  v8.16b,  v4.16b\n\t"
                    "and v10.16b, v10.16b, v4.16b\n\t"
                    "and v11.16b, v11.16b, v4.16b\n\t"
                    "and v12.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // ORR AT, At, APt
                    "orr v9.16b,  v8.16b,  v9.16b\n\t"
                    "orr v13.16b, v10.16b, v13.16b\n\t"
                    "orr v14.16b, v11.16b, v14.16b\n\t"
                    "orr v15.16b, v12.16b, v15.16b\n\t"

                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"
#else
                    // XOR APt, T1, A
                    "eor v9.16b,  v8.16b,  v4.16b\n\t"
                    "eor v13.16b, v10.16b, v4.16b\n\t"
                    "eor v14.16b, v11.16b, v4.16b\n\t"
                    "eor v15.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // USHR T1, W, #7
                    "ushr v8.16b,  v8.16b,  #7\n\t"
                    "ushr v10.16b, v10.16b, #7\n\t"
                    "ushr v11.16b, v11.16b, #7\n\t"
                    "ushr v12.16b, v12.16b, #7\n\t"

                    // add APt, T1, AP
                    "add v9.16b,  v8.16b,  v9.16b\n\t"
                    "add v13.16b, v10.16b, v13.16b\n\t"
                    "add v14.16b, v11.16b, v14.16b\n\t"
                    "add v15.16b, v12.16b, v15.16b\n\t"
#endif

                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    "sadalp v26.8h, v9.16b\n\t"
                    "sadalp v27.8h, v13.16b\n\t"
                    "sadalp v28.8h, v14.16b\n\t"
                    "sadalp v31.8h, v15.16b\n\t"

                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    "addv s23, v23.4s\n\t"
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s30, v30.4s\n\t"

                    "mov v23.s[1], v24.s[0]\n\t"
                    "mov v23.s[2], v25.s[0]\n\t"
                    "mov v23.s[3], v30.s[0]\n\t"

                    "st1 {v23.4s},  [%[dst]]\n\t"

                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[weights], %[weights], %[size], asr #1\n\t"

                    "3:\n\t"

                    : [ dst ] "+r"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
                    : [ activation ] "r"(_activation), [ weights ] "r"(weights),
                      [ size ] "r"(size)
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
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size){}
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount){
                if (input < 0)
                    return 0x01 << shift_amount;
                else
                    return 0;
            }
        }
    }
}
#endif
