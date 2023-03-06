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
        namespace Quaternary{
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 64)?(64 - (shape.size[1] % 64)):(0);
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
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 2);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                // if (k_shape.size[1] % 64)
                //     return Status::SizesMisMatch; 
                if (layout == MemLayout::kColumnMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 4);
                }
                else {
                    // std::cerr << "Filter Qantizer Step #1 with shape of " << get_shape_string(k_shape) << std::endl;
                    int new_weights_length = (k_shape.size[1] / 4) * k_shape.size[0];
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    zero_vector(temp, new_weights_length);
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
                                int cluster_id = j / 64,
                                    container_id = (j % 64) % 16,
                                    shift_amount = ((4 - 1) - ((j % 64) / 16)) * 2;
                                temp_u[i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = j * k_shape.size[0] + i;
                                std::get<7>(access) = input[j * k_shape.size[0] + i];
                                std::get<8>(access) = 0x01 << (shift_amount);
                                std::get<9>(access) = (input[j * k_shape.size[0] + i] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id)];
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    else
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 64,
                                    container_id = (j % 64) % 16,
                                    shift_amount = ((4 - 1) - ((j % 64) / 16)) * 2;
                                temp_u[i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = i * k_shape.size[1] + j;
                                std::get<7>(access) = input[i * k_shape.size[1] + j];
                                std::get<8>(access) = 0x01 << shift_amount;
                                std::get<9>(access) = (input[i * k_shape.size[1] + j] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/4) + ((cluster_id * 16) + container_id)];
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
                        for (int j = 0; j < k_shape.size[1] / 4; j++)
                            std::cout << ((int)temp[i * (k_shape.size[1] / 4) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #4 with shape of (" << k_shape.size[0] << ", " << k_shape.size[1] / 4 << ")" << std::endl;
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 4);
                    #ifdef PRINT_VALUES
                    std::cout << "After Packing:" << std::endl;
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < k_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < k_shape.size[1] / 4; j++)
                            std::cout << ((int)output[i * (k_shape.size[1] / 4) + j]) << ", ";
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
                if (shape.size[shape.number_dims - 1] % 64)
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

                    "add %w[end], %w[i], #64\n\t"
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

                    // SHL W, W, #1
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // USHR T3, W, #7
                    "ushr v8.16b,  v0.16b, #7\n"
                    "ushr v10.16b, v1.16b, #7\n"
                    "ushr v11.16b, v2.16b, #7\n"
                    "ushr v12.16b, v3.16b, #7\n"

                    // SSHL AT, AT, T3
                    "sshl v9.16b,  v9.16b,  v8.16b\n"
                    "sshl v13.16b, v13.16b, v10.16b\n"
                    "sshl v14.16b, v14.16b, v11.16b\n"
                    "sshl v15.16b, v15.16b, v12.16b\n"

                    // SHL W, W, #1
                    "shl v0.16b, v0.16b, #1\n"
                    "shl v1.16b, v1.16b, #1\n"
                    "shl v2.16b, v2.16b, #1\n"
                    "shl v3.16b, v3.16b, #1\n"
                    
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
                    return MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape);
                
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
                    rhs += 4 * (lhs_columns / 4);
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
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_batches = input_shape.size[0] ,
                    lhs_columns = input_shape.size[1] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if (lhs_batches % 4)
                    return Status::NotSupported;

                int8_t* rhs    = const_cast<int8_t*>(kernel);
                int8_t* lhs    = const_cast<int8_t*>(input);
                int32_t* dst_0 = output + 0 * rhs_rows;
                int32_t* dst_1 = output + 1 * rhs_rows;
                int32_t* dst_2 = output + 2 * rhs_rows;
                int32_t* dst_3 = output + 3 * rhs_rows;

                for (int j = 0; j < lhs_batches; j+=4){
                    int i;
                    for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                        dst_0 += 4;
                        dst_1 += 4;
                        dst_2 += 4;
                        dst_3 += 4;
                        rhs += 4 * (lhs_columns / 4);
                    }
                    if (rhs_rows - i == 1){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 2){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 3){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 4){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    lhs   += 4 * lhs_columns;
                    dst_0 += 4 * rhs_rows;
                    dst_1 += 4 * rhs_rows;
                    dst_2 += 4 * rhs_rows;
                    dst_3 += 4 * rhs_rows;
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

                    // SHL W, W, #1
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // USHR T3, W, #7
                    "ushr v8.16b,  v0.16b, #7\n"
                    "ushr v10.16b, v1.16b, #7\n"
                    "ushr v11.16b, v2.16b, #7\n"
                    "ushr v12.16b, v3.16b, #7\n"

                    // SSHL AT, AT, T3
                    "sshl v9.16b,  v9.16b,  v8.16b\n"
                    "sshl v13.16b, v13.16b, v10.16b\n"
                    "sshl v14.16b, v14.16b, v11.16b\n"
                    "sshl v15.16b, v15.16b, v12.16b\n"

                    // SHL W, W, #1
                    "shl v0.16b, v0.16b, #1\n"
                    "shl v1.16b, v1.16b, #1\n"
                    "shl v2.16b, v2.16b, #1\n"
                    "shl v3.16b, v3.16b, #1\n"

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
            #if LWO_PRECISION_QUATERNARY_V1_0
                if      (input ==  1)
                    return 0x00 << shift_amount;
                else if (input == -1)
                    return 0x02 << shift_amount;
                else if (input <  -1)
                    return 0x03 << shift_amount;
                else
                    return 0x01 << shift_amount;
            #elif LWO_PRECISION_QUATERNARY_V1_1
                if      (input ==  1)
                    return 0x00 << shift_amount;
                else if (input == -1)
                    return 0x02 << shift_amount;
                else if (input <  -1)
                    return 0x03 << shift_amount;
                else
                    return 0x01 << shift_amount;
            #endif
            }
        }
    }
}
#endif
