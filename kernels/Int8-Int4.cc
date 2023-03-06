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
        namespace Int4 {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 32)?(32 - (shape.size[1] % 32)):(0);
                Shape new_shape = GetPaddedShape(shape);
                int8_t* new_weight = allocate<int8_t>(new_shape.flatsize);
                for(int i = 0 ; i < shape.size[0] ; i++){
                    for(int j = 0 ; j < shape.size[1] ; j++)
                        new_weight[i * new_shape.size[1] + j] = weight[i * shape.size[1] + j];
                    for (int j = shape.size[1]; j < shape.size[1] + padding_size; j++)
                        new_weight[i * new_shape.size[1] + j] = 0;
                }
                return new_weight;
            }
            Status PaddingWeightsIfNeeded(const int8_t* input_weight, int8_t* output_weight, Shape shape){
                int padding_size = (shape.size[1] % 32)?(32 - (shape.size[1] % 32)):(0);
                Shape new_shape = GetPaddedShape(shape);
                for(int i = 0 ; i < shape.size[0] ; i++){
                    for(int j = 0 ; j < shape.size[1] ; j++)
                        output_weight[i * new_shape.size[1] + j] = input_weight[i * shape.size[1] + j];
                    for (int j = shape.size[1]; j < shape.size[1] + padding_size; j++)
                        output_weight[i * new_shape.size[1] + j] = 0;
                }
                return Status::Success;
            }
            Status PaddingInputsIfNeeded(const int8_t* input, int8_t* output, Shape shape){
                if (shape.number_dims == 1){
                    int padding_size = (shape.size[0] % 32)?(32 - (shape.size[0] % 32)):(0);
                    Shape new_shape = GetPaddedShape(shape);
                    for(int i = 0 ; i < shape.size[0] ; i++){
                        output[i] = input[i];
                    for (int i = shape.size[0]; i < shape.size[0] + padding_size; i++)
                        output[i] = 0;
                    }
                    return Status::Success;
                }
                int padding_size = (shape.size[1] % 32)?(32 - (shape.size[1] % 32)):(0);
                Shape new_shape = GetPaddedShape(shape);
                for(int i = 0 ; i < shape.size[0] ; i++){
                    for(int j = 0 ; j < shape.size[1] ; j++)
                        output[i * new_shape.size[1] + j] = input[i * shape.size[1] + j];
                    for (int j = shape.size[1]; j < shape.size[1] + padding_size; j++)
                        output[i * new_shape.size[1] + j] = 0;
                }
                return Status::Success;
            }
            Shape GetPaddedShape(const Shape shape){
                if (shape.number_dims == 1){
                    int padding_size = (shape.size[0] % 32)?(32 - (shape.size[0] % 32)):(0);
                    Shape new_shape;
                    new_shape.number_dims = shape.number_dims;
                    new_shape.size = new int[new_shape.number_dims];
                    new_shape.size[0] = shape.size[0] + padding_size;
                    new_shape.flatsize = new_shape.size[0];
                    return new_shape;
                }
                int padding_size = (shape.size[1] % 32)?(32 - (shape.size[1] % 32)):(0);
                Shape new_shape;
                new_shape.number_dims = shape.number_dims;
                new_shape.size = new int[new_shape.number_dims];
                new_shape.size[0] = shape.size[0];
                new_shape.size[1] = shape.size[1] + padding_size;
                new_shape.flatsize = new_shape.size[0] * new_shape.size[1];
                return new_shape;
            }
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 4);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeInputWithPadding(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.number_dims != 2 && shape.number_dims != 1)
                    return Status::DimensionsMisMatch;
                if (shape.number_dims == 2 && shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout == MemLayout::kColumnMajor)
                    return Status::WrongMemLayout;
                
                bool is_multibatch = shape.size[0] > 1;
                Shape padded_shape = GetPaddedShape(shape);
                int new_weights_length = 0;
                if (is_multibatch)
                    new_weights_length = (int)(padded_shape.size[1] / 2) * padded_shape.size[0];
                else
                    new_weights_length = padded_shape.flatsize;
                Status padding_status = PaddingInputsIfNeeded(input, output, shape);
                if (padding_status != Status::Success)
                    return padding_status;
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 32)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                // if (is_multibatched && shape.size[0] % 4)
                //     return Status::SizesMisMatch; 
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
            Status QuantizeFilterWithPadding(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (layout == MemLayout::kColumnMajor)
                    return Status::WrongMemLayout;
                
                Shape padded_shape = GetPaddedShape(k_shape);
                int new_weights_length = (int)(padded_shape.size[1] / 2) * padded_shape.size[0];
                int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length + padded_shape.flatsize);
                int8_t* input_padded = temp + new_weights_length;
                Status padding_status = PaddingWeightsIfNeeded(input, input_padded, k_shape);
                if (padding_status != Status::Success)
                    return padding_status;

                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input_padded), output, padded_shape.size[0], padded_shape.size[1] / 2);
                }
                else {
                    // std::cerr << "Filter Qantizer Step #1 with shape of " << get_shape_string(k_shape) << std::endl;
                    zero_vector(temp, new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    // std::cerr << "Filter Qantizer Step #2\n";
                    #ifdef PRINT_VALUES_DETAILED
                    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int>> memory_accesses;
                    std::cout << get_shape_string(padded_shape) << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #3\n";
                    if (layout == MemLayout::kColumnMajor)
                        for (int i = 0 ; i < padded_shape.size[0] ; i++){
                            for (int j = 0 ; j < padded_shape.size[1] ; j++){
                                int cluster_id = j / 32,
                                    container_id = (j % 32) % 16,
                                    shift_amount = ((j % 32) / 16);
                                temp_u[i * (padded_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input_padded[j * padded_shape.size[0] + i], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (padded_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount * 4;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = j * padded_shape.size[0] + i;
                                std::get<7>(access) = input_padded[j * padded_shape.size[0] + i];
                                std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                                std::get<9>(access) = (input_padded[j * padded_shape.size[0] + i] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    else
                        for (int i = 0 ; i < padded_shape.size[0] ; i++){
                            for (int j = 0 ; j < padded_shape.size[1] ; j++){
                                int cluster_id = j / 32,
                                    container_id = (j % 32) % 16,
                                    shift_amount = ((j % 32) / 16);
                                temp_u[i * ((padded_shape.size[1])/2) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input_padded[i * (padded_shape.size[1]) + j], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (padded_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount * 4;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = i * padded_shape.size[1] + j;
                                std::get<7>(access) = input_padded[i * padded_shape.size[1] + j];
                                std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                                std::get<9>(access) = (input_padded[i * padded_shape.size[1] + j] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
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
                                << std::dec
                                << ")\n";
                    #endif
                    #ifdef PRINT_VALUES
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < padded_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < padded_shape.size[1] / 2; j++)
                            std::cout << ((int)temp[i * (padded_shape.size[1] / 2) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #4 with shape of (" << padded_shape.size[0] << ", " << padded_shape.size[1] / 2 << ")" << std::endl;
                    doLowPrecisionWeightPack(temp, output, padded_shape.size[0], padded_shape.size[1] / 2);
                    #ifdef PRINT_VALUES
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < padded_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < padded_shape.size[1] / 2; j++)
                            std::cout << ((int)output[i * (padded_shape.size[1] / 2) + j]) << ", ";
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
            Status MultiplyInt8(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                // In case of multi-batched input
                // std::cout << "Inside MultiplyInt8 with multibatch set to " 
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

                int i;
                for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                    rhs += 4 * (lhs_columns / 2);
                }
                if (rhs_rows - i == 1){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 2){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 3){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 4){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
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

                // if (lhs_columns >= rhs_rows){
                if (true){
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
                }
                else {
                    asm volatile(
                        "mov %w[j], wzr\n\t"

                        "0:\n\t"
                        "dup v23.8h, wzr\n\t"
                        "dup v24.8h, wzr\n\t"
                        "dup v25.8h, wzr\n\t"
                        "dup v30.8h, wzr\n\t"
                        "dup v26.8h, wzr\n\t"
                        "dup v27.8h, wzr\n\t"
                        "dup v28.8h, wzr\n\t"
                        "dup v29.8h, wzr\n\t"
                        "mov %w[i],  wzr\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.16b},  [%[weights]], #16\n\t"
                        "ld1 {v1.16b},  [%[weights]], #16\n\t"
                        "ld1 {v2.16b},  [%[weights]], #16\n\t"
                        "ld1 {v3.16b},  [%[weights]], #16\n\t"

                        "ld1 {v4.16b},  [%[activation]], #16\n\t"

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

                        // SMULL T1.8h, W.8b, A1.8b
                        "smull v13.8h, v8.8b,  v4.8b\n\t"
                        "smull v14.8h, v10.8b, v4.8b\n\t"
                        "smull v15.8h, v11.8b, v4.8b\n\t"
                        "smull v16.8h, v12.8b, v4.8b\n\t"
                        "smull v17.8h, v0.8b,  v4.8b\n\t"
                        "smull v18.8h, v1.8b,  v4.8b\n\t"
                        "smull v19.8h, v2.8b,  v4.8b\n\t"
                        "smull v20.8h, v3.8b,  v4.8b\n\t"

                        // SMLAL2 T1.8h, W.16b, A1.16b
                        "smlal2 v13.8h, v8.16b,  v4.16b\n\t"
                        "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                        "smlal2 v15.8h, v11.16b, v4.16b\n\t"
                        "smlal2 v16.8h, v12.16b, v4.16b\n\t"
                        "smlal2 v17.8h, v0.16b,  v4.16b\n\t"
                        "smlal2 v18.8h, v1.16b,  v4.16b\n\t"
                        "smlal2 v19.8h, v2.16b,  v4.16b\n\t"
                        "smlal2 v20.8h, v3.16b,  v4.16b\n\t"

                        // ACCUMULATE ACC, T1
                        "sadalp v23.4s, v13.8h\n\t"
                        "sadalp v24.4s, v14.8h\n\t"
                        "sadalp v25.4s, v15.8h\n\t"
                        "sadalp v30.4s, v16.8h\n\t"
                        "sadalp v26.4s, v17.8h\n\t"
                        "sadalp v27.4s, v18.8h\n\t"
                        "sadalp v28.4s, v19.8h\n\t"
                        "sadalp v29.4s, v20.8h\n\t"

                        "add %w[i], %w[i], #16\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "addv s23, v23.4s\n\t"
                        "addv s24, v24.4s\n\t"
                        "addv s25, v25.4s\n\t"
                        "addv s30, v30.4s\n\t"
                        "addv s26, v26.4s\n\t"
                        "addv s27, v27.4s\n\t"
                        "addv s28, v28.4s\n\t"
                        "addv s29, v29.4s\n\t"

                        "mov v23.s[1], v24.s[0]\n\t"
                        "mov v23.s[2], v25.s[0]\n\t"
                        "mov v23.s[3], v30.s[0]\n\t"
                        "mov v26.s[1], v27.s[0]\n\t"
                        "mov v26.s[2], v28.s[0]\n\t"
                        "mov v26.s[3], v29.s[0]\n\t"

                        "st1 {v23.4s},  [%[dst]], #16\n\t"
                        "st1 {v26.4s},  [%[dst]], #16\n\t"

                        "mov %[activation], %[lhs_base]\n\t"

                        "3:\n\t"

                        "add %w[j], %w[j], #8\n\t"
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
                }

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
                return Status::Success;
            }
            Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){
                int start_batches   = params.start_batches,
                    start_columns   = params.start_columns,
                    start_rows      = params.start_rows,

                    end_batches     = params.end_batches,
                    end_columns     = params.end_columns,
                    end_rows        = params.end_rows,
                    //TODO: Make sure you are passing strides with respect to quantization and int32 coefficients
                    lhs_stride      = params.lhs_stride,
                    rhs_stride      = params.rhs_stride,
                    dst_stride      = params.dst_stride
                ;
                int8_t*         _kernel     = const_cast<int8_t*>(kernel);
                int8_t*         _kernel_base= const_cast<int8_t*>(kernel);
                int8_t*         _input      = get_pointer_as<int8_t>(const_cast<int8_t*>(input));
                int8_t*         _input_base = get_pointer_as<int8_t>(const_cast<int8_t*>(input));
                int i, j, k, end;
                int32_t*        _output_1   = output + 0 * dst_stride;
                int32_t*        _output_2   = output + 1 * dst_stride;
                int32_t*        _output_3   = output + 2 * dst_stride;
                int32_t*        _output_4   = output + 3 * dst_stride;
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
                    "mov w0, #3\n\t"
                    "mul w0, %w[dst_stride], w0\n\t"
                    "mov w3, %w[start_batches]\n\t"
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"
                    "mov x6, %[weights]\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov w4, %w[start_rows]\n\t"

                    "0:\n\t"
                    "mov w5, %w[start_columns]\n\t"
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
                    "add w5, w5, #32\n\t"
                    "cmp w5, %w[end_columns]\n\t"
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
                    
                    // Load the 4 int32 results
                    "ld1 {v17.4s},  [%[dst_1]], #16\n\t"
                    "ld1 {v21.4s},  [%[dst_2]], #16\n\t"
                    "ld1 {v25.4s},  [%[dst_3]], #16\n\t"
                    "ld1 {v29.4s},  [%[dst_4]], #16\n\t"
                    
                    // Reset output pointer
                    "sub %[dst_1], %[dst_1], #16\n\t"
                    "sub %[dst_2], %[dst_2], #16\n\t"
                    "sub %[dst_3], %[dst_3], #16\n\t"
                    "sub %[dst_4], %[dst_4], #16\n\t"

                    // Accumulate 4 int32 results
                    "add v16.4s, v17.4s, v16.4s\n\t"
                    "add v20.4s, v21.4s, v20.4s\n\t"
                    "add v24.4s, v25.4s, v24.4s\n\t"
                    "add v28.4s, v29.4s, v28.4s\n\t"

                    // Store the 4 int32 results
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                    
                    // Move weights pointer forward for 4 x rhs_stride
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[weights], %[weights]\n\t"
#else
                    "add %[weights], x6, %[rhs_stride], lsl #2\n\t"
                    "add x6, x6, %[rhs_stride], lsl #2\n\t"
#endif

                    // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "mov %[activation], x1\n\t"
#endif

                    // Check if the all the columns of weight matrix are processed
                    "add w4, w4, #4\n\t"
                    "cmp w4, %w[end_rows]\n\t"
                    "b.lt 0b\n\t"

                    // Move weights pointer forward for 4 x lhs_stride
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "add %[activation], x1, %[lhs_stride], lsl #2\n\t"
                    "add x1, x1, %[lhs_stride], lsl #2\n\t"
#endif

                    // Reset the weights to the start
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[weights], %[weights]\n\t"
#else
                    "mov %[weights], x2\n\t"
                    "mov x6, x2\n\t"
#endif

                    // Prepare the destination base for next 4 batches
                    "add %[dst_1], %[dst_1], x0\n\t"
                    "add %[dst_2], %[dst_2], x0\n\t"
                    "add %[dst_3], %[dst_3], x0\n\t"
                    "add %[dst_4], %[dst_4], x0\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add w3, w3, #4\n\t"
                    "cmp w3, %w[end_batches]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                      [ dst_3 ]      "+r" (_output_3),   [ dst_4 ]       "+r" (_output_4)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ start_batches ] "r" (start_batches), [ start_columns ] "r" (start_columns), [ start_rows ] "r" (start_rows),
                      [ end_batches ]   "r" (end_batches),   [ end_columns ]   "r" (end_columns),   [ end_rows ]   "r" (end_rows),
                      [ lhs_stride ]    "r" (lhs_stride),    [ rhs_stride ]    "r" (rhs_stride),    [ dst_stride ] "r" (dst_stride)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31",
                      "x0" , "x1" , "x2" , "x3",
                      "x4" , "x5" , "x6"
                );
                return Status::Success;
            }
            void doMultiplication1Col(const int8_t* activation, 
                            int8_t* weights, 
                            int32_t* dst, int size){
                const int8_t* _activation = activation;
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

                    // "ld1 {v30.4s},  [%[dst]]\n\t"
                    // "add v30.4s, v30.4s, v23.4s\n\t"
                    // "st1 {v30.4s},  [%[dst]]\n\t"
                    "st1 {v23.4s},  [%[dst]]\n\t"

                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"

                    "3:\n\t"

                    : [ dst ] "+r"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
                    : [ activation ] "r"(_activation), [ weights ] "r"(weights),
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
            void doMultiplication(const int8_t* activation,
                            int8_t* weights,
                            int32_t* dst_1, int32_t* dst_2,
                            int32_t* dst_3, int32_t* dst_4,
                            int size){
                const int8_t* _activation = activation;
                int i, end;
                int8_t mask = 0x0F;
                /* Vector assignments:
                    * W    -> v0-3
                    * A    -> v4-7
                    * CW   -> v8-11
                    * T    -> v12-15
                    * ACC1 -> v16-19
                    * ACC2 -> v20-23
                    * ACC3 -> v24-27
                    * ACC4 -> v28-31
                */
                asm volatile(
                    #define SET_ZERO(reg) "movi " #reg ".4s, #0\n\t"

                    #define MULL(dst, src1, src2) "smull " #dst ".8h, " #src1 ".8b, " #src2 ".8b\n\t" \
                                                "smlal2 " #dst ".8h, " #src1 ".16b, " #src2 ".16b\n\t"
                    
                    #define ACC(acc, src) "sadalp " #acc ".4s, " #src ".8h\n\t"

                    // Clear accumulators.
                    SET_ZERO(v16)
                    SET_ZERO(v17)
                    SET_ZERO(v18)
                    SET_ZERO(v19)
                    SET_ZERO(v20)
                    SET_ZERO(v21)
                    SET_ZERO(v22)
                    SET_ZERO(v23)
                    SET_ZERO(v24)
                    SET_ZERO(v25)
                    SET_ZERO(v26)
                    SET_ZERO(v27)
                    SET_ZERO(v28)
                    SET_ZERO(v29)
                    SET_ZERO(v30)
                    SET_ZERO(v31)

                    // "dup v29.16b, %w[mask]\n\t"

                    "mov %w[i], wzr\n\t"
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"

                    // Create Mask Vector
                    "dup v12.16b, %w[mask]\n\t"

                    // LD W1-4,x-y, WP
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // LD A1-4,x  , AP
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    // AND CW, W, M
                    "and v8.16b,  v0.16b, v12.16b\n\t"
                    "and v9.16b,  v1.16b, v12.16b\n\t"
                    "and v10.16b, v2.16b, v12.16b\n\t"
                    "and v11.16b, v3.16b, v12.16b\n\t"

                    // SHL CW, CW, #4
                    "shl v8.16b,  v8.16b,  #4\n\t"
                    "shl v9.16b,  v9.16b,  #4\n\t"
                    "shl v10.16b, v10.16b, #4\n\t"
                    "shl v11.16b, v11.16b, #4\n\t"

                    // SSHR CW, CW, #4
                    "sshr v8.16b,  v8.16b,  #4\n\t"
                    "sshr v9.16b,  v9.16b,  #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // M T, CW, A1,x
                    MULL(v12, v8,  v4)
                    MULL(v13, v9,  v4)
                    MULL(v14, v10, v4)
                    MULL(v15, v11, v4)

                    // ACC ACC1, T
                    ACC(v16, v12)
                    ACC(v17, v13)
                    ACC(v18, v14)
                    ACC(v19, v15)

                    // M T, CW, A2,x
                    MULL(v12, v8,  v5)
                    MULL(v13, v9,  v5)
                    MULL(v14, v10, v5)
                    MULL(v15, v11, v5)

                    // ACC ACC2, T
                    ACC(v20, v12)
                    ACC(v21, v13)
                    ACC(v22, v14)
                    ACC(v23, v15)

                    // M T, CW, A3,x
                    MULL(v12, v8,  v6)
                    MULL(v13, v9,  v6)
                    MULL(v14, v10, v6)
                    MULL(v15, v11, v6)

                    // ACC ACC3, T
                    ACC(v24, v12)
                    ACC(v25, v13)
                    ACC(v26, v14)
                    ACC(v27, v15)

                    // M T, CW, A4,x
                    MULL(v12, v8,  v7)
                    MULL(v13, v9,  v7)
                    MULL(v14, v10, v7)
                    MULL(v15, v11, v7)

                    // ACC ACC4, T
                    ACC(v28, v12)
                    ACC(v29, v13)
                    ACC(v30, v14)
                    ACC(v31, v15)

                    // LD A1-4,y  , AP
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    // SSHR CW, W, #4
                    "sshr v8.16b,  v0.16b, #4\n\t"
                    "sshr v9.16b,  v1.16b, #4\n\t"
                    "sshr v10.16b, v2.16b, #4\n\t"
                    "sshr v11.16b, v3.16b, #4\n\t"

                    // M T, CW, A1,x
                    MULL(v12, v8,  v4)
                    MULL(v13, v9,  v4)
                    MULL(v14, v10, v4)
                    MULL(v15, v11, v4)

                    // ACC ACC1, T
                    ACC(v16, v12)
                    ACC(v17, v13)
                    ACC(v18, v14)
                    ACC(v19, v15)

                    // M T, CW, A2,x
                    MULL(v12, v8,  v5)
                    MULL(v13, v9,  v5)
                    MULL(v14, v10, v5)
                    MULL(v15, v11, v5)

                    // ACC ACC2, T
                    ACC(v20, v12)
                    ACC(v21, v13)
                    ACC(v22, v14)
                    ACC(v23, v15)

                    // M T, CW, A3,x
                    MULL(v12, v8,  v6)
                    MULL(v13, v9,  v6)
                    MULL(v14, v10, v6)
                    MULL(v15, v11, v6)

                    // ACC ACC3, T
                    ACC(v24, v12)
                    ACC(v25, v13)
                    ACC(v26, v14)
                    ACC(v27, v15)

                    // M T, CW, A4,x
                    MULL(v12, v8,  v7)
                    MULL(v13, v9,  v7)
                    MULL(v14, v10, v7)
                    MULL(v15, v11, v7)

                    // ACC ACC4, T
                    ACC(v28, v12)
                    ACC(v29, v13)
                    ACC(v30, v14)
                    ACC(v31, v15)

                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Copied From Ruy::Kernel8bitNeon
                    // Reduce 32bit accumulators horizontally.
                    "addp v16.4s, v16.4s, v17.4s\n"
                    "addp v18.4s, v18.4s, v19.4s\n"
                    "addp v20.4s, v20.4s, v21.4s\n"
                    "addp v22.4s, v22.4s, v23.4s\n"
                    "addp v24.4s, v24.4s, v25.4s\n"
                    "addp v26.4s, v26.4s, v27.4s\n"
                    "addp v28.4s, v28.4s, v29.4s\n"
                    "addp v30.4s, v30.4s, v31.4s\n"

                    // Reduce 32bit accumulators horizontally, second pass
                    // (each pass adds pairwise. we need to add 4-wise).
                    "addp v16.4s, v16.4s, v18.4s\n"
                    "addp v17.4s, v20.4s, v22.4s\n"
                    "addp v18.4s, v24.4s, v26.4s\n"
                    "addp v19.4s, v28.4s, v30.4s\n"

                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v17.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v18.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v19.4s},  [%[dst_4]], #16\n\t"

                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"

                    "3:\n\t"

                    : [ dst_1 ] "+r"(dst_1), [ dst_2 ] "+r"(dst_2),
                    [ dst_3 ] "+r"(dst_3), [ dst_4 ] "+r"(dst_4),
                    [ i ] "+r"(i), [ end ] "+r"(end)
                    : [ activation ] "r"(_activation), [ weights ] "r"(weights),
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
