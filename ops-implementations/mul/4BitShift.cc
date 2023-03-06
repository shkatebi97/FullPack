#ifdef IS_ARM

// #ifndef TFLITE_BUILD
// #include "Mul.h"
// #endif

#include "4BitShift.h"
#include "LowPrecisionPacking.h"
#include <fstream>
#include <streambuf>
#include <arm_neon.h>

namespace Shift4Bit {

//////////////////////////////////////
/////////////// Common ///////////////
//////////////////////////////////////
    Status Prepare(const int8_t* weight, Shape k_shape, void** &output, DataType data_type, MemLayout layout){
        if (k_shape.number_dims != 2)
            return Status::DimensionsMisMatch;
        if (k_shape.size[1] % 32)
            return Status::SizesMisMatch;
        if (data_type == DataType::Int8){
            int new_weights_length = (k_shape.size[1] / 2) * k_shape.size[0];
            int8_t* temp = allocate<int8_t>(new_weights_length);
            uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
            #ifdef PRINT_VALUES
            std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int>> memory_accesses;
            std::cout << get_shape_string(k_shape) << std::endl;
            #endif
            if (layout == MemLayout::kColumnMajor)
                for (int i = 0 ; i < k_shape.size[0] ; i++){
                    for (int j = 0 ; j < k_shape.size[1] ; j++){
                        int cluster_id = j / 32,
                            container_id = (j % 32) % 16,
                            shift_amount = ((j % 32) / 16);
                        temp_u[i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                            quantizeAndPackBitsStep(weight[j * k_shape.size[0] + i], shift_amount);
                        #ifdef PRINT_VALUES
                        std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                        std::get<0>(access) = i;
                        std::get<1>(access) = j;
                        std::get<2>(access) = i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                        std::get<3>(access) = shift_amount * 4;
                        std::get<4>(access) = cluster_id;
                        std::get<5>(access) = container_id;
                        std::get<6>(access) = j * k_shape.size[0] + i;
                        std::get<7>(access) = weight[j * k_shape.size[0] + i];
                        std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                        std::get<9>(access) = (weight[j * k_shape.size[0] + i] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                        memory_accesses.push_back(access);
                        #endif
                    }
                }
            else
                for (int i = 0 ; i < k_shape.size[0] ; i++){
                    for (int j = 0 ; j < k_shape.size[1] ; j++){
                        int cluster_id = j / 32,
                            container_id = (j % 32) % 16,
                            shift_amount = ((j % 32) / 16);
                        temp_u[i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                            quantizeAndPackBitsStep(weight[i * k_shape.size[1] + j], shift_amount);
                        #ifdef PRINT_VALUES
                        std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                        std::get<0>(access) = i;
                        std::get<1>(access) = j;
                        std::get<2>(access) = i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                        std::get<3>(access) = shift_amount * 4;
                        std::get<4>(access) = cluster_id;
                        std::get<5>(access) = container_id;
                        std::get<6>(access) = i * k_shape.size[1] + j;
                        std::get<7>(access) = weight[i * k_shape.size[1] + j];
                        std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                        std::get<9>(access) = (weight[i * k_shape.size[1] + j] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                        memory_accesses.push_back(access);
                        #endif
                    }
                }
            #ifdef PRINT_VALUES
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
            for (int i = 0; i < k_shape.size[0]; i++){
                std::cout << "\t[";
                for (int j = 0; j < k_shape.size[1] / 2; j++)
                    std::cout << ((int)temp[i * (k_shape.size[1] / 2) + j]) << ", ";
                std::cout << "]," << std::endl;
            }
            std::cout << "]";
            std::cout << std::endl;
            #endif
            int8_t* tmp = allocate<int8_t>(new_weights_length);
            doLowPrecisionWeightPack(temp, tmp, k_shape.size[0], k_shape.size[1] / 2);
            #ifdef PRINT_VALUES
            std::cout << "[" << std::endl;
            for (int i = 0; i < k_shape.size[0]; i++){
                std::cout << "\t[";
                for (int j = 0; j < k_shape.size[1] / 2; j++)
                    std::cout << ((int)tmp[i * (k_shape.size[1] / 2) + j]) << ", ";
                std::cout << "]," << std::endl;
            }
            std::cout << "]";
            std::cout << std::endl;
            #endif
            output[0] = get_pointer_as<void>(tmp);
            deallocate(temp);
        }
        else
            return Status::WrongDataType;
        return Status::Success;
    }
    Status Init(){return Status::Success;}
    Status Free(DataType data_type, void** temperories){
        if (data_type == DataType::Int8){
            deallocate(get_pointer_as<int8_t>(temperories[0]));
        }
        else if (data_type == DataType::Float16){
            return Status::NotSupported;
            deallocate(get_pointer_as<int16_t>(temperories[0]));
        }
        else if (data_type == DataType::Float32){
            return Status::NotSupported;
            deallocate(get_pointer_as<int32_t>(temperories[0]));
        }
        else
            return Status::WrongDataType;
        deallocate(temperories);
        return Status::Success;
    }
    Status Eval(const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape){
        int lhs_columns = input_shape.size[0] ,
            rhs_rows = kernel_shape.size[0];
#ifdef TFLITE_BUILD
        int rhs_columns = kernel_shape.size[1];
#else
        int rhs_columns = kernel_shape.size[0];
#endif
        
        if (lhs_columns != rhs_columns)
            return Status::SizesMisMatch;
        int8_t* rhs = const_cast<int8_t*>(kernel);
        const int8_t* lhs = input;

        int i;
        for (i = 0 ; (i+4) < rhs_rows ; i+=4){
            kernel1Col(lhs, rhs, &output[i], lhs_columns);
            rhs += 4 * (lhs_columns / 2);
        }
        if (rhs_rows - i == 1){
            kernel1Col(lhs, rhs, &output[i], lhs_columns);
        }
        else if (rhs_rows - i == 2){
            kernel1Col(lhs, rhs, &output[i], lhs_columns);
        }
        else if (rhs_rows - i == 3){
            kernel1Col(lhs, rhs, &output[i], lhs_columns);
        }
        else if (rhs_rows - i == 4){
            kernel1Col(lhs, rhs, &output[i], lhs_columns);
        }
        return Status::Success;
    }
    void kernel1Col(const int8_t* activation, 
                    int8_t* weights, 
                    int32_t* dst, 
                    int size){
        //////////////////////////////////////////////////////////////////////
        ////
        ////
        ////
        ////                ! Must Complete Below !
        ////
        ////
        ////
        ////
        ////
        //////////////////////////////////////////////////////////////////////


                        
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
            "dup v31.16b, %w[mask]\n\t"
            "dup v16.8h, wzr\n\t"
            "dup v17.8h, wzr\n\t"
            "dup v18.8h, wzr\n\t"
            "dup v19.8h, wzr\n\t"
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
            // "sxtl2 v6.8h, v4.16b\n\t"
            // "sxtl  v4.8h, v4.8b \n\t"

            "ld1 {v5.16b},  [%[activation]], #16\n\t"
            // "sxtl2 v7.8h, v5.16b\n\t"
            // "sxtl  v5.8h, v5.8b \n\t"
        
            // AND WL, W, M
            "and v8.16b,  v0.16b, v31.16b\n\t"
            "and v9.16b,  v1.16b, v31.16b\n\t"
            "and v10.16b, v2.16b, v31.16b\n\t"
            "and v11.16b, v3.16b, v31.16b\n\t"
        
            // SHL WL, WL, #4
            "shl v8.16b,  v8.16b,  #4\n\t"
            "shl v9.16b,  v9.16b,  #4\n\t"
            "shl v10.16b, v10.16b, #4\n\t"
            "shl v11.16b, v11.16b, #4\n\t"
        
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
        
            // SQSHL T1.16b, WL.16b, A1.16b
            "sqshl v12.16b, v4.16b, v8.16b\n\t"
            "sqshl v13.16b, v4.16b, v9.16b\n\t"
            "sqshl v14.16b, v4.16b, v10.16b\n\t"
            "sqshl v15.16b, v4.16b, v11.16b\n\t"
        
            // SQSHL T2.16b, WH.16b, A2.16b
            "sqshl v8.16b,  v5.16b, v0.16b\n\t"
            "sqshl v9.16b,  v5.16b, v1.16b\n\t"
            "sqshl v10.16b, v5.16b, v2.16b\n\t"
            "sqshl v11.16b, v5.16b, v3.16b\n\t"

            // SADDLP T1.8h, T1.16b
            "saddlp v12.8h, v12.16b\n\t"
            "saddlp v13.8h, v13.16b\n\t"
            "saddlp v14.8h, v14.16b\n\t"
            "saddlp v15.8h, v15.16b\n\t"

            // SADDLP T2.8h, T2.16b
            "saddlp v8.8h,  v8.16b \n\t"
            "saddlp v9.8h,  v9.16b\n\t"
            "saddlp v10.8h, v10.16b\n\t"
            "saddlp v11.8h, v11.16b\n\t"
        
            // ACCUMULATE ACC, T1
            "sadalp v16.4s, v12.8h\n\t"
            "sadalp v17.4s, v13.8h\n\t"
            "sadalp v18.4s, v14.8h\n\t"
            "sadalp v19.4s, v15.8h\n\t"
        
            // ACCUMULATE ACC, T2
            "sadalp v16.4s, v8.8h \n\t"
            "sadalp v17.4s, v9.8h\n\t"
            "sadalp v18.4s, v10.8h\n\t"
            "sadalp v19.4s, v11.8h\n\t"

            "add %w[i], %w[i], #32\n\t"
            "cmp %w[i], %w[size]\n\t"
            "b.lt 1b\n\t"
        
            "addv s16, v16.4s\n\t"
            "addv s17, v17.4s\n\t"
            "addv s18, v18.4s\n\t"
            "addv s19, v19.4s\n\t"

            "mov v16.s[1], v17.s[0]\n\t"
            "mov v16.s[2], v18.s[0]\n\t"
            "mov v16.s[3], v19.s[0]\n\t"

            "st1 {v16.4s},  [%[dst]]\n\t"

            "sub %[activation], %[activation], %[size]\n\t"
            "sub %[weights], %[weights], %[size]\n\t"
            "sub %[weights], %[weights], %[size]\n\t"
        
            "3:\n\t"
        
            : [ dst ] "+r"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
            : [ activation ] "r"(activation), [ weights ] "r"(weights),
              [ size ] "r"(size), [ mask ] "r"(mask)
            : "v0",  "v1",  "v2",  "v3",
              "v4",  "v5",  "v6",  "v7",
              "v8",  "v9",  "v10", "v11",
              "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19",
              "v31"
        );
    }
    uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount){
        return ((((int)input) & 0x0f) << (shift_amount * 4));
        // return ((((int)ceil(log2(input))) & 0x0f) << (shift_amount * 4));
    }
}

#else
namespace Shift4Bit{
    Status Prepare(const int8_t* weight, Shape k_shape, void** &output, DataType data_type, MemLayout layout){
        return Status::NotImplemented;
    }
    Status Init(){
        return Status::NotImplemented;
    }
    Status Free(DataType data_type){
        return Status::NotImplemented;
    }
    Status Eval(int8_t* lhs,
                        int32_t* dst, 
                        int lhs_columns, int rhs_rows, int rhs_columns){
        return Status::NotImplemented;
    }
    void kernel1Col(const int8_t* activation, 
                    int8_t* weights, 
                    int32_t* dst,
                    int size){ }
    uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount) { return 0; }
}

#endif





