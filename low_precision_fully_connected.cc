#include "low_precision_fully_connected.h"

#ifndef IS_ARM
#else
namespace LowPrecision{
    unsigned long int Shape::last_id = 0;
    
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        using ::LowPrecision::Matrix;
        using ::LowPrecision::Params;
        using ::LowPrecision::MatrixType;
        using ::LowPrecision::MulParams;
        LowPrecision::Method __default_method;
        LowPrecision::Method get_default_method() { return __default_method; } 
        void set_default_method(LowPrecision::Method method) { __default_method = method; }
        Method GetMethodFromEnv(){
            char * val;
            val = getenv( "LowPrecisionFC" );
            std::string retval = "";                                                          
            if (val != NULL) {
                retval = val;
            }
            if (retval == std::string("I8-I4"))
                return Method::kInt8Int4;
            else if (retval == std::string("I8-Binary"))
                return Method::kInt8Binary;
            else if (retval == std::string("I8-Ternary"))
                return Method::kInt8Ternary;
            else if (retval == std::string("I8-Quaternary"))
                return Method::kInt8QuaTernary;
            else if (retval == std::string("F32-Binary"))
                return Method::kFloat32Binary;
            else if (retval == std::string("I4-I8"))
                return Method::kInt4ActInt8Weight;
            else if (retval == std::string("I4-I4"))
                return Method::kInt4ActInt4Weight;
            else if (retval == std::string("Ternary-I8"))
                return Method::kTernaryActInt8Weight;
            else if (retval == std::string("Ternary-Ternary"))
                return Method::kTernaryActTernaryWeight;
            else if (retval == std::string("Binary-I8"))
                return Method::kBinaryActInt8Weight;
            else if (retval == std::string("Binary-Binary"))
                return Method::kBinaryActBinaryWeight;
            else if (retval == std::string("Binary-Binary-XOR"))
                return Method::kBinaryActBinaryWeightXOR;
            else if (retval == std::string("I3-I3"))
                return Method::kInt3ActInt3Weight;
            else if (retval == std::string("ULPPACK-W1A1"))
                return Method::kULPPACKW1A1;
            else if (retval == std::string("ULPPACK-W2A2"))
                return Method::kULPPACKW2A2;
            else if (retval == std::string("ULPPACK-W3A3"))
                return Method::kULPPACKW3A3;
            else if (retval == std::string("ULPPACK-W4A4"))
                return Method::kULPPACKW4A4;
            else if (retval == std::string("ULPPACK-W5A5"))
                return Method::kULPPACKW5A5;
            else if (retval == std::string("ULPPACK-W6A6"))
                return Method::kULPPACKW6A6;
            else if (retval == std::string("ULPPACK-W7A7"))
                return Method::kULPPACKW7A7;
            else if (retval == std::string("BarrelShift-Mul-W8A8"))
                return Method::kInt8ActInt8WeightBarrelShiftMul;
            else
                return Method::kNoOptimization;
        } 
        std::string GetVariableFromEnv(std::string variable){
            char * val;
            val = getenv( variable.c_str() );
            std::string retval = "";                                                          
            if (val != NULL) {
                retval = val;
            }
            return retval;
        }
        LowPrecision::DataType GetDataType(int type){
            if (type == 1)
                return DataType::Float32;
            else if (type == 2)
                return DataType::Int32;
            else if (type == 6)
                return DataType::Bool;
            else if (type == 7)
                return DataType::Int16;
            else if (type == 9)
                return DataType::Int8;
            else if (type == 10)
                return DataType::Float16;
            else
                return DataType::NotAvailable;
        }
        bool IsAppliable(
            Method method, Shape input_shape, Shape filter_shape,
            DataType input_type, DataType filter_type, 
            DataType output_type, bool Is_FC = false){
            bool multibatched_enabled = !(GetVariableFromEnv( "LowPrecisionMultiBatched" ) == "FALSE") || method & Method::kULPPACK;
            bool singlebatched_enabled = !(GetVariableFromEnv( "LowPrecisionSingleBatched" ) == "FALSE");
            bool input_row_pad_enabled = !(GetVariableFromEnv( "LowPrecisionPadInputRows" ) == "FALSE");
            bool filter_row_pad_enabled = !(GetVariableFromEnv( "LowPrecisionAcceptFilterPadding" ) == "FALSE");
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            
            // Checking for Not-Supported Input DataTypes
            if (
                (input_type != DataType::Int8 && input_type != DataType::Float32 && input_type != DataType::Int32) ||
                filter_type != DataType::Int8 ||
                (output_type != DataType::Int8 && output_type != DataType::Float32 && output_type != DataType::Int32))
                return false;

            // Checking for the conditions of rejection of multi-batched or single-batch input
            if(is_multibatched && !multibatched_enabled)
                return false;
            else if(!is_multibatched && !singlebatched_enabled)
                return false;
            
            if (
                is_multibatched && 
                multibatched_enabled &&
                input_shape.size[0] % 4 &&
                !input_row_pad_enabled
            )
                return false;
            // Checking common conditions
            if (filter_shape.size[filter_shape.number_dims - 2] % 4 && !filter_row_pad_enabled)
                return false;
            // std::cout << "PASSED" << std::endl;

            // Checking conditions of input shape of any method
            // if (method == Method::kInt8Binary ||
            //     method == Method::kFloat32Binary ||
            //     method == Method::kFloat16Binary)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kInt8Ternary ||
            //     method == Method::kFloat32Ternary ||
            //     method == Method::kFloat16Ternary ||
            //     method == Method::kInt8QuaTernary)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kInt8Int4)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kInt4ActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kInt4ActInt4Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kTernaryActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kTernaryActTernaryWeight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kBinaryActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kBinaryActBinaryWeight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kBinaryActBinaryWeightXOR)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            if (method == Method::kInt3ActInt3Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 40);

            // if (method & Method::kULPPACK)
            //     return !(input_shape.size[input_shape.number_dims - 1] % 16);
            if (method == Method::kNoOptimization)
                return false;

            // If none of the aboves
            return true;
        }
        bool IncludesActivationCompression(Method method){
            return 
                (method & Method::kInt4ActInt8Weight)               || 
                (method & Method::kInt4ActInt4Weight)               ||
                (method & Method::kTernaryActInt8Weight)            ||
                (method & Method::kTernaryActTernaryWeight)         ||
                (method & Method::kBinaryActInt8Weight)             ||
                (method & Method::kBinaryActBinaryWeight)           ||
                (method & Method::kBinaryActBinaryWeightXOR)        ||
                (method & Method::kInt3ActInt3Weight)               ||
                (method & Method::kULPPACK)                         ||
                (method & Method::kInt8ActInt8WeightBarrelShiftMul) ||
                (method & Method::kSelfDependentW4A4)
                ;
        }
        bool RequiresOutputUnpacking(Method method){
            return 
                (method & Method::kInt8ActInt8WeightBarrelShiftMul)
                ;
        }
        LowPrecision::PreprocessType    InputPreProcess(Method method){
            switch(method){
                case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                    return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::InputPreProcess();
                case LowPrecision::Method::kULPPACKW1A1:
                case LowPrecision::Method::kULPPACKW2A2:
                case LowPrecision::Method::kULPPACKW3A3:
                case LowPrecision::Method::kULPPACKW4A4:
                    return LowPrecision::FullyConnected::ULPPACK::InputPreProcess();
                case LowPrecision::Method::kInt4ActInt4Weight:
                    return LowPrecision::FullyConnected::Int4InputsInt4Weights::InputPreProcess();
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::InputPreProcess();
                default:
                    return LowPrecision::PreprocessType::Nothing;
            }
        }
        LowPrecision::PreprocessType    FilterPreProcess(Method method){
            switch(method){
                case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                    return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::FilterPreProcess();
                case LowPrecision::Method::kULPPACKW1A1:
                case LowPrecision::Method::kULPPACKW2A2:
                case LowPrecision::Method::kULPPACKW3A3:
                case LowPrecision::Method::kULPPACKW4A4:
                    return LowPrecision::FullyConnected::ULPPACK::FilterPreProcess();
                case LowPrecision::Method::kInt4ActInt4Weight:
                    return LowPrecision::FullyConnected::Int4InputsInt4Weights::FilterPreProcess();
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::FilterPreProcess();
                default:
                    return LowPrecision::PreprocessType::Nothing;
            }
        }
        LowPrecision::PreprocessType    OutputPreProcess(Method method){
            switch(method){
                case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                    return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::OutputPreProcess();
                case LowPrecision::Method::kULPPACKW1A1:
                case LowPrecision::Method::kULPPACKW2A2:
                case LowPrecision::Method::kULPPACKW3A3:
                case LowPrecision::Method::kULPPACKW4A4:
                    return LowPrecision::FullyConnected::ULPPACK::OutputPreProcess();
                case LowPrecision::Method::kInt4ActInt4Weight:
                    return LowPrecision::FullyConnected::Int4InputsInt4Weights::OutputPreProcess();
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::OutputPreProcess();
                default:
                    return LowPrecision::PreprocessType::Nothing;
            }
        }
        LowPrecision::PreprocessType    OutputPostProcess(Method method){
            switch(method){
                case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                    return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::OutputPostProcess();
                case LowPrecision::Method::kULPPACKW1A1:
                case LowPrecision::Method::kULPPACKW2A2:
                case LowPrecision::Method::kULPPACKW3A3:
                case LowPrecision::Method::kULPPACKW4A4:
                    return LowPrecision::FullyConnected::ULPPACK::OutputPostProcess();
                case LowPrecision::Method::kInt4ActInt4Weight:
                    return LowPrecision::FullyConnected::Int4InputsInt4Weights::OutputPostProcess();
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::OutputPostProcess();
                default:
                    return LowPrecision::PreprocessType::Nothing;
            }
        }
        LowPrecision::GEMMType          GEMMSupport(Method method){
            switch(method){
                case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                    return LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::GEMMSupport();
                case LowPrecision::Method::kULPPACKW1A1:
                case LowPrecision::Method::kULPPACKW2A2:
                case LowPrecision::Method::kULPPACKW3A3:
                case LowPrecision::Method::kULPPACKW4A4:
                    return LowPrecision::FullyConnected::ULPPACK::GEMMSupport();
                case LowPrecision::Method::kInt4ActInt4Weight:
                    return LowPrecision::FullyConnected::Int4InputsInt4Weights::GEMMSupport();
                case LowPrecision::Method::kSelfDependentW4A4:
                    return LowPrecision::FullyConnected::SelfDependent::GEMMSupport();
                default:
                    return LowPrecision::GEMMType::SupportsNothing;
            }
        }
        LowPrecision::SelfDependentType IsSelfDependent(Method method){
            switch(method){
                case LowPrecision::Method::kSelfDependentW4A4:
                    #if SelfDependent_Type == SelfDependent_Offset_Vector_Size
                    return LowPrecision::SelfDependentType::Int4SelfDependent16Offset;
                    #elif SelfDependent_Type == SelfDependent_Continious
                    return LowPrecision::SelfDependentType::Int4SelfDependent;
                    #endif
                default:
                    return LowPrecision::SelfDependentType::NotSelfDependent;
            }
        }
        bool NeedPadding(LowPrecision::Method method, LowPrecision::Shape shape){
            return GetPaddedShape(method, shape, true) != shape;
        }
        size_t CalcFlatSize(int* sizes, int num_dims){
            size_t size = 1;
            for (size_t i = 0; i < num_dims; i++)
                size *= sizes[i];
            return size;
        }
        int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape, Method method){
            switch (method)
            {
            case Method::kInt8Int4:
                return Int4::PaddingWeightsIfNeeded(weight, shape);

            case Method::kFloat32Binary:
            case Method::kInt8Binary:
                return Binary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt8Ternary:
                return Ternary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt8QuaTernary:
                return Quaternary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt4ActInt8Weight:
                return Int4InputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt4ActInt4Weight:
                return Int4InputsInt4Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kTernaryActInt8Weight:
                return TernaryInputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kBinaryActInt8Weight:
                return BinaryInputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            default:
                return nullptr;
            }
        }
        size_t TransformFilterShape(LowPrecision::Method method, int* shape, int n_dims){
            int least_dim_size = 16, reduction_coeff = 1;
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                least_dim_size  = 32;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kBinaryActInt8Weight){
                least_dim_size  = 128;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kInt8QuaTernary){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul){
                least_dim_size  = 8;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kSelfDependentW4A4){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }

            int least_row_size = 4;
            if (method & LowPrecision::Method::k8x8){
                least_row_size = 8;
            }
            
            shape[n_dims - 2] = (::ceil(shape[n_dims - 2] / ((float)least_dim_size)) * least_dim_size) / reduction_coeff;
            shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / ((float)least_row_size)) * least_row_size;
            return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
        }
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims){
            int least_dim_size = 16, reduction_coeff = 1;
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kBinaryActInt8Weight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                least_dim_size  = 32;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                least_dim_size  = 128;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8QuaTernary){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kSelfDependentW4A4){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            
            int least_row_size = 4;
            if (method & LowPrecision::Method::k8x8){
                least_row_size = 8;
                least_dim_size = 8;
            }

            shape[n_dims - 1] = (::ceil(shape[n_dims - 1] / ((float)least_dim_size)) * least_dim_size) / reduction_coeff;
            bool is_multibatch = n_dims >= 2 && shape[n_dims - 2] > 1;
            if (is_multibatch && !(method & LowPrecision::Method::k8x8))
                shape[n_dims - 2] = ::ceil(shape[n_dims - 2] / ((float)least_row_size)) * least_row_size;
            else if(method & LowPrecision::Method::k8x8)
                shape[n_dims - 2] = ::ceil(shape[n_dims - 2] / 8.0) * 8;
            return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
        }
        Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
            int8_t* input_ptr = const_cast<int8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, k_shape, true, LowPrecision::MatrixType::Weight);
            bool need_padding = input_padded_shape != k_shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<int8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, k_shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method &  LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeFilter(input_ptr, input_padded_shape, output, layout, 1, 1);
            else if (method &  LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeFilter(input_ptr, input_padded_shape, output, layout, 2, 2);
            else if (method &  LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeFilter(input_ptr, input_padded_shape, output, layout, 3, 3);
            else if (method &  LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeFilter(input_ptr, input_padded_shape, output, layout, 4, 4);
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::QuantizeFilter(method, input_ptr, input_padded_shape, output, layout);
            
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout){
            uint8_t* input_ptr = const_cast<uint8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, k_shape, true, LowPrecision::MatrixType::Weight);
            bool need_padding = input_padded_shape != k_shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<uint8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, k_shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = Status::NotImplemented;
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method &  LowPrecision::Method::kULPPACK)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::QuantizeFilter(method, input_ptr, input_padded_shape, output, layout);
            
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){
            int8_t* input_ptr = const_cast<int8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, shape, true, LowPrecision::MatrixType::Input);
            bool need_padding = input_padded_shape != shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<int8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret = Status::NotSupported;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method &  LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeInput(input_ptr, input_padded_shape, output, layout, 1, 1);
            else if (method &  LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeInput(input_ptr, input_padded_shape, output, layout, 2, 2);
            else if (method &  LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeInput(input_ptr, input_padded_shape, output, layout, 3, 3);
            else if (method &  LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeInput(input_ptr, input_padded_shape, output, layout, 4, 4);
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::QuantizeInput(method, input_ptr, input_padded_shape, output, layout);
            
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout){
            uint8_t* input_ptr = const_cast<uint8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, shape, true, LowPrecision::MatrixType::Input);
            bool need_padding = input_padded_shape != shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<uint8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret = Status::NotSupported;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            if (method == LowPrecision::Method::kInt8Int4)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = Status::NotImplemented;
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method &  LowPrecision::Method::kULPPACK)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::QuantizeInput(method, input_ptr, input_padded_shape, output, layout);
            
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status UnpackOutput(LowPrecision::Method method, const int32_t* input, LowPrecision::Shape shape, int32_t* output){
            LowPrecision::Status ret = Status::NotSupported;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::UnpackOutput(input, shape, output);
            if (method == LowPrecision::Method::kInt8Int4)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = Status::NotImplemented;
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = Status::NotNeeded;
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method &  LowPrecision::Method::kULPPACK)
                ret = Status::NotNeeded;
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = Status::NotNeeded;
            return ret;
        }
        Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params
        ){
            bool use_block_processing = GetVariableFromEnv( "UseBlockProcessing" ) == "TRUE";
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            if (is_multibatched)
                if (use_block_processing)
                    return (Status)(MultiplyInt8MultiBatchedBlockProcessing(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint64_t)Status::MultiMultiplyBlock));
                else
                    return (Status)(MultiplyInt8MultiBatched(method, input, input_shape, kernel, kernel_shape, output, output_shape, params) | ((uint64_t)Status::MultiMultiply));
            else
                return (Status)(MultiplyInt8SingleBatch(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint64_t)Status::SingleMultiply));
        }
        Status Multiply(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params
        ){
            bool use_block_processing = GetVariableFromEnv( "UseBlockProcessing" ) == "TRUE";
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            if (is_multibatched)
                if (use_block_processing)
                    return (Status)(MultiplyInt8MultiBatchedBlockProcessing(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint64_t)Status::MultiMultiplyBlock));
                else
                    return (Status)(MultiplyInt8MultiBatched(method, input, input_shape, kernel, kernel_shape, output, output_shape, params) | ((uint64_t)Status::MultiMultiply));
            else
                return (Status)(MultiplyInt8SingleBatch(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint64_t)Status::SingleMultiply));
        }
        Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Binary)
                ret = LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    1, 1
                );
            else if (method == LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    2, 2
                );
            else if (method == LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    3, 3
                );
            else if (method == LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    4, 4
                );
            else if (method == LowPrecision::Method::kULPPACKW5A5)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    5, 5
                );
            else if (method == LowPrecision::Method::kULPPACKW6A6)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    6, 6
                );
            else if (method == LowPrecision::Method::kULPPACKW7A7)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    7, 7
                );
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::MultiplyInt8SingleBatch(
                    method,
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            return ret;
        }
        Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            // LowPrecision::Status ret;
            // if (method == LowPrecision::Method::kInt4ActInt8Weight)
            //     ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kInt4ActInt4Weight)
            //     ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kTernaryActInt8Weight)
            //     ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
            //     ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kBinaryActInt8Weight)
            //     ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
            //     ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
            //     ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kInt8Int4)
            //     ret = LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kInt8Binary)
            //     ret = LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kInt8Ternary)
            //     ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kInt8QuaTernary)
            //     ret = LowPrecision::FullyConnected::Quaternary::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW1A1)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         1, 1
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW2A2)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         2, 2
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW3A3)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         3, 3
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW4A4)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         4, 4
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW5A5)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         5, 5
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW6A6)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         6, 6
            //     );
            // else if (method == LowPrecision::Method::kULPPACKW7A7)
            //     ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
            //         input, input_shape,
            //         kernel, kernel_shape,
            //         output, output_shape,
            //         7, 7
            //     );
            return Status::NotImplemented;
        }
        Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params
        ){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    1, 1
                );
            else if (method == LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    2, 2
                );
            else if (method == LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    3, 3
                );
            else if (method == LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    4, 4
                );
            else if (method == LowPrecision::Method::kULPPACKW5A5)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    5, 5
                );
            else if (method == LowPrecision::Method::kULPPACKW6A6)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    6, 6
                );
            else if (method == LowPrecision::Method::kULPPACKW7A7)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    7, 7
                );
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kInt8Binary)
                ret = LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::MultiplyInt8MultiBatched(
                    method,
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            // else if (method == LowPrecision::Method::kBinaryActInt8Weight)
            //         ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8MultiBatched(
            //             input, input_shape,
            //             kernel, kernel_shape,
            //             output, output_shape
            //         );
            // else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
            //         ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched(
            //             input, input_shape,
            //             kernel, kernel_shape,
            //             output, output_shape
            //         );
            
            return ret;
        }
        Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params
        ){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul)
                ret = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW1A1)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW2A2)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW3A3)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW4A4)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW5A5)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW6A6)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW7A7)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Binary)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kSelfDependentW4A4)
                ret = LowPrecision::FullyConnected::SelfDependent::MultiplyInt8MultiBatched(
                    method,
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    params
                );
            return ret;
        }
        Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            int input_stride_coeff  = 1,
                kernel_stride_coeff = 1, 
                output_stride_coeff = 4,
                block_num_batches   = 16,
                block_num_columns   = 64,
                block_num_rows      = 64,
                num_batches         = input_shape.size[0],
                num_columns         = input_shape.size[1],
                num_rows            = kernel_shape.size[1]
            ;
            if (GetVariableFromEnv( "BlockBatches" ) != "")
                block_num_batches = std::stoi(GetVariableFromEnv( "BlockBatches" ));
            if (GetVariableFromEnv( "BlockColumns" ) != "")
                block_num_columns = std::stoi(GetVariableFromEnv( "BlockColumns" ));
            if (GetVariableFromEnv( "BlockRows" )    != "")
                block_num_rows    = std::stoi(GetVariableFromEnv( "BlockRows" ));
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 1;
                block_num_columns   = (block_num_columns%8)?(8):(block_num_columns);
                block_num_rows      = (block_num_rows%8)?(8):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt4ActInt8Weight){
                input_stride_coeff  = 2;
                kernel_stride_coeff = 1;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                input_stride_coeff  = 2;
                kernel_stride_coeff = 2;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                input_stride_coeff  = 4;
                kernel_stride_coeff = 1;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                input_stride_coeff  = 4;
                kernel_stride_coeff = 4;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 2;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 8;
                block_num_columns   = (block_num_columns%128)?(128):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 4;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kSelfDependentW4A4){
                input_stride_coeff  = 2;
                kernel_stride_coeff = 2;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else
                return Status::NotImplemented;
            Params current_params;
            Status kernel_return_status;
            current_params.lhs_stride = num_columns / input_stride_coeff;
            current_params.rhs_stride = num_columns / input_stride_coeff;
            current_params.dst_stride = num_rows    *         4         ;

            int8_t*  input_c  = const_cast<int8_t*> (input);
            int8_t*  kernel_c = const_cast<int8_t*> (kernel);
            int32_t* output_c = const_cast<int32_t*>(output);

            for (size_t i = 0; i < num_batches; i += block_num_batches){
                current_params.start_batches         =    i    * block_num_batches;
                current_params.end_batches           = (i + 1) * block_num_batches;

                for (size_t j = 0; j < num_rows; j += block_num_rows){
                    current_params.start_rows        =    j    * block_num_rows;
                    current_params.end_rows          = (j + 1) * block_num_rows;

                    for (size_t k = 0; k < num_columns; k += block_num_columns){
                        current_params.start_columns =    k    * block_num_columns;
                        current_params.end_columns   = (k + 1) * block_num_columns;

                        int8_t*  lhs = input_c  + (i * num_columns) + k;
                        int8_t*  rhs = kernel_c + (j * num_columns) + k;
                        int32_t* dst = output_c + (i * num_rows)    + j;
                        
                        switch (method)
                        {
                        case LowPrecision::Method::kInt8ActInt8WeightBarrelShiftMul:
                            kernel_return_status = LowPrecision::FullyConnected::Int8InputsInt8WeightsBarrelShiftMul::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt4ActInt8Weight:
                            kernel_return_status = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt4ActInt4Weight:
                            kernel_return_status = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kTernaryActInt8Weight:
                            kernel_return_status = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kTernaryActTernaryWeight:
                            kernel_return_status = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Int4:
                            kernel_return_status = LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Binary:
                            kernel_return_status = LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Ternary:
                            kernel_return_status = LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kSelfDependentW4A4:
                            kernel_return_status = LowPrecision::FullyConnected::SelfDependent::MultiplyInt8MultiBatchedBlock(
                                        method, lhs, rhs, dst, current_params
                            ); 
                            break;
                        default:
                            return Status::NotImplemented; 
                            break;
                        }
                        if ((LowPrecision::mask_out_source(kernel_return_status) != LowPrecision::Status::Success))
                            return kernel_return_status;
                    }
                }
            }
            return Status::Success;
        }
        Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            return Status::NotImplemented;
        }
        Shape GetPaddedShape(const LowPrecision::Method method, const Shape& input_shape, bool pad_rows_too, LowPrecision::MatrixType type){
            int least_row_size = 4;
            int least_dim_size = 16;
            if (method == LowPrecision::Method::kInt4ActInt8Weight)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kInt8Int4)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kInt8Binary)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kInt8Ternary)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                least_dim_size = 64;
            else if (method & LowPrecision::Method::k8x8){
                least_row_size = 8;
                least_dim_size = 8;
            }
            else if (method == LowPrecision::Method::kNoOptimization){
                least_row_size = 1;
                least_dim_size = 1;
            }
            if (input_shape.number_dims == 1){
                int padding_size = (input_shape.size[0] % least_dim_size)?(least_dim_size - (input_shape.size[0] % least_dim_size)):(0);
                Shape new_shape;
                new_shape.number_dims = input_shape.number_dims;
                new_shape.size = new int[new_shape.number_dims];
                new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_dim_size)) * least_dim_size;
                new_shape.flatsize = ::LowPrecision::FullyConnected::CalcFlatSize(new_shape.size, 1);
                return new_shape;
            }
            int padding_size = (input_shape.size[1] % least_dim_size)?(least_dim_size - (input_shape.size[1] % least_dim_size)):(0);
            Shape new_shape;
            new_shape.number_dims = input_shape.number_dims;
            new_shape.size = new int[new_shape.number_dims];
            if (type == LowPrecision::MatrixType::Weight){
                if (pad_rows_too)
                    new_shape.size[1] = ::ceil(input_shape.size[1] / ((float)least_row_size)) * least_row_size;
                else
                    new_shape.size[1] = input_shape.size[1];
                new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_dim_size)) * least_dim_size;
            } else if (type == LowPrecision::MatrixType::Output){
                if (pad_rows_too)
                    new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_row_size)) * least_row_size;
                else
                    new_shape.size[0] = input_shape.size[0];
                new_shape.size[1] = ::ceil(input_shape.size[1] / ((float)least_row_size)) * least_row_size;
            } else {
                if (pad_rows_too)
                    new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_row_size)) * least_row_size;
                else
                    new_shape.size[0] = input_shape.size[0];
                new_shape.size[1] = ::ceil(input_shape.size[1] / ((float)least_dim_size)) * least_dim_size;
            }
            new_shape.flatsize = ::LowPrecision::FullyConnected::CalcFlatSize(new_shape.size, 2);
            return new_shape;
        }
        Status TransformShapeToPaddedShape(const LowPrecision::Method method, int* input_sizes, int num_dims, bool pad_rows_too){
            Shape new_shape = LowPrecision::FullyConnected::GetPaddedShape(
                method, 
                LowPrecision::get_shape(input_sizes, num_dims), 
                pad_rows_too);
            for (int i = 0 ; i < num_dims ; i++)
                input_sizes[i] = new_shape.size[i];
            return LowPrecision::Status::Success;
        }
        template <typename Ti, typename To>
        Status PadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape, const To pad_value){
            if (from_shape.number_dims != to_shape.number_dims) return Status::DimensionsMisMatch;
            if (from_shape.number_dims <= 0) return Status::SizesMisMatch;
            int num_dims = from_shape.number_dims;
            if (num_dims == 2){ // We only accept matrix or vector.
                for (int j = 0; j < from_shape.size[0]; j++){
                    std::memcpy(&output[j * to_shape.size[1]], &input[j * from_shape.size[1]], from_shape.size[1]);
                    // std::copy(&input[j * from_shape.size[1]], &input[(j + 1) * from_shape.size[1]], &output[j * to_shape.size[1]]);
                    std::memset(&output[j * to_shape.size[1] + from_shape.size[1]], pad_value, to_shape.size[1] - from_shape.size[1]);
                    // for (int i = from_shape.size[1]; i < to_shape.size[1]; i++)
                    //     output[j * to_shape.size[1] + i] = pad_value;
                }
                std::memset(&output[from_shape.size[0] * to_shape.size[1]], pad_value, (to_shape.size[0] - from_shape.size[0]) * to_shape.size[1]);
                // for (int j = from_shape.size[0]; j < to_shape.size[0]; j++){
                //     for (int i = 0; i < to_shape.size[1]; i++)
                //         output[j * to_shape.size[1] + i] = pad_value;
                // }
            }
            else{
                std::memcpy(output, input, from_shape.size[0]);
                // std::copy(input, &input[from_shape.size[0]], output);
                std::memset(&output[from_shape.size[0]], pad_value, to_shape.size[0] - from_shape.size[0]);
                // for (int i = from_shape.size[0]; i < to_shape.size[0]; i++)
                //     output[i] = pad_value;
            }
            return Status::Success;
        }
        template<typename Ti, typename To>
        Status DePadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape){
            if (from_shape.number_dims < to_shape.number_dims) return Status::DimensionsMisMatch;
            if (from_shape.number_dims <= 0) return Status::SizesMisMatch;
            int num_dims = to_shape.number_dims;
            if (num_dims == 2){ // We only accept matrix or vector.
                for (int j = 0; j < to_shape.size[0]; j++){
                    std::memcpy(&output[j * to_shape.size[1]], &input[j * from_shape.size[1]], to_shape.size[1]);
                    // std::copy(&input[j * from_shape.size[1]], &input[j * from_shape.size[1] + to_shape.size[1]], &output[j * to_shape.size[1]]);
                }
            }
            else{
                std::memcpy(output, input, to_shape.size[0]);
                // std::copy(input, &input[to_shape.size[0]], output);
            }
            return Status::Success;
        }
        Status ApplyDowncast(int32_t* input, int8_t* output, Shape shape, const int32_t downcast_coeff){
#ifdef DOWNCASTING_FUSED_IN_KERNEL
            return Status::Success;
#else
#ifdef VECTORIZED_DOWNCASTING_WITH_SCALAR_DIVISION
            size_t size = shape.flatsize, i = 0;
            asm (

                "1:\n\t"

                "ld1 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[input]], #64\n\t"

                "mov w0,  v0.s[0]\n\t"
                "mov w1,  v0.s[1]\n\t"
                "mov w2,  v0.s[2]\n\t"
                "mov w3,  v0.s[3]\n\t"

                "mov w4,  v1.s[0]\n\t"
                "mov w5,  v1.s[1]\n\t"
                "mov w6,  v1.s[2]\n\t"
                "mov w7,  v1.s[3]\n\t"

                "mov w8,  v2.s[0]\n\t"
                "mov w9,  v2.s[1]\n\t"
                "mov w10, v2.s[2]\n\t"
                "mov w11, v2.s[3]\n\t"

                "mov w12, v3.s[0]\n\t"
                "mov w13, v3.s[1]\n\t"
                "mov w14, v3.s[2]\n\t"
                "mov w15, v3.s[3]\n\t"

                "sdiv w0,  w0,  %w[downcast_coeff]\n\t"
                "sdiv w1,  w1,  %w[downcast_coeff]\n\t"
                "sdiv w2,  w2,  %w[downcast_coeff]\n\t"
                "sdiv w3,  w3,  %w[downcast_coeff]\n\t"
                "sdiv w4,  w4,  %w[downcast_coeff]\n\t"
                "sdiv w5,  w5,  %w[downcast_coeff]\n\t"
                "sdiv w6,  w6,  %w[downcast_coeff]\n\t"
                "sdiv w7,  w7,  %w[downcast_coeff]\n\t"
                "sdiv w8,  w8,  %w[downcast_coeff]\n\t"
                "sdiv w9,  w9,  %w[downcast_coeff]\n\t"
                "sdiv w10, w10, %w[downcast_coeff]\n\t"
                "sdiv w11, w11, %w[downcast_coeff]\n\t"
                "sdiv w12, w12, %w[downcast_coeff]\n\t"
                "sdiv w13, w13, %w[downcast_coeff]\n\t"
                "sdiv w14, w14, %w[downcast_coeff]\n\t"
                "sdiv w15, w15, %w[downcast_coeff]\n\t"

                "mov v0.s[0], w0\n\t"
                "mov v0.s[1], w1\n\t"
                "mov v0.s[2], w2\n\t"
                "mov v0.s[3], w3\n\t"

                "mov v1.s[0], w4\n\t"
                "mov v1.s[1], w5\n\t"
                "mov v1.s[2], w6\n\t"
                "mov v1.s[3], w7\n\t"

                "mov v2.s[0], w8\n\t"
                "mov v2.s[1], w9\n\t"
                "mov v2.s[2], w10\n\t"
                "mov v2.s[3], w11\n\t"

                "mov v3.s[0], w12\n\t"
                "mov v3.s[1], w13\n\t"
                "mov v3.s[2], w14\n\t"
                "mov v3.s[3], w15\n\t"

                "sqxtn  v0.4h,  v0.4s\n\t"
                "sqxtn2 v0.8h,  v1.4s\n\t"

                "sqxtn  v2.4h,  v2.4s\n\t"
                "sqxtn2 v2.8h,  v3.4s\n\t"

                "sqxtn  v0.8b,  v0.8h\n\t"
                "sqxtn2 v0.16b, v2.8h\n\t"

                "st1 {v0.4s},  [%[output]], #16\n\t"

                "add %w[i], %w[i], #64\n\t"
                "cmp %w[i], %w[size]\n\t"
                "b.lt 1b\n\t"

                : [ input ] "+r" (input), [ output ]         "+r" (output)
                : [ size ]  "r" (size)  , [ downcast_coeff ] "r"  (downcast_coeff), 
                  [ i ]     "r" (i)
                : "v0",  "v1",  "v2",  "v3",
                  "w0" , "w1" , "w2" , "w3" ,
                  "w4" , "w5" , "w6" , "w7" ,
                  "w8" , "w8" , "w10", "w11",
                  "w12", "w13", "w14", "w15"
            );
            return Status::Success;
#else
            if (shape.number_dims == 1){
                for (int i = 0; i < shape.size[0]; i++)
                    output[i] = input[i] / downcast_coeff;
            }
            else if (shape.number_dims == 2){
                for (int j = 0; j < shape.size[0]; j++)
                    for (int i = 0; i < shape.size[1]; i++)
                        output[j * shape.size[1] + i] = input[j * shape.size[1] + i] / downcast_coeff;
            }
            else
                return Status::NotSupported;
            return Status::Success;
#endif
#endif
        }
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){
            for(int i = 0 ; i < batch_n ; i++)
                for(int j = 0 ; j < input_n ; j++)
                    output[i * input_n +  j] = input[i * input_n +  j] * scalling_factor[i];
            return;
        }
        
        Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing){
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid() && lhs.getData() == nullptr)
                return (Status)(((uint64_t)Status::LHSNotInitialized) | ((uint64_t)Status::MulAPI));
            if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid() && rhs.getData() == nullptr)
                return (Status)(((uint64_t)Status::RHSNotInitialized) | ((uint64_t)Status::MulAPI));
            if (dst.getNeedScratchpad() && !dst.isScratchpadValid() && dst.getData() == nullptr)
                return (Status)(((uint64_t)Status::DSTNotInitialized) | ((uint64_t)Status::MulAPI));
            if (dst.getNeedScratchpad() && dst.getNeedDowncast())
                return (Status)(((uint64_t)Status::NeedDowncastWScratch) | ((uint64_t)Status::MulAPI));
            if (lhs.getSignStatus() != rhs.getSignStatus())
                return (Status)(((uint64_t)Status::InputsSignsDifferent) | ((uint64_t)Status::MulAPI));
            if (!dst.getSignStatus())
                return (Status)(((uint64_t)Status::DSTCantBeUnsigned) | ((uint64_t)Status::MulAPI));

            bool sign_status = lhs.getSignStatus();
            bool need_downcast = dst.getNeedDowncast();
            LowPrecision::MulParams params;

            
            struct timespec tstart={0,0},
                            tend={0,0};
            if (timing != nullptr && timing->activated())
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid()){
                Status input_ret;
                if (sign_status)
                    input_ret = QuantizeInput(method, lhs.getData(), lhs.getShape(), lhs.getScratchpad(), lhs.getMemLayout());
                else
                    input_ret = QuantizeInput(method, LowPrecision::get_pointer_as<uint8_t>(lhs.getData()), lhs.getShape(), LowPrecision::get_pointer_as<uint8_t>(lhs.getScratchpad()), lhs.getMemLayout());
                if (input_ret == Status::NotNeeded)
                    lhs.setNeedScratchpad(false);
                else if (input_ret != Status::Success)
                    return (Status)(input_ret | ((uint64_t)Status::InputQuantizition));
                else
                    lhs.setScratchpadValid();
            }
            if (timing != nullptr && timing->activated()){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                timing->lhs = calculate_time_diff_seconds(tstart, tend);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
            }
            if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid()){
                Status input_ret;
                if (sign_status)
                    input_ret = QuantizeFilter(method, rhs.getData(), rhs.getShape(), rhs.getScratchpad(), rhs.getMemLayout());
                else
                    input_ret = QuantizeFilter(method, LowPrecision::get_pointer_as<uint8_t>(rhs.getData()), rhs.getShape(), LowPrecision::get_pointer_as<uint8_t>(rhs.getScratchpad()), rhs.getMemLayout());
                if (input_ret != Status::Success)
                    return (Status)(input_ret | ((uint64_t)Status::FilterQuantizition));
                rhs.setScratchpadValid();
            }
            if (timing != nullptr && timing->activated()){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                timing->rhs = calculate_time_diff_seconds(tstart, tend);
            }

            // std::cout << "[" << std::endl << std::hex;
            // for (int i = 0; i < lhs.getShape().size[0]; i++){
            //     std::cout << "\t[ ";
            //     for (int j = 0; j < lhs.getShape().size[1]; j++)
            //         std::cout << "0x" << (int)lhs.getData()[(i * lhs.getShape().size[1]) + j] << ", ";
            //     std::cout << "]" << std::endl;
            // }
            // std::cout << "]";
            // std::cout << std::dec << std::endl;
            //
            // std::cout << "[" << std::endl << std::hex;
            // for (int i = 0; i < lhs.getShape().size[0]; i++){
            //     std::cout << "\t[ ";
            //     for (int j = 0; j < lhs.getShape().size[1] / 4; j++)
            //         std::cout << "0x" << (int)(get_pointer_as<uint16_t>(lhs.getScratchpad()))[(i * lhs.getShape().size[1] / 4) + j] << ", ";
            //     std::cout << "]" << std::endl;
            // }
            // std::cout << "]";
            // std::cout << std::dec << std::endl;

            int8_t*  lhs_p =                                       (lhs.getNeedScratchpad())?(lhs.getScratchpad()):(lhs.getData()) ;
            int8_t*  rhs_p =                                       (rhs.getNeedScratchpad())?(rhs.getScratchpad()):(rhs.getData()) ;
            int32_t* dst_p = LowPrecision::get_pointer_as<int32_t>((dst.getNeedScratchpad())?(dst.getScratchpad()):(dst.getData()));

            int num_batches = 1;
            if (lhs.getShape().number_dims >= 2)
                num_batches = lhs.getShape().size[0];

            bool need_unpacking = RequiresOutputUnpacking(method);
            Shape unpacked_dst_shape;
            int32_t* dst_unpacked_p = nullptr;
            if (need_unpacking){
                dst_unpacked_p = LowPrecision::get_pointer_as<int32_t>(dst.getData());
                if (dst.getNeedScratchpad() && !(num_batches > 1 && num_batches % 8 && method & LowPrecision::Method::k8x8))
                    dst_p = LowPrecision::get_pointer_as<int32_t>(dst.getScratchpad());
                else if (num_batches > 1 && num_batches % 8 && method & LowPrecision::Method::k8x8){
                    unpacked_dst_shape = dst.getShape();
                    if (unpacked_dst_shape.number_dims < 2)
                        unpacked_dst_shape.extend_dims();
                    unpacked_dst_shape.size[unpacked_dst_shape.number_dims - 1] = ::ceil(unpacked_dst_shape.size[unpacked_dst_shape.number_dims - 1] / 8.0) * 8;
                    unpacked_dst_shape.size[unpacked_dst_shape.number_dims - 2] = ::ceil(unpacked_dst_shape.size[unpacked_dst_shape.number_dims - 2] / 8.0) * 8;
                    unpacked_dst_shape.flatsize = CalcFlatSize(unpacked_dst_shape.size, unpacked_dst_shape.number_dims);
                    dst_p = LowPrecision::allocate<int32_t>(unpacked_dst_shape.flatsize);
                }
                else
                    dst_p = LowPrecision::allocate<int32_t>(dst.getShape().flatsize);
            }

            LowPrecision::Status mul_ret_status;

            if (need_downcast)
                params.need_downcasting = need_downcast;

            if (num_batches > 1 && num_batches % 4 && !(method & LowPrecision::Method::k8x8)){
                int32_t* dst_p_backup = nullptr;
                Shape dst_shape, lhs_shape;
                dst_shape = dst.getShape();
                lhs_shape = lhs.getShape();
                
                dst_shape.size[0] = ::ceil(dst_shape.size[0] / 4.0) * 4;
                lhs_shape.size[0] = ::ceil(lhs_shape.size[0] / 4.0) * 4;
                dst_shape.flatsize = CalcFlatSize(dst_shape.size, dst_shape.number_dims);
                lhs_shape.flatsize = CalcFlatSize(lhs_shape.size, lhs_shape.number_dims);
                
                dst_p_backup = LowPrecision::allocate<int32_t>(dst_shape.flatsize);

                if (timing != nullptr && timing->activated())
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
                if (sign_status)
                    mul_ret_status = FullyConnected::Multiply(method, lhs_p, lhs_shape, rhs_p, rhs.getShape(), dst_p_backup, dst.getShape(), params);
                else
                    mul_ret_status = FullyConnected::Multiply(method, LowPrecision::get_pointer_as<uint8_t>(lhs_p), lhs_shape, LowPrecision::get_pointer_as<uint8_t>(rhs_p), rhs.getShape(), dst_p_backup, dst.getShape(), params);
                if (LowPrecision::mask_out_source(mul_ret_status) != Status::Success)
                    return mul_ret_status;
                if (timing != nullptr && timing->activated()){
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                    timing->multiplication = calculate_time_diff_seconds(tstart, tend);
                }

                Status depad_ret = DePadMatrixFromShapeToShape(dst_p_backup, dst_p, dst_shape, dst.getShape());
                if (depad_ret != Status::Success)
                    return (Status)(depad_ret | ((uint64_t)Status::DepadMatrix));
                LowPrecision::deallocate(dst_p_backup, false);
            }
            else if (num_batches % 8 && method & LowPrecision::Method::kULPPACK){
                int32_t* dst_p_backup = nullptr;
                Shape dst_shape, lhs_shape;
                dst_shape = dst.getShape();
                lhs_shape = lhs.getShape();
                if (dst_shape.number_dims < 2)
                    dst_shape.extend_dims();
                if (lhs_shape.number_dims < 2)
                    lhs_shape.extend_dims();
                
                dst_shape.size[dst_shape.number_dims - 2] = ::ceil(dst_shape.size[dst_shape.number_dims - 2] / 8.0) * 8;
                lhs_shape.size[lhs_shape.number_dims - 2] = ::ceil(lhs_shape.size[lhs_shape.number_dims - 2] / 8.0) * 8;
                dst_shape.flatsize = CalcFlatSize(dst_shape.size, dst_shape.number_dims);
                lhs_shape.flatsize = CalcFlatSize(lhs_shape.size, lhs_shape.number_dims);
                
                dst_p_backup = LowPrecision::allocate<int32_t>(dst_shape.flatsize);

                if (timing != nullptr && timing->activated())
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
                mul_ret_status = FullyConnected::Multiply(method, lhs_p, lhs_shape, rhs_p, rhs.getShape(), dst_p_backup, dst_shape, params);
                if (LowPrecision::mask_out_source(mul_ret_status) != Status::Success)
                    return mul_ret_status;
                if (timing != nullptr && timing->activated()){
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                    timing->multiplication = calculate_time_diff_seconds(tstart, tend);
                }

                Status depad_ret = DePadMatrixFromShapeToShape(dst_p_backup, dst_p, dst_shape, dst.getShape());
                if (depad_ret != Status::Success)
                    return (Status)(depad_ret | ((uint64_t)Status::DepadMatrix));
                LowPrecision::deallocate(dst_p_backup, false);
            }
            else if (num_batches > 1 && num_batches % 8 && method & LowPrecision::Method::k8x8){
                int32_t* dst_p_backup = nullptr;
                Shape dst_shape, lhs_shape, rhs_shape;
                dst_shape = dst.getShape();
                lhs_shape = lhs.getShape();
                rhs_shape = rhs.getShape();
                if (dst_shape.number_dims < 2)
                    dst_shape.extend_dims();
                if (lhs_shape.number_dims < 2)
                    lhs_shape.extend_dims();
                
                dst_shape.size[dst_shape.number_dims - 1] = ::ceil(dst_shape.size[dst_shape.number_dims - 1] / 8.0) * 8;
                dst_shape.size[dst_shape.number_dims - 2] = ::ceil(dst_shape.size[dst_shape.number_dims - 2] / 8.0) * 8;
                lhs_shape.size[lhs_shape.number_dims - 1] = ::ceil(lhs_shape.size[lhs_shape.number_dims - 1] / 8.0) * 8;
                lhs_shape.size[lhs_shape.number_dims - 2] = ::ceil(lhs_shape.size[lhs_shape.number_dims - 2] / 8.0) * 8;
                rhs_shape.size[rhs_shape.number_dims - 1] = ::ceil(rhs_shape.size[rhs_shape.number_dims - 1] / 8.0) * 8;
                rhs_shape.size[rhs_shape.number_dims - 2] = ::ceil(rhs_shape.size[rhs_shape.number_dims - 2] / 8.0) * 8;
                dst_shape.flatsize = CalcFlatSize(dst_shape.size, dst_shape.number_dims);
                lhs_shape.flatsize = CalcFlatSize(lhs_shape.size, lhs_shape.number_dims);
                rhs_shape.flatsize = CalcFlatSize(rhs_shape.size, rhs_shape.number_dims);
                if (!need_unpacking)
                    unpacked_dst_shape = dst.getShape();
                
                dst_p_backup = LowPrecision::allocate<int32_t>(dst_shape.flatsize);

                if (timing != nullptr && timing->activated())
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
                mul_ret_status = FullyConnected::Multiply(method, lhs_p, lhs_shape, rhs_p, rhs_shape, dst_p_backup, dst_shape, params);
                if (timing != nullptr && timing->activated()){
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                    timing->multiplication = calculate_time_diff_seconds(tstart, tend);
                }

                if (LowPrecision::mask_out_source(mul_ret_status) != Status::Success)
                    return mul_ret_status;
                Status depad_ret = DePadMatrixFromShapeToShape(dst_p_backup, dst_p, dst_shape, unpacked_dst_shape);
                if (depad_ret != Status::Success)
                    return (Status)(depad_ret | ((uint64_t)Status::DepadMatrix));
                LowPrecision::deallocate(dst_p_backup, false);
            } else {
                if (timing != nullptr && timing->activated())
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
                if (sign_status)
                    mul_ret_status = FullyConnected::Multiply(method, lhs_p, lhs.getShape(), rhs_p, rhs.getShape(), dst_p, dst.getShape(), params);
                else
                    mul_ret_status = FullyConnected::Multiply(method, get_pointer_as<uint8_t>(lhs_p), lhs.getShape(), get_pointer_as<uint8_t>(rhs_p), rhs.getShape(), dst_p, dst.getShape(), params);
                if (LowPrecision::mask_out_source(mul_ret_status) != Status::Success)
                    return mul_ret_status;
                if (timing != nullptr && timing->activated()){
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                    timing->multiplication = calculate_time_diff_seconds(tstart, tend);
                }
            }
            
            if (need_unpacking){
                if (timing != nullptr && timing->activated())
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
                UnpackOutput(method, dst_p, unpacked_dst_shape, dst_unpacked_p);
                if (timing != nullptr && timing->activated()){
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
                    timing->dst = calculate_time_diff_seconds(tstart, tend);
                }
                if (!dst.getNeedScratchpad() || (num_batches > 1 && num_batches % 8 && method & LowPrecision::Method::k8x8))
                    LowPrecision::deallocate(dst_p);
            }

            // if (need_downcast){
            //     Status downcast_ret = ApplyDowncast(dst_p, dst.getData(), dst.getShape(), dst.getDowncastCoeff());
            //     if (downcast_ret != Status::Success)
            //         return (Status)(downcast_ret | ((uint64_t)Status::ApplyDowncast));
            //     LowPrecision::deallocate(dst_p);
            // }

            return mul_ret_status;
        }
    }

    LowPrecision::PreprocessType    InputPreProcess(LowPrecision::Method method)   { return FullyConnected::InputPreProcess(method);   }
    LowPrecision::PreprocessType    FilterPreProcess(LowPrecision::Method method)  { return FullyConnected::FilterPreProcess(method);  }
    LowPrecision::PreprocessType    OutputPreProcess(LowPrecision::Method method)  { return FullyConnected::OutputPreProcess(method);  }
    LowPrecision::PreprocessType    OutputPostProcess(LowPrecision::Method method) { return FullyConnected::OutputPostProcess(method); }
    LowPrecision::GEMMType          GEMMSupport(LowPrecision::Method method)       { return FullyConnected::GEMMSupport(method); }
    LowPrecision::SelfDependentType IsSelfDependent(LowPrecision::Method method)   { return FullyConnected::IsSelfDependent(method); }

    Status MultiplyBackend(
        LowPrecision::Method method,
        const int8_t* input, LowPrecision::Shape input_shape,
        const int8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params
    ){ return FullyConnected::Multiply(method, input, input_shape, kernel, kernel_shape, output, output_shape, params); }
    Status MultiplyBackend(
        LowPrecision::Method method,
        const uint8_t* input, LowPrecision::Shape input_shape,
        const uint8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params
    ){ return FullyConnected::Multiply(method, input, input_shape, kernel, kernel_shape, output, output_shape, params); }

    Status PrepareMatrixAsFilterForMethod(Matrix& matrix, Method method, TimingDetailes* timing){
        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::FilterPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;

        bool contain_scratchpad         = matrix.getNeedScratchpad(),
             contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();

        bool isvalid_scratchpad         = matrix.isScratchpadValid(),
             isvalid_padding_scratchpad = matrix.isPaddedDataValid();

        Shape unpacked_shape, padded_shape;
        bool process_unsinged = !matrix.getSignStatus();
        int8_t* unpacked_data =  matrix.getData();
        unpacked_shape        =  matrix.getShape();
        padded_shape          =  FullyConnected::GetPaddedShape(method, matrix.getShape(), true, LowPrecision::MatrixType::Weight);

        struct timespec tstart = {0,0},
                        tend = {0,0};
        
        matrix.setPreparedShape(matrix.getShape());

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_padding && padded_shape != unpacked_shape){
            if (contain_padding_scratchpad){
                if (!isvalid_padding_scratchpad){
                    if (padded_shape != matrix.getShape()){
                        if (matrix.getPaddedData() == nullptr)
                            return (Status)(Status::NotAllocated | ((uint64_t)Status::PreparingFilter));
                        Status pad_ret;
                        if (process_unsinged)
                            pad_ret = FullyConnected::PadMatrixFromShapeToShape(
                                LowPrecision::get_pointer_as<uint8_t>(matrix.getData()), 
                                LowPrecision::get_pointer_as<uint8_t>(matrix.getPaddedData()), 
                                matrix.getShape(), 
                                padded_shape);
                        else
                            pad_ret = FullyConnected::PadMatrixFromShapeToShape(
                                matrix.getData(), 
                                matrix.getPaddedData(), 
                                matrix.getShape(), 
                                padded_shape);
                        unpacked_data = matrix.getPaddedData();
                        unpacked_shape = padded_shape;
                        if (pad_ret != Status::Success)
                            return (Status)(pad_ret | ((uint64_t)Status::PreparingFilter));
                        else
                            matrix.setPaddedDataValid();
                    }
                }
                matrix.setPreparedShape(unpacked_shape);
            } else {
                return (Status)(((uint64_t)Status::NeedPaddingScratchpad) | ((uint64_t)Status::PreparingFilter));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::RHSPadding);

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_packing){
            Shape packed_shape;
            packed_shape = unpacked_shape;
            if (contain_scratchpad){
                if (!isvalid_scratchpad){
                    packed_shape.flatsize = FullyConnected::TransformFilterShape(method, packed_shape.size, packed_shape.number_dims);
                    Status packing_ret;
                    if (process_unsinged)
                        packing_ret = FullyConnected::QuantizeFilter(
                            method, 
                            LowPrecision::get_pointer_as<uint8_t>(unpacked_data), 
                            unpacked_shape, 
                            LowPrecision::get_pointer_as<uint8_t>(matrix.getScratchpad()), 
                            matrix.getMemLayout());
                    else
                        packing_ret = FullyConnected::QuantizeFilter(
                            method, 
                            unpacked_data, 
                            unpacked_shape, 
                            matrix.getScratchpad(), 
                            matrix.getMemLayout());
                    if (packing_ret != Status::Success)
                        return (Status)(packing_ret | ((uint64_t)Status::PreparingFilter));
                    else
                        matrix.setScratchpadValid();
                }
            } else {
                return (Status)(((uint64_t)Status::NeedPackingScratchpad) | ((uint64_t)Status::PreparingFilter));
            }
            // matrix.setPreparedShape(packed_shape);
            matrix.setPreparedShape(unpacked_shape);
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::RHSPacking);

        return Status::Success;
    }
    Status PrepareMatrixAsInputForMethod(Matrix& matrix, Method method, TimingDetailes* timing){
        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::InputPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;

        bool contain_scratchpad         = matrix.getNeedScratchpad(),
             contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();

        bool isvalid_scratchpad         = matrix.isScratchpadValid(),
             isvalid_padding_scratchpad = matrix.isPaddedDataValid();
        
        bool using_single_scratchpad    = matrix.isUseSingleScratchpad();

        bool process_unsinged = !matrix.getSignStatus();
        int8_t* unpacked_data = matrix.getData();
        Shape unpacked_shape  = matrix.getShape();

        struct timespec tstart = {0,0},
                        tend = {0,0};
        
        matrix.setPreparedShape(matrix.getShape());
        Shape padded_shape = FullyConnected::GetPaddedShape(method, matrix.getShape(), true, LowPrecision::MatrixType::Input);

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_padding && padded_shape != unpacked_shape){
            if (contain_padding_scratchpad || using_single_scratchpad){
                if (!isvalid_padding_scratchpad){
                    if (padded_shape != matrix.getShape()){
                        if (using_single_scratchpad){
                            Shape scratchpad_shape;
                            scratchpad_shape = padded_shape;
                            scratchpad_shape.flatsize = FullyConnected::TransformInputShape(method, scratchpad_shape.size, scratchpad_shape.number_dims);
                            int8_t* padding_scratchpad;
                            if (process_unsinged)
                                padding_scratchpad = LowPrecision::get_pointer_as<int8_t>(LowPrecision::get_pointer_as<uint8_t>(matrix.getScratchpad()) + scratchpad_shape.flatsize);
                            else
                                padding_scratchpad = matrix.getScratchpad() + scratchpad_shape.flatsize;
                            matrix.setPaddingScratchpad(padding_scratchpad);
                            matrix.setPaddingScratchpadSetting();
                            contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();
                        }
                        Status pad_ret;
                        if (process_unsinged)
                            pad_ret = FullyConnected::PadMatrixFromShapeToShape(
                                LowPrecision::get_pointer_as<uint8_t>(matrix.getData()),
                                LowPrecision::get_pointer_as<uint8_t>(matrix.getPaddedData()), 
                                matrix.getShape(), 
                                padded_shape);
                        else
                            pad_ret = FullyConnected::PadMatrixFromShapeToShape(
                                matrix.getData(),
                                matrix.getPaddedData(), 
                                matrix.getShape(), 
                                padded_shape);
                        unpacked_data = matrix.getPaddedData();
                        unpacked_shape = padded_shape;
                        if (pad_ret != Status::Success)
                            return (Status)(pad_ret | ((uint64_t)Status::PreparingInput));
                        else
                            matrix.setPaddedDataValid();
                    }
                }
                matrix.setPreparedShape(unpacked_shape);
            } else {
                return (Status)(((uint64_t)Status::NeedPaddingScratchpad) | ((uint64_t)Status::PreparingInput));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::LHSPadding);

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_packing){
            Shape packed_shape;
            packed_shape = unpacked_shape;
            if (contain_scratchpad){
                if (!isvalid_scratchpad){
                    packed_shape.flatsize = FullyConnected::TransformInputShape(method, packed_shape.size, packed_shape.number_dims);
                    Status packing_ret;
                    if (process_unsinged)
                        packing_ret = FullyConnected::QuantizeInput(
                            method, 
                            LowPrecision::get_pointer_as<uint8_t>(unpacked_data), 
                            unpacked_shape, 
                            LowPrecision::get_pointer_as<uint8_t>(matrix.getScratchpad()), 
                            matrix.getMemLayout());
                    else
                        packing_ret = FullyConnected::QuantizeInput(
                            method, 
                            unpacked_data, 
                            unpacked_shape, 
                            matrix.getScratchpad(), 
                            matrix.getMemLayout());
                    if (packing_ret != Status::Success)
                        return (Status)(packing_ret | ((uint64_t)Status::PreparingInput));
                    else
                        matrix.setScratchpadValid();
                }
            } else {
                return (Status)(((uint64_t)Status::NeedPackingScratchpad) | ((uint64_t)Status::PreparingInput));
            }
            // matrix.setPreparedShape(packed_shape);
            matrix.setPreparedShape(unpacked_shape);
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::LHSPacking);

        return Status::Success;
    }
    Status PrepareMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing){
        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::OutputPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;

        bool contain_scratchpad         = matrix.getNeedScratchpad(),
             contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();
        
        bool using_single_scratchpad    = matrix.isUseSingleScratchpad();

        int8_t* unpacked_data = matrix.getData();
        Shape padded_shape = FullyConnected::GetPaddedShape(method, matrix.getShape(), true, LowPrecision::MatrixType::Output);

        struct timespec tstart = {0,0},
                        tend = {0,0};
        
        matrix.setPreparedShape(matrix.getShape());

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_padding && padded_shape != matrix.getShape()){
            if (contain_padding_scratchpad || using_single_scratchpad){
                if (using_single_scratchpad && padded_shape != matrix.getShape()){
                    Shape scratchpad_shape;
                    scratchpad_shape = padded_shape;
                    scratchpad_shape.flatsize = FullyConnected::TransformInputShape(method, scratchpad_shape.size, scratchpad_shape.number_dims);
                    if (matrix.getDataType() == LowPrecision::DataType::Int32)
                        matrix.setPaddingScratchpad(LowPrecision::get_pointer_as<int32_t>(matrix.getScratchpad()) + scratchpad_shape.flatsize);
                    else 
                        matrix.setPaddingScratchpad(matrix.getScratchpad() + scratchpad_shape.flatsize);
                    matrix.setPaddingScratchpadSetting();
                    contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();
                }
                matrix.setPreparedShape(padded_shape);
            } else {
                return (Status)(((uint64_t)Status::NeedPaddingScratchpad) | ((uint64_t)Status::PreparingOutput));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::DSTPadding);

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_packing){
            if (!contain_scratchpad){
                return (Status)(((uint64_t)Status::NeedPackingScratchpad) | ((uint64_t)Status::PreparingOutput));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::DSTPacking);
        
        return Status::Success;
    }
    Status PostprocessMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing){
        LowPrecision::PreprocessType method_required_postprocess
                                        = LowPrecision::FullyConnected::OutputPostProcess(method);
        bool requires_packing           = method_required_postprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_postprocess & LowPrecision::PreprocessType::PaddingIfNeccessery,
             requires_downcasting       = matrix.getNeedDowncast();

        bool contain_scratchpad         = matrix.getNeedScratchpad(),
             contain_padding_scratchpad = matrix.getPaddingScratchpadSetting();

        int8_t* final_data      = matrix.getData();
        int8_t* unpacked_data   = nullptr;

        if (requires_padding && contain_padding_scratchpad)
            unpacked_data = matrix.getPaddedData();
        else 
            unpacked_data = matrix.getData();

        struct timespec tstart = {0,0},
                        tend = {0,0};

        Shape padded_shape   = matrix.getFinalShape();
        Shape original_shape = matrix.getShape();
        
        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_packing){
            if (contain_scratchpad){
                LowPrecision::Status unpacking_ret;
                unpacking_ret = LowPrecision::FullyConnected::UnpackOutput(method, get_pointer_as<int32_t>(matrix.getScratchpad()), padded_shape, get_pointer_as<int32_t>(unpacked_data));
                if (LowPrecision::mask_out_source(unpacking_ret) != LowPrecision::Status::Success)
                    return unpacking_ret;
            } else {
                return (Status)(((uint64_t) Status::NeedPackingScratchpad) | ((uint64_t) Status::PostprocessingOutput));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::DSTUnPacking);

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (requires_padding && padded_shape != original_shape){
            if (contain_padding_scratchpad){
                LowPrecision::Status unpadding_ret;
                if (requires_downcasting)
                    unpadding_ret = LowPrecision::FullyConnected::DePadMatrixFromShapeToShape(
                                                            get_pointer_as<int32_t>(matrix.getPaddedData()),
                                                            get_pointer_as<int8_t>(matrix.getData()),
                                                            padded_shape, original_shape
                                    );
                else
                    unpadding_ret = LowPrecision::FullyConnected::DePadMatrixFromShapeToShape(
                                                            get_pointer_as<int32_t>(matrix.getPaddedData()),
                                                            get_pointer_as<int32_t>(matrix.getData()),
                                                            padded_shape, original_shape
                                    );
                if (LowPrecision::mask_out_source(unpadding_ret) != LowPrecision::Status::Success)
                    return unpadding_ret;
            } else {
                return (Status)(((uint64_t)Status::NeedPaddingScratchpad) | ((uint64_t)Status::PreparingOutput));
            }
        }
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::DSTUnPadding);

        return Status::Success;
    }

    LowPrecision::ShapeList GetInputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape){
        ShapeList list;

        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::InputPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
                requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;
        Shape padded_shape = FullyConnected::GetPaddedShape(method, base_shape, true, LowPrecision::MatrixType::Input);
        if (padded_shape != base_shape && requires_padding)
            list.push_back(padded_shape);
        
        Shape packed_shape;
        packed_shape = padded_shape;
        packed_shape.flatsize = LowPrecision::FullyConnected::TransformInputShape(method, packed_shape.size, packed_shape.number_dims);

        if (requires_packing)
            list.push_back(packed_shape);

        return list;
    }
    LowPrecision::ShapeList GetFilterShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape){
        ShapeList list;

        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::FilterPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;
        Shape padded_shape = FullyConnected::GetPaddedShape(method, base_shape, true, LowPrecision::MatrixType::Weight);
        if (padded_shape != base_shape && requires_padding)
            list.push_back(padded_shape);
        
        Shape packed_shape;
        packed_shape = padded_shape;
        LowPrecision::FullyConnected::TransformFilterShape(method, packed_shape.size, packed_shape.number_dims);
        packed_shape.flatsize = LowPrecision::FullyConnected::CalcFlatSize(packed_shape.size, packed_shape.number_dims);

        if (requires_packing)
            list.push_back(packed_shape);

        return list;
    }
    LowPrecision::ShapeList GetOutputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, LowPrecision::Shape output_shape){
        ShapeList list;

        LowPrecision::PreprocessType method_required_preprocess
                                        = LowPrecision::FullyConnected::OutputPreProcess(method);
        bool requires_packing           = method_required_preprocess & LowPrecision::PreprocessType::Packing,
             requires_padding           = method_required_preprocess & LowPrecision::PreprocessType::PaddingIfNeccessery;
        Shape input_padded_shape        = FullyConnected::GetPaddedShape(method, input_shape, true, LowPrecision::MatrixType::Input),
              filter_padded_shape       = FullyConnected::GetPaddedShape(method, filter_shape, true, LowPrecision::MatrixType::Weight);
        bool input_is_padded  = input_padded_shape != input_shape,
             filter_is_padded = filter_padded_shape != filter_shape,
             output_need_unpadding = input_is_padded || filter_is_padded;

        Shape output_padded_shape;
        output_padded_shape = input_padded_shape;
        output_padded_shape.size[output_padded_shape.number_dims - 1] = filter_padded_shape.size[filter_padded_shape.number_dims - 1];

        if (output_need_unpadding && output_padded_shape != output_shape){
            assert(requires_padding == output_need_unpadding);
            list.push_back(output_padded_shape);
        }
        if (requires_packing)
            list.push_back(output_padded_shape);

        return list;
    }

    LowPrecision::Status GEMM(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing){
        // Check if LHS matrix is processed and ready.
        if (lhs.getPaddingScratchpadSetting() && !lhs.isPaddedDataValid())
            return (Status)(((uint64_t)Status::LHSNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid())
            return (Status)(((uint64_t)Status::LHSNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (lhs.getData() == nullptr)
            return (Status)(((uint64_t)Status::LHSNotInitialized) | ((uint64_t)Status::GEMMAPI));
        else if (!Shape::Validate(lhs.getFinalShape()))
            return (Status)(((uint64_t)Status::LHSFinalShapeNotValid) | ((uint64_t)Status::GEMMAPI));

        // Check if RHS matrix is processed and ready.
        if (rhs.getPaddingScratchpadSetting() && !rhs.isPaddedDataValid())
            return (Status)(((uint64_t)Status::RHSNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid())
            return (Status)(((uint64_t)Status::RHSNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (rhs.getData() == nullptr)
            return (Status)(((uint64_t)Status::RHSNotInitialized) | ((uint64_t)Status::GEMMAPI));
        else if (!Shape::Validate(rhs.getFinalShape()))
            return (Status)(((uint64_t)Status::RHSFinalShapeNotValid) | ((uint64_t)Status::GEMMAPI));

        // Check if DST matrix is processed and ready.
        if (dst.getPaddingScratchpadSetting() && dst.getPaddedData() == nullptr)
            return (Status)(((uint64_t)Status::DSTNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (dst.getNeedScratchpad() && dst.getScratchpad() == nullptr)
            return (Status)(((uint64_t)Status::DSTNotReady) | ((uint64_t)Status::GEMMAPI));
        else if (dst.getData() == nullptr)
            return (Status)(((uint64_t)Status::DSTNotInitialized) | ((uint64_t)Status::GEMMAPI));
        else if (!Shape::Validate(dst.getFinalShape()))
            return (Status)(((uint64_t)Status::DSTFinalShapeNotValid) | ((uint64_t)Status::GEMMAPI));
        
        // Ensure that LHS and RHS matrix signes are the same
        if (lhs.getSignStatus() != rhs.getSignStatus())
            return (Status)(((uint64_t)Status::InputsSignsDifferent) | ((uint64_t)Status::GEMMAPI));
        // Ensure that DST matrix is singed
        if (!dst.getSignStatus())
            return (Status)(((uint64_t)Status::DSTCantBeUnsigned) | ((uint64_t)Status::GEMMAPI));
        
        bool process_unsigned = !lhs.getSignStatus();

        int8_t*  lhs_pointer = nullptr;
        int8_t*  rhs_pointer = nullptr;
        int32_t* dst_pointer = nullptr;

        if (lhs.getNeedScratchpad())
            lhs_pointer = lhs.getScratchpad();
        else if (lhs.getPaddingScratchpadSetting())
            lhs_pointer = lhs.getPaddedData();
        else
            lhs_pointer = lhs.getData();

        if (rhs.getNeedScratchpad())
            rhs_pointer = rhs.getScratchpad();
        else if (rhs.getPaddingScratchpadSetting())
            rhs_pointer = rhs.getPaddedData();
        else
            rhs_pointer = rhs.getData();

        if (dst.getNeedScratchpad())
            dst_pointer = get_pointer_as<int32_t>(dst.getScratchpad());
        else if (dst.getPaddingScratchpadSetting())
            dst_pointer = get_pointer_as<int32_t>(dst.getPaddedData());
        else
            dst_pointer = get_pointer_as<int32_t>(dst.getData());
        
        Shape lhs_final_shape, rhs_final_shape, dst_final_shape;

        lhs_final_shape = lhs.getFinalShape();
        rhs_final_shape = rhs.getFinalShape();
        dst_final_shape = dst.getFinalShape();

        LowPrecision::MulParams params;
        LowPrecision::Status mul_backend_ret;

        struct timespec tstart = {0,0},
                        tend = {0,0};

        TimingDetailes::SaveTimestamp(timing, tstart);
        if (process_unsigned)
            mul_backend_ret = MultiplyBackend(
                method, 
                LowPrecision::get_pointer_as<uint8_t>(lhs_pointer), lhs_final_shape, 
                LowPrecision::get_pointer_as<uint8_t>(rhs_pointer), rhs_final_shape, 
                dst_pointer, dst_final_shape,
                params);
        else
            mul_backend_ret = MultiplyBackend(
                method, 
                lhs_pointer, lhs_final_shape, 
                rhs_pointer, rhs_final_shape, 
                dst_pointer, dst_final_shape, 
                params);
        TimingDetailes::SaveTimestamp(timing, tend);

        TimingDetailes::SaveDifference(timing, tstart, tend, TimingDetailes::TimingElement::GEMM);

        if (LowPrecision::mask_out_source(mul_backend_ret) != LowPrecision::Status::Success)
            return mul_backend_ret;
        
        LowPrecision::Status postprocess_ret;
        postprocess_ret = LowPrecision::PostprocessMatrixAsOutputForMethod(dst, method, timing);

        if (LowPrecision::mask_out_source(postprocess_ret) != LowPrecision::Status::Success)
            return postprocess_ret;

        return mul_backend_ret;
    }

    void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                        int batch_n, int input_n){
        FullyConnected::doScallingFactorMultiplication(input, scalling_factor, output, batch_n, input_n);
    }

}
#endif