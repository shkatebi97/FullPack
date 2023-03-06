#ifndef LOW_PRECISION_FULLY_CONNECTED_H_
#define LOW_PRECISION_FULLY_CONNECTED_H_
#include "common/types.h"
// #include "common/cvector.hpp"
#include "ops-implementations/mul/LowPrecisionPacking.h"
#include "common/flags.h"
#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cstring>

#ifdef IS_ARM
#include <arm_neon.h>
#endif

#ifdef IS_ARM
// #define PRINT_VALUES true
// #define PRINT_VALUES_DETAILED false
namespace LowPrecision {
    class Matrix {
        bool                     _data_is_int32          = false; 

        bool                     _use_single_scratchpad  = false;

        bool                     _delete_data            = false;
        bool                     _delete_scratchpad      = false;
        bool                     _delete_padded_data     = false;

        int8_t*                  _data                   = nullptr;
        int8_t*                  _scratchpad             = nullptr;
        int8_t*                  _padded_data            = nullptr;

        bool                     _data_is_in_scratchpad  = false;
        bool                     _need_scratchpad        = false;

        bool                     _padded_data_is_valid   = false;
        bool                     _padding_scrathpad_set  = false;

        LowPrecision::MemLayout  _mem_layout             = LowPrecision::MemLayout::kColumnMajor;
        LowPrecision::Shape      _shape;
        LowPrecision::Shape      _prepared_shape;
        LowPrecision::MatrixType _type;
        bool                     _need_downcast          = false;
        int32_t                  _downcast_coeff         = 1;
        bool                     _process_unsigned       = false;
    public:
        Matrix(
            LowPrecision::MatrixType type = LowPrecision::MatrixType::Unknown,
            LowPrecision::MemLayout memLayout = LowPrecision::MemLayout::kColumnMajor
        ){ _mem_layout = memLayout; _type = type; }
        Matrix(const Matrix& var) {
            this->_data                     = var._data;
            this->_scratchpad               = var._scratchpad;
            this->_padded_data              = var._padded_data;
            
            this->_data_is_in_scratchpad    = var._data_is_in_scratchpad;
            this->_need_scratchpad          = var._need_scratchpad;

            this->_padded_data_is_valid     = var._padded_data_is_valid;
            this->_padding_scrathpad_set    = var._padding_scrathpad_set;

            this->_mem_layout               = var._mem_layout;
            this->_shape                    = var._shape;
            this->_type                     = var._type;
        }
        ~Matrix(){
            if(_delete_data)
                LowPrecision::deallocate(_data);
            if(_delete_scratchpad)
                LowPrecision::deallocate(_scratchpad);
            if(_delete_padded_data)
                LowPrecision::deallocate(_padded_data);
        }
        
        bool isUseSingleScratchpad()                                    { return _use_single_scratchpad; }

        bool isScratchpadValid()                                        { return _data_is_in_scratchpad; }
        bool getNeedScratchpad()                                        { return _need_scratchpad; }

        bool isPaddedDataValid()                                        { return _padded_data_is_valid; }
        bool getPaddingScratchpadSetting()                              { return _padding_scrathpad_set; }

        bool getNeedDowncast()                                          { return _need_downcast; }

        int8_t* getData()                                               { return _data; }
        int8_t* getScratchpad()                                         { return _scratchpad; }
        int8_t* getPaddedData()                                         { return _padded_data; }

        int32_t getDowncastCoeff()                                      { return _downcast_coeff; }
        LowPrecision::Shape getShape()                                  { return _shape; }
        LowPrecision::Shape getPreparedShape()                          { return _prepared_shape; }
        LowPrecision::Shape getFinalShape()                             { return _prepared_shape; }
        LowPrecision::MemLayout getMemLayout()                          { return _mem_layout; }
        LowPrecision::MatrixType getMatrixType()                        { return _type; }
        LowPrecision::DataType getDataType()                            { return ((_data_is_int32)?(LowPrecision::DataType::Int32):(LowPrecision::DataType::Int8)); }
        bool getSignStatus()                                            { return !_process_unsigned; }
    
        void useSingleScratchpad(bool enable = true)                    { _use_single_scratchpad = enable; }

        void setScratchpadValid(bool enable_scratchpad = true)          { _data_is_in_scratchpad= enable_scratchpad; }
        void setNeedScratchpad(bool need_scratchpad = true)             { _need_scratchpad      = need_scratchpad; }

        void setPaddedDataValid(bool enable_padded_data = true)         { _padded_data_is_valid = enable_padded_data; }
        void setPaddingScratchpadSetting(bool set_padding_sp = true)    { _padding_scrathpad_set= set_padding_sp; }

        void setData(int8_t* data)                                      { _data                 = data; }
        void setScratchpad(int8_t* data)                                { _scratchpad           = data; }
        void setPaddingScratchpad(int8_t* data)                         { _padded_data          = data; }

        void setData(int32_t* data)                                     { _data                 = LowPrecision::get_pointer_as<int8_t>(data); _data_is_int32 = true; }
        void setScratchpad(int32_t* data)                               { _scratchpad           = LowPrecision::get_pointer_as<int8_t>(data); _data_is_int32 = true; }
        void setPaddingScratchpad(int32_t* data)                        { _padded_data          = LowPrecision::get_pointer_as<int8_t>(data); _data_is_int32 = true; }

        void setDataAndScratchpad(int8_t* data, int8_t* scratchpad)     { _data                 = data;
                                                                          _scratchpad           = scratchpad; }
        void setDataAndScratchpad(int32_t* data, int32_t* scratchpad)   { _data                 = LowPrecision::get_pointer_as<int8_t>(data);
                                                                          _scratchpad           = LowPrecision::get_pointer_as<int8_t>(scratchpad);
                                                                          _data_is_int32        = true; }

        void setDataAndScratchpadAndShape(const int8_t* data, const int8_t* scratchpad, LowPrecision::Shape shape)
                                                                        { _data                 = const_cast<int8_t*>(data);
                                                                          _scratchpad           = const_cast<int8_t*>(scratchpad);
                                                                          _shape                = shape; }
        void setDataAndScratchpadAndShape(const int32_t* data, const int32_t* scratchpad, LowPrecision::Shape shape)
                                                                        { _data                 = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(data));
                                                                          _scratchpad           = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(scratchpad));
                                                                          _shape                = shape;
                                                                          _data_is_int32        = true; }

        void setDataAndPaddingAndScratchpadAndShape(const int8_t* data, const int8_t* scratchpad, const int8_t* padded_data, LowPrecision::Shape shape)
                                                                        { _data                 = const_cast<int8_t*>(data);
                                                                          _scratchpad           = const_cast<int8_t*>(scratchpad);
                                                                          _padded_data          = const_cast<int8_t*>(padded_data);
                                                                          _shape                = shape; }
        void setDataAndPaddingAndScratchpadAndShape(const int32_t* data, const int32_t* scratchpad, const int32_t* padded_data, LowPrecision::Shape shape)
                                                                        { _data                 = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(data));
                                                                          _scratchpad           = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(scratchpad));
                                                                          _padded_data          = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(padded_data));
                                                                          _shape                = shape;
                                                                          _data_is_int32        = true; }

        void setShape(LowPrecision::Shape shape)                        { _shape                = shape; }
        void setPreparedShape(LowPrecision::Shape shape)                { _prepared_shape       = shape; }
        void setMemLayout(LowPrecision::MemLayout mem_layout)           { _mem_layout           = mem_layout; }
        void setDowncastCoeff(int32_t downcast_coeff)                   { _downcast_coeff       = downcast_coeff; 
                                                                          _need_downcast        = true; }
        void setSignStatus(bool is_signed = true)                       { _process_unsigned     = !is_signed; }

        void setDataForDelete(bool _delete = true)                      { _delete_data          = _delete; }
        void setScratchpadForDelete(bool _delete = true)                { _delete_scratchpad    = _delete; }
        void setPaddedDataForDelete(bool _delete = true)                { _delete_padded_data   = _delete; }
    };
    class Params{
    public:
        int start_batches;
        int start_columns;
        int start_rows;

        int end_batches;
        int end_columns;
        int end_rows;

        int lhs_stride;
        int rhs_stride;
        int dst_stride;
        Params(){}
    };
    class MulParams{
    public:
        bool need_downcasting;
        MulParams(){
            need_downcasting = false;
        }
    };
    class TimingDetailes{
        bool _activated = false;
        long int _id = -1;
    public:
        typedef enum {
            Multiplication,
            LHS,
            RHS,
            DST,
            Packing,
            LHSPacking,
            RHSPacking,
            DSTPacking,
            DSTUnPacking,
            Padding,
            LHSPadding,
            RHSPadding,
            DSTPadding,
            DSTUnPadding,
            GEMM,
        } TimingElement;
        TimingDetailes():_activated(false){}
        TimingDetailes(const TimingDetailes&) = delete;
        inline bool activated(){ return _activated; }
        inline bool activate(bool activate = true){ _activated = activate; return _activated; }
        inline long int ID(long int id = -1){ _id = ((id >= 0)?(id):(_id)); return _id; }
        long double multiplication = 0;
        long double lhs = 0;
        long double rhs = 0;
        long double dst = 0;
        long double lhs_packing = 0;
        long double rhs_packing = 0;
        long double dst_packing = 0;
        long double dst_unpacking = 0;
        long double lhs_padding = 0;
        long double rhs_padding = 0;
        long double dst_padding = 0;
        long double dst_unpadding = 0;
        long double gemm = 0;
        inline double total() { return multiplication + lhs + rhs + dst + 
                                       lhs_packing + rhs_packing + dst_packing + 
                                       lhs_padding + rhs_padding + dst_padding + 
                                       dst_unpacking + dst_unpadding +
                                       gemm; }
        static inline void SaveTimestamp(TimingDetailes* timing, struct timespec& t){
            if (timing != nullptr && timing->activated())
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
        }
        static inline void SaveDifference(TimingDetailes* timing, struct timespec& ts, struct timespec& te, TimingElement element){
            if (timing != nullptr && timing->activated()){
                long double t = calculate_time_diff_seconds(ts, te);
                switch (element){
                case TimingDetailes::TimingElement::Multiplication:
                    timing->multiplication += t;
                    break;
                case TimingDetailes::TimingElement::LHS:
                    timing->lhs += t;
                    break;
                case TimingDetailes::TimingElement::RHS:
                    timing->rhs += t;
                    break;
                case TimingDetailes::TimingElement::DST:
                    timing->dst += t;
                    break;
                case TimingDetailes::TimingElement::LHSPacking:
                    timing->lhs_packing += t;
                    break;
                case TimingDetailes::TimingElement::RHSPacking:
                    timing->rhs_packing += t;
                    break;
                case TimingDetailes::TimingElement::DSTPacking:
                    timing->dst_packing += t;
                    break;
                case TimingDetailes::TimingElement::DSTUnPacking:
                    timing->dst_unpacking += t;
                    break;
                case TimingDetailes::TimingElement::LHSPadding:
                    timing->lhs_padding += t;
                    break;
                case TimingDetailes::TimingElement::RHSPadding:
                    timing->rhs_padding += t;
                    break;
                case TimingDetailes::TimingElement::DSTPadding:
                    timing->dst_padding += t;
                    break;
                case TimingDetailes::TimingElement::DSTUnPadding:
                    timing->dst_unpadding += t;
                    break;
                case TimingDetailes::TimingElement::GEMM:
                    timing->gemm += t;
                    break;
                default:
                    break;
                }
            }
        }
    };
    class TimingManager{
        std::vector<TimingDetailes*> _timings;
    public:
        TimingManager(){}
        TimingManager(const TimingManager&) = delete;
        ~TimingManager(){
            size_t count = _timings.size();
            double total = 0, gemm = 0, 
                   dst_unpacking = 0, dst_unpadding = 0,
                   lhs_packing   = 0, lhs_padding   = 0,
                   rhs_packing   = 0, rhs_padding   = 0,
                   dst_packing   = 0, dst_padding   = 0;
            for (TimingDetailes* timer : _timings){
                total         += timer->total() - timer->rhs_packing - timer->rhs_padding;
                gemm          += timer->gemm;
                dst_unpacking += timer->dst_unpacking;
                dst_unpadding += timer->dst_unpadding;
                dst_packing   += timer->dst_packing;
                dst_padding   += timer->dst_padding;
                lhs_packing   += timer->lhs_packing;
                lhs_padding   += timer->lhs_padding;
                rhs_packing   += timer->rhs_packing;
                rhs_padding   += timer->rhs_padding;
                // delete timer;
            }
            if (count > 0){
                std::cout << "Total GEMM API Timing   : " << total * 1000000 << std::endl;
                std::cout << "\t" << "GEMM            : " << gemm * 1000000 << std::endl;
                std::cout << "\t" << "Input  Packing  : " << lhs_packing * 1000000 << std::endl;
                std::cout << "\t" << "Filter Packing  : " << rhs_packing * 1000000 << std::endl;
                std::cout << "\t" << "Output Packing  : " << dst_packing * 1000000 << std::endl;
                std::cout << "\t" << "Output UnPacking: " << dst_unpacking * 1000000 << std::endl;
                std::cout << "\t" << "Input  Padding  : " << lhs_padding * 1000000 << std::endl;
                std::cout << "\t" << "Filter Padding  : " << rhs_padding * 1000000 << std::endl;
                std::cout << "\t" << "Output Padding  : " << dst_padding * 1000000 << std::endl;
                std::cout << "\t" << "Output UnPadding: " << dst_unpadding * 1000000 << std::endl;
            }
        }
        void addTimingDetail(TimingDetailes* timer){ _timings.push_back(timer); }
        size_t getCount(){ return _timings.size(); }
    };
    typedef std::vector<Shape> ShapeList;
    static TimingManager timingManager;
    namespace FullyConnected {
        static long int id = 0;
        LowPrecision::Method get_default_method();
        void set_default_method(LowPrecision::Method method);
        LowPrecision::Method GetMethodFromEnv();
        std::string GetVariableFromEnv(std::string variable);
        LowPrecision::DataType GetDataType(int type);
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type,
            LowPrecision::DataType output_type, bool Is_FC);
        bool IncludesActivationCompression(LowPrecision::Method method);
        bool RequiresOutputUnpacking(LowPrecision::Method method);
        LowPrecision::PreprocessType InputPreProcess(Method method);
        LowPrecision::PreprocessType FilterPreProcess(Method method);
        LowPrecision::PreprocessType OutputPreProcess(Method method);
        LowPrecision::PreprocessType OutputPostProcess(Method method);
        LowPrecision::GEMMType GEMMSupport(Method method);
        size_t CalcFlatSize(int* sizes, int num_dims);
        int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape, Method method);
        size_t TransformFilterShape(LowPrecision::Method method, int* shape, int n_dims);
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims);
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status UnpackOutput(LowPrecision::Method method, const int32_t* input, LowPrecision::Shape shape, int32_t* output);
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape,
            LowPrecision::MulParams params = LowPrecision::MulParams());
        LowPrecision::Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const uint8_t* input, LowPrecision::Shape input_shape,
            const uint8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        Shape GetPaddedShape(const LowPrecision::Method method, const Shape& input_shape, bool pad_rows_too = false, LowPrecision::MatrixType type = LowPrecision::MatrixType::Unknown);
        Status TransformShapeToPaddedShape(const LowPrecision::Method method, int* input_sizes, int num_dims, bool pad_rows_too = true);
        template <typename Ti, typename To>
        Status PadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape, const To pad_value = 0);
        template<typename Ti, typename To>
        Status DePadMatrixFromShapeToShape(const Ti* input, To* output, Shape from_shape, Shape to_shape);
        Status ApplyDowncast(int32_t* input, int8_t* output, Shape shape, const int32_t downcast_coeff);

        namespace Int8InputsInt8WeightsBarrelShiftMul {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            template <typename T>
            LowPrecision::Status QuantizeFilter(const T* input, LowPrecision::Shape k_shape, T* output, LowPrecision::MemLayout layout);
            template <typename T>
            LowPrecision::Status QuantizeInput(const T* input, LowPrecision::Shape shape, T* output, LowPrecision::MemLayout layout);
            LowPrecision::Status UnpackOutput(const int32_t* input, Shape shape, int32_t* output);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void unpack_8x8_block_barrelshift_mul(
                uint16x8_t& vACC_Ar76543210_x_Wc76543210,
                uint16x8_t& vACC_Ar76543210_x_Wc07654321,
                uint16x8_t& vACC_Ar76543210_x_Wc10765432,
                uint16x8_t& vACC_Ar76543210_x_Wc21076543,
                uint16x8_t& vACC_Ar76543210_x_Wc32107654,
                uint16x8_t& vACC_Ar76543210_x_Wc43210765,
                uint16x8_t& vACC_Ar76543210_x_Wc54321076,
                uint16x8_t& vACC_Ar76543210_x_Wc65432107
            );
            void unpack_8x8_block_barrelshift_mul(
                int16x8_t& vACC_Ar76543210_x_Wc76543210,
                int16x8_t& vACC_Ar76543210_x_Wc07654321,
                int16x8_t& vACC_Ar76543210_x_Wc10765432,
                int16x8_t& vACC_Ar76543210_x_Wc21076543,
                int16x8_t& vACC_Ar76543210_x_Wc32107654,
                int16x8_t& vACC_Ar76543210_x_Wc43210765,
                int16x8_t& vACC_Ar76543210_x_Wc54321076,
                int16x8_t& vACC_Ar76543210_x_Wc65432107
            );
            inline void unpack_8x8_block_barrelshift(const int32_t* O, int32_t* O_unpack, size_t offset);
            LowPrecision::PreprocessType InputPreProcess();
            LowPrecision::PreprocessType FilterPreProcess();
            LowPrecision::PreprocessType OutputPreProcess();
            LowPrecision::PreprocessType OutputPostProcess();
            LowPrecision::GEMMType GEMMSupport();
        }
        namespace Int4 {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            Status PaddingWeightsIfNeeded(const int8_t* input_weight, int8_t* output_weight, Shape shape);
            Status PaddingInputsIfNeeded(const int8_t* input, int8_t* output, Shape shape);
            Shape GetPaddedShape(const Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilterWithPadding(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInputWithPadding(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Binary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Ternary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Quaternary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int4InputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int4InputsInt4Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
            void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
            void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
        }
        namespace BinaryInputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace BinaryInputsBinaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace BinaryInputsBinaryWeightsXOR {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace TernaryInputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace TernaryInputsTernaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int3InputsInt3Weights {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int8InputsInt4PowerWeights {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape);
        }
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout, size_t Wb, size_t Ab);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout, size_t Wb, size_t Ab);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            LowPrecision::PreprocessType InputPreProcess();
            LowPrecision::PreprocessType FilterPreProcess();
            LowPrecision::PreprocessType OutputPreProcess();
            LowPrecision::PreprocessType OutputPostProcess();
            LowPrecision::GEMMType GEMMSupport();
        }

        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n);
        LowPrecision::Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing=nullptr);
    }
    
    LowPrecision::PreprocessType InputPreProcess(Method method);
    LowPrecision::PreprocessType FilterPreProcess(Method method);
    LowPrecision::PreprocessType OutputPreProcess(Method method);
    LowPrecision::PreprocessType OutputPostProcess(Method method);
    LowPrecision::GEMMType       GEMMSupport(Method method);

    LowPrecision::Status MultiplyBackend(
        LowPrecision::Method method,
        const int8_t* input, LowPrecision::Shape input_shape,
        const int8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params = LowPrecision::MulParams());
    LowPrecision::Status MultiplyBackend(
        LowPrecision::Method method,
        const uint8_t* input, LowPrecision::Shape input_shape,
        const uint8_t* kernel, LowPrecision::Shape kernel_shape,
        int32_t* output, LowPrecision::Shape output_shape,
        LowPrecision::MulParams params = LowPrecision::MulParams());

    LowPrecision::Status PrepareMatrixAsFilterForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PrepareMatrixAsInputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PrepareMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);
    LowPrecision::Status PostprocessMatrixAsOutputForMethod(Matrix& matrix, Method method, TimingDetailes* timing=nullptr);

    LowPrecision::ShapeList GetInputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape);
    LowPrecision::ShapeList GetFilterShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape base_shape);
    LowPrecision::ShapeList GetOutputShapeListForMethod(LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, LowPrecision::Shape output_shape);

    LowPrecision::Status GEMM(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method, TimingDetailes* timing=nullptr);

    void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                        int batch_n, int input_n);
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type,
            LowPrecision::DataType output_type, bool Is_FC) { return false; }
        LowPrecision::Method GetMethodFromEnv() { return Method::kNoOptimization; }
        std::string GetVariableFromEnv(std::string variable) { return std::string(); }
        LowPrecision::DataType GetDataType(int type) { return LowPrecision::DataType::NotAvailable; }
        LowPrecision::Status QuantizeFilterToInt4(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout) { return LowPrecision::Status::NotSupported; }
        LowPrecision::Status MultiplyInt8Int4(
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape) { return LowPrecision::Status::NotSupported; }
        void do4BitMultiplication(const int8_t* activation, 
                                int8_t* weights, 
                                int32_t& dst_1, int32_t& dst_2,
                                int32_t& dst_3, int32_t& dst_4, 
                                int size) {  }
        uint8_t quantizeTo4BitIntAndPackBitsStep(const int8_t& input, int shift_amount){ return 0; }
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){ }
    }
}
#endif
#endif