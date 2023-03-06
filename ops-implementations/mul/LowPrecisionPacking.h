#ifdef IS_ARM
#include <arm_neon.h>
#endif
#include <iostream>

//////////////////////////////////////
///////////     Int8         /////////
//////////////////////////////////////

void doLowPrecisionPack(const int8_t* src, int8_t* packed, int rows, int columns);
void doLowPrecisionWeightPackImpl(int8_t* src_ptr_1, int8_t* src_ptr_2,
                            int8_t* src_ptr_3, int8_t* src_ptr_4,
                            int8_t* dst_ptr_r, int size);
void doLowPrecisionWeightPackImpl(int8_t* src_ptr_1, int8_t* src_ptr_2,
                            int8_t* src_ptr_3, int8_t* src_ptr_4,
                            int8_t* dst_ptr_r, int size, int rows);

void doLowPrecisionWeightPack(int8_t* src, int8_t* packed, 
                        int rows, int columns);

void doLowPrecision2BatchInputPack(
                        int8_t* src, int8_t* packed, 
                        int rows, int columns);

//////////////////////////////////////
/////////       Float32       ////////
//////////////////////////////////////

void doLowPrecisionPack(const int32_t* src, int32_t* packed, int rows, int columns);
void doLowPrecisionWeightPackImpl(int32_t* src_ptr_1, int32_t* src_ptr_2,
                            int32_t* src_ptr_3, int32_t* src_ptr_4,
                            int32_t* dst_ptr_r, int size);

void doLowPrecisionWeightPack(int32_t* src, int32_t* packed, 
                        int rows, int columns);

//////////////////////////////////////
/////////       Float16       ////////
//////////////////////////////////////

void doLowPrecisionPack(const int16_t* src, int16_t* packed, int rows, int columns);
void doLowPrecisionWeightPackImpl(int16_t* src_ptr_1, int16_t* src_ptr_2,
                            int16_t* src_ptr_3, int16_t* src_ptr_4,
                            int16_t* dst_ptr_r,   int size);

void doLowPrecisionWeightPack(int16_t* src, int16_t* packed, 
                        int rows, int columns);