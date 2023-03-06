#include "LowPrecisionPacking.h"
#include "../../common/asmutility.h"
#include <iostream>

#ifdef IS_ARM
//////////////////////////////////////
///////////     Int8         /////////
//////////////////////////////////////

void doLowPrecisionPack(const int8_t* src, int8_t* packed, int rows, int columns){
    doLowPrecisionWeightPack(const_cast<int8_t*>(src), packed, rows, columns);
}

void doLowPrecisionWeightPack(int8_t* src, int8_t* packed, int rows, int columns){
    int i;

    int8_t *src_ptr_1 = src + 0 * columns;
    int8_t *src_ptr_2 = src + 1 * columns;
    int8_t *src_ptr_3 = src + 2 * columns;
    int8_t *src_ptr_4 = src + 3 * columns;
    int8_t *packed_ptr = packed;
    return doLowPrecisionWeightPackImpl(
        src_ptr_1, src_ptr_2, src_ptr_3, src_ptr_4, 
        packed_ptr, columns, rows 
    );
    for (i = 0 ; (i+4) <= rows ; i+=4){
        // std::cout << "Packing " 
        //           << ((void*)src_ptr_1) << ", " 
        //           << ((void*)src_ptr_2) << ", " 
        //           << ((void*)src_ptr_3) << ", " 
        //           << ((void*)src_ptr_4) << " - "
        //           << ((void*)packed_ptr) << " - "
        //           << columns << " " << rows << " - " 
        //           << i << std::endl;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
        src_ptr_1 +=  4 * columns;
        src_ptr_2 +=  4 * columns;
        src_ptr_3 +=  4 * columns;
        src_ptr_4 +=  4 * columns;
        packed_ptr += 4 * columns;
    }
    i = rows - (i - 4);
    if (i == 1){
        // std::cout << "Also executing pack for i == " << i << std::endl;
        src_ptr_2 = src_ptr_1;
        src_ptr_3 = src_ptr_1;
        src_ptr_4 = src_ptr_1;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 2){
        // std::cout << "Also executing pack for i == " << i << std::endl;
        src_ptr_3 = src_ptr_2;
        src_ptr_4 = src_ptr_2;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 3){
        // std::cout << "Also executing pack for i == " << i << std::endl;
        src_ptr_4 = src_ptr_3;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 4 && packed_ptr < packed + rows * columns){
        // std::cout << "Also executing pack for i == " << i << std::endl;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
}

void doLowPrecisionWeightPackImpl(int8_t* src_ptr_1, int8_t* src_ptr_2,
                            int8_t* src_ptr_3, int8_t* src_ptr_4,
                            int8_t* dst_ptr_r,   int size){
    int i;
    auto dst_ptr = dst_ptr_r;
    asm volatile(
        "mov %w[i], wzr\n"
        "mov x0, %[src_ptr_1]\n"
        "mov x1, %[src_ptr_2]\n"
        "mov x2, %[src_ptr_3]\n"
        "mov x3, %[src_ptr_4]\n"
        "mov x4, %[dst_ptr]\n"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "add %w[i], %w[i], #16\n"

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.16b}, [x0], #16\n"
        "ld1 {v2.16b}, [x1], #16\n"
        "ld1 {v3.16b}, [x2], #16\n"
        "ld1 {v4.16b}, [x3], #16\n"

        //////////////////////////////////////////////////////////////////

        "st1 {v1.16b},  [x4], #16\n"
        "st1 {v2.16b},  [x4], #16\n"
        "st1 {v3.16b},  [x4], #16\n"
        "st1 {v4.16b},  [x4], #16\n"

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        PACK_SIMPLE_ONE_DATA(0)
        PACK_SIMPLE_ONE_DATA(1)
        PACK_SIMPLE_ONE_DATA(2)
        PACK_SIMPLE_ONE_DATA(3)
        PACK_SIMPLE_ONE_DATA(4)
        PACK_SIMPLE_ONE_DATA(5)
        PACK_SIMPLE_ONE_DATA(6)
        PACK_SIMPLE_ONE_DATA(7)
        PACK_SIMPLE_ONE_DATA(8)
        PACK_SIMPLE_ONE_DATA(9)
        PACK_SIMPLE_ONE_DATA(10)
        PACK_SIMPLE_ONE_DATA(11)
        PACK_SIMPLE_ONE_DATA(12)
        PACK_SIMPLE_ONE_DATA(13)
        PACK_SIMPLE_ONE_DATA(14)
        PACK_SIMPLE_ONE_DATA(15)

        "5:\n"
        "7:\n"

        "sub x0, x0, %[size]\n"
        "sub x1, x1, %[size]\n"
        "sub x2, x2, %[size]\n"
        "sub x3, x3, %[size]\n"

        "sub x4, x4, %[size]\n"
        "sub x4, x4, %[size]\n"
        "sub x4, x4, %[size]\n"
        "sub x4, x4, %[size]\n"

        : [ dst_ptr ]   "+r"(dst_ptr),  [ i ]          "+r"(i)
        : [ src_ptr_1 ] "r" (src_ptr_1), [ src_ptr_2 ] "r" (src_ptr_2),
          [ src_ptr_3 ] "r" (src_ptr_3), [ src_ptr_4 ] "r" (src_ptr_4),
          [ size ]      "r" (size)
        : "v1", "v2", "v3", "v4", "x0", "x1", "x2", "x3", "x4"
    );
}

void doLowPrecisionWeightPackImpl(int8_t* src_ptr_1, int8_t* src_ptr_2,
                            int8_t* src_ptr_3, int8_t* src_ptr_4,
                            int8_t* dst_ptr_r, int size, int rows){
    int i, j;
    auto dst_ptr = dst_ptr_r;
    asm volatile(
        "mov %w[j], wzr\n\t"
        "mov x0, %[src_ptr_1]\n"
        "mov x1, %[src_ptr_2]\n"
        "mov x2, %[src_ptr_3]\n"
        "mov x3, %[src_ptr_4]\n"
        "mov x4, %[dst_ptr]\n"

        "0:\n\t"

        "cmp %w[size], #16\n"
        "blt 3f\n"

        "mov %w[i], wzr\n"
        "add %w[i], %w[i], #16\n"

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.16b}, [x0], #16\n"
        "ld1 {v2.16b}, [x1], #16\n"
        "ld1 {v3.16b}, [x2], #16\n"
        "ld1 {v4.16b}, [x3], #16\n"

        //////////////////////////////////////////////////////////////////

        "st1 {v1.16b},  [x4], #16\n"
        "st1 {v2.16b},  [x4], #16\n"
        "st1 {v3.16b},  [x4], #16\n"
        "st1 {v4.16b},  [x4], #16\n"

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #16\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        PACK_SIMPLE_ONE_DATA(0)
        PACK_SIMPLE_ONE_DATA(1)
        PACK_SIMPLE_ONE_DATA(2)
        PACK_SIMPLE_ONE_DATA(3)
        PACK_SIMPLE_ONE_DATA(4)
        PACK_SIMPLE_ONE_DATA(5)
        PACK_SIMPLE_ONE_DATA(6)
        PACK_SIMPLE_ONE_DATA(7)
        PACK_SIMPLE_ONE_DATA(8)
        PACK_SIMPLE_ONE_DATA(9)
        PACK_SIMPLE_ONE_DATA(10)
        PACK_SIMPLE_ONE_DATA(11)
        PACK_SIMPLE_ONE_DATA(12)
        PACK_SIMPLE_ONE_DATA(13)
        PACK_SIMPLE_ONE_DATA(14)
        PACK_SIMPLE_ONE_DATA(15)

        "5:\n"

        "mov x0, x3\n\t"
        "add x1, x0, %[size]\n\t"
        "add x2, x1, %[size]\n\t"
        "add x3, x2, %[size]\n\t"

        "add %w[j], %w[j], #4\n\t"
        "cmp %w[j], %w[rows]\n\t"
        "b.lt 0b\n\t"

        "7:\n"

        : [ dst_ptr ]   "+r"(dst_ptr),   [ i ]         "+r"(i),
          [ j ]         "+r"(j)
        : [ src_ptr_1 ] "r" (src_ptr_1), [ src_ptr_2 ] "r" (src_ptr_2),
          [ src_ptr_3 ] "r" (src_ptr_3), [ src_ptr_4 ] "r" (src_ptr_4),
          [ size ]      "r" (size),      [ rows ]      "r" (rows)
        : "v1", "v2", "v3", "v4", "x0", "x1", "x2", "x3", "x4"
    );
}

void doLowPrecision2BatchInputPack(
                        int8_t* src, int8_t* packed, 
                        int rows, int columns){
    int i, j, depth = rows * columns;
    int8_t *src_ptr_1 = src + 0 * columns;
    int8_t *src_ptr_2 = src + 1 * columns;
    int8_t *packed_ptr = packed;
    asm volatile(
        "mov %w[j], wzr\n"

        "cmp %w[size], #0\n"
        "blt 3f\n"
        "cmp %w[rows], #0\n"
        "blt 3f\n"

        "0:\n"
        "mov %w[i], wzr\n"

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.16b}, [%[src_ptr_1]], #16\n"
        "ld1 {v2.16b}, [%[src_ptr_2]], #16\n"

        //////////////////////////////////////////////////////////////////

        "st1 {v1.16b},  [%[packed_ptr]], #16\n"
        "st1 {v2.16b},  [%[packed_ptr]], #16\n"

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #16\n\t"
        "cmp %w[i], %w[size]\n\t"
        "b.lt 1b\n\t"

        "add %[src_ptr_1], %[src_ptr_2], %[size]\n"
        "add %[src_ptr_2], %[src_ptr_2], %[size]\n"
        "add %w[j], %w[j], #2\n\t"
        "cmp %w[j], %w[rows]\n\t"
        "b.lt 0b\n\t"

        "sub %[src_ptr_1],  %[src_ptr_1],  %[depth]\n"
        "sub %[src_ptr_2],  %[src_ptr_2],  %[depth]\n"

        "sub %[packed_ptr], %[packed_ptr], %[depth]\n"

        "3:\n"

        : [ packed_ptr ]    "+r"(packed_ptr),   [ i ]           "+r"(i),
          [ j ]             "+r"(j)
        : [ src_ptr_1 ]     "r"(src_ptr_1),     [ src_ptr_2 ]   "r"(src_ptr_2),
          [ size ]          "r"(columns),       [ rows ]        "r"(rows),
          [ depth ]         "r"(depth)
        : "v1", "v2", "v3", "v4"
    );
}

//////////////////////////////////////
/////////       Float32       ////////
//////////////////////////////////////

void doLowPrecisionPack(const int32_t* src, int32_t* packed, int rows, int columns){
    doLowPrecisionWeightPack(const_cast<int32_t*>(src), packed, rows, columns);
}

void doLowPrecisionWeightPack(int32_t* src, int32_t* packed, int rows, int columns){
    int i;

    int32_t *src_ptr_1 = src + 0 * columns;
    int32_t *src_ptr_2 = src + 1 * columns;
    int32_t *src_ptr_3 = src + 2 * columns;
    int32_t *src_ptr_4 = src + 3 * columns;

    int32_t *packed_ptr = packed;

    for (i = 0 ; (i+4) <= rows ; i+=4){
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
        src_ptr_1 += 4 * columns;
        src_ptr_2 += 4 * columns;
        src_ptr_3 += 4 * columns;
        src_ptr_4 += 4 * columns;
    }
    i = rows - (i - 4);
    if (i == 1){
        src_ptr_2 = src_ptr_1;
        src_ptr_3 = src_ptr_1;
        src_ptr_4 = src_ptr_1;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 2){
        src_ptr_3 = src_ptr_2;
        src_ptr_4 = src_ptr_2;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 3){
        src_ptr_4 = src_ptr_3;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 4){
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
}

void doLowPrecisionWeightPackImpl(int32_t* src_ptr_1, int32_t* src_ptr_2,
                            int32_t* src_ptr_3, int32_t* src_ptr_4,
                            int32_t* dst_ptr_r,   int size){
    int i;

    auto dst_ptr = dst_ptr_r;
    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #4\n"
        "blt 3f\n"

        "add %w[i], %w[i], #4\n"

        // Start of Main Loop
        "1:\n"

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

        "add %w[i], %w[i], #4\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #4\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        "dup v4.4s, wzr\n"

        PACK_SIMPLE_ONE_DATA_F32(0)
        PACK_SIMPLE_ONE_DATA_F32(1)
        PACK_SIMPLE_ONE_DATA_F32(2)
        PACK_SIMPLE_ONE_DATA_F32(3)

        "5:\n"
        "7:\n"

        : [ dst_ptr ] "+r"(dst_ptr), [ i ] "+r"(i)
        : [ src_ptr_1 ] "r"(src_ptr_1), [ src_ptr_2 ] "r"(src_ptr_2),
          [ src_ptr_3 ] "r"(src_ptr_3), [ src_ptr_4 ] "r"(src_ptr_4),
          [ size ] "r"(size)
        : "v1", "v2", "v3", "v4"
    );
}

//////////////////////////////////////
/////////       Float16       ////////
//////////////////////////////////////

void doLowPrecisionPack(const int16_t* src, int16_t* packed, int rows, int columns){
    doLowPrecisionWeightPack(const_cast<int16_t*>(src), packed, rows, columns);
}

void doLowPrecisionWeightPack(int16_t* src, int16_t* packed, int rows, int columns){
    int i;

    int16_t *src_ptr_1 = src + 0 * columns;
    int16_t *src_ptr_2 = src + 1 * columns;
    int16_t *src_ptr_3 = src + 2 * columns;
    int16_t *src_ptr_4 = src + 3 * columns;

    int16_t *packed_ptr = packed;

    for (i = 0 ; (i+4) <= rows ; i+=4){
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
        src_ptr_1 += 4 * columns;
        src_ptr_2 += 4 * columns;
        src_ptr_3 += 4 * columns;
        src_ptr_4 += 4 * columns;
    }
    i = rows - (i - 4);
    if (i == 1){
        src_ptr_2 = src_ptr_1;
        src_ptr_3 = src_ptr_1;
        src_ptr_4 = src_ptr_1;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 2){
        src_ptr_3 = src_ptr_2;
        src_ptr_4 = src_ptr_2;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 3){
        src_ptr_4 = src_ptr_3;
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
    else if (i == 4){
        doLowPrecisionWeightPackImpl(
            src_ptr_1, src_ptr_2,
            src_ptr_3, src_ptr_4,
            packed_ptr,
            columns);
    }
}

void doLowPrecisionWeightPackImpl(int16_t* src_ptr_1, int16_t* src_ptr_2,
                            int16_t* src_ptr_3, int16_t* src_ptr_4,
                            int16_t* dst_ptr_r,   int size){
    int i;
    auto dst_ptr = dst_ptr_r;

    asm volatile(
        "mov %w[i], wzr\n"

        "cmp %w[size], #8\n"
        "blt 3f\n"

        "add %w[i], %w[i], #8\n"

        // Start of Main Loop
        "1:\n"

        "ld1 {v1.8h}, [%[src_ptr_1]], #16\n"
        "ld1 {v2.8h}, [%[src_ptr_2]], #16\n"
        "ld1 {v3.8h}, [%[src_ptr_3]], #16\n"
        "ld1 {v4.8h}, [%[src_ptr_4]], #16\n"

        //////////////////////////////////////////////////////////////////

        "st1 {v1.8h},  [%[dst_ptr]], #16\n"
        "st1 {v2.8h},  [%[dst_ptr]], #16\n"
        "st1 {v3.8h},  [%[dst_ptr]], #16\n"
        "st1 {v4.8h},  [%[dst_ptr]], #16\n"

        //////////////////////////////////////////////////////////////////

        "add %w[i], %w[i], #8\n"
        "cmp %w[i], %w[size]\n"
        "b.le 1b\n"
        "sub %w[i], %w[i], #8\n"
        "sub %w[i], %w[size], %w[i]\n"

        "3:\n"
        
        "cmp %w[i], #0\n"
        "beq 7f\n"

        "dup v1.8h, wzr\n"
        "dup v2.8h, wzr\n"
        "dup v3.8h, wzr\n"
        "dup v4.8h, wzr\n"

        PACK_SIMPLE_ONE_DATA_F16(0)
        PACK_SIMPLE_ONE_DATA_F16(1)
        PACK_SIMPLE_ONE_DATA_F16(2)
        PACK_SIMPLE_ONE_DATA_F16(3)
        PACK_SIMPLE_ONE_DATA_F16(4)
        PACK_SIMPLE_ONE_DATA_F16(5)
        PACK_SIMPLE_ONE_DATA_F16(6)
        PACK_SIMPLE_ONE_DATA_F16(7)

        "5:\n"
        "7:\n"

        : [ dst_ptr ] "+r"(dst_ptr), [ i ] "+r"(i)
        : [ src_ptr_1 ] "r"(src_ptr_1), [ src_ptr_2 ] "r"(src_ptr_2),
          [ src_ptr_3 ] "r"(src_ptr_3), [ src_ptr_4 ] "r"(src_ptr_4),
          [ size ] "r"(size)
        : "v1", "v2", "v3", "v4"
    );
}

#else
void doLowPrecisionPack(const int8_t* src, int8_t* packed, int rows, int columns){  }
void doLowPrecisionWeightPack(int8_t* src, int8_t* packed, int rows, int columns){ }
void doLowPrecisionWeightPackImpl(int8_t* src_ptr_1, int8_t* src_ptr_2,
                            int8_t* src_ptr_3, int8_t* src_ptr_4,
                            int8_t* dst_ptr_r,   int size){ }
void doLowPrecisionPack(const int32_t* src, int32_t* packed, int rows, int columns){  }
void doLowPrecisionWeightPack(int32_t* src, int32_t* packed, int rows, int columns){ }
void doLowPrecisionWeightPackImpl(int32_t* src_ptr_1, int32_t* src_ptr_2,
                            int32_t* src_ptr_3, int32_t* src_ptr_4,
                            int32_t* dst_ptr_r,   int size){ }
void doLowPrecisionPack(const int16_t* src, int16_t* packed, int rows, int columns){  }
void doLowPrecisionWeightPack(int16_t* src, int16_t* packed, int rows, int columns){ }
void doLowPrecisionWeightPackImpl(int16_t* src_ptr_1, int16_t* src_ptr_2,
                            int16_t* src_ptr_3, int16_t* src_ptr_4,
                            int16_t* dst_ptr_r,   int size){ }
#endif
