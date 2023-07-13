#ifndef FLAGS_H_
#define FLAGS_H_

#define DO_PREFETCHING false
#define DO_LOG_PREFETCHING DO_PREFETCHING
#define DO_FLOAT_PREFETCHING DO_PREFETCHING
#define PREFETCH_OFFSET 16
#define PREFETCH_MANTISA_OFFSET_1 16
#define PREFETCH_MANTISA_OFFSET_2 16
#define PREFETCH_FLOAT_OFFSET 32
#define USE_FUSED_LOG_MULTIPLICATION true
#define USE_RUY_FOR_FLOAT true
#ifndef DEBUG
#define DEBUG false
#endif
#define USE_32BIT_SIGN
// #define MULTIPLY_SIGN
#define EXTEND_FOR_ACCURACY
#ifdef MULTIPLY_SIGN
#define IN_KERNEL_EXTEND
#else
#define IN_KERNEL_EXTEND
#endif
#define TENSORFLOW_LIKE_RUY_MATRIX_MAKE false

#define LWO_PRECISION_QUATERNARY_V1_0 true
#define LWO_PRECISION_QUATERNARY_V1_1 false

#define LWO_PRECISION_TERNARY 2
// #define DISABLE_PROFILE
// #define USE_SINGLE_ROW_BINARY_OP true
// #define VECTORIZED_DOWNCASTING_WITH_SCALAR_DIVISION true
#define DOWNCASTING_FUSED_IN_KERNEL true


#define SelfDependent_Continious 2
#define SelfDependent_Offset_Vector_Size 1
#define SelfDependent_Type SelfDependent_Continious

#define SelfDependent_Simple_Packing 1
#define SelfDependent_ASM_Packing 2
#define SelfDependent_ASM_TLB_Packing 3
#define SelfDependent_LHS_Packing SelfDependent_ASM_TLB_Packing
#define SelfDependent_RHS_Packing SelfDependent_Simple_Packing

#endif