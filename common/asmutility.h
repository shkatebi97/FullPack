#define LOAD_ONE_DATA(R)                                \
  "cmp %w[i], #" #R "\n"                                \
  "beq 5f\n"                                            \
  "ld1 { v0.b }[" #R "], [%[src_ptr1_exp]], #1\n"       \
  "ld1 { v1.b }[" #R "], [%[src_ptr2_1]], #1\n"         \
  "ld1 { v2.b }[" #R "], [%[src_ptr2_2]], #1\n"         \
  "ld1 { v3.b }[" #R "], [%[src_ptr2_3]], #1\n"         \
  "ld1 { v4.b }[" #R "], [%[src_ptr2_4]], #1\n"

#define STORE_ONE_DATA(R)                               \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "st1 { v2.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "st1 { v3.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "st1 { v4.b }[" #R "], [%[differ_from_max_1]], #1\n"

#define PACK_LOAD_ONE_DATA(R)                           \
  "cmp %w[i], #" #R "\n"                                \
  "beq 5f\n"                                            \
  "ld1 { v1.b }[" #R "], [%[src_ptr_1]], #1\n"          \
  "ld1 { v2.b }[" #R "], [%[src_ptr_2]], #1\n"          \
  "ld1 { v3.b }[" #R "], [%[src_ptr_3]], #1\n"          \
  "ld1 { v4.b }[" #R "], [%[src_ptr_4]], #1\n"          \
  "ld1 { v5.b }[" #R "], [%[src_ptr_s_1]], #1\n"        \
  "ld1 { v6.b }[" #R "], [%[src_ptr_s_2]], #1\n"        \
  "ld1 { v7.b }[" #R "], [%[src_ptr_s_3]], #1\n"        \
  "ld1 { v8.b }[" #R "], [%[src_ptr_s_4]], #1\n"

#define PACK_STORE_ONE_DATA(R)                          \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.b }[" #R "],  [%[dst_ptr_ref]], #1\n"       \
  "st1 { v9.b }[" #R "],  [%[dst_ptr_ref]], #1\n"       \
  "st1 { v2.b }[" #R "],  [%[dst_ptr_ref]], #1\n"       \
  "st1 { v10.b }[" #R "], [%[dst_ptr_ref]], #1\n"       \
  "st1 { v3.b }[" #R "],  [%[dst_ptr_ref]], #1\n"       \
  "st1 { v11.b }[" #R "], [%[dst_ptr_ref]], #1\n"       \
  "st1 { v4.b }[" #R "],  [%[dst_ptr_ref]], #1\n"       \
  "st1 { v12.b }[" #R "], [%[dst_ptr_ref]], #1\n"

#ifdef IN_KERNEL_EXTEND
#define PACK_STORE_ONE_DATA_16BIT_1(R,R2)               \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v9.b }[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v2.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v10.b}[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v3.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v11.b}[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v4.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v12.b}[" #R "],   [%[dst_ptr_ref]], #1\n"
#else
#define PACK_STORE_ONE_DATA_16BIT_1(R,R2)               \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v1.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v9.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v2.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v10.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v3.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v11.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v4.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v12.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"
#endif

#ifdef IN_KERNEL_EXTEND
#define PACK_STORE_ONE_DATA_16BIT_2(R,R2)               \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v5.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v9.b }[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v6.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v10.b}[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v7.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v11.b}[" #R "],   [%[dst_ptr_ref]], #1\n"      \
  "st1 { v8.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v12.b}[" #R "],   [%[dst_ptr_ref]], #1\n"
#else
#define PACK_STORE_ONE_DATA_16BIT_2(R,R2)               \
  "cmp %w[i], #" #R "\n"                                \
  "beq 6f\n"                                            \
  "st1 { v5.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v13.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v6.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v14.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v7.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v15.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v8.h }[" #R2 "],  [%[dst_ptr_ref]], #2\n"      \
  "st1 { v16.h}[" #R2 "],  [%[dst_ptr_ref]], #2\n"
#endif

#ifdef USE_32BIT_SIGN
#define LOAD_ONE_DATA_32_1(R)                           \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v1.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v2.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v3.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v4.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v17.s}[" #R "], [%[src_ptr1_mantise]],  #4\n"  \
  "ld1 { v19.s}[" #R "], [%[src_ptr2_s_1]],  #4\n"      \
  "ld1 { v20.s}[" #R "], [%[src_ptr2_s_2]],  #4\n"      \
  "ld1 { v21.s}[" #R "], [%[src_ptr2_s_3]],  #4\n"      \
  "ld1 { v22.s}[" #R "], [%[src_ptr2_s_4]],  #4\n"
#else
#define LOAD_ONE_DATA_32_1(R)                           \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v1.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v2.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v3.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v4.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v17.s}[" #R "], [%[src_ptr1_mantise]],  #4\n"  \
  "ld1 { v19.b}[" #R "], [%[src_ptr2_s_1]],  #1\n"      \
  "ld1 { v20.b}[" #R "], [%[src_ptr2_s_2]],  #1\n"      \
  "ld1 { v21.b}[" #R "], [%[src_ptr2_s_3]],  #1\n"      \
  "ld1 { v22.b}[" #R "], [%[src_ptr2_s_4]],  #1\n"
#endif

#ifdef USE_32BIT_SIGN
#define LOAD_ONE_DATA_32_2(R,R2)                        \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v1.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v2.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v3.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v4.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v18.s}[" #R2 "], [%[src_ptr1_mantise]], #4\n"  \
  "ld1 { v23.s}[" #R2 "], [%[src_ptr2_s_1]],  #4\n"     \
  "ld1 { v24.s}[" #R2 "], [%[src_ptr2_s_2]],  #4\n"     \
  "ld1 { v25.s}[" #R2 "], [%[src_ptr2_s_3]],  #4\n"     \
  "ld1 { v26.s}[" #R2 "], [%[src_ptr2_s_4]],  #4\n"
#else
#define LOAD_ONE_DATA_32_2(R,R2)                        \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v1.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v2.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v3.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v4.b }[" #R "], [%[differ_from_max_1]], #1\n"  \
  "ld1 { v18.s}[" #R2 "], [%[src_ptr1_mantise]], #4\n"  \
  "ld1 { v23.b}[" #R "], [%[src_ptr2_s_1]],  #1\n"      \
  "ld1 { v24.b}[" #R "], [%[src_ptr2_s_2]],  #1\n"      \
  "ld1 { v25.b}[" #R "], [%[src_ptr2_s_3]],  #1\n"      \
  "ld1 { v26.b}[" #R "], [%[src_ptr2_s_4]],  #1\n"
#endif

#ifdef USE_32BIT_SIGN
#define LOAD_ONE_MANTISE_32_1(R,R2)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v17.s}[" #R2 "], [%[src_ptr1_mantise]],  #4\n" \
  "ld1 { v23.s}[" #R2 "], [%[src_ptr2_s_1]],  #4\n"     \
  "ld1 { v24.s}[" #R2 "], [%[src_ptr2_s_2]],  #4\n"     \
  "ld1 { v25.s}[" #R2 "], [%[src_ptr2_s_3]],  #4\n"     \
  "ld1 { v26.s}[" #R2 "], [%[src_ptr2_s_4]],  #4\n"

#define LOAD_ONE_MANTISE_32_2(R,R2)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v18.s}[" #R2 "], [%[src_ptr1_mantise]], #4\n"  \
  "ld1 { v27.s}[" #R2 "], [%[src_ptr2_s_1]],  #4\n"     \
  "ld1 { v28.s}[" #R2 "], [%[src_ptr2_s_2]],  #4\n"     \
  "ld1 { v29.s}[" #R2 "], [%[src_ptr2_s_3]],  #4\n"     \
  "ld1 { v30.s}[" #R2 "], [%[src_ptr2_s_4]],  #4\n"
#else
#define LOAD_ONE_MANTISE_32_1(R,R2)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v17.s}[" #R2 "], [%[src_ptr1_mantise]],  #4\n" \
  "ld1 { v23.b}[" #R2 "], [%[src_ptr2_s_1]],  #1\n"     \
  "ld1 { v24.b}[" #R2 "], [%[src_ptr2_s_2]],  #1\n"     \
  "ld1 { v25.b}[" #R2 "], [%[src_ptr2_s_3]],  #1\n"     \
  "ld1 { v26.b}[" #R2 "], [%[src_ptr2_s_4]],  #1\n"

#define LOAD_ONE_MANTISE_32_2(R,R2)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v18.s}[" #R2 "], [%[src_ptr1_mantise]], #4\n"  \
  "ld1 { v23.b}[" #R2 "], [%[src_ptr2_s_1]],  #1\n"     \
  "ld1 { v24.b}[" #R2 "], [%[src_ptr2_s_2]],  #1\n"     \
  "ld1 { v25.b}[" #R2 "], [%[src_ptr2_s_3]],  #1\n"     \
  "ld1 { v26.b}[" #R2 "], [%[src_ptr2_s_4]],  #1\n"
#endif

#ifdef IN_KERNEL_EXTEND
#define LOAD_ONE_DATA_AND_SIGN_EXT_1(R,R2)              \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v0.b  }[" #R "],  [%[src_ptr1_exp]],  #1\n"    \
  "ld1 { v1.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v23.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v2.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v24.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v3.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v25.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v4.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v26.b }[" #R "],  [%[src_ptr2]],      #1\n"
#else
#define LOAD_ONE_DATA_AND_SIGN_EXT_1(R,R2)              \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v0.h  }[" #R2 "], [%[src_ptr1_exp]],  #2\n"    \
  "ld1 { v1.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v23.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v2.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v24.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v3.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v25.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v4.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v26.h }[" #R2 "], [%[src_ptr2]],      #2\n"
#endif

#ifdef IN_KERNEL_EXTEND
#define LOAD_ONE_DATA_AND_SIGN_EXT_2(R,R2)              \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v31.b }[" #R "],  [%[src_ptr1_exp]],  #1\n"    \
  "ld1 { v1.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v23.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v2.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v24.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v3.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v25.b }[" #R "],  [%[src_ptr2]],      #1\n"    \
  "ld1 { v4.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v26.b }[" #R "],  [%[src_ptr2]],      #1\n"
#else
#define LOAD_ONE_DATA_AND_SIGN_EXT_2(R,R2)              \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v31.h }[" #R2 "], [%[src_ptr1_exp]],  #2\n"    \
  "ld1 { v1.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v27.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v2.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v28.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v3.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v29.h }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v4.h  }[" #R2 "], [%[src_ptr2]],      #2\n"    \
  "ld1 { v30.h }[" #R2 "], [%[src_ptr2]],      #2\n"
#endif

#define LOAD_ONE_DATA_AND_SIGN(R)                       \
  "cmp %w[i], #" #R "\n"                                \
  "beq 15f\n"                                           \
  "ld1 { v0.b  }[" #R "], [%[src_ptr1_exp]],  #1\n"     \
  "ld1 { v1.b  }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v23.b }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v2.b  }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v24.b }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v3.b  }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v25.b }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v4.b  }[" #R "], [%[src_ptr2]],      #1\n"     \
  "ld1 { v26.b }[" #R "], [%[src_ptr2]],      #1\n"

#define SET_ZERO_IF_OUT(R)                              \
  "cmp %w[i], #" #R "\n"                                \
  "bge 110f\n"                                          \
  "ins v5.b[" #R "], wzr\n"                             \
  "ins v6.b[" #R "], wzr\n"                             \
  "ins v7.b[" #R "], wzr\n"                             \
  "ins v8.b[" #R "], wzr\n"

#define SET_ZERO_IF_OUT_2_2(R,R2)                       \
  "cmp %w[i], #" #R "\n"                                \
  "bge 110f\n"                                          \
  "ins v5.4s[" #R2 "], wzr\n"                           \
  "ins v6.4s[" #R2 "], wzr\n"                           \
  "ins v7.4s[" #R2 "], wzr\n"                           \
  "ins v8.4s[" #R2 "], wzr\n"

#define SET_ZERO_IF_OUT_2_1(R,R2)                       \
  "cmp %w[i], #" #R "\n"                                \
  "bge 110f\n"                                          \
  "ins v1.4s[" #R2 "], wzr\n"                           \
  "ins v2.4s[" #R2 "], wzr\n"                           \
  "ins v3.4s[" #R2 "], wzr\n"                           \
  "ins v4.4s[" #R2 "], wzr\n"


#define SET_ZERO_IF_OUT_1(R)                           \
  "cmp %w[i], #" #R "\n"                               \
  "bge 110f\n"                                         \
  "ins v1.4s[" #R "], wzr\n"                           \
  "ins v2.4s[" #R "], wzr\n"                           \
  "ins v3.4s[" #R "], wzr\n"                           \
  "ins v4.4s[" #R "], wzr\n"

#define SET_ZERO_IF_OUT_2(R,R_2)                       \
  "cmp %w[i], #" #R "\n"                               \
  "bge 110f\n"                                         \
  "ins v5.4s[" #R_2 "], wzr\n"                         \
  "ins v6.4s[" #R_2 "], wzr\n"                         \
  "ins v7.4s[" #R_2 "], wzr\n"                         \
  "ins v8.4s[" #R_2 "], wzr\n"

#define PACK_SIMPLE_ONE_DATA(R)                         \
  "cmp %w[i], #" #R "\n"                                \
  "beq 5f\n"                                            \
  "ld1 { v1.b }[" #R "], [x0], #1\n"                    \
  "ld1 { v2.b }[" #R "], [x1], #1\n"                    \
  "ld1 { v3.b }[" #R "], [x2], #1\n"                    \
  "ld1 { v4.b }[" #R "], [x3], #1\n"

#define PACK_SIMPLE_ONE_DATA_F32(R)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 5f\n"                                            \
  "ld1 { v1.s }[" #R "], [%[src_ptr_1]], #4\n"          \
  "ld1 { v2.s }[" #R "], [%[src_ptr_2]], #4\n"          \
  "ld1 { v3.s }[" #R "], [%[src_ptr_3]], #4\n"          \
  "ld1 { v4.s }[" #R "], [%[src_ptr_4]], #4\n"

#define PACK_SIMPLE_ONE_DATA_F16(R)                     \
  "cmp %w[i], #" #R "\n"                                \
  "beq 5f\n"                                            \
  "ld1 { v1.h }[" #R "], [%[src_ptr_1]], #2\n"          \
  "ld1 { v2.h }[" #R "], [%[src_ptr_2]], #2\n"          \
  "ld1 { v3.h }[" #R "], [%[src_ptr_3]], #2\n"          \
  "ld1 { v4.h }[" #R "], [%[src_ptr_4]], #2\n"

#define PREFETCH_DATA(SRC,OFFSET) "prfm PLDL1STRM, [%[" SRC "], #" #OFFSET "]\n"