#include <arm_neon.h>
// #include "test.h"

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter8(
    size_t mr, size_t nr, size_t k, const uint8_t* __restrict__ a,
    size_t a_stride, const uint8_t* __restrict__ w, int32_t* __restrict__ c,
    size_t c_stride) {
  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;

  int32x4_t vacc0x0123 = vld1q_s32(c0);
  int32x4_t vacc0x4567 = vld1q_s32(c0 + 4);
  int32x4_t vacc1x0123 = vld1q_s32(c1);
  int32x4_t vacc1x4567 = vld1q_s32(c1 + 4);
  int32x4_t vacc2x0123 = vld1q_s32(c2);
  int32x4_t vacc2x4567 = vld1q_s32(c2 + 4);
  int32x4_t vacc3x0123 = vld1q_s32(c3);
  int32x4_t vacc3x4567 = vld1q_s32(c3 + 4);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint16x8_t vacc0, vacc1, vacc2, vacc3;
  uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
  uint16x8_t vacc1x = vacc0x;
  uint16x8_t vacc2x = vacc0x;
  uint16x8_t vacc3x = vacc0x;

  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c2, vxa1, 2);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c2, vxa2, 2);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c0, vxa0, 0);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c0, vxa1, 0);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c0, vxa2, 0);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c2, vxa1, 2);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c2, vxa2, 2);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);
  }

  uint32x4_t vacc0x0123_;
  uint32x4_t vacc0x4567_;
  uint32x4_t vacc1x0123_;
  uint32x4_t vacc1x4567_;
  uint32x4_t vacc2x0123_;
  uint32x4_t vacc2x4567_;
  uint32x4_t vacc3x0123_;
  uint32x4_t vacc3x4567_;
  vacc0x0123_ = vmovl_u16(vget_low_u16(vacc0x));
  vacc0x4567_ = vmovl_u16(vget_high_u16(vacc0x));
  vacc1x0123_ = vmovl_u16(vget_low_u16(vacc1x));
  vacc1x4567_ = vmovl_u16(vget_high_u16(vacc1x));
  vacc2x0123_ = vmovl_u16(vget_low_u16(vacc2x));
  vacc2x4567_ = vmovl_u16(vget_high_u16(vacc2x));
  vacc3x0123_ = vmovl_u16(vget_low_u16(vacc3x));
  vacc3x4567_ = vmovl_u16(vget_high_u16(vacc3x));

  vacc0x0123 = vaddq_s32(vacc0x0123, vreinterpretq_s32_u32(vacc0x0123_));
  vacc1x0123 = vaddq_s32(vacc1x0123, vreinterpretq_s32_u32(vacc1x0123_));
  vacc2x0123 = vaddq_s32(vacc2x0123, vreinterpretq_s32_u32(vacc2x0123_));
  vacc3x0123 = vaddq_s32(vacc3x0123, vreinterpretq_s32_u32(vacc3x0123_));
  vacc0x4567 = vaddq_s32(vacc0x4567, vreinterpretq_s32_u32(vacc0x4567_));
  vacc1x4567 = vaddq_s32(vacc1x4567, vreinterpretq_s32_u32(vacc1x4567_));
  vacc2x4567 = vaddq_s32(vacc2x4567, vreinterpretq_s32_u32(vacc2x4567_));
  vacc3x4567 = vaddq_s32(vacc3x4567, vreinterpretq_s32_u32(vacc3x4567_));

  vst1q_s32(c0, vacc0x0123);
  vst1q_s32(c0 + 4, vacc0x4567);
  vst1q_s32(c1, vacc1x0123);
  vst1q_s32(c1 + 4, vacc1x4567);
  vst1q_s32(c2, vacc2x0123);
  vst1q_s32(c2 + 4, vacc2x4567);
  vst1q_s32(c3, vacc3x0123);
  vst1q_s32(c3 + 4, vacc3x4567);
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter4(
    size_t mr, size_t nr, size_t k, const uint8_t* __restrict__ a,
    size_t a_stride, const uint8_t* __restrict__ w, int32_t* __restrict__ c,
    size_t c_stride) {
  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint16x8_t vacc0, vacc1, vacc2, vacc3;
  uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
  uint16x8_t vacc1x = vacc0x;
  uint16x8_t vacc2x = vacc0x;
  uint16x8_t vacc3x = vacc0x;

  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c2, vxa1, 2);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c2, vxa2, 2);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c2, vxa1, 2);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c2, vxa2, 2);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);
  }

  uint32x4_t vacc0x0123_;
  uint32x4_t vacc0x4567_;
  uint32x4_t vacc1x0123_;
  uint32x4_t vacc1x4567_;
  uint32x4_t vacc2x0123_;
  uint32x4_t vacc2x4567_;
  uint32x4_t vacc3x0123_;
  uint32x4_t vacc3x4567_;
  vacc0x0123_ = vmovl_u16(vget_low_u16(vacc0x));
  vacc0x4567_ = vmovl_u16(vget_high_u16(vacc0x));
  vacc1x0123_ = vmovl_u16(vget_low_u16(vacc1x));
  vacc1x4567_ = vmovl_u16(vget_high_u16(vacc1x));
  vacc2x0123_ = vmovl_u16(vget_low_u16(vacc2x));
  vacc2x4567_ = vmovl_u16(vget_high_u16(vacc2x));
  vacc3x0123_ = vmovl_u16(vget_low_u16(vacc3x));
  vacc3x4567_ = vmovl_u16(vget_high_u16(vacc3x));
  int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vacc0x0123_);
  int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vacc1x0123_);
  int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vacc2x0123_);
  int32x4_t vacc3x0123 = vreinterpretq_s32_u32(vacc3x0123_);
  int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vacc0x4567_);
  int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vacc1x4567_);
  int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vacc2x4567_);
  int32x4_t vacc3x4567 = vreinterpretq_s32_u32(vacc3x4567_);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;

  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0), vacc0x0123));
  vst1q_s32(c0 + 4, vaddq_s32(vld1q_s32(c0 + 4), vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1), vacc1x0123));
  vst1q_s32(c1 + 4, vaddq_s32(vld1q_s32(c1 + 4), vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2), vacc2x0123));
  vst1q_s32(c2 + 4, vaddq_s32(vld1q_s32(c2 + 4), vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3), vacc3x0123));
  vst1q_s32(c3 + 4, vaddq_s32(vld1q_s32(c3 + 4), vacc3x4567));
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter2(
    size_t mr, size_t nr, size_t k, const uint8_t* __restrict__ a,
    size_t a_stride, const uint8_t* __restrict__ w, int32_t* __restrict__ c,
    size_t c_stride) {
  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint16x8_t vacc0, vacc1, vacc2, vacc3;
  uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
  uint16x8_t vacc1x = vacc0x;
  uint16x8_t vacc2x = vacc0x;
  uint16x8_t vacc3x = vacc0x;

  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c2, vxa0, 2);
    vacc1 = vmulq_lane_u16(vxb01234567c2, vxa1, 2);
    vacc2 = vmulq_lane_u16(vxb01234567c2, vxa2, 2);
    vacc3 = vmulq_lane_u16(vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c1, vxa1, 1);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c1, vxa2, 1);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c1, vxa3, 1);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c2, vxa0, 2);
    vacc1 = vmulq_lane_u16(vxb01234567c2, vxa1, 2);
    vacc2 = vmulq_lane_u16(vxb01234567c2, vxa2, 2);
    vacc3 = vmulq_lane_u16(vxb01234567c2, vxa3, 2);

    vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
    vacc1 = vmlaq_lane_u16(vacc1, vxb01234567c3, vxa1, 3);
    vacc2 = vmlaq_lane_u16(vacc2, vxb01234567c3, vxa2, 3);
    vacc3 = vmlaq_lane_u16(vacc3, vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);
  }

  uint32x4_t vacc0x0123_;
  uint32x4_t vacc0x4567_;
  uint32x4_t vacc1x0123_;
  uint32x4_t vacc1x4567_;
  uint32x4_t vacc2x0123_;
  uint32x4_t vacc2x4567_;
  uint32x4_t vacc3x0123_;
  uint32x4_t vacc3x4567_;
  vacc0x0123_ = vmovl_u16(vget_low_u16(vacc0x));
  vacc0x4567_ = vmovl_u16(vget_high_u16(vacc0x));
  vacc1x0123_ = vmovl_u16(vget_low_u16(vacc1x));
  vacc1x4567_ = vmovl_u16(vget_high_u16(vacc1x));
  vacc2x0123_ = vmovl_u16(vget_low_u16(vacc2x));
  vacc2x4567_ = vmovl_u16(vget_high_u16(vacc2x));
  vacc3x0123_ = vmovl_u16(vget_low_u16(vacc3x));
  vacc3x4567_ = vmovl_u16(vget_high_u16(vacc3x));
  int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vacc0x0123_);
  int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vacc1x0123_);
  int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vacc2x0123_);
  int32x4_t vacc3x0123 = vreinterpretq_s32_u32(vacc3x0123_);
  int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vacc0x4567_);
  int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vacc1x4567_);
  int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vacc2x4567_);
  int32x4_t vacc3x4567 = vreinterpretq_s32_u32(vacc3x4567_);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;

  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0), vacc0x0123));
  vst1q_s32(c0 + 4, vaddq_s32(vld1q_s32(c0 + 4), vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1), vacc1x0123));
  vst1q_s32(c1 + 4, vaddq_s32(vld1q_s32(c1 + 4), vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2), vacc2x0123));
  vst1q_s32(c2 + 4, vaddq_s32(vld1q_s32(c2 + 4), vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3), vacc3x0123));
  vst1q_s32(c3 + 4, vaddq_s32(vld1q_s32(c3 + 4), vacc3x4567));
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter1(
    size_t mr, size_t nr, size_t k, const uint8_t* __restrict__ a,
    size_t a_stride, const uint8_t* __restrict__ w, int32_t* __restrict__ c,
    size_t c_stride) {
  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint16x8_t vacc0, vacc1, vacc2, vacc3;
  uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
  uint16x8_t vacc1x = vacc0x;
  uint16x8_t vacc2x = vacc0x;
  uint16x8_t vacc3x = vacc0x;

  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c1, vxa0, 1);
    vacc1 = vmulq_lane_u16(vxb01234567c1, vxa1, 1);
    vacc2 = vmulq_lane_u16(vxb01234567c1, vxa2, 1);
    vacc3 = vmulq_lane_u16(vxb01234567c1, vxa3, 1);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c2, vxa0, 2);
    vacc1 = vmulq_lane_u16(vxb01234567c2, vxa1, 2);
    vacc2 = vmulq_lane_u16(vxb01234567c2, vxa2, 2);
    vacc3 = vmulq_lane_u16(vxb01234567c2, vxa3, 2);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c3, vxa0, 3);
    vacc1 = vmulq_lane_u16(vxb01234567c3, vxa1, 3);
    vacc2 = vmulq_lane_u16(vxb01234567c3, vxa2, 3);
    vacc3 = vmulq_lane_u16(vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vxa0 = vld1_u16((uint16_t*)a0);
    a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1);
    a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2);
    a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3);
    a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w);
    w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w);
    w += 16;

    vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
    vacc1 = vmulq_lane_u16(vxb01234567c0, vxa1, 0);
    vacc2 = vmulq_lane_u16(vxb01234567c0, vxa2, 0);
    vacc3 = vmulq_lane_u16(vxb01234567c0, vxa3, 0);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c1, vxa0, 1);
    vacc1 = vmulq_lane_u16(vxb01234567c1, vxa1, 1);
    vacc2 = vmulq_lane_u16(vxb01234567c1, vxa2, 1);
    vacc3 = vmulq_lane_u16(vxb01234567c1, vxa3, 1);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c2, vxa0, 2);
    vacc1 = vmulq_lane_u16(vxb01234567c2, vxa1, 2);
    vacc2 = vmulq_lane_u16(vxb01234567c2, vxa2, 2);
    vacc3 = vmulq_lane_u16(vxb01234567c2, vxa3, 2);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);

    vacc0 = vmulq_lane_u16(vxb01234567c3, vxa0, 3);
    vacc1 = vmulq_lane_u16(vxb01234567c3, vxa1, 3);
    vacc2 = vmulq_lane_u16(vxb01234567c3, vxa2, 3);
    vacc3 = vmulq_lane_u16(vxb01234567c3, vxa3, 3);

    vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
    vacc1x = vsraq_n_u16(vacc1x, vacc1, 8);
    vacc2x = vsraq_n_u16(vacc2x, vacc2, 8);
    vacc3x = vsraq_n_u16(vacc3x, vacc3, 8);
  }

  uint32x4_t vacc0x0123_;
  uint32x4_t vacc0x4567_;
  uint32x4_t vacc1x0123_;
  uint32x4_t vacc1x4567_;
  uint32x4_t vacc2x0123_;
  uint32x4_t vacc2x4567_;
  uint32x4_t vacc3x0123_;
  uint32x4_t vacc3x4567_;
  vacc0x0123_ = vmovl_u16(vget_low_u16(vacc0x));
  vacc0x4567_ = vmovl_u16(vget_high_u16(vacc0x));
  vacc1x0123_ = vmovl_u16(vget_low_u16(vacc1x));
  vacc1x4567_ = vmovl_u16(vget_high_u16(vacc1x));
  vacc2x0123_ = vmovl_u16(vget_low_u16(vacc2x));
  vacc2x4567_ = vmovl_u16(vget_high_u16(vacc2x));
  vacc3x0123_ = vmovl_u16(vget_low_u16(vacc3x));
  vacc3x4567_ = vmovl_u16(vget_high_u16(vacc3x));
  int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vacc0x0123_);
  int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vacc1x0123_);
  int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vacc2x0123_);
  int32x4_t vacc3x0123 = vreinterpretq_s32_u32(vacc3x0123_);
  int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vacc0x4567_);
  int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vacc1x4567_);
  int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vacc2x4567_);
  int32x4_t vacc3x4567 = vreinterpretq_s32_u32(vacc3x4567_);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;

  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0), vacc0x0123));
  vst1q_s32(c0 + 4, vaddq_s32(vld1q_s32(c0 + 4), vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1), vacc1x0123));
  vst1q_s32(c1 + 4, vaddq_s32(vld1q_s32(c1 + 4), vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2), vacc2x0123));
  vst1q_s32(c2 + 4, vaddq_s32(vld1q_s32(c2 + 4), vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3), vacc3x0123));
  vst1q_s32(c3 + 4, vaddq_s32(vld1q_s32(c3 + 4), vacc3x4567));
}

static size_t iter_cnt[][7] = {{8, 8, 8, 8, 4, 2, 1}, {8, 8, 4, 2, 1, 0, 0},
                               {8, 4, 2, 1, 0, 0, 0}, {8, 2, 1, 0, 0, 0, 0},
                               {4, 1, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0, 0},
                               {1, 0, 0, 0, 0, 0, 0}};

static void pack_qnnpack4x8multi(uint8_t* W, uint8_t* W_pack, size_t M,
                                 size_t N) {
  int p = 0;
  for (size_t j = 0; j < N; j += 8)
    for (size_t i = 0; i < M; i += 2)
      for (size_t k = j; k < j + 8; k++) {
        W_pack[p++] = W[i * N + k + N];
        W_pack[p++] = W[i * N + k];
      }
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter8(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
    size_t c_stride,
    int shift) {

  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
  uint32x4_t vacc0x4567_=vacc0x0123_; 
  uint32x4_t vacc1x0123_=vacc0x0123_; 
  uint32x4_t vacc1x4567_=vacc0x0123_; 
  uint32x4_t vacc2x0123_=vacc0x0123_; 
  uint32x4_t vacc2x4567_=vacc0x0123_; 
  uint32x4_t vacc3x0123_=vacc0x0123_; 
  uint32x4_t vacc3x4567_=vacc0x0123_; 

  uint32x4_t vacc0x0; 
  uint32x4_t vacc0x4; 
  uint32x4_t vacc1x0; 
  uint32x4_t vacc1x4; 
  uint32x4_t vacc2x0; 
  uint32x4_t vacc2x4; 
  uint32x4_t vacc3x0; 
  uint32x4_t vacc3x4; 

  //4,4 : 24-12-0
  uint32x4_t mask = vdupq_n_u32(((1<<shift)-1)<<shift);
  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    uint16x4_t vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    uint16x4_t vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;


    //uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
   
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    //vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);
  }

  vacc0x0123_ = vshrq_n_u32( vacc0x0123_, shift);
  vacc0x4567_ = vshrq_n_u32( vacc0x4567_, shift);
  vacc1x0123_ = vshrq_n_u32( vacc1x0123_, shift);
  vacc1x4567_ = vshrq_n_u32( vacc1x4567_, shift);
  vacc2x0123_ = vshrq_n_u32( vacc2x0123_, shift);
  vacc2x4567_ = vshrq_n_u32( vacc2x4567_, shift);
  vacc3x0123_ = vshrq_n_u32( vacc3x0123_, shift);
  vacc3x4567_ = vshrq_n_u32( vacc3x4567_, shift);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;
  
  vst1q_s32(c0,vacc0x0123_);
  vst1q_s32(c0+4, vacc0x4567_);
  vst1q_s32(c1, vacc1x0123_);
  vst1q_s32(c1+4, vacc1x4567_);
  vst1q_s32(c2, vacc2x0123_);
  vst1q_s32(c2+4, vacc2x4567_);
  vst1q_s32(c3, vacc3x0123_);
  vst1q_s32(c3+4, vacc3x4567_);
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter4(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
    size_t c_stride,
    int shift) {

  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
  uint32x4_t vacc0x4567_=vacc0x0123_; 
  uint32x4_t vacc1x0123_=vacc0x0123_; 
  uint32x4_t vacc1x4567_=vacc0x0123_; 
  uint32x4_t vacc2x0123_=vacc0x0123_; 
  uint32x4_t vacc2x4567_=vacc0x0123_; 
  uint32x4_t vacc3x0123_=vacc0x0123_; 
  uint32x4_t vacc3x4567_=vacc0x0123_; 

  uint32x4_t vacc0x0; 
  uint32x4_t vacc0x4; 
  uint32x4_t vacc1x0; 
  uint32x4_t vacc1x4; 
  uint32x4_t vacc2x0; 
  uint32x4_t vacc2x4; 
  uint32x4_t vacc3x0; 
  uint32x4_t vacc3x4; 

  //4,4 : 24-12-0
  uint32x4_t mask = vdupq_n_u32(((1<<shift)-1)<<shift);
  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    uint16x4_t vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    uint16x4_t vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;

    //uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
   
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    //vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);
  }

  vacc0x0123_ = vshrq_n_u32( vacc0x0123_, shift);
  vacc0x4567_ = vshrq_n_u32( vacc0x4567_, shift);
  vacc1x0123_ = vshrq_n_u32( vacc1x0123_, shift);
  vacc1x4567_ = vshrq_n_u32( vacc1x4567_, shift);
  vacc2x0123_ = vshrq_n_u32( vacc2x0123_, shift);
  vacc2x4567_ = vshrq_n_u32( vacc2x4567_, shift);
  vacc3x0123_ = vshrq_n_u32( vacc3x0123_, shift);
  vacc3x4567_ = vshrq_n_u32( vacc3x4567_, shift);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;
  
  vst1q_s32(c0,vacc0x0123_);
  vst1q_s32(c0+4, vacc0x4567_);
  vst1q_s32(c1, vacc1x0123_);
  vst1q_s32(c1+4, vacc1x4567_);
  vst1q_s32(c2, vacc2x0123_);
  vst1q_s32(c2+4, vacc2x4567_);
  vst1q_s32(c3, vacc3x0123_);
  vst1q_s32(c3+4, vacc3x4567_);
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter2(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
    size_t c_stride,
    int shift) {

  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
  uint32x4_t vacc0x4567_=vacc0x0123_; 
  uint32x4_t vacc1x0123_=vacc0x0123_; 
  uint32x4_t vacc1x4567_=vacc0x0123_; 
  uint32x4_t vacc2x0123_=vacc0x0123_; 
  uint32x4_t vacc2x4567_=vacc0x0123_; 
  uint32x4_t vacc3x0123_=vacc0x0123_; 
  uint32x4_t vacc3x4567_=vacc0x0123_; 

  uint32x4_t vacc0x0; 
  uint32x4_t vacc0x4; 
  uint32x4_t vacc1x0; 
  uint32x4_t vacc1x4; 
  uint32x4_t vacc2x0; 
  uint32x4_t vacc2x4; 
  uint32x4_t vacc3x0; 
  uint32x4_t vacc3x4; 

  //4,4 : 24-12-0
  uint32x4_t mask = vdupq_n_u32(((1<<shift)-1)<<shift);
  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    uint16x4_t vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    uint16x4_t vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;

    //uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
   
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    //vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 1);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 2);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmlal_lane_u16(vacc0x0, vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmlal_lane_u16(vacc0x4, vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmlal_lane_u16(vacc1x0, vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmlal_lane_u16(vacc1x4, vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmlal_lane_u16(vacc2x0, vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmlal_lane_u16(vacc2x4, vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmlal_lane_u16(vacc3x0, vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmlal_lane_u16(vacc3x4, vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);
  }

  vacc0x0123_ = vshrq_n_u32( vacc0x0123_, shift);
  vacc0x4567_ = vshrq_n_u32( vacc0x4567_, shift);
  vacc1x0123_ = vshrq_n_u32( vacc1x0123_, shift);
  vacc1x4567_ = vshrq_n_u32( vacc1x4567_, shift);
  vacc2x0123_ = vshrq_n_u32( vacc2x0123_, shift);
  vacc2x4567_ = vshrq_n_u32( vacc2x4567_, shift);
  vacc3x0123_ = vshrq_n_u32( vacc3x0123_, shift);
  vacc3x4567_ = vshrq_n_u32( vacc3x4567_, shift);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;
  
  vst1q_s32(c0,vacc0x0123_);
  vst1q_s32(c0+4, vacc0x4567_);
  vst1q_s32(c1, vacc1x0123_);
  vst1q_s32(c1+4, vacc1x4567_);
  vst1q_s32(c2, vacc2x0123_);
  vst1q_s32(c2+4, vacc2x4567_);
  vst1q_s32(c3, vacc3x0123_);
  vst1q_s32(c3+4, vacc3x4567_);
}

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_type2_iter1(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
    size_t c_stride,
    int shift) {

  const uint8_t* a0 = a;
  const uint8_t* a1 = (a0 + a_stride);
  const uint8_t* a2 = (a1 + a_stride);
  const uint8_t* a3 = (a2 + a_stride);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint32x4_t vacc0x0123_=veorq_u32(vacc0x0123_, vacc0x0123_); 
  uint32x4_t vacc0x4567_=vacc0x0123_; 
  uint32x4_t vacc1x0123_=vacc0x0123_; 
  uint32x4_t vacc1x4567_=vacc0x0123_; 
  uint32x4_t vacc2x0123_=vacc0x0123_; 
  uint32x4_t vacc2x4567_=vacc0x0123_; 
  uint32x4_t vacc3x0123_=vacc0x0123_; 
  uint32x4_t vacc3x4567_=vacc0x0123_; 

  uint32x4_t vacc0x0; 
  uint32x4_t vacc0x4; 
  uint32x4_t vacc1x0; 
  uint32x4_t vacc1x4; 
  uint32x4_t vacc2x0; 
  uint32x4_t vacc2x4; 
  uint32x4_t vacc3x0; 
  uint32x4_t vacc3x4; 

  //4,4 : 24-12-0
  uint32x4_t mask = vdupq_n_u32(((1<<shift)-1)<<shift);
  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    uint16x4_t vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    uint16x4_t vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;

    //uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
   
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 1);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 2);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;
    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    //vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    //vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 0);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 0);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 0);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 0);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 0);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 0);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 0);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 0);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 1);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 1);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 1);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 1);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 1);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 1);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 1);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 1);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 2);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 2);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 2);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 2);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 2);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 2);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 2);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 2);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);

    vxb01234567c0l = vld1_u16((uint16_t*)w); w += 8;
    vxb01234567c0h = vld1_u16((uint16_t*)w); w += 8;
    vacc0x0 = vmull_lane_u16(vxb01234567c0l, vxa0, 3);
    vacc0x4 = vmull_lane_u16(vxb01234567c0h, vxa0, 3);
    vacc1x0 = vmull_lane_u16(vxb01234567c0l, vxa1, 3);
    vacc1x4 = vmull_lane_u16(vxb01234567c0h, vxa1, 3);
    vacc2x0 = vmull_lane_u16(vxb01234567c0l, vxa2, 3);
    vacc2x4 = vmull_lane_u16(vxb01234567c0h, vxa2, 3);
    vacc3x0 = vmull_lane_u16(vxb01234567c0l, vxa3, 3);
    vacc3x4 = vmull_lane_u16(vxb01234567c0h, vxa3, 3);

    vacc0x0123_ = vaddq_u32(vandq_u32(vacc0x0, mask),  vacc0x0123_);
    vacc0x4567_ = vaddq_u32(vandq_u32(vacc0x4, mask),  vacc0x4567_);
    vacc1x0123_ = vaddq_u32(vandq_u32(vacc1x0, mask),  vacc1x0123_);
    vacc1x4567_ = vaddq_u32(vandq_u32(vacc1x4, mask),  vacc1x4567_);
    vacc2x0123_ = vaddq_u32(vandq_u32(vacc2x0, mask),  vacc2x0123_);
    vacc2x4567_ = vaddq_u32(vandq_u32(vacc2x4, mask),  vacc2x4567_);
    vacc3x0123_ = vaddq_u32(vandq_u32(vacc3x0, mask),  vacc3x0123_);
    vacc3x4567_ = vaddq_u32(vandq_u32(vacc3x4, mask),  vacc3x4567_);
  }

  vacc0x0123_ = vshrq_n_u32( vacc0x0123_, shift);
  vacc0x4567_ = vshrq_n_u32( vacc0x4567_, shift);
  vacc1x0123_ = vshrq_n_u32( vacc1x0123_, shift);
  vacc1x4567_ = vshrq_n_u32( vacc1x4567_, shift);
  vacc2x0123_ = vshrq_n_u32( vacc2x0123_, shift);
  vacc2x4567_ = vshrq_n_u32( vacc2x4567_, shift);
  vacc3x0123_ = vshrq_n_u32( vacc3x0123_, shift);
  vacc3x4567_ = vshrq_n_u32( vacc3x4567_, shift);

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;
  
  vst1q_s32(c0,vacc0x0123_);
  vst1q_s32(c0+4, vacc0x4567_);
  vst1q_s32(c1, vacc1x0123_);
  vst1q_s32(c1+4, vacc1x4567_);
  vst1q_s32(c2, vacc2x0123_);
  vst1q_s32(c2+4, vacc2x4567_);
  vst1q_s32(c3, vacc3x0123_);
  vst1q_s32(c3+4, vacc3x4567_);
}

static size_t iter_cnt_type2[][7] = {
  {8,8,8,8,8,8,2},
  {8,8,8,8,8,2,0},
  {8,8,8,8,4,1,0},
  {8,8,8,8,2,0,0},
  {8,8,4,2,1,0,0},
  {8,2,1,0,0,0,0},
  {2,0,0,0,0,0,0}
};

static void pack_qnnpack4x8multi_type2_lhs(uint8_t *A, uint8_t *A_pack, size_t M, size_t N, size_t bitw) {
  size_t shift = 8-bitw;
  for (size_t i=1;i<M*N;i+=2) {
    A_pack[i] = A[i] << shift;
  }
}

static void pack_qnnpack4x8multi_type2_rhs(uint8_t *W, uint8_t *W_pack, size_t M, size_t N, size_t bitw) {
  int p = 0;
  size_t shift = 8-bitw;
  for (size_t j=0;j<N;j+=8)
    for (size_t i=0;i<M;i+=2)
      for (size_t k=j;k<j+8;k++) {
        W_pack[p++] = W[i*N+k+N];
        W_pack[p++] = (W[i*N+k] << shift);
      }
}