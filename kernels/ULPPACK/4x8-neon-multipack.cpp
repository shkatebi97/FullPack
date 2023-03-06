/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>
#include "test.h"

static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter8(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
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
  int32x4_t vacc0x4567 = vld1q_s32(c0+4);
  int32x4_t vacc1x0123 = vld1q_s32(c1);
  int32x4_t vacc1x4567 = vld1q_s32(c1+4);
  int32x4_t vacc2x0123= vld1q_s32(c2);
  int32x4_t vacc2x4567= vld1q_s32(c2+4);
  int32x4_t vacc3x0123= vld1q_s32(c3);
  int32x4_t vacc3x4567= vld1q_s32(c3+4);

  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  uint16x8_t vacc0, vacc1, vacc2, vacc3;
  uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
  uint16x8_t vacc1x = vacc0x;
  uint16x8_t vacc2x = vacc0x;
  uint16x8_t vacc3x = vacc0x;

  for (; k >= 16; k -= 16) {
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
    
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

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

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
  vacc1x0123 = vaddq_s32(vacc1x0123,  vreinterpretq_s32_u32(vacc1x0123_));
  vacc2x0123 = vaddq_s32(vacc2x0123,  vreinterpretq_s32_u32(vacc2x0123_));
  vacc3x0123 = vaddq_s32(vacc3x0123,  vreinterpretq_s32_u32(vacc3x0123_));
  vacc0x4567 = vaddq_s32(vacc0x4567,  vreinterpretq_s32_u32(vacc0x4567_));
  vacc1x4567 = vaddq_s32(vacc1x4567,  vreinterpretq_s32_u32(vacc1x4567_));
  vacc2x4567 = vaddq_s32(vacc2x4567,  vreinterpretq_s32_u32(vacc2x4567_));
  vacc3x4567 = vaddq_s32(vacc3x4567,  vreinterpretq_s32_u32(vacc3x4567_));
  
  vst1q_s32(c0, vacc0x0123);
  vst1q_s32(c0+4, vacc0x4567);
  vst1q_s32(c1, vacc1x0123);
  vst1q_s32(c1+4, vacc1x4567);
  vst1q_s32(c2, vacc2x0123);
  vst1q_s32(c2+4, vacc2x4567);
  vst1q_s32(c3, vacc3x0123);
  vst1q_s32(c3+4, vacc3x4567);
}



static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter4(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
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
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
    
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

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

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
  
  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0),vacc0x0123));
  vst1q_s32(c0+4, vaddq_s32(vld1q_s32(c0+4),vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1),vacc1x0123));
  vst1q_s32(c1+4, vaddq_s32(vld1q_s32(c1+4),vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2),vacc2x0123));
  vst1q_s32(c2+4, vaddq_s32(vld1q_s32(c2+4),vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3),vacc3x0123));
  vst1q_s32(c3+4, vaddq_s32(vld1q_s32(c3+4),vacc3x4567));
}



static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter2(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
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
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
    
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

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

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
  
  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0),vacc0x0123));
  vst1q_s32(c0+4, vaddq_s32(vld1q_s32(c0+4),vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1),vacc1x0123));
  vst1q_s32(c1+4, vaddq_s32(vld1q_s32(c1+4),vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2),vacc2x0123));
  vst1q_s32(c2+4, vaddq_s32(vld1q_s32(c2+4),vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3),vacc3x0123));
  vst1q_s32(c3+4, vaddq_s32(vld1q_s32(c3+4),vacc3x4567));
}



static void pytorch_q8gemm_ukernel_4x8__neon_multipack_iter1(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* __restrict__ a,
    size_t a_stride,
    const uint8_t* __restrict__ w,
    int32_t* __restrict__ c,
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
    uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    uint16x4_t vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    uint16x4_t vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    uint16x4_t vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
    
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

    vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;
    vxa1 = vld1_u16((uint16_t*)a1); a1 += 8;
    vxa2 = vld1_u16((uint16_t*)a2); a2 += 8;
    vxa3 = vld1_u16((uint16_t*)a3); a3 += 8;

    vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
    vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

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
  
  vst1q_s32(c0, vaddq_s32(vld1q_s32(c0),vacc0x0123));
  vst1q_s32(c0+4, vaddq_s32(vld1q_s32(c0+4),vacc0x4567));
  vst1q_s32(c1, vaddq_s32(vld1q_s32(c1),vacc1x0123));
  vst1q_s32(c1+4, vaddq_s32(vld1q_s32(c1+4),vacc1x4567));
  vst1q_s32(c2, vaddq_s32(vld1q_s32(c2),vacc2x0123));
  vst1q_s32(c2+4, vaddq_s32(vld1q_s32(c2+4),vacc2x4567));
  vst1q_s32(c3, vaddq_s32(vld1q_s32(c3),vacc3x0123));
  vst1q_s32(c3+4, vaddq_s32(vld1q_s32(c3+4),vacc3x4567));
}

static size_t iter_cnt[][7] = {
  {8,8,8,8,4,2,1},
  {8,8,4,2,1,0,0},
  {8,4,2,1,0,0,0},
  {8,2,1,0,0,0,0},
  {4,1,0,0,0,0,0},
  {2,0,0,0,0,0,0},
  {1,0,0,0,0,0,0}
};

static void pack_qnnpack4x8multi(uint8_t *W, uint8_t *W_pack, size_t M, size_t N) {
  int p = 0;
  for (size_t j=0;j<N;j+=8)
    for (size_t i=0;i<M;i+=2)
      for (size_t k=j;k<j+8;k++) {
        W_pack[p++] = W[i*N+k+N];
        W_pack[p++] = W[i*N+k];
      }
}

std::pair<double,double> calc_qnnpack4x8multi(uint8_t *A, uint8_t *B_before_pack, int32_t *C, size_t M, size_t K, size_t N, size_t Wb, size_t Ab) {
  if (Wb>7 || Ab>7) return NOT_SUPPORTED;

  size_t mr_block_size = 4, nr_block_size = 8, kr_block_size = 512;
  size_t iter = iter_cnt[Wb-1][Ab-1];

  void (*ukernel)(size_t,size_t,size_t,const uint8_t*,size_t,const uint8_t*,int32_t*,size_t);
  if (iter == 8) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter8;
  else if (iter == 4) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter4;
  else if (iter == 2) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter2;
  else if (iter == 1) ukernel = pytorch_q8gemm_ukernel_4x8__neon_multipack_iter1;
  else return NOT_SUPPORTED;

  mat_initialize(C,M,N,-1);

  uint8_t *B;
  mat_new(&B,K,N);

  double t_elapsed = 0, t_pack = 0.;
  
  TIMEIT(t_pack, pack_qnnpack4x8multi(B_before_pack,B,K,N););
  TIMEIT(t_elapsed,
    do {
      for (size_t mr_block_start = 0; mr_block_start < M; mr_block_start += mr_block_size) {
        for (size_t nr_block_start = 0; nr_block_start < N; nr_block_start += nr_block_size) {
          for (size_t kr_block_start = 0; kr_block_start < K; kr_block_start += kr_block_size) {
            ukernel(
              mr_block_size,
              nr_block_size,
              std::min(kr_block_size, K-kr_block_start),
              A + mr_block_start * K + kr_block_start,
              K,
              B + nr_block_start * K + nr_block_size * kr_block_start,
              C + mr_block_start * N + nr_block_start,
              N
            );
          }
        }
      }
    } while(0);
  )

  mat_del(&B);
  return {t_elapsed,t_pack};
}
