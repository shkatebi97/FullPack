#ifndef CVECTOR_H
#include <arm_neon.h>

#define CVECTOR_STRING_VALUE(n) #n
#define CVECTOR_STRING(s) CVECTOR_STRING_UNEXPANDED(s)
#define CVECTOR_STRING_UNEXPANDED(s) #s

class v_uint64x1_t;
class v_uint64x2_t;

class v_uint32x2_t;
class v_uint32x4_t;

class v_uint16x4_t;
class v_uint16x8_t;

class v_uint8x8_t;
class v_uint8x16_t;

class v_int64x1_t;
class v_int64x2_t;

class v_int32x2_t;
class v_int32x4_t;

class v_int16x4_t;
class v_int16x8_t;

class v_int8x8_t;
class v_int8x16_t;

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

class v_uint64x1_t{
public:
    uint64x1_t v;
    v_uint64x1_t(){}
    v_uint64x1_t(v_uint64x1_t& _v):v(_v.v){}
    v_uint64x1_t(uint64x1_t _v):v(_v){}
    inline v_uint64x1_t operator+(v_uint64x1_t b);
    inline v_uint64x1_t operator+=(v_uint64x1_t const b);
    inline v_uint64x1_t& operator=(const int& a);
    inline v_uint64x1_t& operator=(const uint64_t* a);
    inline v_uint64x1_t& operator=(const uint64x1_t& a);
    inline v_uint64x1_t operator&=(v_uint64x1_t const b);
    inline v_uint64x1_t operator<<=(int const& n);
    inline v_uint64x1_t operator|(v_uint64x1_t const& b);
    inline v_uint64x1_t& operator()(uint64_t* p);
    inline operator int();
    inline operator uint64_t();
    inline operator uint64x1_t();
};
class v_uint64x2_t{
public:
    uint64x2_t v;
    v_uint64x1_t high();
    v_uint64x1_t low();
    v_uint64x2_t(){}
    v_uint64x2_t(v_uint64x2_t& _v):v(_v.v){}
    v_uint64x2_t(uint64x2_t _v):v(_v){}
    inline v_uint64x2_t operator+(v_uint64x2_t const b);
    inline v_uint64x2_t operator+=(v_uint64x2_t const b);
    inline v_uint64x2_t& operator=(const int& a);
    inline v_uint64x2_t& operator=(const uint64_t* a);
    inline v_uint64x2_t& operator=(const uint64x2_t& a);
    inline v_uint64x2_t operator&=(v_uint64x2_t const b);
    inline v_uint64x2_t operator<<=(int const& n);
    inline v_uint64x2_t operator|(v_uint64x2_t const& b);
    inline v_uint64x2_t& operator()(uint64_t* p);
    inline operator int();
    inline operator uint64_t();
    inline operator uint64x2_t();
};

class v_uint32x2_t{
public:
    uint32x2_t v;
    v_uint32x2_t(){}
    v_uint32x2_t(v_uint32x2_t& _v):v(_v.v){}
    v_uint32x2_t(uint32x2_t _v):v(_v){}
    inline v_uint64x2_t operator+(v_uint32x2_t const b);
    inline v_uint32x2_t operator+=(v_uint32x2_t const b);
    inline v_uint32x2_t& operator=(const int& a);
    inline v_uint32x2_t& operator=(const uint32_t* a);
    inline v_uint32x2_t& operator=(const uint32x2_t& a);
    inline v_uint32x2_t operator&=(v_uint32x2_t const b);
    inline v_uint32x2_t operator<<=(int const& n);
    inline v_uint32x2_t operator|(v_uint32x2_t const& b);
    inline v_uint32x2_t& operator()(uint32_t* p);
    inline v_uint64x2_t operator*(v_uint32x2_t& b);
    inline operator int();
    inline operator uint32_t();
    inline operator v_uint64x2_t();
    inline operator uint32x2_t();
};
class v_uint32x4_t{
public:
    uint32x4_t v;
    v_uint32x4_t& zero();
    v_uint32x2_t high();
    v_uint32x2_t low();
    v_uint32x4_t(){}
    v_uint32x4_t(v_uint32x4_t& _v):v(_v.v){}
    v_uint32x4_t(uint32x4_t _v):v(_v){}
    inline v_uint64x2_t operator+(v_uint32x4_t const b);
    inline v_uint32x4_t operator+=(v_uint32x4_t const b);
    inline v_uint32x4_t& operator+=(v_uint16x8_t const b);
    inline v_uint32x4_t& operator=(const int& a);
    inline v_uint32x4_t& operator=(const uint32_t* a);
    inline v_uint32x4_t& operator=(const uint32x4_t& a);
    inline v_uint32x4_t operator&=(v_uint32x4_t const b);
    inline v_uint32x4_t operator<<=(int const& n);
    inline v_uint32x4_t operator|(v_uint32x4_t const& b);
    inline v_uint32x4_t& operator()(uint32_t* p);
    inline operator int();
    inline operator uint32_t();
    inline operator v_uint64x2_t();
    inline operator uint32x4_t();
    inline v_uint32x4_t& MAC(v_uint16x4_t& a, v_uint16x4_t& b);
    template <int n> inline v_uint32x4_t& MAC_lane(v_uint16x4_t& a, v_uint16x4_t& b);
    template <int n> inline v_uint32x4_t& MAC_lane(v_uint16x4_t& a, v_uint16x8_t& b);
};

class v_uint16x4_t{
public:
    uint16x4_t v;
    v_uint16x4_t(){}
    v_uint16x4_t(v_uint16x4_t& _v):v(_v.v){}
    v_uint16x4_t(uint16x4_t _v):v(_v){}
    inline v_uint32x4_t operator+(v_uint16x4_t const b);
    inline v_uint16x4_t operator+=(v_uint16x4_t const b);
    inline v_uint16x4_t& operator=(const int& a);
    inline v_uint16x4_t& operator=(const uint16_t* a);
    inline v_uint16x4_t& operator=(const uint16x4_t& a);
    inline v_uint16x4_t operator&=(v_uint16x4_t const b);
    inline v_uint16x4_t operator&(v_uint16x4_t const b);
    inline v_uint16x4_t operator&(v_uint16x8_t& b);
    inline v_uint16x4_t operator<<=(int const& n);
    inline v_uint16x4_t operator>>=(int const& n);
    inline v_uint16x4_t operator|(v_uint16x4_t const& b);
    inline v_uint16x4_t& operator|=(v_uint16x4_t const& b);
    inline v_uint16x4_t& operator()(uint16_t* p);
    inline v_uint32x4_t operator*(v_uint16x4_t& b);
    inline operator int();
    inline operator uint16_t();
    inline operator v_uint32x4_t();
    inline operator uint16x4_t();
};
class v_uint16x8_t{
public:
    uint16x8_t v;
    v_uint16x8_t& zero();
    v_uint16x4_t high();
    v_uint16x4_t low();
    v_uint16x8_t& high(v_uint16x4_t& a);
    v_uint16x8_t& low(v_uint16x4_t& a);
    v_uint16x8_t(){}
    v_uint16x8_t(v_uint16x8_t& _v):v(_v.v){}
    v_uint16x8_t(uint16x8_t _v):v(_v){}
    inline v_uint32x4_t operator+(v_uint16x8_t const b);
    inline v_uint16x8_t operator+=(v_uint16x8_t const b);
    inline v_uint16x8_t& operator+=(v_uint8x16_t const b);
    inline v_uint16x8_t& operator=(const int& a);
    inline v_uint16x8_t& operator=(const uint16_t* a);
    inline v_uint16x8_t& operator=(const uint16x8_t& a);
    inline v_uint16x8_t& operator=(v_uint8x8_t& a);
    inline v_uint16x8_t operator&=(v_uint16x8_t const b);
    inline v_uint16x8_t operator&(v_uint16x8_t& b);
    inline v_uint16x8_t& operator<<=(int const& n);
    inline v_uint16x8_t& operator>>=(int const& n);
    inline v_uint16x8_t operator|(v_uint16x8_t const& b);
    inline v_uint16x8_t& operator()(uint16_t* p);
    inline operator int();
    inline operator uint16_t();
    inline operator v_uint32x4_t();
    inline operator uint16x8_t();
    inline v_uint16x8_t& MAC(v_uint8x8_t& a, v_uint8x8_t& b);
    template <int n> inline v_uint16x8_t& MAC_lane(v_uint8x8_t& a, v_uint8x8_t& b);
    template <int n> inline v_uint16x8_t& MAC_lane(v_uint8x8_t& a, v_uint8x16_t& b);
    inline v_uint32x4_t expand_high();
    inline v_uint32x4_t expand_low();
};

class v_uint8x8_t{
public:
    uint8x8_t v;
    v_uint8x8_t(){}
    v_uint8x8_t(v_uint8x8_t& _v):v(_v.v){}
    v_uint8x8_t(uint8x8_t _v):v(_v){}
    inline v_uint16x8_t operator+(v_uint8x8_t const b);
    inline v_uint8x8_t operator+=(v_uint8x8_t const b);
    inline v_uint8x8_t& operator=(const int& a);
    inline v_uint8x8_t& operator=(const uint8_t* a);
    inline v_uint8x8_t& operator=(const uint8x8_t& a);
    inline v_uint8x8_t operator&=(v_uint8x8_t const b);
    inline v_uint8x8_t operator&(v_uint8x16_t& b);
    inline v_uint8x8_t operator<<=(int const& n);
    inline v_uint8x8_t& operator>>=(int const& n);
    inline v_uint8x8_t operator>>(int const& n);
    inline v_uint8x8_t operator|(v_uint8x8_t const& b);
    inline v_uint8x8_t& operator()(uint8_t* p);
    inline v_uint16x8_t operator*(v_uint8x8_t& b);
    inline operator int();
    inline operator uint8_t();
    inline operator v_uint16x8_t();
    inline operator uint8x8_t();
};
class v_uint8x16_t{
public:
    uint8x16_t v;
    v_uint8x8_t high();
    v_uint8x8_t low();
    v_uint8x16_t(){}
    v_uint8x16_t(v_uint8x16_t& _v):v(_v.v){}
    v_uint8x16_t(uint8x16_t _v):v(_v){}
    inline v_uint16x8_t operator+(v_uint8x16_t const b);
    inline v_uint8x16_t operator+=(v_uint8x16_t const b);
    inline v_uint8x16_t& operator=(const int& a);
    inline v_uint8x16_t& operator=(const uint8_t* a);
    inline v_uint8x16_t& operator=(const uint8x16_t& a);
    inline v_uint8x16_t operator&=(v_uint8x16_t const b);
    inline v_uint8x16_t operator&(v_uint8x16_t& b);
    inline v_uint8x16_t operator<<=(int const& n);
    inline v_uint8x16_t& operator>>=(int const& n);
    inline v_uint8x16_t operator>>(int const& n);
    inline v_uint8x16_t operator|(v_uint8x16_t const& b);
    inline v_uint8x16_t& operator()(uint8_t* p);
    inline operator int();
    inline operator uint8_t();
    inline operator v_uint16x8_t();
    inline operator uint8x16_t();
};

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

inline v_uint64x1_t v_uint64x1_t::operator+(v_uint64x1_t b){
    return vadd_u64(v, b.v);
}
inline v_uint64x1_t v_uint64x1_t::operator+=(v_uint64x1_t const b){
    v = vadd_u64(v, b.v);
    return *this;
}
inline v_uint64x1_t& v_uint64x1_t::operator=(const int& a){
    this->v = vdup_n_u64(a);
    return *this;
}
inline v_uint64x1_t& v_uint64x1_t::operator=(const uint64_t* a){
    this->v = vld1_u64(a);
    return *this;
}
inline v_uint64x1_t& v_uint64x1_t::operator=(const uint64x1_t& a){
    this->v = a;
    return *this;
}
inline v_uint64x1_t v_uint64x1_t::operator&=(v_uint64x1_t const b){
    v = vand_u64(v, b.v);
    return *this;
}
inline v_uint64x1_t v_uint64x1_t::operator<<=(int const& n){
    v = vshl_n_u64(v, n);
    return *this;
}
inline v_uint64x1_t v_uint64x1_t::operator|(v_uint64x1_t const& b){
    return vorr_u64(v, b.v);
}
inline v_uint64x1_t& v_uint64x1_t::operator()(uint64_t* p){
    vst1_u64(p,v);
    return *this;
}
inline v_uint64x1_t::operator int(){
    return vget_lane_u64(v, 0); 
}
inline v_uint64x1_t::operator uint64_t(){
    return vget_lane_u64(v, 0);
}
inline v_uint64x1_t::operator uint64x1_t(){
    return v; 
}

inline v_uint64x1_t v_uint64x2_t::high(){
    return vget_high_u64(v);
}
inline v_uint64x1_t v_uint64x2_t::low(){
    return vget_low_u64(v);
}
inline v_uint64x2_t v_uint64x2_t::operator+(v_uint64x2_t const b){
    return vaddq_u64(v, b.v);
}
inline v_uint64x2_t v_uint64x2_t::operator+=(v_uint64x2_t const b){
    v = vaddq_u64(v, b.v);
    return *this;
}
inline v_uint64x2_t& v_uint64x2_t::operator=(const int& a){
    this->v = vdupq_n_u64(a);
    return *this;
}
inline v_uint64x2_t& v_uint64x2_t::operator=(const uint64_t* a){
    this->v = vld1q_u64(a);
    return *this;
}
inline v_uint64x2_t& v_uint64x2_t::operator=(const uint64x2_t& a){
    this->v = a;
    return *this;
}
inline v_uint64x2_t v_uint64x2_t::operator&=(v_uint64x2_t const b){
    v = vandq_u64(v, b.v);
    return *this;
}
inline v_uint64x2_t v_uint64x2_t::operator<<=(int const& n){
    v = vshlq_n_u64(v, n);
    return *this;
}
inline v_uint64x2_t v_uint64x2_t::operator|(v_uint64x2_t const& b){
    return vorrq_u64(v, b.v);
}
inline v_uint64x2_t& v_uint64x2_t::operator()(uint64_t* p){
    vst1q_u64(p,v);
    return *this;
}
inline v_uint64x2_t::operator int(){
    return vaddvq_u64(v); 
}
inline v_uint64x2_t::operator uint64_t(){
    return vaddvq_u64(v); 
}
inline v_uint64x2_t::operator uint64x2_t(){
    return v; 
}

inline v_uint64x2_t v_uint32x2_t::operator+(v_uint32x2_t const b){
    return vaddl_u32(v, b.v);
}
inline v_uint32x2_t v_uint32x2_t::operator+=(v_uint32x2_t const b){
    v = vadd_u32(v, b.v);
    return *this;
}
inline v_uint32x2_t& v_uint32x2_t::operator=(const int& a){
    this->v = vdup_n_u32(a);
    return *this;
}
inline v_uint32x2_t& v_uint32x2_t::operator=(const uint32_t* a){
    this->v = vld1_u32(a);
    return *this;
}
inline v_uint32x2_t& v_uint32x2_t::operator=(const uint32x2_t& a){
    this->v = a;
    return *this;
}
inline v_uint32x2_t v_uint32x2_t::operator&=(v_uint32x2_t const b){
    v = vand_u32(v, b.v);
    return *this;
}
inline v_uint32x2_t v_uint32x2_t::operator<<=(int const& n){
    v = vshl_n_u32(v, n);
    return *this;
}
inline v_uint32x2_t v_uint32x2_t::operator|(v_uint32x2_t const& b){
    return vorr_u32(v, b.v);
}
inline v_uint32x2_t& v_uint32x2_t::operator()(uint32_t* p){
    vst1_u32(p,v);
    return *this;
}
inline v_uint64x2_t v_uint32x2_t::operator*(v_uint32x2_t& b){
    return vmull_u32(v, b.v);
}
inline v_uint32x2_t::operator int(){
    return vaddv_u32(v); 
}
inline v_uint32x2_t::operator uint32_t(){
    return vaddv_u32(v); 
}
inline v_uint32x2_t::operator v_uint64x2_t(){
    return vmovl_u32(v);
}
inline v_uint32x2_t::operator uint32x2_t(){
    return v; 
}

inline v_uint32x4_t& v_uint32x4_t::zero(){
    v = veorq_u32(v, v);
    return *this;
}
inline v_uint32x2_t v_uint32x4_t::high(){
    return vget_high_u32(v);
}
inline v_uint32x2_t v_uint32x4_t::low(){
    return vget_low_u32(v);
}
inline v_uint64x2_t v_uint32x4_t::operator+(v_uint32x4_t const b){
    return vaddq_u64(vaddl_u32(vget_low_u32(v), vget_low_u32(b.v)), vaddl_u32(vget_high_u32(v), vget_high_u32(b.v)));
}
inline v_uint32x4_t v_uint32x4_t::operator+=(v_uint32x4_t const b){
    v = vaddq_u32(v, b.v);
    return *this;
}
inline v_uint32x4_t& v_uint32x4_t::operator+=(v_uint16x8_t const b){
    v = vpadalq_u16(v, b.v);
    return *this;
}
inline v_uint32x4_t& v_uint32x4_t::operator=(const int& a){
    this->v = vdupq_n_u32(a);
    return *this;
}
inline v_uint32x4_t& v_uint32x4_t::operator=(const uint32_t* a){
    this->v = vld1q_u32(a);
    return *this;
}
inline v_uint32x4_t& v_uint32x4_t::operator=(const uint32x4_t& a){
    this->v = a;
    return *this;
}
inline v_uint32x4_t v_uint32x4_t::operator&=(v_uint32x4_t const b){
    v = vandq_u32(v, b.v);
    return *this;
}
inline v_uint32x4_t v_uint32x4_t::operator<<=(int const& n){
    v = vshlq_n_u32(v, n);
    return *this;
}
inline v_uint32x4_t v_uint32x4_t::operator|(v_uint32x4_t const& b){
    return vorrq_u32(v, b.v);
}
inline v_uint32x4_t& v_uint32x4_t::operator()(uint32_t* p){
    vst1q_u32(p,v);
    return *this;
}
inline v_uint32x4_t::operator int(){
    return vaddvq_u32(v); 
}
inline v_uint32x4_t::operator uint32_t(){
    return vaddvq_u32(v); 
}
inline v_uint32x4_t::operator v_uint64x2_t(){
    return vpaddlq_u32(v);
}
inline v_uint32x4_t::operator uint32x4_t(){
    return v; 
}
inline v_uint32x4_t& v_uint32x4_t::MAC(v_uint16x4_t& a, v_uint16x4_t& b){
    v = vmlal_u16(v, a.v, b.v);
    return *this;
}
template <int n> inline v_uint32x4_t& v_uint32x4_t::MAC_lane(v_uint16x4_t& a, v_uint16x4_t& b){
    v = vmlal_lane_u16(v, a.v, b.v, n);
    return *this;
}
template <int n> inline v_uint32x4_t& v_uint32x4_t::MAC_lane(v_uint16x4_t& a, v_uint16x8_t& b){
    v = vmlal_laneq_u16(v, a.v, b.v, n);
    return *this;
}

inline v_uint32x4_t v_uint16x4_t::operator+(v_uint16x4_t const b){
    return vaddl_u16(v, b.v);
}
inline v_uint16x4_t v_uint16x4_t::operator+=(v_uint16x4_t const b){
    v = vadd_u16(v, b.v);
    return *this;
}
inline v_uint16x4_t& v_uint16x4_t::operator=(const int& a){
    this->v = vdup_n_u16(a);
    return *this;
}
inline v_uint16x4_t& v_uint16x4_t::operator=(const uint16_t* a){
    this->v = vld1_u16(a);
    return *this;
}
inline v_uint16x4_t& v_uint16x4_t::operator=(const uint16x4_t& a){
    this->v = a;
    return *this;
}
inline v_uint16x4_t v_uint16x4_t::operator&=(v_uint16x4_t const b){
    v = vand_u16(v, b.v);
    return *this;
}
inline v_uint16x4_t v_uint16x4_t::operator&(v_uint16x4_t const b){
    return vand_u16(v, b.v);
}
inline v_uint16x4_t v_uint16x4_t::operator&(v_uint16x8_t& b){
    uint16x4_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_uint16x4_t v_uint16x4_t::operator<<=(int const& n){
    v = vshl_n_u16(v, n);
    return *this;
}
inline v_uint16x4_t v_uint16x4_t::operator>>=(int const& n){
    v = vshr_n_u16(v, n);
    return *this;
}
inline v_uint16x4_t v_uint16x4_t::operator|(v_uint16x4_t const& b){
    return vorr_u16(v, b.v);
}
inline v_uint16x4_t& v_uint16x4_t::operator|=(v_uint16x4_t const& b){
    v = vorr_u16(v, b.v);
    return *this;
}
inline v_uint16x4_t& v_uint16x4_t::operator()(uint16_t* p){
    vst1_u16(p,v);
    return *this;
}
inline v_uint32x4_t v_uint16x4_t::operator*(v_uint16x4_t& b){
    return vmull_u16(v, b.v);
}
inline v_uint16x4_t::operator int(){
    return vaddv_u16(v); 
}
inline v_uint16x4_t::operator uint16_t(){
    return vaddv_u16(v); 
}
inline v_uint16x4_t::operator v_uint32x4_t(){
    return vmovl_u16(v);
}
inline v_uint16x4_t::operator uint16x4_t(){
    return v; 
}

inline v_uint16x8_t& v_uint16x8_t::zero(){
    v = veorq_u16(v, v);
    return *this;
}
inline v_uint16x4_t v_uint16x8_t::high(){
    return vget_high_u16(v);
}
inline v_uint16x4_t v_uint16x8_t::low(){
    return vget_low_u16(v);
}
inline v_uint16x8_t& v_uint16x8_t::high(v_uint16x4_t& a){
    asm volatile("mov %[v].d[1], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::low(v_uint16x4_t& a){
    asm volatile("mov %[v].d[0], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_uint32x4_t v_uint16x8_t::operator+(v_uint16x8_t const b){
    return vaddq_u32(vaddl_u16(vget_low_u16(v), vget_low_u16(b.v)), vaddl_u16(vget_high_u16(v), vget_high_u16(b.v)));
}
inline v_uint16x8_t v_uint16x8_t::operator+=(v_uint16x8_t const b){
    v = vaddq_u16(v, b.v);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator+=(v_uint8x16_t const b){
    v = vpadalq_u8(v, b.v);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator=(const int& a){
    this->v = vdupq_n_u16(a);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator=(const uint16_t* a){
    this->v = vld1q_u16(a);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator=(const uint16x8_t& a){
    this->v = a;
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator=(v_uint8x8_t& a){
    this->v = vmovl_u8(a.v);
    return *this;
}
inline v_uint16x8_t v_uint16x8_t::operator&=(v_uint16x8_t const b){
    v = vandq_u16(v, b.v);
    return *this;
}
inline v_uint16x8_t v_uint16x8_t::operator&(v_uint16x8_t& b){
    return vandq_u16(v, b.v);
}
inline v_uint16x8_t& v_uint16x8_t::operator<<=(int const& n){
    v = vshlq_n_u16(v, n);
    return *this;
}
inline v_uint16x8_t& v_uint16x8_t::operator>>=(int const& n){
    v = vshrq_n_u16(v, n);
    return *this;
}
inline v_uint16x8_t v_uint16x8_t::operator|(v_uint16x8_t const& b){
    return vorrq_u16(v, b.v);
}
inline v_uint16x8_t& v_uint16x8_t::operator()(uint16_t* p){
    vst1q_u16(p,v);
    return *this;
}
inline v_uint16x8_t::operator int(){
    return vaddvq_u16(v); 
}
inline v_uint16x8_t::operator uint16_t(){
    return vaddvq_u16(v); 
}
inline v_uint16x8_t::operator v_uint32x4_t(){
    return vpaddlq_u16(v);
}
inline v_uint16x8_t::operator uint16x8_t(){
    return v; 
}
inline v_uint16x8_t& v_uint16x8_t::MAC(v_uint8x8_t& a, v_uint8x8_t& b){
    v = vmlal_u8(v, a.v, b.v);
    return *this;
}
template <int n> inline v_uint16x8_t& v_uint16x8_t::MAC_lane(v_uint8x8_t& a, v_uint8x8_t& b){
    switch (n)
    {
    case 0:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[0]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 1:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[1]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 2:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[2]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 3:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[3]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 4:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[4]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 5:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[5]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 6:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[6]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 7:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[7]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    default:
        break;
    }
    return *this;
}
template <int n> inline v_uint16x8_t& v_uint16x8_t::MAC_lane(v_uint8x8_t& a, v_uint8x16_t& b){
    switch (n)
    {
    case 0:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[0]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 1:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[1]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 2:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[2]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 3:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[3]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 4:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[4]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 5:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[5]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 6:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[6]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 7:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[7]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 8:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[8]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 9:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[9]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 10:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[10]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 11:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[11]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 12:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[12]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 13:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[13]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 14:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[14]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 15:
        asm volatile("UMLAL %[v].8h, %[a].8b, %[b].b[15]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    default:
        break;
    }
    return *this;
}
inline v_uint32x4_t v_uint16x8_t::expand_high(){
    uint32x4_t o;
    asm volatile("USHLL2 %[o].4s, %[v].8h, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_uint32x4_t v_uint16x8_t::expand_low(){
    uint32x4_t o;
    asm volatile("USHLL %[o].4s, %[v].4h, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}

inline v_uint16x8_t v_uint8x8_t::operator+(v_uint8x8_t const b){
    return vaddl_u8(v, b.v);
}
inline v_uint8x8_t v_uint8x8_t::operator+=(v_uint8x8_t const b){
    v = vadd_u8(v, b.v);
    return *this;
}
inline v_uint8x8_t& v_uint8x8_t::operator=(const int& a){
    this->v = vdup_n_u8(a);
    return *this;
}
inline v_uint8x8_t& v_uint8x8_t::operator=(const uint8_t* a){
    this->v = vld1_u8(a);
    return *this;
}
inline v_uint8x8_t& v_uint8x8_t::operator=(const uint8x8_t& a){
    this->v = a;
    return *this;
}
inline v_uint8x8_t v_uint8x8_t::operator&=(v_uint8x8_t const b){
    v = vand_u8(v, b.v);
    return *this;
}
inline v_uint8x8_t v_uint8x8_t::operator&(v_uint8x16_t& b){
    uint8x8_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_uint8x8_t v_uint8x8_t::operator<<=(int const& n){
    v = vshl_n_u8(v, n);
    return *this;
}
inline v_uint8x8_t& v_uint8x8_t::operator>>=(int const& n){
    v = vshr_n_u8(v, n);
    return *this;
}
inline v_uint8x8_t v_uint8x8_t::operator>>(int const& n){
    return vshr_n_u8(v, n);
}
inline v_uint8x8_t v_uint8x8_t::operator|(v_uint8x8_t const& b){
    return vorr_u8(v, b.v);
}
inline v_uint8x8_t& v_uint8x8_t::operator()(uint8_t* p){
    vst1_u8(p,v);
    return *this;
}
inline v_uint16x8_t v_uint8x8_t::operator*(v_uint8x8_t& b){
    return vmull_u8(v, b.v);
}
inline v_uint8x8_t::operator int(){
    return vaddv_u8(v); 
}
inline v_uint8x8_t::operator uint8_t(){
    return vaddv_u8(v); 
}
inline v_uint8x8_t::operator v_uint16x8_t(){
    return vmovl_u8(v); 
}
inline v_uint8x8_t::operator uint8x8_t(){
    return v; 
}

inline v_uint8x8_t v_uint8x16_t::high(){
    return vget_high_u8(v);
}
inline v_uint8x8_t v_uint8x16_t::low(){
    return vget_low_u8(v);
}
inline v_uint16x8_t v_uint8x16_t::operator+(v_uint8x16_t const b){
    return vaddq_u16(vaddl_u8(vget_low_u8(v), vget_low_u8(b.v)), vaddl_u8(vget_high_u8(v), vget_high_u8(b.v)));
}
inline v_uint8x16_t v_uint8x16_t::operator+=(v_uint8x16_t const b){
    v = vaddq_u8(v, b.v);
    return *this;
}
inline v_uint8x16_t& v_uint8x16_t::operator=(const int& a){
    this->v = vdupq_n_u8(a);
    return *this;
}
inline v_uint8x16_t& v_uint8x16_t::operator=(const uint8_t* a){
    this->v = vld1q_u8(a);
    return *this;
}
inline v_uint8x16_t& v_uint8x16_t::operator=(const uint8x16_t& a){
    this->v = a;
    return *this;
}
inline v_uint8x16_t v_uint8x16_t::operator&=(v_uint8x16_t const b){
    v = vandq_u8(v, b.v);
    return *this;
}
inline v_uint8x16_t v_uint8x16_t::operator&(v_uint8x16_t& b){
    return vandq_u8(v, b.v);
}
inline v_uint8x16_t v_uint8x16_t::operator<<=(int const& n){
    v = vshlq_n_u8(v, n);
    return *this;
}
inline v_uint8x16_t& v_uint8x16_t::operator>>=(int const& n){
    v = vshrq_n_u8(v, n);
    return *this;
}
inline v_uint8x16_t v_uint8x16_t::operator>>(int const& n){
    return vshrq_n_u8(v, n);
}
inline v_uint8x16_t v_uint8x16_t::operator|(v_uint8x16_t const& b){
    return vorrq_u8(v, b.v);
}
inline v_uint8x16_t& v_uint8x16_t::operator()(uint8_t* p){
    vst1q_u8(p,v);
    return *this;
}
inline v_uint8x16_t::operator int(){
    return vaddvq_u8(v); 
}
inline v_uint8x16_t::operator uint8_t(){
    return vaddvq_u8(v); 
}
inline v_uint8x16_t::operator v_uint16x8_t(){
    return vpaddlq_u8(v); 
}
inline v_uint8x16_t::operator uint8x16_t(){
    return v; 
}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

class v_int64x1_t{
public:
    int64x1_t v;
    v_int64x1_t(){}
    v_int64x1_t(v_int64x1_t& _v):v(_v.v){}
    v_int64x1_t(int64x1_t _v):v(_v){}
    inline v_int64x1_t  operator+(v_int64x1_t const& b);
    inline v_int64x1_t& operator+=(v_int64x1_t const& b);
    inline v_int64x1_t& operator=(const int& a);
    inline v_int64x1_t& operator=(const int64_t* a);
    inline v_int64x1_t& operator=(const int64x1_t& a);
    inline v_int64x1_t& operator&=(v_int64x1_t const& b);
    inline v_int64x1_t& operator&=(v_int64x2_t const& b);
    inline v_int64x1_t  operator&(v_int64x1_t const& b);
    inline v_int64x1_t  operator&(v_int64x2_t const& b);
    inline v_int64x1_t& operator<<=(int const& n);
    inline v_int64x1_t& operator>>=(int const& n);
    inline v_int64x1_t  operator>>(int const& n);
    inline v_int64x1_t  operator<<(int const& n);
    inline v_int64x1_t& operator|=(v_int64x1_t const& b);
    inline v_int64x1_t& operator|=(v_int64x2_t const& b);
    inline v_int64x1_t  operator|(v_int64x1_t const& b);
    inline v_int64x1_t  operator|(v_int64x2_t const& b);
    inline v_int64x1_t& operator()(int64_t* p);
    inline operator int();
    inline operator int64_t();
    inline operator int64x1_t();
    inline v_int64x1_t& zero();
};
class v_int64x2_t{
public:
    int64x2_t v;
    v_int64x2_t(){}
    v_int64x2_t(v_int64x2_t& _v):v(_v.v){}
    v_int64x2_t(int64x2_t _v):v(_v){}
    inline v_int64x2_t  operator+(v_int64x2_t const& b);
    inline v_int64x2_t& operator+=(v_int64x2_t const& b);
    inline v_int64x2_t& operator+=(v_int32x4_t const& b);
    inline v_int64x2_t& operator=(const int& a);
    inline v_int64x2_t& operator=(const int64_t* a);
    inline v_int64x2_t& operator=(const int64x2_t& a);
    inline v_int64x2_t& operator=(const int32x2_t& a);
    inline v_int64x2_t& operator&=(v_int64x2_t const& b);
    inline v_int64x2_t  operator&(v_int64x2_t const& b);
    inline v_int64x2_t& operator<<=(int const& n);
    inline v_int64x2_t& operator>>=(int const& n);
    inline v_int64x2_t  operator>>(int const& n);
    inline v_int64x2_t  operator<<(int const& n);
    inline v_int64x2_t& operator|=(v_int64x2_t const& b);
    inline v_int64x2_t  operator|(v_int64x2_t const& b);
    inline v_int64x2_t& operator()(int64_t* p);
    inline operator int();
    inline operator int64_t();
    inline operator int64x2_t();
    inline v_int64x2_t& MAC(v_int32x2_t const& a, v_int32x2_t const& b);
    template <int n> inline v_int64x2_t& MAC_lane(v_int32x2_t const& a, v_int32x2_t const& b);
    template <int n> inline v_int64x2_t& MAC_lane(v_int32x2_t const& a, v_int32x4_t const& b);
    v_int64x2_t& zero();
    v_int64x1_t high();
    v_int64x1_t low();
    v_int64x2_t& high(v_int64x1_t const& a);
    v_int64x2_t& low(v_int64x1_t const& a);
};

class v_int32x2_t{
public:
    int32x2_t v;
    v_int32x2_t(){}
    v_int32x2_t(v_int32x2_t& _v):v(_v.v){}
    v_int32x2_t(int32x2_t _v):v(_v){}
    inline v_int64x2_t  operator+(v_int32x2_t const& b);
    inline v_int32x2_t& operator+=(v_int32x2_t const& b);
    inline v_int32x2_t& operator=(const int& a);
    inline v_int32x2_t& operator=(const int32_t* a);
    inline v_int32x2_t& operator=(const int32x2_t& a);
    inline v_int32x2_t& operator&=(v_int32x2_t const& b);
    inline v_int32x2_t& operator&=(v_int32x4_t const& b);
    inline v_int32x2_t  operator&(v_int32x2_t const& b);
    inline v_int32x2_t  operator&(v_int32x4_t const& b);
    inline v_int32x2_t& operator<<=(int const& n);
    inline v_int32x2_t& operator>>=(int const& n);
    inline v_int32x2_t  operator>>(int const& n);
    inline v_int32x2_t  operator<<(int const& n);
    inline v_int32x2_t& operator|=(v_int32x2_t const& b);
    inline v_int32x2_t& operator|=(v_int32x4_t const& b);
    inline v_int32x2_t  operator|(v_int32x2_t const& b);
    inline v_int32x2_t  operator|(v_int32x4_t const& b);
    inline v_int32x2_t& operator()(int32_t* p);
    inline v_int64x2_t  operator*(v_int32x2_t& b);
    inline operator int();
    inline operator v_int64x2_t();
    inline operator int32x2_t();
    inline v_int32x2_t& zero();
};
class v_int32x4_t{
public:
    int32x4_t v;
    v_int32x4_t(){}
    v_int32x4_t(v_int32x4_t& _v):v(_v.v){}
    v_int32x4_t(int32x4_t _v):v(_v){}
    inline v_int64x2_t  operator+(v_int32x4_t const& b);
    inline v_int32x4_t& operator+=(v_int32x4_t const& b);
    inline v_int32x4_t& operator+=(v_int16x8_t const& b);
    inline v_int32x4_t& operator=(const int& a);
    inline v_int32x4_t& operator=(const int32_t* a);
    inline v_int32x4_t& operator=(const int32x4_t& a);
    inline v_int32x4_t& operator=(const int16x4_t& a);
    inline v_int32x4_t& operator&=(v_int32x4_t const& b);
    inline v_int32x4_t  operator&(v_int32x4_t const& b);
    inline v_int32x4_t& operator<<=(int const& n);
    inline v_int32x4_t& operator>>=(int const& n);
    inline v_int32x4_t  operator>>(int const& n);
    inline v_int32x4_t  operator<<(int const& n);
    inline v_int32x4_t& operator|=(v_int32x4_t const& b);
    inline v_int32x4_t  operator|(v_int32x4_t const& b);
    inline v_int32x4_t& operator()(int32_t* p);
    inline operator int();
    inline operator v_int64x2_t();
    inline operator int32x4_t();
    inline v_int32x4_t& MAC(v_int16x4_t& a, v_int16x4_t& b);
    template <int n> inline v_int32x4_t& MAC_lane(v_int16x4_t& a, v_int16x4_t& b);
    template <int n> inline v_int32x4_t& MAC_lane(v_int16x4_t& a, v_int16x8_t& b);
    inline v_int64x2_t expand_high();
    inline v_int64x2_t expand_low();
    v_int32x4_t& zero();
    v_int32x2_t high();
    v_int32x2_t low();
    inline v_int32x4_t& high(v_int32x2_t const& a);
    inline v_int32x4_t& low(v_int32x2_t const& a);
};

class v_int16x4_t{
public:
    int16x4_t v;
    v_int16x4_t(){}
    v_int16x4_t(v_int16x4_t& _v):v(_v.v){}
    v_int16x4_t(int16x4_t _v):v(_v){}
    inline v_int32x4_t  operator+(v_int16x4_t const& b);
    inline v_int16x4_t& operator+=(v_int16x4_t const& b);
    inline v_int16x4_t& operator=(const int& a);
    inline v_int16x4_t& operator=(const int16_t* a);
    inline v_int16x4_t& operator=(const int16x4_t& a);
    inline v_int16x4_t& operator&=(v_int16x4_t const& b);
    inline v_int16x4_t& operator&=(v_int16x8_t const& b);
    inline v_int16x4_t  operator&(v_int16x4_t const& b);
    inline v_int16x4_t  operator&(v_int16x8_t& b);
    inline v_int16x4_t& operator<<=(int const& n);
    inline v_int16x4_t& operator>>=(int const& n);
    inline v_int16x4_t  operator>>(int const& n);
    inline v_int16x4_t  operator<<(int const& n);
    inline v_int16x4_t& operator|=(v_int16x4_t const& b);
    inline v_int16x4_t& operator|=(v_int16x8_t const& b);
    inline v_int16x4_t  operator|(v_int16x4_t const& b);
    inline v_int16x4_t  operator|(v_int16x8_t const& b);
    inline v_int16x4_t& operator()(int16_t* p);
    inline v_int32x4_t  operator*(v_int16x4_t& b);
    inline operator int();
    inline operator int16_t();
    inline operator v_int32x4_t();
    inline operator int16x4_t();
    inline v_int16x4_t& zero();
};
class v_int16x8_t{
public:
    int16x8_t v;
    v_int16x8_t(){}
    v_int16x8_t(int16x8_t _v):v(_v){}
    v_int16x8_t(v_int16x8_t& _v):v(_v.v){}
    inline v_int32x4_t  operator+(v_int16x8_t const& b);
    inline v_int16x8_t& operator+=(v_int16x8_t const& b);
    inline v_int16x8_t& operator+=(v_int8x16_t const& b);
    inline v_int16x8_t& operator=(const int& a);
    inline v_int16x8_t& operator=(const int16_t* a);
    inline v_int16x8_t& operator=(const int16x8_t& a);
    inline v_int16x8_t& operator=(v_int8x8_t const& a);
    inline v_int16x8_t& operator&=(v_int16x8_t const& b);
    inline v_int16x8_t  operator&(v_int16x8_t& b);
    inline v_int16x8_t& operator<<=(int const& n);
    inline v_int16x8_t& operator>>=(int const& n);
    inline v_int16x8_t  operator>>(int const& n);
    inline v_int16x8_t  operator<<(int const& n);
    inline v_int16x8_t& operator|=(v_int16x8_t const& b);
    inline v_int16x8_t operator|(v_int16x8_t const& b);
    inline v_int16x8_t& operator()(int16_t* p);
    inline operator int();
    inline operator int16_t();
    inline operator v_int32x4_t();
    inline operator int16x8_t();
    inline v_int16x8_t& MAC(v_int8x8_t& a, v_int8x8_t& b);
    template <int n> inline v_int16x8_t& MAC_lane(v_int8x8_t& a, v_int8x8_t& b);
    template <int n> inline v_int16x8_t& MAC_lane(v_int8x8_t& a, v_int8x16_t& b);
    inline v_int32x4_t expand_high();
    inline v_int32x4_t expand_low();
    inline v_int16x8_t& zero();
    inline v_int16x4_t high();
    inline v_int16x4_t low();
    inline v_int16x8_t& high(v_int16x4_t& a);
    inline v_int16x8_t& low(v_int16x4_t& a);
};

class v_int8x8_t{
public:
    int8x8_t v;
    v_int8x8_t(){}
    v_int8x8_t(v_int8x8_t& _v):v(_v.v){}
    v_int8x8_t(int8x8_t _v):v(_v){}
    inline v_int16x8_t operator+(v_int8x8_t const& b);
    inline v_int8x8_t& operator+=(v_int8x8_t const& b);
    inline v_int8x8_t& operator=(const int& a);
    inline v_int8x8_t& operator=(const int8_t* a);
    inline v_int8x8_t& operator=(const int8x8_t& a);
    inline v_int8x8_t& operator&=(v_int8x8_t const& b);
    inline v_int8x8_t& operator&=(v_int8x16_t const& b);
    inline v_int8x8_t  operator&(v_int8x8_t& b);
    inline v_int8x8_t  operator&(v_int8x16_t& b);
    inline v_int8x8_t& operator<<=(int const& n);
    inline v_int8x8_t& operator>>=(int const& n);
    inline v_int8x8_t  operator<<(int const& n);
    inline v_int8x8_t  operator>>(int const& n);
    inline v_int8x8_t& operator|=(v_int8x8_t const& b);
    inline v_int8x8_t& operator|=(v_int8x16_t const& b);
    inline v_int8x8_t operator|(v_int8x8_t const& b);
    inline v_int8x8_t operator|(v_int8x16_t const& b);
    inline v_int8x8_t& operator()(int8_t* p);
    inline v_int16x8_t operator*(v_int8x8_t& b);
    inline operator int();
    inline operator int8_t();
    inline operator v_int16x8_t();
    inline operator int8x8_t();
    inline v_int8x8_t& zero();
    template <int lane> inline v_int8x8_t& fill(v_int8x16_t& a);
};
class v_int8x16_t{
public:
    int8x16_t v;
    v_int8x16_t(){}
    v_int8x16_t(v_int8x16_t& _v):v(_v.v){}
    v_int8x16_t(int8x16_t _v):v(_v){}
    inline v_int16x8_t  operator+(v_int8x16_t const& b);
    inline v_int8x16_t& operator+=(v_int8x16_t const& b);
    inline v_int8x16_t& operator=(const int& a);
    inline v_int8x16_t& operator=(const int8_t* a);
    inline v_int8x16_t& operator=(const int8x16_t& a);
    inline v_int8x16_t& operator&=(v_int8x16_t const& b);
    inline v_int8x16_t  operator&(v_int8x16_t& b);
    inline v_int8x16_t& operator<<=(int const& n);
    inline v_int8x16_t& operator>>=(int const& n);
    inline v_int8x16_t  operator>>(int const& n);
    inline v_int8x16_t  operator<<(int const& n);
    inline v_int8x16_t& operator|=(v_int8x16_t const& b);
    inline v_int8x16_t  operator|(v_int8x16_t const& b);
    inline v_int8x16_t& operator()(int8_t* p);
    inline operator int();
    inline operator int8_t();
    inline operator v_int16x8_t();
    inline operator int8x16_t();
    inline v_int16x8_t expand_high();
    inline v_int16x8_t expand_low();
    inline v_int8x16_t& zero();
    v_int8x8_t high();
    v_int8x8_t low();
    inline v_int8x16_t& high(v_int8x8_t const& a);
    inline v_int8x16_t& low(v_int8x8_t const& a);
    template <int lane> inline v_int8x16_t& fill(v_int8x16_t& a);
};

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

inline v_int64x1_t& v_int64x1_t::operator&=(v_int64x2_t const& b){
    asm volatile("and %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int64x1_t  v_int64x1_t::operator&(v_int64x1_t const& b){
    return vand_s64(v, b.v);
}
inline v_int64x1_t  v_int64x1_t::operator&(v_int64x2_t const& b){
    int64x1_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int64x1_t& v_int64x1_t::operator>>=(int const& n){
    v = vshr_n_s64(v, n);
    return *this;
}
inline v_int64x1_t  v_int64x1_t::operator>>(int const& n){
    return vshr_n_s64(v, n);
}
inline v_int64x1_t  v_int64x1_t::operator<<(int const& n){
    return vshl_n_s64(v, n);
}
inline v_int64x1_t& v_int64x1_t::operator|=(v_int64x1_t const& b){
    v = vorr_s64(v, b.v);
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator|=(v_int64x2_t const& b){
    asm volatile("orr %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int64x1_t  v_int64x1_t::operator|(v_int64x2_t const& b){
    int64x1_t o;
    asm volatile("orr %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int64x1_t& v_int64x1_t::zero(){
    v = veor_s64(v, v);
    return *this;
}
inline v_int64x1_t  v_int64x1_t::operator+(v_int64x1_t const& b){
    return vadd_s64(v, b.v);
}
inline v_int64x1_t& v_int64x1_t::operator+=(v_int64x1_t const& b){
    v = vadd_s64(v, b.v);
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator=(const int& a){
    this->v = vdup_n_s64(a);
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator=(const int64_t* a){
    this->v = vld1_s64(a);
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator=(const int64x1_t& a){
    this->v = a;
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator&=(v_int64x1_t const& b){
    v = vand_s64(v, b.v);
    return *this;
}
inline v_int64x1_t& v_int64x1_t::operator<<=(int const& n){
    v = vshl_n_s64(v, n);
    return *this;
}
inline v_int64x1_t  v_int64x1_t::operator|(v_int64x1_t const& b){
    return vorr_s64(v, b.v);
}
inline v_int64x1_t& v_int64x1_t::operator()(int64_t* p){
    vst1_s64(p,v);
    return *this;
}
inline v_int64x1_t::operator int(){
    return vget_lane_s64(v, 0); 
}
inline v_int64x1_t::operator int64_t(){
    return vget_lane_s64(v, 0);
}
inline v_int64x1_t::operator int64x1_t(){
    return v; 
}

inline v_int64x2_t& v_int64x2_t::operator+=(v_int32x4_t const& b){
    v = vpadalq_s32(v, b.v);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator=(const int32x2_t& a){
    v = vmovl_s32(a);
    return *this;
}
inline v_int64x2_t  v_int64x2_t::operator&(v_int64x2_t const& b){
    return vandq_s64(v, b.v);
}
inline v_int64x2_t& v_int64x2_t::operator>>=(int const& n){
    v = vshrq_n_s64(v, n);
    return *this;
}
inline v_int64x2_t  v_int64x2_t::operator>>(int const& n){
    return vshrq_n_s64(v, n);
}
inline v_int64x2_t  v_int64x2_t::operator<<(int const& n){
    return vshlq_n_s64(v, n);
}
inline v_int64x2_t& v_int64x2_t::operator|=(v_int64x2_t const& b){
    v = vorrq_s64(v, b.v);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::MAC(v_int32x2_t const& a, v_int32x2_t const& b){
    v = vmlal_s32(v, a.v, b.v);
    return *this;
}
template <int n> inline v_int64x2_t& v_int64x2_t::MAC_lane(v_int32x2_t const& a, v_int32x2_t const& b){
    v = vmlal_lane_s32(v, a.v, b.v, n);
    return *this;
}
template <int n> inline v_int64x2_t& v_int64x2_t::MAC_lane(v_int32x2_t const& a, v_int32x4_t const& b){
    v = vmlal_laneq_s32(v, a.v, b.v, n);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::zero(){
    v = veorq_s64(v, v);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::high(v_int64x1_t const& a){
    asm volatile("mov %[v].d[1], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::low(v_int64x1_t const& a){
    asm volatile("mov %[v].d[0], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int64x1_t  v_int64x2_t::high(){
    return vget_high_s64(v);
}
inline v_int64x1_t  v_int64x2_t::low(){
    return vget_low_s64(v);
}
inline v_int64x2_t  v_int64x2_t::operator+(v_int64x2_t const& b){
    return vaddq_s64(v, b.v);
}
inline v_int64x2_t& v_int64x2_t::operator+=(v_int64x2_t const& b){
    v = vaddq_s64(v, b.v);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator=(const int& a){
    v = vdupq_n_s64(a);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator=(const int64_t* a){
    v = vld1q_s64(a);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator=(const int64x2_t& a){
    v = a;
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator&=(v_int64x2_t const& b){
    v = vandq_s64(v, b.v);
    return *this;
}
inline v_int64x2_t& v_int64x2_t::operator<<=(int const& n){
    v = vshlq_n_s64(v, n);
    return *this;
}
inline v_int64x2_t  v_int64x2_t::operator|(v_int64x2_t const& b){
    return vorrq_s64(v, b.v);
}
inline v_int64x2_t& v_int64x2_t::operator()(int64_t* p){
    vst1q_s64(p,v);
    return *this;
}
inline v_int64x2_t::operator int(){
    return vaddvq_s64(v); 
}
inline v_int64x2_t::operator int64_t(){
    return vaddvq_s64(v); 
}
inline v_int64x2_t::operator int64x2_t(){
    return v; 
}

inline v_int32x2_t& v_int32x2_t::operator&=(v_int32x4_t const& b){
    asm volatile("and %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int32x2_t  v_int32x2_t::operator&(v_int32x2_t const& b){
    return vand_s32(v, b.v);
}
inline v_int32x2_t  v_int32x2_t::operator&(v_int32x4_t const& b){
    int32x2_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int32x2_t& v_int32x2_t::operator>>=(int const& n){
    v = vshr_n_s32(v, n);
    return *this;
}
inline v_int32x2_t  v_int32x2_t::operator>>(int const& n){
    return vshr_n_s32(v, n);
}
inline v_int32x2_t  v_int32x2_t::operator<<(int const& n){
    return vshl_n_s32(v, n);
}
inline v_int32x2_t& v_int32x2_t::operator|=(v_int32x4_t const& b){
    asm volatile("orr %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator|=(v_int32x2_t const& b){
    v = vorr_s32(v, b.v);
    return *this;
}
inline v_int32x2_t  v_int32x2_t::operator|(v_int32x4_t const& b){
    int32x2_t o;
    asm volatile("orr %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int32x2_t& v_int32x2_t::zero(){
    v = veor_s32(v, v);
    return *this;
}
inline v_int64x2_t  v_int32x2_t::operator+(v_int32x2_t const& b){
    return vaddl_s32(v, b.v);
}
inline v_int32x2_t& v_int32x2_t::operator+=(v_int32x2_t const& b){
    v = vadd_s32(v, b.v);
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator=(const int& a){
    this->v = vdup_n_s32(a);
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator=(const int32_t* a){
    this->v = vld1_s32(a);
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator=(const int32x2_t& a){
    this->v = a;
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator&=(v_int32x2_t const& b){
    v = vand_s32(v, b.v);
    return *this;
}
inline v_int32x2_t& v_int32x2_t::operator<<=(int const& n){
    v = vshl_n_s32(v, n);
    return *this;
}
inline v_int32x2_t  v_int32x2_t::operator|(v_int32x2_t const& b){
    return vorr_s32(v, b.v);
}
inline v_int32x2_t& v_int32x2_t::operator()(int32_t* p){
    vst1_s32(p,v);
    return *this;
}
inline v_int64x2_t  v_int32x2_t::operator*(v_int32x2_t& b){
    return vmull_s32(v, b.v);
}
inline v_int32x2_t::operator int(){
    return vaddv_s32(v); 
}
inline v_int32x2_t::operator v_int64x2_t(){
    return vmovl_s32(v);
}
inline v_int32x2_t::operator int32x2_t(){
    return v; 
}

inline v_int32x4_t& v_int32x4_t::operator=(const int16x4_t& a){
    v = vmovl_s16(a);
    return *this;
}
inline v_int32x4_t  v_int32x4_t::operator&(v_int32x4_t const& b){
    return vandq_s32(v, b.v);
}
inline v_int32x4_t& v_int32x4_t::operator>>=(int const& n){
    v = vshrq_n_s32(v, n);
    return *this;
}
inline v_int32x4_t  v_int32x4_t::operator>>(int const& n){
    return vshrq_n_s32(v, n);
}
inline v_int32x4_t  v_int32x4_t::operator<<(int const& n){
    return vshlq_n_s32(v, n);
}
inline v_int32x4_t& v_int32x4_t::operator|=(v_int32x4_t const& b){
    v = vorrq_s32(v, b.v);
    return *this;
}
inline v_int64x2_t  v_int32x4_t::expand_high(){
    int64x2_t o;
    asm volatile("SSHLL2 %[o].2d, %[v].4s, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_int64x2_t  v_int32x4_t::expand_low(){
    int64x2_t o;
    asm volatile("SSHLL %[o].2d, %[v].2s, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_int32x4_t& v_int32x4_t::high(v_int32x2_t const& a){
    asm volatile("mov %[v].d[1], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::low(v_int32x2_t const& a){
    asm volatile("mov %[v].d[0], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::zero(){
    v = veorq_s32(v, v);
    return *this;
}
inline v_int32x2_t  v_int32x4_t::high(){
    return vget_high_s32(v);
}
inline v_int32x2_t  v_int32x4_t::low(){
    return vget_low_s32(v);
}
inline v_int64x2_t  v_int32x4_t::operator+(v_int32x4_t const& b){
    return vaddq_s64(vaddl_s32(vget_low_s32(v), vget_low_s32(b.v)), vaddl_s32(vget_high_s32(v), vget_high_s32(b.v)));
}
inline v_int32x4_t& v_int32x4_t::operator+=(v_int32x4_t const& b){
    v = vaddq_s32(v, b.v);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator+=(v_int16x8_t const& b){
    v = vpadalq_s16(v, b.v);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator=(const int& a){
    v = vdupq_n_s32(a);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator=(const int32_t* a){
    v = vld1q_s32(a);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator=(const int32x4_t& a){
    v = a;
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator&=(v_int32x4_t const& b){
    v = vandq_s32(v, b.v);
    return *this;
}
inline v_int32x4_t& v_int32x4_t::operator<<=(int const& n){
    v = vshlq_n_s32(v, n);
    return *this;
}
inline v_int32x4_t  v_int32x4_t::operator|(v_int32x4_t const& b){
    return vorrq_s32(v, b.v);
}
inline v_int32x4_t& v_int32x4_t::operator()(int32_t* p){
    vst1q_s32(p,v);
    return *this;
}
inline v_int32x4_t::operator int(){
    return vaddvq_s32(v); 
}
inline v_int32x4_t::operator v_int64x2_t(){
    return vpaddlq_s32(v);
}
inline v_int32x4_t::operator int32x4_t(){
    return v; 
}
inline v_int32x4_t& v_int32x4_t::MAC(v_int16x4_t& a, v_int16x4_t& b){
    v = vmlal_s16(v, a.v, b.v);
    return *this;
}
template <int n> inline v_int32x4_t& v_int32x4_t::MAC_lane(v_int16x4_t& a, v_int16x4_t& b){
    v = vmlal_lane_s16(v, a.v, b.v, n);
    return *this;
}
template <int n> inline v_int32x4_t& v_int32x4_t::MAC_lane(v_int16x4_t& a, v_int16x8_t& b){
    v = vmlal_laneq_s16(v, a.v, b.v, n);
    return *this;
}

inline v_int16x4_t& v_int16x4_t::operator&=(v_int16x8_t const& b){
    asm volatile("and %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int16x4_t  v_int16x4_t::operator>>(int const& n){
    return vshr_n_s16(v, n);
}
inline v_int16x4_t  v_int16x4_t::operator<<(int const& n){
    return vshl_n_s16(v, n);
}
inline v_int16x4_t& v_int16x4_t::operator|=(v_int16x8_t const& b){
    asm volatile("orr %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int16x4_t  v_int16x4_t::operator|(v_int16x8_t const& b){
    int16x4_t o;
    asm volatile("orr %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int16x4_t& v_int16x4_t::zero(){
    v = veor_s16(v, v);
    return *this;
}
inline v_int32x4_t  v_int16x4_t::operator+(v_int16x4_t const& b){
    return vaddl_s16(v, b.v);
}
inline v_int16x4_t& v_int16x4_t::operator+=(v_int16x4_t const& b){
    v = vadd_s16(v, b.v);
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator=(const int& a){
    v = vdup_n_s16(a);
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator=(const int16_t* a){
    v = vld1_s16(a);
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator=(const int16x4_t& a){
    v = a;
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator&=(v_int16x4_t const& b){
    v = vand_s16(v, b.v);
    return *this;
}
inline v_int16x4_t  v_int16x4_t::operator&(v_int16x4_t const& b){
    return vand_s16(v, b.v);
}
inline v_int16x4_t  v_int16x4_t::operator&(v_int16x8_t& b){
    int16x4_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int16x4_t& v_int16x4_t::operator<<=(int const& n){
    v = vshl_n_s16(v, n);
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator>>=(int const& n){
    v = vshr_n_s16(v, n);
    return *this;
}
inline v_int16x4_t  v_int16x4_t::operator|(v_int16x4_t const& b){
    return vorr_s16(v, b.v);
}
inline v_int16x4_t& v_int16x4_t::operator|=(v_int16x4_t const& b){
    v = vorr_s16(v, b.v);
    return *this;
}
inline v_int16x4_t& v_int16x4_t::operator()(int16_t* p){
    vst1_s16(p,v);
    return *this;
}
inline v_int32x4_t  v_int16x4_t::operator*(v_int16x4_t& b){
    return vmull_s16(v, b.v);
}
inline v_int16x4_t::operator int(){
    return vaddv_s16(v); 
}
inline v_int16x4_t::operator int16_t(){
    return vaddv_s16(v); 
}
inline v_int16x4_t::operator v_int32x4_t(){
    return vmovl_s16(v);
}
inline v_int16x4_t::operator int16x4_t(){
    return v; 
}

inline v_int16x8_t  v_int16x8_t::operator>>(int const& n){
    return vshrq_n_s16(v, n);
}
inline v_int16x8_t  v_int16x8_t::operator<<(int const& n){
    return vshlq_n_s16(v, n);
}
inline v_int16x8_t& v_int16x8_t::operator|=(v_int16x8_t const& b){
    v = vorrq_s16(v, b.v);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::zero(){
    v = veorq_s16(v, v);
    return *this;
}
inline v_int16x4_t  v_int16x8_t::high(){
    return vget_high_s16(v);
}
inline v_int16x4_t  v_int16x8_t::low(){
    return vget_low_s16(v);
}
inline v_int16x8_t& v_int16x8_t::high(v_int16x4_t& a){
    asm volatile("mov %[v].d[1], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::low(v_int16x4_t& a){
    asm volatile("mov %[v].d[0], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int32x4_t  v_int16x8_t::operator+(v_int16x8_t const& b){
    return vaddq_s32(vaddl_s16(vget_low_s16(v), vget_low_s16(b.v)), vaddl_s16(vget_high_s16(v), vget_high_s16(b.v)));
}
inline v_int16x8_t& v_int16x8_t::operator+=(v_int16x8_t const& b){
    v = vaddq_s16(v, b.v);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator+=(v_int8x16_t const& b){
    v = vpadalq_s8(v, b.v);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator=(const int& a){
    this->v = vdupq_n_s16(a);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator=(const int16_t* a){
    this->v = vld1q_s16(a);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator=(const int16x8_t& a){
    this->v = a;
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator=(v_int8x8_t const& a){
    this->v = vmovl_s8(a.v);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator&=(v_int16x8_t const& b){
    v = vandq_s16(v, b.v);
    return *this;
}
inline v_int16x8_t  v_int16x8_t::operator&(v_int16x8_t& b){
    return vandq_s16(v, b.v);
}
inline v_int16x8_t& v_int16x8_t::operator<<=(int const& n){
    v = vshlq_n_s16(v, n);
    return *this;
}
inline v_int16x8_t& v_int16x8_t::operator>>=(int const& n){
    v = vshrq_n_s16(v, n);
    return *this;
}
inline v_int16x8_t  v_int16x8_t::operator|(v_int16x8_t const& b){
    return vorrq_s16(v, b.v);
}
inline v_int16x8_t& v_int16x8_t::operator()(int16_t* p){
    vst1q_s16(p,v);
    return *this;
}
inline v_int16x8_t::operator int(){
    return vaddvq_s16(v); 
}
inline v_int16x8_t::operator int16_t(){
    return vaddvq_s16(v); 
}
inline v_int16x8_t::operator v_int32x4_t(){
    return vpaddlq_s16(v);
}
inline v_int16x8_t::operator int16x8_t(){
    return v; 
}
inline v_int16x8_t& v_int16x8_t::MAC(v_int8x8_t& a, v_int8x8_t& b){
    v = vmlal_s8(v, a.v, b.v);
    return *this;
}
template <int n> inline v_int16x8_t& v_int16x8_t::MAC_lane(v_int8x8_t& a, v_int8x8_t& b){
    switch (n)
    {
    case 0:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[0]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 1:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[1]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 2:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[2]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 3:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[3]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 4:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[4]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 5:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[5]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 6:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[6]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 7:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[7]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    default:
        break;
    }
    return *this;
}
template <int n> inline v_int16x8_t& v_int16x8_t::MAC_lane(v_int8x8_t& a, v_int8x16_t& b){
    switch (n)
    {
    case 0:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[0]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 1:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[1]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 2:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[2]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 3:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[3]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 4:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[4]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 5:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[5]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 6:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[6]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 7:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[7]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 8:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[8]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 9:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[9]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 10:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[10]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 11:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[11]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 12:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[12]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 13:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[13]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 14:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[14]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    case 15:
        asm volatile("SMLAL %[v].8h, %[a].8b, %[b].b[15]":[v]"=w"(v):[a]"w"(a.v),[b]"w"(b.v):);
        break;
    default:
        break;
    }
    return *this;
}
inline v_int32x4_t v_int16x8_t::expand_high(){
    int32x4_t o;
    asm volatile("SSHLL2 %[o].4s, %[v].8h, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_int32x4_t v_int16x8_t::expand_low(){
    int32x4_t o;
    asm volatile("SSHLL %[o].4s, %[v].4h, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}

inline v_int8x8_t& v_int8x8_t::operator&=(v_int8x16_t const& b){
    asm volatile("and %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int8x8_t  v_int8x8_t::operator&(v_int8x8_t& b){
    return vand_s8(v, b.v);
}
inline v_int8x8_t  v_int8x8_t::operator<<(int const& n){
    return vshl_n_s8(v, n);
}
inline v_int8x8_t& v_int8x8_t::operator|=(v_int8x8_t const& b){
    v = vorr_s8(v, b.v);
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator|=(v_int8x16_t const& b){
    asm volatile("orr %[v].8b, %[v].8b, %[b].8b":[v]"=w"(v):[b]"w"(b.v):);
    return *this;
}
inline v_int8x8_t  v_int8x8_t::operator|(v_int8x16_t const& b){
    int8x8_t o;
    asm volatile("orr %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int8x8_t& v_int8x8_t::zero(){
    v = veor_s8(v, v);
    return *this;
}
inline v_int16x8_t v_int8x8_t::operator+(v_int8x8_t const& b){
    return vaddl_s8(v, b.v);
}
inline v_int8x8_t& v_int8x8_t::operator+=(v_int8x8_t const& b){
    v = vadd_s8(v, b.v);
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator=(const int& a){
    this->v = vdup_n_s8(a);
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator=(const int8_t* a){
    this->v = vld1_s8(a);
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator=(const int8x8_t& a){
    this->v = a;
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator&=(v_int8x8_t const& b){
    v = vand_s8(v, b.v);
    return *this;
}
inline v_int8x8_t  v_int8x8_t::operator&(v_int8x16_t& b){
    int8x8_t o;
    asm volatile("and %[o].8b, %[v].8b, %[b].8b":[o]"=w"(o):[v]"w"(v),[b]"w"(b.v):);
    return o;
}
inline v_int8x8_t& v_int8x8_t::operator<<=(int const& n){
    v = vshl_n_s8(v, n);
    return *this;
}
inline v_int8x8_t& v_int8x8_t::operator>>=(int const& n){
    v = vshr_n_s8(v, n);
    return *this;
}
inline v_int8x8_t  v_int8x8_t::operator>>(int const& n){
    return vshr_n_s8(v, n);
}
inline v_int8x8_t  v_int8x8_t::operator|(v_int8x8_t const& b){
    return vorr_s8(v, b.v);
}
inline v_int8x8_t& v_int8x8_t::operator()(int8_t* p){
    vst1_s8(p,v);
    return *this;
}
inline v_int16x8_t v_int8x8_t::operator*(v_int8x8_t& b){
    return vmull_s8(v, b.v);
}
inline v_int8x8_t::operator int(){
    return vaddv_s8(v); 
}
inline v_int8x8_t::operator int8_t(){
    return vaddv_s8(v); 
}
inline v_int8x8_t::operator v_int16x8_t(){
    return vmovl_s8(v); 
}
inline v_int8x8_t::operator int8x8_t(){
    return v; 
}
template <int lane> 
inline v_int8x8_t& v_int8x8_t::fill(v_int8x16_t& a){
    v = vdup_laneq_s8(a.v, lane);
    return *this;
}

inline v_int8x16_t  v_int8x16_t::operator<<(int const& n){
    return vshlq_n_s8(v, n);
}
inline v_int8x16_t& v_int8x16_t::operator|=(v_int8x16_t const& b){
    v = vorrq_s8(v, b.v);
    return *this;
}
inline v_int16x8_t  v_int8x16_t::expand_high(){
    int16x8_t o;
    asm volatile("SSHLL2 %[o].8h, %[v].16b, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_int16x8_t  v_int8x16_t::expand_low(){
    int16x8_t o;
    asm volatile("SSHLL %[o].8h, %[v].8b, #0":[o]"=w"(o):[v]"w"(v):);
    return o;
}
inline v_int8x16_t& v_int8x16_t::zero(){
    v = veorq_s8(v, v);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::high(v_int8x8_t const& a){
    asm volatile("mov %[v].d[1], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::low(v_int8x8_t const& a){
    asm volatile("mov %[v].d[0], %[a].d[0]":[v]"=w"(v):[a]"w"(a.v):);
    return *this;
}
inline v_int8x8_t   v_int8x16_t::high(){
    return vget_high_s8(v);
}
inline v_int8x8_t   v_int8x16_t::low(){
    return vget_low_s8(v);
}
inline v_int16x8_t  v_int8x16_t::operator+(v_int8x16_t const& b){
    return vaddq_s16(vaddl_s8(vget_low_s8(v), vget_low_s8(b.v)), vaddl_s8(vget_high_s8(v), vget_high_s8(b.v)));
}
inline v_int8x16_t& v_int8x16_t::operator+=(v_int8x16_t const& b){
    v = vaddq_s8(v, b.v);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::operator=(const int& a){
    v = vdupq_n_s8(a);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::operator=(const int8_t* a){
    v = vld1q_s8(a);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::operator=(const int8x16_t& a){
    v = a;
    return *this;
}
inline v_int8x16_t& v_int8x16_t::operator&=(v_int8x16_t const& b){
    v = vandq_s8(v, b.v);
    return *this;
}
inline v_int8x16_t  v_int8x16_t::operator&(v_int8x16_t& b){
    return vandq_s8(v, b.v);
}
inline v_int8x16_t& v_int8x16_t::operator<<=(int const& n){
    v = vshlq_n_s8(v, n);
    return *this;
}
inline v_int8x16_t& v_int8x16_t::operator>>=(int const& n){
    v = vshrq_n_s8(v, n);
    return *this;
}
inline v_int8x16_t  v_int8x16_t::operator>>(int const& n){
    return vshrq_n_s8(v, n);
}
inline v_int8x16_t  v_int8x16_t::operator|(v_int8x16_t const& b){
    return vorrq_s8(v, b.v);
}
inline v_int8x16_t& v_int8x16_t::operator()(int8_t* p){
    vst1q_s8(p,v);
    return *this;
}
inline v_int8x16_t::operator int(){
    return vaddvq_s8(v); 
}
inline v_int8x16_t::operator int8_t(){
    return vaddvq_s8(v); 
}
inline v_int8x16_t::operator v_int16x8_t(){
    return vpaddlq_s8(v); 
}
inline v_int8x16_t::operator int8x16_t(){
    return v; 
}
template <int lane> 
inline v_int8x16_t& v_int8x16_t::fill(v_int8x16_t& a){
    v = vdupq_laneq_s8(a.v, lane);
    return *this;
}

#define CVECTOR_H
#endif // CVECTOR_H