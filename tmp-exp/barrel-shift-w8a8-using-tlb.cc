#include <arm_neon.h>
#include <iostream>

using namespace std;

void stepTLB(int16_t* i, int32_t* o, size_t rows, size_t columns, int8_t* scratchpad=nullptr){
    int16x8_t vBuffer0, vBuffer1, vBuffer2, vBuffer3,
              vBuffer4, vBuffer5, vBuffer6, vBuffer7;

    vBuffer0 = vld1q_s16(i + 0 * columns);
    vBuffer1 = vld1q_s16(i + 1 * columns);
    vBuffer2 = vld1q_s16(i + 2 * columns);
    vBuffer3 = vld1q_s16(i + 3 * columns);

    vBuffer4 = vld1q_s16(i + 4 * columns);
    vBuffer5 = vld1q_s16(i + 5 * columns);
    vBuffer6 = vld1q_s16(i + 6 * columns);
    vBuffer7 = vld1q_s16(i + 7 * columns);

    uint8x16_t vidxs, offset, offset_pos;
    int8x16x4_t vTABLE;
    int32x4_t vO;
    asm volatile(
        "mov w1, 0xFFFF\n\t"
        "mov w0, 0x0100\n\t"
        "ins %[vidxs].h[0], w0\n\t"
        "ins %[vidxs].h[1], w1\n\t"
        "mov w0, 0x1110\n\t"
        "ins %[vidxs].h[2], w0\n\t"
        "ins %[vidxs].h[3], w1\n\t"
        "mov w0, 0x2120\n\t"
        "ins %[vidxs].h[4], w0\n\t"
        "ins %[vidxs].h[5], w1\n\t"
        "mov w0, 0x3130\n\t"
        "ins %[vidxs].h[6], w0\n\t"
        "ins %[vidxs].h[7], w1\n\t"
        
        "mov w0, 0x02\n\t"
        "dup %[offset].16b, w0\n\t"

        "mov w1, 0x0000\n\t"
        "ins %[offset].h[1], w1\n\t"
        "ins %[offset].h[3], w1\n\t"
        "ins %[offset].h[5], w1\n\t"
        "ins %[offset].h[7], w1\n\t"
        :[ vidxs ]"=w"( vidxs ), [ offset_pos ]"=w"( offset_pos ), [ offset ]"=w"( offset )::"w0","w1"
    );

    vTABLE = { vreinterpretq_s8_s16(vBuffer0), vreinterpretq_s8_s16(vBuffer1),
               vreinterpretq_s8_s16(vBuffer2), vreinterpretq_s8_s16(vBuffer3) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer4), vreinterpretq_s8_s16(vBuffer5),
               vreinterpretq_s8_s16(vBuffer6), vreinterpretq_s8_s16(vBuffer7) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer7), vreinterpretq_s8_s16(vBuffer0),
               vreinterpretq_s8_s16(vBuffer1), vreinterpretq_s8_s16(vBuffer2) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer3), vreinterpretq_s8_s16(vBuffer4),
               vreinterpretq_s8_s16(vBuffer5), vreinterpretq_s8_s16(vBuffer6) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer6), vreinterpretq_s8_s16(vBuffer7),
               vreinterpretq_s8_s16(vBuffer0), vreinterpretq_s8_s16(vBuffer1) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer2), vreinterpretq_s8_s16(vBuffer3),
               vreinterpretq_s8_s16(vBuffer4), vreinterpretq_s8_s16(vBuffer5) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer5), vreinterpretq_s8_s16(vBuffer6),
               vreinterpretq_s8_s16(vBuffer7), vreinterpretq_s8_s16(vBuffer0) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer1), vreinterpretq_s8_s16(vBuffer2),
               vreinterpretq_s8_s16(vBuffer3), vreinterpretq_s8_s16(vBuffer4) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer4), vreinterpretq_s8_s16(vBuffer5),
               vreinterpretq_s8_s16(vBuffer6), vreinterpretq_s8_s16(vBuffer7) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer0), vreinterpretq_s8_s16(vBuffer1),
               vreinterpretq_s8_s16(vBuffer2), vreinterpretq_s8_s16(vBuffer3) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer3), vreinterpretq_s8_s16(vBuffer4),
               vreinterpretq_s8_s16(vBuffer5), vreinterpretq_s8_s16(vBuffer6) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer7), vreinterpretq_s8_s16(vBuffer0),
               vreinterpretq_s8_s16(vBuffer1), vreinterpretq_s8_s16(vBuffer2) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer2), vreinterpretq_s8_s16(vBuffer3),
               vreinterpretq_s8_s16(vBuffer4), vreinterpretq_s8_s16(vBuffer5) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer6), vreinterpretq_s8_s16(vBuffer7),
               vreinterpretq_s8_s16(vBuffer0), vreinterpretq_s8_s16(vBuffer1) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;

    vidxs = vaddq_u8(vidxs, offset);

    vTABLE = { vreinterpretq_s8_s16(vBuffer1), vreinterpretq_s8_s16(vBuffer2),
               vreinterpretq_s8_s16(vBuffer3), vreinterpretq_s8_s16(vBuffer4) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
    vTABLE = { vreinterpretq_s8_s16(vBuffer5), vreinterpretq_s8_s16(vBuffer6),
               vreinterpretq_s8_s16(vBuffer7), vreinterpretq_s8_s16(vBuffer0) };
    vO = vreinterpretq_s32_s8(vqtbl4q_s8(vTABLE, vidxs));
    vst1q_s32(o, vO); o += 4;
}

void stepOriginal(int16_t* in, int32_t* out, size_t rows, size_t columns, int8_t* scratchpad=nullptr){
    for (size_t i = 0 ; i < 8 ; i++)
        for (size_t j = 0 ; j < 8 ; j++){
            // cout << j * columns + ((j + i) % 8) << " <- " << i * columns + j << endl;
            out[j * columns + ((j + i) % 8)] = in[i * columns + j];
        }
}

template <typename T> void fillZero(T* pointer, size_t size){
    for (size_t i = 0 ; i < size ; i++) pointer[i] = 0;
}
template <typename T> void fillIncremental(T* pointer, size_t size){
    for (size_t i = 0 ; i < size ; i++) pointer[i] = i + 1;
}
template <typename T> void print(T* pointer, size_t rows, size_t columns, string name = "Matrix", bool print_in_hex=false){
    cout << name << " = [" << endl;
    if (print_in_hex)
        cout << hex;
    for (size_t i = 0 ; i < rows ; i++){
        cout << "\t[ ";
        for (size_t j = 0 ; j < columns ; j++) 
            cout << ((uint)pointer[i * columns + j]) << ", ";
        cout << "]," << endl;
    }
    if(print_in_hex)
        cout << dec;
    cout << "]" << endl;
}
template <typename Tf,
          typename Ts> bool areEqual(Tf* first, Ts* second, size_t rows, size_t columns){
    bool are_equal = true; 
    for (size_t i = 0 ; i < rows ; i++)
        for (size_t j = 0 ; j < columns ; j++)
            are_equal &= (first[i * columns + j] == second[i * columns + j]);
    return are_equal;
}

inline long double calculate_time_diff_seconds(timespec start, timespec end){
    long double tstart_ext_ld, tend_ext_ld, tdiff_ext_ld;
	tstart_ext_ld = (long double)start.tv_sec + start.tv_nsec/1.0e+9;
	tend_ext_ld = (long double)end.tv_sec + end.tv_nsec/1.0e+9;
	tdiff_ext_ld = (long double)(tend_ext_ld - tstart_ext_ld);
    return tdiff_ext_ld;
}

int main(){
    int16_t *i;
    int32_t *o_n, *o_o;
    int8_t t[4 * 16];
    size_t rows = 8, columns = 8, size = rows * columns, num_iterations = 1'000'000;

    i   = new int16_t[size];
    o_n = new int32_t[size];
    o_o = new int32_t[size];
    
    fillIncremental(i, size);
    fillZero(o_n, size);
    fillZero(o_o, size);

    stepTLB     (i, o_n, rows, columns, t);
    stepOriginal(i, o_o, rows, columns, t);

    if (!areEqual(o_n, o_o, rows, columns)){
        cout << "Sanity Check FAILED" << endl;
        print(i,   rows, columns, "i");
        print(o_o, rows, columns, "Output Original");
        print(o_n, rows, columns, "Output New");
    } else {
        cout << "Sanity Check PASS" << endl;
    }

    double tlb_time = 0, original_time = 0;

    {
        struct timespec tstart = {0,0},
                        tend = {0,0};
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
        for (int c = 0 ; c < num_iterations ; c++)
            stepTLB(i, o_n, rows, columns);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
        tlb_time = calculate_time_diff_seconds(tstart, tend);
    }
    {
        struct timespec tstart = {0,0},
                        tend = {0,0};
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
        for (int c = 0 ; c < num_iterations ; c++)
            stepOriginal(i, o_n, rows, columns);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
        original_time = calculate_time_diff_seconds(tstart, tend);
    }
    
    cout << "Original Time for " << num_iterations << " iterations: " << original_time << " seconds." << endl;
    cout << "   TLB   Time for " << num_iterations << " iterations: " << tlb_time << " seconds." << endl;
    return 0;
}