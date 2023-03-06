#include<cstddef>
#include<cstdint>
#include<chrono>
#include<random>

using namespace std::chrono;

#define TIMEIT(t_elapsed,x) do {\
  high_resolution_clock::time_point __t1, __t2; \
  __t1 = high_resolution_clock::now(); \
  x \
  __t2 = high_resolution_clock::now(); \
  t_elapsed += duration_cast<duration<double>>(__t2 - __t1).count(); \
} while(0);

#define NOT_SUPPORTED std::pair<double,double>(1e18,1e18)

std::pair<double,double> calc_naive(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb = 8, size_t Ab = 8);

std::pair<double,double> calc_gemmlowp(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb = 8, size_t Ab = 8);

std::pair<double,double> calc_cgo_bitserial(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t a_bits, size_t b_bits);

std::pair<double,double> calc_bitserial(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t a_bits, size_t b_bits);

std::pair<double,double> calc_qnnpack8x8(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb = 8, size_t Ab = 8);

std::pair<double,double> calc_qnnpack4x8multi(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb, size_t Ab);

std::pair<double,double> calc_qnnpack4x8multi_type2(uint8_t *A_before_pack, uint8_t *B_before_pack, int32_t *C, size_t M, size_t K, size_t N, size_t Wb, size_t Ab);

// double calc_qnnpack4x8(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb = 8, size_t Ab = 8);

// double calc_qnnpack(uint8_t *A, uint8_t *B, int32_t *C, size_t M, size_t K, size_t N, size_t Wb = 8, size_t Ab = 8);


static std::random_device rd;
static std::mt19937 gen(rd());

template<typename T>
void mat_new(T **A, size_t M, size_t N) {*A = new T[M*N];}

template<typename T>
void mat_del(T **A) {delete[] *A;}

template<typename T>
void mat_initialize(T *A, size_t M, size_t N, size_t bitw=8, T value=0) {
  if (bitw==-1) {
    for (size_t i=0;i<M;i++) for (size_t j=0;j<N;j++) A[i*N+j] = (T)value;
  }
  else {
    std::uniform_int_distribution<int> dis(0,(1<<bitw)-1);
    for (size_t i=0;i<M;i++) for (size_t j=0;j<N;j++) A[i*N+j] = (T)dis(gen);
  }
}

template<typename T, typename S>
bool mat_equal(T *A, S *B, size_t M, size_t N) {
  for (size_t i=0;i<M;i++) for (size_t j=0;j<N;j++) if ((int)A[i*N+j]!=(int)B[i*N+j]) return false;
  return true;
}

template<typename T>
void mat_print(T *A, size_t M, size_t N) {
  for (size_t i=0;i<M;i++) {
    for (size_t j=0;j<N;j++) printf("%3d ",(int)A[i*N+j]);
    puts("");
  }
  puts("");
}

template<typename T>
void mat_transpose(T *A, T *A_packed, size_t M, size_t N) {
  for (size_t i=0;i<M;i++) {
    for (size_t j=0;j<N;j++) {
      A_packed[j*M+i] = A[i*N+j];
    }
  }
}