#include <gem5/m5ops.h>
#include <cassert>
int main(){
    asm volatile ("nop\n\t"::);

    m5_dump_reset_stats(0, 0);

    asm volatile ("nop\n\t"::);

    m5_switch_cpu();

    asm volatile ("nop\n\t"::);

    assert(10 == 11);

    asm volatile ("nop\n\t"::);

    return 0;
}