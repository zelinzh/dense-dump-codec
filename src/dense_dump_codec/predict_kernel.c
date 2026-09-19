#include <float.h>
#include <stddef.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "DDC requires IEEE binary32");

int ddc_predict_abi(void) {
    return 1;
}

void ddc_predict_add(size_t count, const float *restrict start,
                     const float *restrict end, const float *restrict residual,
                     float start_weight, float end_weight, float *restrict output) {
    for (size_t index = 0; index < count; ++index) {
        const float first = start_weight * start[index];
        const float second = end_weight * end[index];
        const float predicted = first + second;
        output[index] = predicted + residual[index];
    }
}
