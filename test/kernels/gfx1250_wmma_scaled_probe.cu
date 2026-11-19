using v8i = int __attribute__((ext_vector_type(8)));
using v16i = int __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));

extern "C" __device__ __attribute__((always_inline, used)) v8f
gfx1250_wmma_scaled_probe(v8i a, v16i b, v8f c, int scale_a, int scale_b) {
    v8f out;
    asm volatile(
        "v_wmma_scale_f32_16x16x128_f8f6f4 "
        "%0, %1, %2, %3, %4, %5 matrix_a_fmt:MATRIX_FMT_FP4"
        : "=v"(out)
        : "v"(a), "v"(b), "v"(c), "v"(scale_a), "v"(scale_b)
    );
    return out;
}
