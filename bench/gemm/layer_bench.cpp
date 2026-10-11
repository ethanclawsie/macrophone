// Profiles one Whisper large-v3-turbo encoder layer on ggml's Metal backend, op by op, by timing graphs
// that contain only that op (repeated). Shapes match whisper.cpp's encoder graph (flash attention path).
//
// Usage: layer_bench [q4_0|q8_0|f16] [n_frames] [n_keys]
//   n_frames = queries (frames computed), n_keys = keys attended (1500 for full / silence-tail mode)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_backend_t backend;

// Builds `reps` copies of the op (independent outputs), times the graph, returns ms per op.
static double time_op(const char * name, int reps, double flops,
                      std::function<ggml_tensor *(ggml_context *, int)> build) {
    ggml_init_params ip = { (size_t) 4096*ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    for (int r = 0; r < reps; ++r) ggml_build_forward_expand(gf, build(ctx, r));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_clear(buf, 0);
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);
    const int iters = 6;
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
    const double ms = (now_ms() - t0)/(iters*reps);
    printf("  %-22s %8.4f ms   %s\n", name, ms, flops > 0 ? (std::to_string(flops/(ms*1e-3)/1e12).substr(0, 5) + " TFLOPS").c_str() : "");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms;
}

int main(int argc, char ** argv) {
    const std::string tname = argc > 1 ? argv[1] : "q4_0";
    const int N  = argc > 2 ? std::atoi(argv[2]) : 1500; // frames computed
    const int NK = argc > 3 ? std::atoi(argv[3]) : 1500; // keys attended
    const ggml_type wtype = tname == "f16" ? GGML_TYPE_F16 : tname == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    const int D = 1280, H = 20, DH = 64, F = 5120;
    const int NKP = (NK + 255)/256*256;
    const int reps = 8;

    backend = ggml_backend_metal_init();
    { // ramp GPU clocks
        ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, F), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1024)));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buf, 0);
        const double t0 = now_ms();
        while (now_ms() - t0 < 500) ggml_backend_graph_compute(backend, gf);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    printf("encoder layer, weights %s, %d frames, %d keys (padded %d)\n", tname.c_str(), N, NK, NKP);
    double total = 0;
    auto x  = [&](ggml_context * c) { return ggml_new_tensor_2d(c, GGML_TYPE_F32, D, N); };
    auto w  = [&](ggml_context * c, int rows, int cols) { return ggml_new_tensor_2d(c, wtype, cols, rows); };
    auto v1 = [&](ggml_context * c, int n) { return ggml_new_tensor_1d(c, GGML_TYPE_F32, n); };

    // norm + affine (x2 per layer: attention LN, MLP LN)
    total += 2*time_op("layernorm (x2)", reps, 0, [&](ggml_context * c, int) {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x(c), 1e-5f), v1(c, D)), v1(c, D));
    });
    // Q, K, V, O projections (+ bias add)
    total += 4*time_op("q/k/v/o proj (x4)", reps, 2.0*D*D*N, [&](ggml_context * c, int) {
        return ggml_add(c, ggml_mul_mat(c, w(c, D, D), x(c)), v1(c, D));
    });
    // fused alternative for Q, K, V: one [3D x D] matmul + one bias add (compare with 3x the line above)
    time_op("qkv fused (x1 = 3 proj)", reps, 2.0*3*D*D*N, [&](ggml_context * c, int) {
        return ggml_add(c, ggml_mul_mat(c, w(c, 3*D, D), x(c)), v1(c, 3*D));
    });
    // K/V copies into the padded F16 buffer (x2)
    total += 2*time_op("k/v cpy to f16 (x2)", reps, 0, [&](ggml_context * c, int) {
        return ggml_cpy(c, x(c), ggml_new_tensor_2d(c, GGML_TYPE_F16, D, N));
    });
    // flash attention: Q [DH, N, H] over K/V [DH, NKP, H] with an F16 mask
    total += time_op("flash attention", reps, 4.0*N*NK*D, [&](ggml_context * c, int) {
        ggml_tensor * q = ggml_permute(c, ggml_reshape_3d(c, x(c), DH, H, N), 0, 2, 1, 3);
        ggml_tensor * k = ggml_new_tensor_3d(c, GGML_TYPE_F16, DH, NKP, H);
        ggml_tensor * v = ggml_new_tensor_3d(c, GGML_TYPE_F16, DH, NKP, H);
        ggml_tensor * m = NKP != NK || NK != N ? ggml_new_tensor_2d(c, GGML_TYPE_F16, NKP, N) : nullptr;
        return ggml_flash_attn_ext(c, q, k, v, m, 1.0f/sqrtf(DH), 0.0f, 0.0f);
    });
    // residual adds (x2)
    total += 2*time_op("residual add (x2)", reps, 0, [&](ggml_context * c, int) { return ggml_add(c, x(c), x(c)); });
    // MLP
    total += time_op("mlp up (+bias)", reps, 2.0*D*F*N, [&](ggml_context * c, int) {
        return ggml_add(c, ggml_mul_mat(c, w(c, F, D), x(c)), v1(c, F));
    });
    total += time_op("gelu", reps, 0, [&](ggml_context * c, int) {
        return ggml_gelu(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, F, N));
    });
    total += time_op("mlp down (+bias)", reps, 2.0*D*F*N, [&](ggml_context * c, int) {
        return ggml_add(c, ggml_mul_mat(c, w(c, D, F), ggml_new_tensor_2d(c, GGML_TYPE_F32, F, N)), v1(c, D));
    });
    printf("  %-22s %8.4f ms   x32 layers = %.1f ms\n", "layer total", total, 32*total);

    ggml_backend_free(backend);
    return 0;
}
