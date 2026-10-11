// Decoder-step matrix-vector products (one token) on ggml's Metal backend: effective weight bandwidth.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <chrono>
#include <cstdio>
#include <string>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const std::string tname = argc > 1 ? argv[1] : "q4_0";
    const ggml_type wtype = tname == "f16" ? GGML_TYPE_F16 : tname == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    ggml_backend_t backend = ggml_backend_metal_init();
    struct S { int rows, cols; const char * what; } shapes[] = {
        {1280, 1280, "attn proj"}, {5120, 1280, "mlp up"}, {1280, 5120, "mlp down"}, {51866, 1280, "logits"} };
    for (auto s : shapes) {
        const int reps = 16;
        ggml_init_params ip = { (size_t) (reps*3 + 8)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int r = 0; r < reps; ++r) {
            // distinct weights per rep so nothing stays cached
            ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, s.cols, s.rows);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.cols, 1);
            ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W, x));
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buf, 0);
        for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(backend, gf);
        const int iters = 20;
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
        const double ms = (now_ms() - t0)/(iters*reps);
        const double bytes = (double) ggml_row_size(wtype, s.cols)*s.rows;
        printf("%-10s %6d x %-5d %8.4f ms  %6.1f GB/s\n", s.what, s.rows, s.cols, ms, bytes/(ms*1e-3)/1e9);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
