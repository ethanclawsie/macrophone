// Per-op latency floor on ggml's Metal backend: a chain of N dependent tiny ops in one graph.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <chrono>
#include <cstdio>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    ggml_backend_t backend = ggml_backend_metal_init();
    for (int kind = 0; kind < 3; ++kind) {
        const int n = 64;
        ggml_init_params ip = { (size_t) (4*n + 16)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1280);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 1280, 1280);
        ggml_tensor * W8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 1280, 1280);
        ggml_tensor * cur = x;
        for (int i = 0; i < n; ++i) {
            if (kind == 0) cur = ggml_add(ctx, cur, x);                                  // elementwise
            else if (kind == 1) cur = ggml_mul_mat(ctx, W, ggml_reshape_2d(ctx, cur, 1280, 1));  // q4_0 gemv
            else cur = ggml_mul_mat(ctx, W8, ggml_reshape_2d(ctx, cur, 1280, 1));        // q8_0 gemv
            cur = ggml_reshape_1d(ctx, cur, 1280);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cur);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buf, 0);
        for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(backend, gf);
        const int iters = 50;
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
        const double ms = (now_ms() - t0)/iters;
        printf("%-22s chain of %d: %.3f ms/graph = %.1f us per dependent op\n",
               kind == 0 ? "add (1280 floats)" : kind == 1 ? "gemv 1280x1280 q4_0" : "gemv 1280x1280 q8_0", n, ms, 1000*ms/n);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
