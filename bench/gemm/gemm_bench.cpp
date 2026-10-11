// GEMM micro-benchmark for the Whisper encoder's matmul shapes on ggml's Metal backend.
//
//   y[M, N] = W[M, K] · x[K, N]   (W = weights, x = activations, N = encoder frames)
//
// Build: see bench/gemm/build.sh. Usage: gemm_bench [q8_0|q4_0|f16] [reps]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const std::string tname = argc > 1 ? argv[1] : "q8_0";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 16;
    const bool act_f16 = argc > 3 && std::string(argv[3]) == "f16";
    const ggml_type xtype = act_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    const ggml_type wtype = tname == "f16" ? GGML_TYPE_F16 : tname == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "no Metal backend\n");
        return 1;
    }

    struct Shape { int M, K; const char * what; };
    const Shape shapes[] = {
        {1280, 1280, "qkvo"},
        {5120, 1280, "mlp_up"},
        {1280, 5120, "mlp_down"},
    };
    const int frames[] = {64, 128, 256, 512, 1024, 1500};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    // Ramp GPU clocks up before measuring anything (~0.5 s of matmuls).
    {
        ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1280, 5120);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, 1024);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, a, b));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buf, 0);
        const double t0 = now_ms();
        while (now_ms() - t0 < 500) ggml_backend_graph_compute(backend, gf);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    printf("weights %s, activations %s, %d matmuls per graph\n", tname.c_str(), act_f16 ? "f16" : "f32", reps);
    printf("%-14s %6s %6s %6s %10s %10s\n", "shape", "M", "K", "N", "ms/matmul", "TFLOPS");

    for (const Shape & s : shapes) {
        for (int N : frames) {
            ggml_init_params ip = { (size_t) (reps*4 + 8)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);

            ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, s.K, s.M);
            std::vector<ggml_tensor *> xs;
            for (int r = 0; r < reps; ++r) xs.push_back(ggml_new_tensor_2d(ctx, xtype, s.K, N));

            ggml_cgraph * gf = ggml_new_graph(ctx);
            for (int r = 0; r < reps; ++r) ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W, xs[r]));

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

            // weights: random floats, quantized to wtype
            {
                std::vector<float> wf((size_t) s.M*s.K);
                for (auto & v : wf) v = dist(rng);
                std::vector<uint8_t> wq(ggml_nbytes(W));
                if (wtype == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) wq.data(), (int64_t) wf.size());
                } else {
                    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, s.M, s.K, nullptr);
                }
                ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
            }
            {
                std::vector<float> xf((size_t) s.K*N);
                for (auto & v : xf) v = dist(rng);
                std::vector<ggml_fp16_t> xh(xf.size());
                ggml_fp32_to_fp16_row(xf.data(), xh.data(), (int64_t) xf.size());
                for (ggml_tensor * x : xs) ggml_backend_tensor_set(x, act_f16 ? (const void *) xh.data() : (const void *) xf.data(), 0, ggml_nbytes(x));
            }

            for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);

            const int iters = 8;
            const double t0 = now_ms();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
            const double ms = (now_ms() - t0)/(iters*reps);
            const double tflops = 2.0*s.M*s.K*N/(ms*1e-3)/1e12;

            printf("%-14s %6d %6d %6d %10.4f %10.2f\n", s.what, s.M, s.K, N, ms, tflops);

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }

    ggml_backend_free(backend);
    return 0;
}
