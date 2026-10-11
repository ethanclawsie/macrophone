// whisper_bench.cpp - latency / accuracy harness for stock whisper.cpp (libwhisper).
//
// Protocol: load the model once, decode every audio file to 16 kHz mono float32 up front
// (AudioToolbox ExtAudioFile, outside timing), run warm-up transcriptions, then time only
// whisper_full() (wall clock, steady_clock) and read the segment text afterwards.
//
// Output rows (no header):
//   latency: clip_name \t run_index(1-based) \t ms \t text
//   eval:    file_name \t ms \t text
//
// Usage:
//   whisper_bench --model M.bin --preset greedy|cli-defaults [--threads 4]
//                 [--clip name=path.wav ...] [--runs 5] [--latency-out out.tsv]
//                 [--eval-list eval.tsv] [--eval-out out.tsv] [--eval-limit N]
//                 [--warmup 4] [--warmup-file path ...] [--require-coreml] [--label name]
//   whisper_bench --check-audio --eval-list eval.tsv [--clip name=path ...]   (decode only, no model)

#include "whisper.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_coreml_loaded{false};
std::atomic<bool> g_coreml_failed{false};

void log_cb(enum ggml_log_level level, const char * text, void * /*user*/) {
    (void) level;
    if (text == nullptr) return;
    if (strstr(text, "Core ML model loaded"))        g_coreml_loaded = true;
    if (strstr(text, "failed to load Core ML model")) g_coreml_failed = true;
    fputs(text, stderr);
    fflush(stderr);
}

[[noreturn]] void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("whisper_bench: error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputs("\n", stderr);
    va_end(ap);
    exit(1);
}

std::string basename_of(const std::string & p) {
    const auto pos = p.find_last_of('/');
    return pos == std::string::npos ? p : p.substr(pos + 1);
}

// Decode any CoreAudio-readable file (WAV, FLAC, ...) to 16 kHz mono float32.
// The bench inputs are already 16 kHz mono 16-bit, so this is a pure int16 -> float32
// conversion (x / 32768, same as whisper-cli's miniaudio path); no resampling happens.
std::vector<float> load_audio(const std::string & path) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(path.c_str()), (CFIndex) path.size(), false);
    if (!url) die("cannot create URL for %s", path.c_str());

    ExtAudioFileRef f = nullptr;
    OSStatus st = ExtAudioFileOpenURL(url, &f);
    CFRelease(url);
    if (st != noErr || !f) die("ExtAudioFileOpenURL failed (%d) for %s", (int) st, path.c_str());

    AudioStreamBasicDescription src = {};
    UInt32 sz = sizeof(src);
    st = ExtAudioFileGetProperty(f, kExtAudioFileProperty_FileDataFormat, &sz, &src);
    if (st != noErr) die("cannot read file format (%d) for %s", (int) st, path.c_str());
    if (src.mSampleRate != 16000.0 || src.mChannelsPerFrame != 1) {
        fprintf(stderr, "whisper_bench: warning: %s is %.0f Hz / %u ch; CoreAudio will convert to 16 kHz mono\n",
                path.c_str(), src.mSampleRate, (unsigned) src.mChannelsPerFrame);
    }

    AudioStreamBasicDescription dst = {};
    dst.mSampleRate       = 16000.0;
    dst.mFormatID         = kAudioFormatLinearPCM;
    dst.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagsNativeEndian;
    dst.mBitsPerChannel   = 32;
    dst.mChannelsPerFrame = 1;
    dst.mFramesPerPacket  = 1;
    dst.mBytesPerFrame    = 4;
    dst.mBytesPerPacket   = 4;
    st = ExtAudioFileSetProperty(f, kExtAudioFileProperty_ClientDataFormat, sizeof(dst), &dst);
    if (st != noErr) die("cannot set client format (%d) for %s", (int) st, path.c_str());

    std::vector<float> out;
    std::vector<float> buf(16384);
    for (;;) {
        AudioBufferList abl;
        abl.mNumberBuffers              = 1;
        abl.mBuffers[0].mNumberChannels = 1;
        abl.mBuffers[0].mDataByteSize   = (UInt32) (buf.size() * sizeof(float));
        abl.mBuffers[0].mData           = buf.data();
        UInt32 frames = (UInt32) buf.size();
        st = ExtAudioFileRead(f, &frames, &abl);
        if (st != noErr) die("ExtAudioFileRead failed (%d) for %s", (int) st, path.c_str());
        if (frames == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + frames);
    }
    ExtAudioFileDispose(f);

    if (out.empty()) die("no audio decoded from %s", path.c_str());
    return out;
}

struct Item {
    std::string name;  // clip name or eval file name
    std::string path;
    double      ref_seconds = 0.0;
    std::vector<float> pcm;
};

std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) { parts.push_back(line.substr(start)); break; }
        parts.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

std::vector<Item> read_eval_list(const std::string & path, int limit) {
    std::ifstream in(path);
    if (!in) die("cannot open eval list %s", path.c_str());
    std::vector<Item> items;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto parts = split_tabs(line);
        if (parts.size() < 2) die("bad eval row: %s", line.c_str());
        Item it;
        it.path        = parts[0];
        it.name        = basename_of(parts[0]);
        it.ref_seconds = atof(parts[1].c_str());
        items.push_back(std::move(it));
        if (limit > 0 && (int) items.size() >= limit) break;
    }
    return items;
}

std::string collect_text(whisper_context * ctx) {
    std::string text;
    const int n = whisper_full_n_segments(ctx);
    for (int i = 0; i < n; ++i) {
        const char * s = whisper_full_get_segment_text(ctx, i);
        if (s) text += s;
    }
    for (char & c : text) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    }
    const auto b = text.find_first_not_of(' ');
    if (b == std::string::npos) return "";
    const auto e = text.find_last_not_of(' ');
    return text.substr(b, e - b + 1);
}

whisper_full_params make_params(const std::string & preset, int n_threads) {
    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    if (preset == "greedy") {
        wp.strategy         = WHISPER_SAMPLING_GREEDY;
        wp.n_threads        = n_threads;
        wp.language         = "en";
        wp.detect_language  = false;
        wp.no_timestamps    = true;
        wp.temperature      = 0.0f;
        wp.temperature_inc  = 0.0f;   // no temperature fallback
        wp.initial_prompt   = nullptr;
        wp.prompt_tokens    = nullptr;
        wp.prompt_n_tokens  = 0;
        wp.no_context       = true;
        wp.print_special    = false;
        wp.print_progress   = false;
        wp.print_realtime   = false;
        wp.print_timestamps = false;
    } else if (preset == "cli-defaults") {
        // Mirrors examples/cli/cli.cpp at v1.9.5 with no CLI flags given:
        // whisper_full_default_params(GREEDY), then strategy = BEAM_SEARCH because beam_size (5) > 1,
        // best_of 5, beam_size 5, temperature 0, temperature_inc 0.2 (fallback on), entropy 2.4,
        // logprob -1, no_speech 0.6, timestamps ON (no_timestamps = false), language "en".
        wp.strategy               = WHISPER_SAMPLING_BEAM_SEARCH;
        wp.n_threads              = n_threads;
        wp.language               = "en";
        wp.detect_language        = false;
        wp.translate              = false;
        wp.greedy.best_of         = 5;
        wp.beam_search.beam_size  = 5;
        wp.temperature            = 0.0f;
        wp.temperature_inc        = 0.2f;
        wp.entropy_thold          = 2.40f;
        wp.logprob_thold          = -1.00f;
        wp.no_speech_thold        = 0.6f;
        wp.no_timestamps          = false;
        wp.token_timestamps       = false;
        wp.suppress_nst           = false;
        wp.initial_prompt         = nullptr;   // whisper-cli passes "" which tokenizes to nothing
        wp.carry_initial_prompt   = false;
        wp.print_special          = false;
        wp.print_progress         = false;
        wp.print_realtime         = false;
        wp.print_timestamps       = false;     // only affects print_realtime output
    } else {
        die("unknown preset '%s' (expected greedy or cli-defaults)", preset.c_str());
    }
    return wp;
}

void dump_params(const whisper_full_params & p) {
    fprintf(stderr,
        "whisper_bench: params: strategy=%s n_threads=%d n_max_text_ctx=%d language=%s detect_language=%d "
        "no_context=%d no_timestamps=%d single_segment=%d token_timestamps=%d translate=%d "
        "temperature=%.2f temperature_inc=%.2f entropy_thold=%.2f logprob_thold=%.2f no_speech_thold=%.2f "
        "greedy.best_of=%d beam_search.beam_size=%d suppress_blank=%d suppress_nst=%d max_initial_ts=%.2f "
        "length_penalty=%.2f audio_ctx=%d initial_prompt=%s vad=%d print_progress=%d print_realtime=%d\n",
        p.strategy == WHISPER_SAMPLING_GREEDY ? "greedy" : "beam_search",
        p.n_threads, p.n_max_text_ctx, p.language ? p.language : "(null)", (int) p.detect_language,
        (int) p.no_context, (int) p.no_timestamps, (int) p.single_segment, (int) p.token_timestamps, (int) p.translate,
        p.temperature, p.temperature_inc, p.entropy_thold, p.logprob_thold, p.no_speech_thold,
        p.greedy.best_of, p.beam_search.beam_size, (int) p.suppress_blank, (int) p.suppress_nst, p.max_initial_ts,
        p.length_penalty, p.audio_ctx, p.initial_prompt ? p.initial_prompt : "(null)", (int) p.vad,
        (int) p.print_progress, (int) p.print_realtime);
}

// Times only whisper_full(). The per-run breakdown (whisper_get_timings, averages per call since the
// reset just before the run) is read afterwards and logged to stderr for diagnosis only.
double transcribe(whisper_context * ctx, const whisper_full_params & wp, const std::vector<float> & pcm,
                  std::string & text, const std::string & tag) {
    whisper_reset_timings(ctx);
    const auto t0 = std::chrono::steady_clock::now();
    const int rc  = whisper_full(ctx, wp, pcm.data(), (int) pcm.size());
    const auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) die("whisper_full failed (%d)", rc);
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    text = collect_text(ctx);

    int n_tokens = 0;
    for (int i = 0; i < whisper_full_n_segments(ctx); ++i) n_tokens += whisper_full_n_tokens(ctx, i);
    if (whisper_timings * tm = whisper_get_timings(ctx)) {
        fprintf(stderr, "whisper_bench: run %s: total=%.2f ms | per call: encode=%.2f decode=%.2f batchd=%.2f prompt=%.2f sample=%.2f ms | segments=%d tokens=%d\n",
                tag.c_str(), ms, tm->encode_ms, tm->decode_ms, tm->batchd_ms, tm->prompt_ms, tm->sample_ms,
                whisper_full_n_segments(ctx), n_tokens);
        delete tm;
    }
    return ms;
}

FILE * open_out(const std::string & path, std::string & tmp) {
    tmp = path + ".partial";
    FILE * f = fopen(tmp.c_str(), "w");
    if (!f) die("cannot open %s for writing", tmp.c_str());
    return f;
}

void close_out(FILE * f, const std::string & tmp, const std::string & path) {
    fclose(f);
    if (rename(tmp.c_str(), path.c_str()) != 0) die("cannot rename %s -> %s", tmp.c_str(), path.c_str());
}

} // namespace

int main(int argc, char ** argv) {
    std::string model, preset = "greedy", latency_out, eval_list, eval_out, label = "whisper.cpp";
    std::vector<Item> clips;
    std::vector<std::string> warmup_files;
    int n_threads = 4, runs = 5, warmup = -1, eval_limit = 0;
    bool require_coreml = false, check_audio = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for %s", a.c_str());
            return argv[++i];
        };
        if      (a == "--model")          model = next();
        else if (a == "--preset")         preset = next();
        else if (a == "--threads")        n_threads = atoi(next().c_str());
        else if (a == "--runs")           runs = atoi(next().c_str());
        else if (a == "--warmup")         warmup = atoi(next().c_str());
        else if (a == "--warmup-file")    warmup_files.push_back(next());
        else if (a == "--latency-out")    latency_out = next();
        else if (a == "--eval-list")      eval_list = next();
        else if (a == "--eval-out")       eval_out = next();
        else if (a == "--eval-limit")     eval_limit = atoi(next().c_str());
        else if (a == "--require-coreml") require_coreml = true;
        else if (a == "--check-audio")    check_audio = true;
        else if (a == "--label")          label = next();
        else if (a == "--clip") {
            const std::string v = next();
            const auto eq = v.find('=');
            if (eq == std::string::npos) die("--clip expects name=path, got %s", v.c_str());
            Item it;
            it.name = v.substr(0, eq);
            it.path = v.substr(eq + 1);
            clips.push_back(std::move(it));
        } else {
            die("unknown argument %s", a.c_str());
        }
    }

    // ---- decode all audio up front (outside any timing) ----
    for (auto & c : clips) c.pcm = load_audio(c.path);
    std::vector<Item> evals;
    if (!eval_list.empty()) {
        evals = read_eval_list(eval_list, eval_limit);
        for (auto & e : evals) e.pcm = load_audio(e.path);
    }
    std::vector<Item> warm;
    for (const auto & p : warmup_files) { Item it; it.name = basename_of(p); it.path = p; it.pcm = load_audio(p); warm.push_back(std::move(it)); }
    if (warm.empty()) warm = clips;  // default: warm up on the latency clips

    if (check_audio) {
        double max_diff = 0.0;
        for (const auto & c : clips) {
            double sum_abs = 0.0, peak = 0.0;
            for (float x : c.pcm) { sum_abs += std::fabs(x); peak = std::max(peak, (double) std::fabs(x)); }
            printf("clip %-8s %8zu samples  %.3f s  sum|x|=%.6f  peak=%.6f\n", c.name.c_str(), c.pcm.size(), c.pcm.size() / 16000.0, sum_abs, peak);
        }
        for (const auto & e : evals) max_diff = std::max(max_diff, std::fabs(e.pcm.size() / 16000.0 - e.ref_seconds));
        printf("eval: %zu files decoded, max |decoded - listed| duration = %.4f s\n", evals.size(), max_diff);
        return 0;
    }

    if (model.empty()) die("--model is required");
    if (!latency_out.empty() && clips.empty()) die("--latency-out given but no --clip");
    if (!eval_out.empty() && evals.empty()) die("--eval-out given but no --eval-list");
    if (warmup < 0) warmup = std::max(2, (int) warm.size());
    if (warmup > 0 && warm.empty()) die("no warm-up audio (give --clip or --warmup-file)");

    // ---- load the model once ----
    whisper_log_set(log_cb, nullptr);
    fprintf(stderr, "whisper_bench: label=%s whisper_version=%s preset=%s model=%s\n", label.c_str(), whisper_version(), preset.c_str(), model.c_str());
    fprintf(stderr, "whisper_bench: system_info: %s\n", whisper_print_system_info());

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = true;  // flash_attn left at its default
    fprintf(stderr, "whisper_bench: context params: use_gpu=%d flash_attn=%d gpu_device=%d\n", (int) cp.use_gpu, (int) cp.flash_attn, cp.gpu_device);

    const auto tl0 = std::chrono::steady_clock::now();
    whisper_context * ctx = whisper_init_from_file_with_params(model.c_str(), cp);
    const auto tl1 = std::chrono::steady_clock::now();
    if (!ctx) die("failed to load model %s", model.c_str());
    const double load_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();
    fprintf(stderr, "whisper_bench: model load %.0f ms, coreml_loaded=%d\n", load_ms, (int) g_coreml_loaded.load());
    printf("[%s] model loaded in %.0f ms%s\n", label.c_str(), load_ms, g_coreml_loaded ? " (Core ML encoder loaded)" : "");
    fflush(stdout);
    if (require_coreml && (!g_coreml_loaded || g_coreml_failed)) die("Core ML encoder was not loaded (fallback would use the Metal encoder)");

    const whisper_full_params wp = make_params(preset, n_threads);
    dump_params(wp);

    std::string text;

    // ---- warm-up ----
    for (int i = 0; i < warmup; ++i) {
        const auto & w = warm[i % warm.size()];
        const double ms = transcribe(ctx, wp, w.pcm, text, "warmup " + std::to_string(i + 1) + " " + w.name);
        printf("[%s] warmup %d (%s): %.1f ms | %s\n", label.c_str(), i + 1, w.name.c_str(), ms, text.c_str());
        fflush(stdout);
    }

    // ---- latency clips ----
    if (!latency_out.empty()) {
        std::string tmp;
        FILE * f = open_out(latency_out, tmp);
        for (const auto & c : clips) {
            for (int r = 1; r <= runs; ++r) {
                const double ms = transcribe(ctx, wp, c.pcm, text, c.name + " " + std::to_string(r));
                fprintf(f, "%s\t%d\t%.2f\t%s\n", c.name.c_str(), r, ms, text.c_str());
                fflush(f);
                printf("[%s] %s run %d: %.2f ms | %s\n", label.c_str(), c.name.c_str(), r, ms, text.c_str());
                fflush(stdout);
            }
        }
        close_out(f, tmp, latency_out);
    }

    // ---- accuracy set ----
    if (!eval_out.empty()) {
        std::string tmp;
        FILE * f = open_out(eval_out, tmp);
        size_t i = 0;
        for (const auto & e : evals) {
            const double ms = transcribe(ctx, wp, e.pcm, text, e.name);
            fprintf(f, "%s\t%.2f\t%s\n", e.name.c_str(), ms, text.c_str());
            fflush(f);
            ++i;
            printf("[%s] eval %zu/%zu %s (%.2f s): %.2f ms | %s\n", label.c_str(), i, evals.size(), e.name.c_str(), e.pcm.size() / 16000.0, ms, text.c_str());
            fflush(stdout);
        }
        close_out(f, tmp, eval_out);
    }

    whisper_free(ctx);
    return 0;
}
