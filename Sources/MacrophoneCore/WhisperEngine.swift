import CWhisper
import Foundation
import Synchronization

/// Result of one Whisper pass.
public struct Transcript: Sendable {
    public var text: String
    /// Text tokens; `draft` adds timestamp tokens — the sequence to hand to a later pass as its draft.
    public var tokenCount: Int
    public var draft: [Int32] = []
    /// Draft tokens this pass accepted without a decoder step.
    public var draftAccepted = 0
    public var audioSeconds: Double
    /// Encoder frames actually processed (1500 = the full 30 s window).
    public var audioCtx: Int
    public var noSpeechProb: Float
    public var milliseconds: Double
    public var aborted: Bool
    /// Where the time went (ms): log-mel, encoder (incl. cross K/V), prompt prefill, per-token decode, sampling.
    public var breakdown = Breakdown()

    public struct Breakdown: Sendable {
        public var mel = 0.0, encode = 0.0, prompt = 0.0, decode = 0.0, sample = 0.0
        public var decodeSteps = 0
    }
}

/// Latency-focused wrapper around a whisper.cpp context.
///
/// Not thread-safe: drive it from a single serial queue (see `InferenceRunner`).
public final class WhisperEngine: @unchecked Sendable {
    public enum Window: Sendable, Equatable {
        /// Whisper's native 30 s window: robust, but the encoder always costs ~300 ms on a base M5.
        case full
        /// Encoder context truncated to the audio: its frames plus `margin` frames of trailing silence,
        /// rounded up to `roundTo`. Fast, but stock turbo loops on short clips in this mode.
        case fitted(margin: Int, roundTo: Int)
        /// Silence-tail cache: compute only the audio's frames (+ margin, rounded) while every encoder
        /// layer and the decoder still attend over the full 30 s window, whose silent part comes from a
        /// cached encode of silence (see `whisper_macrophone_init_tail`). `encoderTail: false` caches only the
        /// decoder's view (ablation).
        case tail(margin: Int, roundTo: Int, encoderTail: Bool)

        public static let `default` = Window.tail(margin: 16, roundTo: 16, encoderTail: true)
    }

    public static let sampleRate = 16_000
    /// One encoder frame covers 20 ms of audio.
    public static let samplesPerFrame = 320
    public static let maxFrames = 1500

    private let ctx: OpaquePointer
    public let threads: Int32
    public let language: String
    private let abortRequested = Atomic<Bool>(false)

    /// Whether the silence-tail cache is built (`Window.tail` falls back to `.full` without it).
    public private(set) var hasTailCache = false

    /// - Parameter melPeak: log-mel peak every utterance is normalized to (makes the silence level
    ///   constant, which the tail cache requires). `nil` keeps whisper.cpp's stock normalization and
    ///   disables the tail cache.
    public init(modelPath: String, threads: Int32 = 4, language: String = "en", flashAttention: Bool = true,
                melPeak: Float? = 1.5) throws {
        Log.installWhisperLogFilter()
        var params = whisper_context_default_params()
        params.use_gpu = true
        params.flash_attn = flashAttention
        guard let ctx = whisper_init_from_file_with_params(modelPath, params) else {
            throw MacrophoneError.modelLoad(modelPath)
        }
        self.ctx = ctx
        self.threads = threads
        self.language = language
        if let melPeak {
            hasTailCache = whisper_macrophone_init_tail(ctx, melPeak, threads) == 0
        }
    }

    /// Log-mel peak normalization for every mode (0 = whisper.cpp's stock normalization).
    public func setMelPeak(_ peak: Float) { whisper_macrophone_set_mel_peak(ctx, peak) }

    deinit { whisper_free(ctx) }

    /// Asks the pass currently running (if any) to stop early. Thread-safe.
    public func requestAbort() { abortRequested.store(true, ordering: .relaxed) }

    public func encoderFrames(forSamples n: Int, window: Window) -> Int {
        switch window {
        case .full:
            return Self.maxFrames
        case let .fitted(margin, roundTo), let .tail(margin, roundTo, _):
            var frames = (n + Self.samplesPerFrame - 1) / Self.samplesPerFrame + margin
            if roundTo > 1 { frames = (frames + roundTo - 1) / roundTo * roundTo }
            return min(frames, Self.maxFrames)
        }
    }

    /// - Parameter draft: tokens of an earlier pass over a prefix of the same audio (see `Transcript.draft`);
    ///   matching tokens are verified in one batched decoder step instead of one step each.
    public func transcribe(_ samples: UnsafeBufferPointer<Float>, prompt: String? = nil, window requestedWindow: Window = .default,
                           draft: [Int32] = []) -> Transcript {
        abortRequested.store(false, ordering: .relaxed)
        let n = samples.count
        let seconds = Double(n) / Double(Self.sampleRate)
        // Longer than one window: Whisper's own long-form decoding (seek to the last timestamp, next window).
        // The dictation pipeline cuts segments well before this; it matters for whole-file use.
        let longForm = n > Self.sampleRate * 30
        let window = longForm ? .full : requestedWindow
        let audioCtx = encoderFrames(forSamples: n, window: window)

        var p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        p.n_threads = threads
        p.no_context = true // context is passed explicitly through `prompt`
        // Timestamp mode lets whisper.cpp stop as soon as a timestamp reaches the end of the audio.
        // Without it, a fitted encoder window makes turbo loop on short clips.
        p.no_timestamps = false
        p.single_segment = !longForm // exactly one encoder pass
        p.print_special = false
        p.print_progress = false
        p.print_realtime = false
        p.print_timestamps = false
        p.token_timestamps = false
        p.detect_language = false
        p.suppress_blank = true
        p.suppress_nst = true
        p.temperature = 0
        p.temperature_inc = 0 // no temperature fallback: one decode, predictable latency
        p.greedy.best_of = 1
        p.max_tokens = longForm ? 0 : Int32(24 + seconds * 10)
        switch window {
        case .full:
            p.audio_ctx = 0
            whisper_macrophone_set_live(ctx, 0, 0)
        case .fitted:
            p.audio_ctx = Int32(audioCtx)
            whisper_macrophone_set_live(ctx, 0, 0)
        case let .tail(_, _, encoderTail):
            // The decoder sees the whole window; only `audioCtx` encoder frames are computed.
            p.audio_ctx = 0
            whisper_macrophone_set_live(ctx, hasTailCache && audioCtx < Self.maxFrames ? Int32(audioCtx) : 0, encoderTail ? 2 : 1)
        }
        p.abort_callback = { userData in
            guard let userData else { return false }
            return Unmanaged<WhisperEngine>.fromOpaque(userData).takeUnretainedValue()
                .abortRequested.load(ordering: .relaxed)
        }
        p.abort_callback_user_data = Unmanaged.passUnretained(self).toOpaque()

        whisper_reset_timings(ctx)
        draft.withUnsafeBufferPointer { whisper_macrophone_set_draft(ctx, $0.baseAddress, Int32($0.count)) }
        let start = Clock.now
        let rc: Int32 = language.withCString { lang in
            p.language = lang
            return withOptionalCString(prompt) { promptPtr in
                p.initial_prompt = promptPtr
                return whisper_full(ctx, p, samples.baseAddress, Int32(n))
            }
        }
        let elapsed = Clock.ms(since: start)
        let aborted = rc != 0 || abortRequested.load(ordering: .relaxed)

        var ms = [Float](repeating: 0, count: 6)
        var counts = [Int32](repeating: 0, count: 5)
        whisper_macrophone_get_timings(ctx, &ms, &counts)
        let breakdown = Transcript.Breakdown(
            mel: Double(ms[0]), encode: Double(ms[2]), prompt: Double(ms[5]),
            decode: Double(ms[3] + ms[4]), sample: Double(ms[1]), decodeSteps: Int(counts[2] + counts[3]))

        var text = ""
        var tokens = 0
        var draftOut: [Int32] = []
        var noSpeech: Float = 0
        if !aborted {
            let eot = whisper_token_eot(ctx)
            let beg = whisper_token_beg(ctx)
            for i in 0..<whisper_full_n_segments(ctx) {
                if let s = whisper_full_get_segment_text(ctx, i) { text += String(cString: s) }
                for j in 0..<whisper_full_n_tokens(ctx, i) {
                    let id = whisper_full_get_token_id(ctx, i, j)
                    if id < eot { tokens += 1 }
                    if id < eot || id >= beg { draftOut.append(id) }
                }
                if i == 0 { noSpeech = whisper_full_get_segment_no_speech_prob(ctx, 0) }
            }
        }
        var accepted: Int32 = 0
        var batches: Int32 = 0
        whisper_macrophone_draft_stats(ctx, &accepted, &batches)
        return Transcript(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            tokenCount: tokens,
            draft: draftOut,
            draftAccepted: Int(accepted),
            audioSeconds: seconds,
            audioCtx: audioCtx,
            noSpeechProb: noSpeech,
            milliseconds: elapsed,
            aborted: aborted,
            breakdown: breakdown
        )
    }

    /// Compiles the Metal pipelines used by short and long passes and faults in the weights,
    /// so the first real dictation runs at full speed.
    public func warmup() {
        var rng = SystemRandomNumberGenerator()
        let noise = (0..<(Self.sampleRate * 8)).map { _ in Float.random(in: -0.002...0.002, using: &rng) }
        noise.withUnsafeBufferPointer { all in
            _ = transcribe(UnsafeBufferPointer(rebasing: all[0..<Self.sampleRate]))
            _ = transcribe(all, prompt: "Warm up the decoder with a prompt.")
        }
    }
}
