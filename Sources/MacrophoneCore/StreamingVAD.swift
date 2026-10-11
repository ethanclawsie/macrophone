import CWhisper

/// Silero VAD (whisper.cpp's ggml port) fed incrementally while the key is held.
/// The LSTM state carries across calls, so each 32 ms window is processed exactly once.
public final class StreamingVAD: @unchecked Sendable {
    /// Samples per VAD window (32 ms at 16 kHz).
    public static let windowSize = 512

    private let vctx: OpaquePointer
    private var carry: [Float] = []

    public init(modelPath: String) throws {
        Log.installWhisperLogFilter()
        var params = whisper_vad_default_context_params()
        params.n_threads = 1
        params.use_gpu = false // tiny model; keeps the GPU free for Whisper
        guard let vctx = whisper_vad_init_from_file_with_params(modelPath, params) else {
            throw MacrophoneError.modelLoad(modelPath)
        }
        self.vctx = vctx
        carry.reserveCapacity(Self.windowSize * 8)
    }

    deinit { whisper_vad_free(vctx) }

    public func reset() {
        whisper_vad_reset_state(vctx)
        carry.removeAll(keepingCapacity: true)
    }

    /// Feeds samples and returns one speech probability per completed window.
    public func process(_ samples: UnsafeBufferPointer<Float>) -> [Float] {
        carry.append(contentsOf: samples)
        let windows = carry.count / Self.windowSize
        guard windows > 0 else { return [] }
        let n = windows * Self.windowSize
        let ok = carry.withUnsafeBufferPointer {
            whisper_vad_detect_speech_no_reset(vctx, $0.baseAddress, Int32(n))
        }
        var probs = [Float](repeating: 0, count: windows)
        if ok, let p = whisper_vad_probs(vctx) {
            for i in 0..<min(windows, Int(whisper_vad_n_probs(vctx))) { probs[i] = p[i] }
        }
        carry.removeFirst(n)
        return probs
    }
}
