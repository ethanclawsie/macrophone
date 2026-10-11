import Foundation

public enum ReleaseStrategy: String, Sendable, Codable {
    /// Keep the last speculative pass and transcribe only the speech after it (lowest latency).
    case tailOnly
    /// Re-transcribe the whole open segment on release (more context per pass, more latency).
    case openSegment
    /// `tailOnly` when the last speculation ended at a real pause (a natural phrase break),
    /// otherwise `openSegment`, so a cut never splits a phrase.
    case adaptive
}

public struct PipelineConfig: Sendable {
    /// Silero probability to enter speech, and to stay in it (hysteresis).
    public var speechThreshold: Float = 0.5
    public var silenceThreshold: Float = 0.35
    /// Silence after speech that triggers a speculative pass while the key is still held.
    public var speculateAfterMs = 160
    /// Once the open segment is this long, a finished speculative pass is committed, which bounds
    /// how much audio can be left to transcribe when the key is released.
    public var commitAfterSeconds = 6.0
    /// Only cut (commit) at pauses at least this long: shorter ones are often mid-phrase.
    public var commitMinPauseMs = 300
    /// Unbroken speech longer than this is cut regardless (Whisper's window is 30 s).
    public var maxSegmentSeconds = 24.0
    public var leadingPadMs = 200
    public var trailingPadMs = 200
    /// Less speech than this after the last pass is treated as noise.
    public var minSpeechMs = 96
    public var window: WhisperEngine.Window = .default
    /// Optional glossary passed to Whisper as prompt text (biases spelling of names and jargon).
    public var vocabularyPrompt: String?
    public var releaseStrategy: ReleaseStrategy = .adaptive

    public init() {}
}

/// Serial executor for Whisper passes; owns the engine.
public final class InferenceRunner: @unchecked Sendable {
    public let engine: WhisperEngine
    private let queue = DispatchQueue(label: "macrophone.inference", qos: .userInteractive)

    public init(engine: WhisperEngine) { self.engine = engine }

    public func transcribe(
        _ samples: [Float], prompt: String?, window: WhisperEngine.Window, draft: [Int32] = [],
        completion: @escaping @Sendable (Transcript) -> Void
    ) {
        queue.async { [engine] in
            let t = samples.withUnsafeBufferPointer { engine.transcribe($0, prompt: prompt, window: window, draft: draft) }
            completion(t)
        }
    }

    /// Stops the pass in flight (if any) early.
    public func abortCurrent() { engine.requestAbort() }

    /// Runs a tiny pass so GPU clocks and memory residency are up before the first real pass.
    public func kick() {
        queue.async { [engine] in
            let silence = [Float](repeating: 0, count: WhisperEngine.sampleRate / 2)
            _ = silence.withUnsafeBufferPointer { engine.transcribe($0) }
        }
    }

    public func sync<T>(_ body: (WhisperEngine) -> T) -> T { queue.sync { body(engine) } }
}

/// One hold of the hotkey.
///
/// Latency strategy: while the key is held, every pause in speech triggers a speculative pass over
/// the open segment, so when the key is released the text is usually already there. If speech
/// continued after the last pass, only the tail after it is transcribed (`ReleaseStrategy.tailOnly`).
///
/// All methods must be called on `queue`; pass completions hop back onto it.
public final class DictationSession: @unchecked Sendable {
    public struct Stats: Sendable {
        public var audioSeconds = 0.0
        public var speechSeconds = 0.0
        public var passes = 0
        public var finalPassMs: Double?
        public var finalPassSeconds: Double?
        public var waitedForInflightMs = 0.0
        public var releaseToTextMs = 0.0
        public var answeredFromSpeculation = false
        /// Tokens verified from drafts instead of decoded one by one, over all passes.
        public var draftAccepted = 0
    }

    public struct Result: Sendable {
        public var text: String
        public var stats: Stats
    }

    public let queue: DispatchQueue
    private let runner: InferenceRunner
    private let vad: StreamingVAD
    private let config: PipelineConfig
    private let sr = WhisperEngine.sampleRate
    private let win = StreamingVAD.windowSize

    private var audio: [Float] = []
    private var speechFlags: [Bool] = [] // one per VAD window
    private var inSpeech = false
    private var silenceWindows = 0
    private var lastSpeechEnd = 0 // sample index just past the last speech window
    private var speechWindowCount = 0

    private var committed: [String] = []
    private var openFloor = 0 // the open segment can't start before the last cut
    private var openStart = 0
    private var openHasSpeech = false
    /// Text for [openStart, end); `speechEnd` is where speech stopped before the pause that triggered it.
    private var speculation: (end: Int, speechEnd: Int, text: String, draft: [Int32])?
    private var inflight: (start: Int, end: Int, speechEnd: Int, final: Bool)?
    private var speculationWanted = false

    private var finishing = false
    private var cancelled = false
    private var releasedAt: UInt64 = 0
    private var waitStartedAt: UInt64 = 0
    private var stats = Stats()
    private var completion: ((Result) -> Void)?

    public init(runner: InferenceRunner, vad: StreamingVAD, config: PipelineConfig, queue: DispatchQueue) {
        self.runner = runner
        self.vad = vad
        self.config = config
        self.queue = queue
        vad.reset()
        audio.reserveCapacity(sr * 90)
        speechFlags.reserveCapacity(sr * 90 / win)
    }

    public var capturedSeconds: Double { Double(audio.count) / Double(sr) }

    // MARK: Audio in

    public func append(_ samples: [Float]) {
        guard !finishing, !cancelled, !samples.isEmpty else { return }
        audio.append(contentsOf: samples)
        let probs = samples.withUnsafeBufferPointer { vad.process($0) }
        for p in probs { onWindow(probability: p) }
    }

    private func onWindow(probability p: Float) {
        let index = speechFlags.count
        let end = (index + 1) * win
        let speech = p >= (inSpeech ? config.silenceThreshold : config.speechThreshold)
        speechFlags.append(speech)
        inSpeech = speech

        if speech {
            speechWindowCount += 1
            silenceWindows = 0
            lastSpeechEnd = end
            if !openHasSpeech {
                openHasSpeech = true
                openStart = max(openFloor, index * win - samples(ms: config.leadingPadMs))
            }
            if seconds(end - openStart) > config.maxSegmentSeconds, inflight == nil {
                // Unbroken speech: cut here even though it may split a word.
                startPass(start: openStart, end: end, final: false)
            }
        } else {
            silenceWindows += 1
            if openHasSpeech, silenceWindows == windows(ms: config.speculateAfterMs) { speculate() }
            if silenceWindows == windows(ms: config.commitMinPauseMs) { commitIfDue() }
        }
    }

    // MARK: Passes

    private func speculate() {
        guard !finishing, !cancelled else { return }
        let end = min(audio.count, lastSpeechEnd + samples(ms: config.trailingPadMs))
        if let speculation, speculation.end >= end { return }
        if inflight != nil {
            speculationWanted = true
            return
        }
        startPass(start: openStart, end: end, final: false)
    }

    private func startPass(start: Int, end: Int, final: Bool) {
        let speechEnd = lastSpeechEnd
        inflight = (start, end, speechEnd, final)
        let slice = Array(audio[start..<end])
        // A previous pass over a prefix of the same audio makes a near-perfect draft for this one.
        let draft = speculation.flatMap { $0.end <= end && start == openStart ? $0.draft : nil } ?? []
        // Strong capture on purpose: once the key is released nothing else references the session,
        // and the pass in flight must still be able to deliver.
        runner.transcribe(slice, prompt: promptText(), window: config.window, draft: draft) { transcript in
            self.queue.async { self.passFinished(start: start, end: end, speechEnd: speechEnd, final: final, transcript) }
        }
    }

    private func passFinished(start: Int, end: Int, speechEnd: Int, final: Bool, _ t: Transcript) {
        inflight = nil
        stats.passes += 1
        stats.draftAccepted += t.draftAccepted
        guard !cancelled else { return }

        if final {
            stats.finalPassMs = t.milliseconds
            stats.finalPassSeconds = t.audioSeconds
            if !t.aborted { committed.append(t.text) }
            openHasSpeech = false
            deliver()
            return
        }

        if !t.aborted, start == openStart { speculation = (end, speechEnd, t.text, t.draft) }
        if finishing {
            finalize()
            return
        }
        commitIfDue()
        if speculationWanted {
            speculationWanted = false
            speculate()
        }
    }

    /// Commits the speculation once the open segment is long and the pause after it is a real one.
    private func commitIfDue() {
        guard let s = speculation, seconds(s.end - openStart) >= config.commitAfterSeconds else { return }
        if s.speechEnd > lastSpeechEnd { return }
        let pause = pauseMs(after: s.speechEnd)
        // An unbroken stretch hitting the hard limit is cut regardless.
        guard pause >= config.commitMinPauseMs || seconds(s.end - openStart) > config.maxSegmentSeconds else { return }
        commitSpeculation()
    }

    /// Freezes the current speculation as final text and starts a new open segment after it.
    private func commitSpeculation() {
        guard let s = speculation else { return }
        committed.append(s.text)
        speculation = nil
        openFloor = s.end
        openStart = s.end
        if let first = firstSpeechWindow(atOrAfter: s.end) {
            openHasSpeech = true
            openStart = max(openFloor, first * win - samples(ms: config.leadingPadMs))
        } else {
            openHasSpeech = false
        }
    }

    private func promptText() -> String? {
        var parts: [String] = []
        if let v = config.vocabularyPrompt, !v.isEmpty { parts.append(v) }
        let prior = committed.joined(separator: " ")
        if !prior.isEmpty { parts.append(String(prior.suffix(400))) }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }

    // MARK: Release

    /// The key was released: deliver the text as soon as possible.
    public func finish(completion: @escaping (Result) -> Void) {
        guard !finishing, !cancelled else { return }
        finishing = true
        releasedAt = Clock.now
        self.completion = completion
        stats.audioSeconds = capturedSeconds
        stats.speechSeconds = Double(speechWindowCount * win) / Double(sr)
        finalize()
    }

    public func cancel() {
        cancelled = true
        completion = nil
        if inflight != nil { runner.abortCurrent() }
    }

    private func finalize() {
        if inflight != nil {
            if waitStartedAt == 0 { waitStartedAt = Clock.now }
            return // passFinished calls back here
        }
        if waitStartedAt != 0 { stats.waitedForInflightMs = Clock.ms(since: waitStartedAt) }
        guard openHasSpeech else { return deliver() }

        let end = min(audio.count, lastSpeechEnd + samples(ms: config.trailingPadMs))
        if let s = speculation, speechMs(atOrAfter: s.end) < config.minSpeechMs {
            committed.append(s.text)
            stats.answeredFromSpeculation = true
            return deliver()
        }

        var start = openStart
        let cutAtSpeculation: Bool
        switch config.releaseStrategy {
        case .tailOnly: cutAtSpeculation = true
        case .openSegment: cutAtSpeculation = false
        case .adaptive: cutAtSpeculation = speculation.map { pauseMs(after: $0.speechEnd) >= config.commitMinPauseMs } ?? false
        }
        if cutAtSpeculation, let s = speculation {
            committed.append(s.text)
            speculation = nil
            let first = firstSpeechWindow(atOrAfter: s.end) ?? s.end / win
            start = max(s.end, first * win - samples(ms: config.leadingPadMs))
        }
        guard end - start >= samples(ms: config.minSpeechMs) else { return deliver() }
        startPass(start: start, end: end, final: true)
    }

    private func deliver() {
        guard let completion else { return }
        self.completion = nil
        stats.releaseToTextMs = Clock.ms(since: releasedAt)
        let text = committed.filter { !$0.isEmpty }.joined(separator: " ")
        completion(Result(text: text, stats: stats))
    }

    // MARK: Helpers

    private func samples(ms: Int) -> Int { ms * sr / 1000 }
    private func windows(ms: Int) -> Int { max(1, (samples(ms: ms) + win - 1) / win) }
    private func seconds(_ samples: Int) -> Double { Double(samples) / Double(sr) }

    private func firstSpeechWindow(atOrAfter sample: Int) -> Int? {
        var i = (sample + win - 1) / win
        while i < speechFlags.count {
            if speechFlags[i] { return i }
            i += 1
        }
        return nil
    }

    /// Length of the silence that starts at `sample` (until the next speech window or the end).
    private func pauseMs(after sample: Int) -> Int {
        var i = (sample + win - 1) / win
        var count = 0
        while i < speechFlags.count, !speechFlags[i] {
            count += 1
            i += 1
        }
        return count * win * 1000 / sr
    }

    private func speechMs(atOrAfter sample: Int) -> Int {
        var count = 0
        var i = (sample + win - 1) / win
        while i < speechFlags.count {
            if speechFlags[i] { count += 1 }
            i += 1
        }
        return count * win * 1000 / sr
    }
}
