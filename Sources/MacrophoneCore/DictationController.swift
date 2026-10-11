import Foundation

/// Long-lived owner of the engine, VAD and microphone; runs one `DictationSession` per hotkey hold.
/// Thread-safe: every entry point hops onto the pipeline queue.
public final class DictationController: @unchecked Sendable {
    public struct Outcome: Sendable {
        public var result: DictationSession.Result
        /// Hotkey press → first microphone buffer (ms), when known.
        public var micStartMs: Double?
    }

    public let runner: InferenceRunner
    public let queue = DispatchQueue(label: "macrophone.pipeline", qos: .userInteractive)
    private let vad: StreamingVAD
    private let capture = AudioCapture()

    /// Read on the pipeline queue at the start of each hold.
    nonisolated(unsafe) public var config = PipelineConfig()
    /// Keep the microphone open this long after a hold so rapid follow-ups start instantly.
    /// 0 closes it right away (the orange privacy dot goes off).
    nonisolated(unsafe) public var micKeepAlive: TimeInterval = 0
    nonisolated(unsafe) public var kickGPUOnPress = false

    private var session: DictationSession?
    private var pump: DispatchSourceTimer?
    private var micStop: DispatchWorkItem?
    private var pressedAt: UInt64 = 0

    public init(runner: InferenceRunner, vad: StreamingVAD) {
        self.runner = runner
        self.vad = vad
    }

    /// Builds the microphone graph ahead of the first hold (doesn't start the device).
    public func prepareMicrophone(preferBuiltIn: Bool) {
        queue.async { [self] in
            capture.preferBuiltInMic = preferBuiltIn
            try? capture.prepare()
        }
    }

    /// Hotkey went down.
    public func press(onError: @escaping @Sendable (Error) -> Void) {
        let pressed = Clock.now
        queue.async { [self] in
            micStop?.cancel()
            micStop = nil
            session?.cancel()
            do {
                try capture.start()
            } catch {
                onError(error)
                return
            }
            pressedAt = pressed
            capture.beginTake()
            session = DictationSession(runner: runner, vad: vad, config: config, queue: queue)
            if kickGPUOnPress { runner.kick() }
            startPump()
        }
    }

    /// Hotkey went up: finish the take and deliver the text.
    public func release(completion: @escaping @Sendable (Outcome) -> Void) {
        queue.async { [self] in
            guard let session else { return }
            stopPump()
            session.append(capture.drain16k(flush: true))
            let micStartMs = capture.firstBufferTime.map { Double(Int64(bitPattern: $0 &- pressedAt)) / 1e6 }
            self.session = nil
            scheduleMicStop()
            session.finish { result in
                completion(Outcome(result: result, micStartMs: micStartMs))
            }
        }
    }

    /// Hotkey hold abandoned (chord, Esc, too short): discard everything.
    public func cancel() {
        queue.async { [self] in
            stopPump()
            session?.cancel()
            session = nil
            scheduleMicStop()
        }
    }

    private func startPump() {
        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
        timer.schedule(deadline: .now() + .milliseconds(10), repeating: .milliseconds(10), leeway: .milliseconds(2))
        timer.setEventHandler { [weak self] in
            guard let self, let session = self.session else { return }
            session.append(self.capture.drain16k())
        }
        timer.resume()
        pump = timer
    }

    private func stopPump() {
        pump?.cancel()
        pump = nil
    }

    private func scheduleMicStop() {
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.session == nil else { return }
            self.capture.stop()
        }
        micStop = work
        queue.asyncAfter(deadline: .now() + micKeepAlive, execute: work)
    }
}
