import CoreML
import Foundation
import WhisperKit

// whisperkit-bench: latency + accuracy harness for WhisperKit (argmax-oss-swift) running
// Whisper large-v3-turbo Core ML models.
//
//   whisperkit-bench download --hf-base DIR --variant NAME [--revision SHA] [--tokenizer-revision SHA]
//   whisperkit-bench bench --model DIR --tokenizer-base DIR [--label NAME]
//                          [--mel gpu] [--encoder ane] [--decoder ane]      (cpu | gpu | ane | all)
//                          [--warmup 2] [--warmup-clip FILE]
//                          [--audio-dir DIR --clips short,medium,long,xlong --runs 5 --latency-out FILE]
//                          [--eval-tsv FILE --eval-out FILE]
//                          [--breakdown]
//   whisperkit-bench check-audio [--audio-dir DIR] [--eval-tsv FILE]     (decode-only input check, no model)

let modelRepo = "argmaxinc/whisperkit-coreml"
let modelRepoRevision = "0f63a7800b00dd0226abd051b906c246e1907482"      // HF main @ 2026-08-19
let tokenizerRepo = "openai/whisper-large-v3"                               // same tokenizer as large-v3-turbo
let tokenizerRepoRevision = "06f233fe06e710322aca913c1bc4249a0d71fce1"

/// Default --audio-dir when the binary is run by hand (run.sh always passes one): the repo's bench/audio,
/// found from this file's compile-time path, <repo>/bench/competitors/whisperkit/Sources/whisperkit-bench/Bench.swift.
let defaultAudioDir: String = {
    var bench = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { bench.deleteLastPathComponent() }  // Bench.swift, whisperkit-bench, Sources, whisperkit, competitors
    return bench.appending(path: "audio").path
}()

struct BenchError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

struct Options {
    var command = ""
    var values: [String: String] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) {
        var rest = argv[...]
        if let first = rest.first, !first.hasPrefix("--") {
            command = first
            rest = rest.dropFirst()
        }
        while let arg = rest.first {
            rest = rest.dropFirst()
            guard arg.hasPrefix("--") else { continue }
            if let next = rest.first, !next.hasPrefix("--") {
                values[arg] = next
                rest = rest.dropFirst()
            } else {
                flags.insert(arg)
            }
        }
    }

    func string(_ key: String) -> String? { values[key] }
    func string(_ key: String, _ fallback: String) -> String { values[key] ?? fallback }
    func int(_ key: String, _ fallback: Int) -> Int { values[key].flatMap { Int($0) } ?? fallback }
    func has(_ flag: String) -> Bool { flags.contains(flag) }
}

func eprint(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func computeUnits(_ name: String) throws -> MLComputeUnits {
    switch name.lowercased() {
    case "cpu": return .cpuOnly
    case "gpu": return .cpuAndGPU
    case "ane": return .cpuAndNeuralEngine
    case "all": return .all
    default: throw BenchError("unknown compute units '\(name)' (use cpu | gpu | ane | all)")
    }
}

func unitsName(_ units: MLComputeUnits) -> String {
    switch units {
    case .cpuOnly: return "cpuOnly"
    case .cpuAndGPU: return "cpuAndGPU"
    case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
    case .all: return "all"
    @unknown default: return "unknown"
    }
}

/// Fastest legitimate dictation configuration:
/// English forced (no language detection), transcribe, greedy temperature 0 with no fallback,
/// no timestamps (<|notimestamps|> prompt, no timestamp logits filter), no word timestamps,
/// no prompt / previous-text conditioning, no chunking.
func benchDecodingOptions() -> DecodingOptions {
    DecodingOptions(
        verbose: false,
        task: .transcribe,
        language: "en",
        temperature: 0.0,
        temperatureFallbackCount: 0,
        usePrefillPrompt: true,          // force <|en|><|transcribe|><|notimestamps|>
        detectLanguage: false,
        skipSpecialTokens: true,
        withoutTimestamps: true,
        wordTimestamps: false,
        promptTokens: nil,
        prefixTokens: nil,
        // Fallback triggers are meaningless without fallback. firstTokenLogProbThreshold would
        // otherwise end a window after its first token (empty text) when that token is unlikely.
        compressionRatioThreshold: nil,
        logProbThreshold: nil,
        firstTokenLogProbThreshold: nil,
        noSpeechThreshold: nil,
        chunkingStrategy: nil
    )
}

func milliseconds(_ duration: Duration) -> Double {
    let c = duration.components
    return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
}

func fmt(_ value: Double, _ digits: Int = 2) -> String { String(format: "%.\(digits)f", value) }

func tsvSafe(_ text: String) -> String {
    text.replacingOccurrences(of: "\t", with: " ")
        .replacingOccurrences(of: "\r", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
        .trimmingCharacters(in: .whitespaces)
}

/// Writes rows to FILE.partial and renames to FILE only when complete.
final class TSVWriter {
    let path: String
    let partialPath: String
    let handle: FileHandle

    init(path: String) throws {
        self.path = path
        partialPath = path + ".partial"
        try FileManager.default.createDirectory(
            atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: partialPath, contents: nil)
        handle = try FileHandle(forWritingTo: URL(fileURLWithPath: partialPath))
    }

    func row(_ fields: [String]) throws {
        try handle.write(contentsOf: Data((fields.joined(separator: "\t") + "\n").utf8))
    }

    func finish() throws {
        try handle.close()
        if FileManager.default.fileExists(atPath: path) { try FileManager.default.removeItem(atPath: path) }
        try FileManager.default.moveItem(atPath: partialPath, toPath: path)
    }
}

func loadAudio16kMono(_ path: String) throws -> [Float] {
    // WhisperKit's own loader (AVAudioFile; reads WAV/FLAC, resamples to 16 kHz mono if needed).
    let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
    guard !samples.isEmpty else { throw BenchError("no samples decoded from \(path)") }
    return samples
}

struct Timed {
    let ms: Double
    let text: String
    let timings: TranscriptionTimings?
}

/// The timed region: in-memory 16 kHz samples -> final text.
func timedTranscribe(_ pipe: WhisperKit, _ samples: [Float], _ options: DecodingOptions) async throws -> Timed {
    let clock = ContinuousClock()
    let start = clock.now
    let results = try await pipe.transcribe(audioArray: samples, decodeOptions: options)
    let text = results.map(\.text).joined(separator: " ")
    let elapsed = clock.now - start
    return Timed(ms: milliseconds(elapsed), text: text, timings: results.first?.timings)
}

/// WhisperKit's per-call component timings. `decodingLoop` spans the whole window loop
/// (padding + mel + encoder + decoder), so the decoder share is what remains after the rest.
func breakdownLine(_ t: TranscriptionTimings?) -> String {
    guard let t else { return "" }
    let steps = max(t.totalDecodingLoops, 1)
    let decoder = t.decodingLoop - t.audioProcessing - t.logmels - t.encoding - t.decodingWindowing
    return "pad \(fmt(t.audioProcessing * 1000, 1)) | mel \(fmt(t.logmels * 1000, 1)) | enc \(fmt(t.encoding * 1000, 1))"
        + " | dec \(fmt(decoder * 1000, 1)) ms = \(Int(t.totalDecodingLoops)) steps x (predict \(fmt(t.decodingPredictions * 1000 / steps, 2))"
        + " + host \(fmt(t.decodingNonPrediction * 1000 / steps, 2)) ms) | windows \(Int(t.totalDecodingWindows))"
}

// MARK: - download

func runDownload(_ o: Options) async throws {
    guard let base = o.string("--hf-base"), let variant = o.string("--variant") else {
        throw BenchError("download needs --hf-base DIR --variant NAME")
    }
    let hub = HubApiWrapper(downloadBase: URL(fileURLWithPath: base))
    let revision = o.string("--revision", modelRepoRevision)
    let clock = ContinuousClock()
    let start = clock.now
    var lastReport = -1
    let folder = try await hub.snapshot(
        from: HubApiWrapper.Repo(id: modelRepo), revision: revision, matching: ["\(variant)/*"]
    ) { progress in
        let pct = Int(progress.fractionCompleted * 100)
        if pct / 10 != lastReport / 10 {
            lastReport = pct
            eprint("  \(variant): \(pct)%")
        }
    }
    eprint("model \(variant) @ \(revision) -> \(folder.appending(path: variant).path) (\(fmt(milliseconds(clock.now - start) / 1000, 1)) s)")

    let tokRevision = o.string("--tokenizer-revision", tokenizerRepoRevision)
    let tokFolder = try await hub.snapshot(
        from: HubApiWrapper.Repo(id: tokenizerRepo), revision: tokRevision,
        matching: ["config.json", "tokenizer.json", "tokenizer_config.json"])
    eprint("tokenizer \(tokenizerRepo) @ \(tokRevision) -> \(tokFolder.path)")
}

// MARK: - bench

func runBench(_ o: Options) async throws {
    guard let modelDir = o.string("--model") else { throw BenchError("bench needs --model DIR") }
    let label = o.string("--label", "whisperkit")
    let mel = try computeUnits(o.string("--mel", "gpu"))
    let encoder = try computeUnits(o.string("--encoder", "ane"))
    let decoder = try computeUnits(o.string("--decoder", "ane"))
    let runs = o.int("--runs", 5)
    let warmups = o.int("--warmup", 2)
    let showBreakdown = o.has("--breakdown")
    let options = benchDecodingOptions()
    let clock = ContinuousClock()

    // 1. Load the model once (download disabled: everything comes from local folders).
    let config = WhisperKitConfig(
        modelFolder: modelDir,
        tokenizerFolder: o.string("--tokenizer-base").map { URL(fileURLWithPath: $0) },
        computeOptions: ModelComputeOptions(melCompute: mel, audioEncoderCompute: encoder, textDecoderCompute: decoder),
        verbose: false,
        logLevel: .none,
        prewarm: false,
        load: true,
        download: false
    )
    let loadStart = clock.now
    let pipe = try await WhisperKit(config)
    let loadMs = milliseconds(clock.now - loadStart)
    let ct = pipe.currentTimings
    print("[\(label)] \((modelDir as NSString).lastPathComponent)  mel=\(unitsName(mel)) encoder=\(unitsName(encoder)) decoder=\(unitsName(decoder))")
    print("[\(label)] load \(fmt(loadMs, 0)) ms (encoder \(fmt(ct.encoderLoadTime * 1000, 0)) ms, decoder \(fmt(ct.decoderLoadTime * 1000, 0)) ms, tokenizer \(fmt(ct.tokenizerLoadTime * 1000, 0)) ms)")

    // 2. Decode all audio to 16 kHz mono float arrays before any timing.
    var clips: [(name: String, samples: [Float])] = []
    if o.string("--latency-out") != nil {
        let audioDir = o.string("--audio-dir", defaultAudioDir)
        for name in o.string("--clips", "short,medium,long,xlong").split(separator: ",").map(String.init) {
            clips.append((name, try loadAudio16kMono((audioDir as NSString).appendingPathComponent("\(name).wav"))))
        }
    }
    var evalItems: [(file: String, samples: [Float])] = []
    if let evalTSV = o.string("--eval-tsv") {
        guard o.string("--eval-out") != nil else { throw BenchError("--eval-tsv needs --eval-out") }
        let rows = try String(contentsOfFile: evalTSV, encoding: .utf8).split(separator: "\n")
        for row in rows {
            let path = String(row.split(separator: "\t", maxSplits: 2)[0])
            guard !path.isEmpty else { continue }
            evalItems.append(((path as NSString).lastPathComponent, try loadAudio16kMono(path)))
        }
    }

    // 3. Warm up.
    let warmupSamples: [Float]
    if let warmupClip = o.string("--warmup-clip") {
        warmupSamples = try loadAudio16kMono(warmupClip)
    } else if let last = clips.last {
        warmupSamples = last.samples
    } else if let first = evalItems.first {
        warmupSamples = first.samples
    } else {
        throw BenchError("nothing to run: pass --latency-out and/or --eval-tsv/--eval-out")
    }
    for i in 0..<warmups {
        let r = try await timedTranscribe(pipe, warmupSamples, options)
        eprint("[\(label)] warmup \(i + 1): \(fmt(r.ms, 1)) ms  \(r.text)")
    }

    // 4. Latency clips: each clip `runs` times.
    if let latencyOut = o.string("--latency-out") {
        let writer = try TSVWriter(path: latencyOut)
        for clip in clips {
            var times: [Double] = []
            var lastText = ""
            for run in 1...runs {
                let r = try await timedTranscribe(pipe, clip.samples, options)
                times.append(r.ms)
                lastText = tsvSafe(r.text)
                try writer.row([clip.name, String(run), fmt(r.ms), lastText])
                if showBreakdown {
                    eprint("[\(label)] \(clip.name) run \(run): \(fmt(r.ms, 1)) ms  [\(breakdownLine(r.timings))]")
                }
            }
            let sorted = times.sorted()
            print("[\(label)] \(clip.name) (\(fmt(Double(clip.samples.count) / 16000, 2)) s): median \(fmt(sorted[sorted.count / 2], 1)) ms  min \(fmt(sorted.first ?? 0, 1)) ms  runs \(times.map { fmt($0, 1) }.joined(separator: ", "))")
            print("[\(label)]     \(lastText)")
        }
        try writer.finish()
        print("[\(label)] latency -> \(latencyOut)")
    }

    // 5. Accuracy set: each utterance once.
    if let evalOut = o.string("--eval-out"), !evalItems.isEmpty {
        let writer = try TSVWriter(path: evalOut)
        var total = 0.0
        for (i, item) in evalItems.enumerated() {
            let r = try await timedTranscribe(pipe, item.samples, options)
            total += r.ms
            try writer.row([item.file, fmt(r.ms), tsvSafe(r.text)])
            if showBreakdown || (i + 1) % 20 == 0 {
                eprint("[\(label)] eval \(i + 1)/\(evalItems.count) \(item.file): \(fmt(r.ms, 1)) ms" + (showBreakdown ? "  [\(breakdownLine(r.timings))]" : ""))
            }
        }
        try writer.finish()
        print("[\(label)] eval: \(evalItems.count) utterances, mean \(fmt(total / Double(evalItems.count), 1)) ms -> \(evalOut)")
    }
}

// MARK: - check-audio (no model, no inference): verify every input decodes to the expected length

func runCheckAudio(_ o: Options) throws {
    let audioDir = o.string("--audio-dir", defaultAudioDir)
    for name in o.string("--clips", "short,medium,long,xlong").split(separator: ",") {
        let samples = try loadAudio16kMono((audioDir as NSString).appendingPathComponent("\(name).wav"))
        print("\(name).wav: \(samples.count) samples = \(fmt(Double(samples.count) / 16000, 3)) s")
    }
    guard let evalTSV = o.string("--eval-tsv") else { return }
    var count = 0, mismatched = 0, totalSeconds = 0.0, peak: Float = 0
    for row in try String(contentsOfFile: evalTSV, encoding: .utf8).split(separator: "\n") {
        let fields = row.split(separator: "\t", maxSplits: 2).map(String.init)
        guard fields.count == 3, let expected = Double(fields[1]) else { throw BenchError("bad eval row: \(row)") }
        let samples = try loadAudio16kMono(fields[0])
        let seconds = Double(samples.count) / 16000
        peak = max(peak, samples.map(abs).max() ?? 0)
        if abs(seconds - expected) > 0.01 {
            mismatched += 1
            print("length mismatch: \(fields[0]) decoded \(fmt(seconds, 3)) s, tsv \(fields[1]) s")
        }
        count += 1
        totalSeconds += seconds
    }
    print("eval: \(count) files decoded, \(fmt(totalSeconds, 1)) s total, \(mismatched) length mismatches, peak |x| \(fmt(Double(peak), 3))")
}

func benchMain() async -> Int32 {
    let o = Options(Array(CommandLine.arguments.dropFirst()))
    do {
        switch o.command {
        case "download": try await runDownload(o)
        case "bench": try await runBench(o)
        case "check-audio": try runCheckAudio(o)
        default:
            eprint("usage: whisperkit-bench download|bench|check-audio [options]  (see Sources/whisperkit-bench/Bench.swift)")
            return 2
        }
        return 0
    } catch {
        eprint("error: \(error)")
        return 1
    }
}
