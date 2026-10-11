import AVFoundation
import Foundation
import MacrophoneCore

// macrophone-bench — measure the engine and the dictation pipeline on WAV files.
//
//   macrophone-bench transcribe [--full] [--margin N] [--round N] [--runs N] [--prompt TEXT] FILE...
//   macrophone-bench simulate   [--tail-ms N] [--strategy tailOnly|openSegment] [--speculate-ms N] FILE...
//   macrophone-bench idle       [--sleep S,S,...] [--kick] FILE
//   macrophone-bench coldstart  FILE
//   macrophone-bench protocol   --name NAME            the published benchmark → bench/results/NAME-*.tsv
//   macrophone-bench eval       [--per-bucket N] [--full] [--out FILE]   quick WER (simple normalizer) + latency
//   macrophone-bench table | manifest | pick | score   result tables and eval-set tools
//
// Common: --model PATH  --vad PATH  --threads N

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print("usage: macrophone-bench transcribe|simulate|idle|coldstart|protocol|eval|table|manifest|pick|score [options] [FILE...]")
    exit(2)
}
args.removeFirst()

var options: [String: String] = [:]
var flags: Set<String> = []
var files: [String] = []
let valued: Set<String> = ["--model", "--vad", "--threads", "--margin", "--round", "--runs", "--prompt",
                           "--tail-ms", "--strategy", "--speculate-ms", "--sleep", "--commit-s",
                           "--manifest", "--per-bucket", "--out", "--list", "--name"]
var i = 0
while i < args.count {
    let a = args[i]
    if valued.contains(a), i + 1 < args.count {
        options[a] = args[i + 1]
        i += 2
    } else if a.hasPrefix("--") {
        flags.insert(a)
        i += 1
    } else {
        files.append(a)
        i += 1
    }
}

let modelPath = options["--model"] ?? Paths.defaultModel.path
let vadPath = options["--vad"] ?? Paths.defaultVADModel.path
let threads = Int32(options["--threads"] ?? "4") ?? 4

func loadAudio16k(_ path: String) throws -> [Float] {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let inFormat = file.processingFormat
    guard let buffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
        throw MacrophoneError.audioFile(path)
    }
    try file.read(into: buffer)
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    if inFormat.sampleRate == 16_000 && inFormat.channelCount == 1 {
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }
    let converter = AVAudioConverter(from: inFormat, to: target)!
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16_000 / inFormat.sampleRate) + 1024
    let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)!
    var done = false
    var error: NSError?
    converter.convert(to: out, error: &error) { _, status in
        if done {
            status.pointee = .endOfStream
            return nil
        }
        done = true
        status.pointee = .haveData
        return buffer
    }
    return Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
}

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s.isEmpty ? 0 : s[s.count / 2]
}

func f(_ x: Double, _ digits: Int = 1) -> String { String(format: "%.\(digits)f", x) }

func loadEngine() -> WhisperEngine {
    let t0 = Clock.now
    let peak = Float(options["--mel-peak"] ?? "1.5") ?? 1.5
    let engine = try! WhisperEngine(modelPath: modelPath, threads: threads, flashAttention: !flags.contains("--no-fa"),
                                    melPeak: flags.contains("--stock-mel") ? nil : peak)
    let loadMs = Clock.ms(since: t0)
    let t1 = Clock.now
    engine.warmup()
    print("model: \((modelPath as NSString).lastPathComponent)  load \(f(loadMs, 0)) ms  warmup \(f(Clock.ms(since: t1), 0)) ms  tail cache: \(engine.hasTailCache)")
    return engine
}

/// --full | --fitted | (default) tail;  --margin N  --round N  --decoder-tail-only
func windowOption() -> WhisperEngine.Window {
    let margin = Int(options["--margin"] ?? "16") ?? 16
    let round = Int(options["--round"] ?? "16") ?? 16
    if flags.contains("--full") { return .full }
    if flags.contains("--fitted") { return .fitted(margin: margin, roundTo: round) }
    return .tail(margin: margin, roundTo: round, encoderTail: !flags.contains("--decoder-tail-only"))
}

func windowLabel() -> String {
    let norm = flags.contains("--stock-mel") ? "stock-mel" : "mel-peak \(options["--mel-peak"] ?? "1.5")"
    switch windowOption() {
    case .full: return "full, \(norm)"
    case let .fitted(m, r): return "fitted(margin \(m), round \(r)), \(norm)"
    case let .tail(m, r, enc): return "tail(margin \(m), round \(r)\(enc ? "" : ", decoder only")), \(norm)"
    }
}

switch command {
case "transcribe":
    let engine = loadEngine()
    let runs = Int(options["--runs"] ?? "5") ?? 5
    let window = windowOption()
    print("window: \(windowLabel())")
    for path in files {
        let audio = try loadAudio16k(path)
        var times: [Double] = []
        var last: Transcript?
        audio.withUnsafeBufferPointer { buf in
            for _ in 0..<runs {
                let t = engine.transcribe(buf, prompt: options["--prompt"], window: window)
                times.append(t.milliseconds)
                last = t
            }
        }
        guard let t = last else { continue }
        let name = (path as NSString).lastPathComponent
        print("\(name)  \(f(t.audioSeconds))s  ctx=\(t.audioCtx)  median \(f(median(times))) ms  min \(f(times.min() ?? 0)) ms  tokens=\(t.tokenCount)")
        let b = t.breakdown
        print("    mel \(f(b.mel)) · encode \(f(b.encode)) · prompt \(f(b.prompt)) · decode \(f(b.decode)) (\(b.decodeSteps) steps, \(f(b.decode / Double(max(1, b.decodeSteps)), 2)) ms/step) · sample \(f(b.sample)) · other \(f(t.milliseconds - b.mel - b.encode - b.prompt - b.decode - b.sample))")
        print("    \(t.text)")
    }

case "simulate":
    let engine = loadEngine()
    let runner = InferenceRunner(engine: engine)
    let vad = try StreamingVAD(modelPath: vadPath)
    var config = PipelineConfig()
    if let s = options["--strategy"], let strategy = ReleaseStrategy(rawValue: s) { config.releaseStrategy = strategy }
    if let ms = options["--speculate-ms"].flatMap(Int.init) { config.speculateAfterMs = ms }
    if let s = options["--commit-s"].flatMap(Double.init) { config.commitAfterSeconds = s }
    let tailMs = Int(options["--tail-ms"] ?? "0") ?? 0
    let queue = DispatchQueue(label: "bench.pipeline", qos: .userInteractive)
    let chunk = 160 // 10 ms, like the app's pump

    for path in files {
        var audio = try loadAudio16k(path)
        // Room-tone tail between the end of speech and the key release.
        audio += (0..<(tailMs * 16)).map { _ in Float.random(in: -0.0005...0.0005) }
        let session = queue.sync { DictationSession(runner: runner, vad: vad, config: config, queue: queue) }
        let start = Clock.now
        var offset = 0
        while offset < audio.count {
            let end = min(offset + chunk, audio.count)
            let piece = Array(audio[offset..<end])
            queue.async { session.append(piece) }
            offset = end
            if !flags.contains("--fast") {
                let due = start + UInt64(offset) * 1_000_000_000 / 16_000
                let now = Clock.now
                if due > now { usleep(UInt32((due - now) / 1000)) }
            }
        }
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: DictationSession.Result?
        queue.async {
            session.finish { r in
                result = r
                done.signal()
            }
        }
        done.wait()
        guard let r = result else { continue }
        let s = r.stats
        let name = (path as NSString).lastPathComponent
        let finalPass = s.finalPassMs.map { "final pass \(f($0)) ms over \(f(s.finalPassSeconds ?? 0))s" } ?? "no final pass"
        print("\(name)  audio \(f(s.audioSeconds))s  release→text \(f(s.releaseToTextMs)) ms  (\(s.answeredFromSpeculation ? "speculation" : finalPass), waited \(f(s.waitedForInflightMs)) ms, passes \(s.passes), draft tokens \(s.draftAccepted))")
        print("    \(r.text)")
    }

case "idle":
    // How much slower is the first pass after the GPU has been idle?
    let engine = loadEngine()
    let runner = InferenceRunner(engine: engine)
    guard let path = files.first else { exit(2) }
    let audio = try loadAudio16k(path)
    let sleeps = (options["--sleep"] ?? "0,2,10,30").split(separator: ",").compactMap { Double($0) }
    for s in sleeps {
        usleep(UInt32(s * 1_000_000))
        if flags.contains("--kick") {
            runner.kick()
            usleep(150_000)
        }
        let t = runner.sync { e in audio.withUnsafeBufferPointer { e.transcribe($0) } }
        let again = runner.sync { e in audio.withUnsafeBufferPointer { e.transcribe($0) } }
        print("after \(f(s, 0)) s idle: \(f(t.milliseconds)) ms   (immediately again: \(f(again.milliseconds)) ms)")
    }

case "coldstart":
    // macrophone-bench coldstart FILE — pass latency after 1.5 s of GPU idle (as while you talk), with GPU
    // heartbeats of different intensity filling the gap.
    let engine = loadEngine()
    guard let path = files.first, let warmer = GPUWarmer() else { exit(2) }
    let audio = try loadAudio16k(path)
    func pass() -> Double { audio.withUnsafeBufferPointer { engine.transcribe($0).milliseconds } }
    for _ in 0..<3 { _ = pass() }
    print("warm (back to back): \(f(median((0..<5).map { _ in pass() }))) ms")
    let gap: UInt32 = 1_500_000
    var idle: [Double] = []
    for _ in 0..<4 { usleep(gap); idle.append(pass()) }
    print("after 1.5 s idle: \(f(median(idle))) ms")
    for (threads, iterations, interval) in [(1 << 16, 512, 8), (1 << 18, 512, 8), (1 << 18, 1024, 8), (1 << 19, 1024, 10), (1 << 20, 1024, 16)] {
        warmer.threads = threads
        warmer.iterations = Int32(iterations)
        let t0 = Clock.now
        warmer.beat(wait: true)
        let cost = Clock.ms(since: t0)
        var v: [Double] = []
        for _ in 0..<4 {
            warmer.start(intervalMs: interval); usleep(gap); warmer.stop(); usleep(3000)
            v.append(pass())
        }
        print("heartbeat \(f(cost, 2)) ms every \(interval) ms (~\(f(100 * cost / Double(interval), 0))% duty): \(f(median(v))) ms")
    }

case "manifest":
    // macrophone-bench manifest LIBRISPEECH_DIR OUT.tsv  →  path \t seconds \t reference
    guard files.count == 2 else { exit(2) }
    let root = URL(fileURLWithPath: files[0])
    var lines: [String] = []
    let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator where url.lastPathComponent.hasSuffix(".trans.txt") {
        for line in try String(contentsOf: url, encoding: .utf8).split(separator: "\n") {
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let audio = url.deletingLastPathComponent().appendingPathComponent("\(parts[0]).flac")
            guard let file = try? AVAudioFile(forReading: audio) else { continue }
            let seconds = Double(file.length) / file.fileFormat.sampleRate
            lines.append("\(audio.path)\t\(String(format: "%.3f", seconds))\t\(parts[1])")
        }
    }
    lines.sort()
    try lines.joined(separator: "\n").write(toFile: files[1], atomically: true, encoding: .utf8)
    print("\(lines.count) utterances → \(files[1])")

case "protocol":
    // macrophone-bench protocol --name macrophone [model/window options]
    // Same protocol as the competitor harnesses (bench/competitors/*/run.sh): load once, warm up,
    // time only the transcription of in-memory audio. Writes bench/results/<name>-{latency,eval}.tsv.
    let engine = loadEngine()
    let window = windowOption()
    let name = options["--name"] ?? "macrophone"
    let root = FileManager.default.currentDirectoryPath
    func clean(_ s: String) -> String { s.replacingOccurrences(of: "\t", with: " ").replacingOccurrences(of: "\n", with: " ") }
    var latency: [String] = []
    for clip in ["short", "medium", "long", "xlong"] {
        let audio = try loadAudio16k("\(root)/bench/audio/\(clip).wav")
        audio.withUnsafeBufferPointer { b in
            _ = engine.transcribe(b, window: window) // per-clip warm-up, like the other harnesses' global warm-up
            for run in 1...5 {
                let t = engine.transcribe(b, window: window)
                latency.append("\(clip)\t\(run)\t\(f(t.milliseconds, 2))\t\(clean(t.text))")
            }
        }
    }
    var eval: [String] = []
    for item in pickEvalSet(perBucket: 40) {
        let audio = try loadAudio16k(item.path)
        let t = audio.withUnsafeBufferPointer { engine.transcribe($0, window: window) }
        eval.append("\((item.path as NSString).lastPathComponent)\t\(f(t.milliseconds, 2))\t\(clean(t.text))")
    }
    try latency.joined(separator: "\n").write(toFile: "\(root)/bench/results/\(name)-latency.tsv", atomically: true, encoding: .utf8)
    try eval.joined(separator: "\n").write(toFile: "\(root)/bench/results/\(name)-eval.tsv", atomically: true, encoding: .utf8)
    print("\(name): \(windowLabel()) → bench/results/\(name)-{latency,eval}.tsv")

case "table":
    // macrophone-bench table — markdown table of the median latency per clip from every bench/results/*-latency.tsv
    // (accuracy is scored by bench/score.py with Whisper's English text normalizer)
    let root = FileManager.default.currentDirectoryPath + "/bench/results"
    let names = try FileManager.default.contentsOfDirectory(atPath: root)
        .filter { $0.hasSuffix("-latency.tsv") }.map { String($0.dropLast("-latency.tsv".count)) }
    var lines: [(Double, String)] = []
    for name in names {
        var byClip: [String: [Double]] = [:]
        let text = (try? String(contentsOfFile: "\(root)/\(name)-latency.tsv", encoding: .utf8)) ?? ""
        for row in text.split(separator: "\n").map({ $0.split(separator: "\t", maxSplits: 3, omittingEmptySubsequences: false) })
        where row.count >= 3 {
            byClip[String(row[0]), default: []].append(Double(row[2]) ?? 0)
        }
        let clips = ["short", "medium", "long", "xlong"].map { byClip[$0].map { f(median($0), 0) } ?? "–" }
        lines.append((byClip["short"].map(median) ?? .infinity, "| \(name) | \(clips.joined(separator: " | ")) |"))
    }
    print("| implementation | 1.1 s clip | 4.4 s | 13.1 s | 24.6 s |")
    print("|---|---:|---:|---:|---:|")
    for line in lines.sorted(by: { $0.0 < $1.0 }) { print(line.1) }

case "pick":
    // macrophone-bench pick --per-bucket N OUT.tsv — the fixed evaluation subset shared by every harness.
    guard let out = files.first else { exit(2) }
    let picked = pickEvalSet(perBucket: Int(options["--per-bucket"] ?? "40") ?? 40)
    try picked.map { "\($0.path)\t\(f($0.seconds, 3))\t\($0.reference)" }.joined(separator: "\n")
        .write(toFile: out, atomically: true, encoding: .utf8)
    print("\(picked.count) utterances → \(out)")

case "score":
    // macrophone-bench score --list eval.tsv RESULTS.tsv...  (results: id \t ms \t text, id = file name)
    let list = try String(contentsOfFile: options["--list"] ?? "bench/data/eval-160.tsv", encoding: .utf8)
        .split(separator: "\n").map { $0.split(separator: "\t", maxSplits: 2).map(String.init) }
    var refs: [String: (Double, String)] = [:]
    for row in list where row.count == 3 { refs[(row[0] as NSString).lastPathComponent] = (Double(row[1])!, row[2]) }
    let bucketNames = ["<3s", "3-6s", "6-12s", "12s+"]
    func bucket(_ s: Double) -> Int { s < 3 ? 0 : s < 6 ? 1 : s < 12 ? 2 : 3 }
    for path in files {
        var errors = [Int](repeating: 0, count: 4), words = [Int](repeating: 0, count: 4)
        var lat = [[Double]](repeating: [], count: 4)
        var missing = 0
        let rows = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
            .map { $0.split(separator: "\t", maxSplits: 2, omittingEmptySubsequences: false).map(String.init) }
        var seen = Set<String>()
        for row in rows where row.count >= 2 {
            guard let (seconds, reference) = refs[row[0]] else { continue }
            seen.insert(row[0])
            let b = bucket(seconds)
            let ref = normalizeForWER(reference)
            errors[b] += editDistance(ref, normalizeForWER(row.count > 2 ? row[2] : ""))
            words[b] += ref.count
            if let ms = Double(row[1]) { lat[b].append(ms) }
        }
        missing = refs.count - seen.count
        print("\((path as NSString).lastPathComponent)\(missing > 0 ? "  (missing \(missing))" : "")")
        for b in 0..<4 where words[b] > 0 {
            print("  \(bucketNames[b].padding(toLength: 6, withPad: " ", startingAt: 0)) n=\(lat[b].count)  WER \(f(100 * Double(errors[b]) / Double(words[b]), 2))%  median \(f(median(lat[b]))) ms")
        }
        print("  all    n=\(seen.count)  WER \(f(100 * Double(errors.reduce(0, +)) / Double(max(1, words.reduce(0, +))), 2))%  median \(f(median(lat.flatMap { $0 }))) ms")
    }

case "eval":
    // macrophone-bench eval --manifest M [--per-bucket N] [--full | --margin N --round N] [--out results.tsv]
    let engine = loadEngine()
    let perBucket = Int(options["--per-bucket"] ?? "50") ?? 50
    let window = windowOption()
    let buckets: [(String, ClosedRange<Double>)] = [("<3s", 0...3), ("3-6s", 3...6), ("6-12s", 6...12), ("12s+", 12...60)]
    let picked = pickEvalSet(perBucket: perBucket)

    var errors = [Int](repeating: 0, count: buckets.count)
    var words = [Int](repeating: 0, count: buckets.count)
    var latency = [[Double]](repeating: [], count: buckets.count)
    var report: [String] = []
    for (n, item) in picked.enumerated() {
        let audio = try loadAudio16k(item.path)
        let t = audio.withUnsafeBufferPointer { engine.transcribe($0, window: window) }
        let ref = normalizeForWER(item.reference)
        let hyp = normalizeForWER(t.text)
        let e = editDistance(ref, hyp)
        errors[item.bucket] += e
        words[item.bucket] += ref.count
        latency[item.bucket].append(t.milliseconds)
        report.append([(item.path as NSString).lastPathComponent, f(item.seconds, 2), String(e), String(ref.count),
                       f(t.milliseconds), String(t.audioCtx), t.text].joined(separator: "\t"))
        if flags.contains("--verbose"), e > 0 {
            print("[\(n)] \(f(item.seconds))s err=\(e)/\(ref.count)\n    ref: \(ref.joined(separator: " "))\n    hyp: \(hyp.joined(separator: " "))")
        }
    }
    print("window: \(windowLabel())")
    for (b, bucket) in buckets.enumerated() where words[b] > 0 {
        let wer = 100 * Double(errors[b]) / Double(words[b])
        print("  \(bucket.0.padding(toLength: 6, withPad: " ", startingAt: 0)) n=\(latency[b].count)  WER \(f(wer, 2))%  median \(f(median(latency[b]))) ms")
    }
    let total = 100 * Double(errors.reduce(0, +)) / Double(max(1, words.reduce(0, +)))
    print("  all    n=\(picked.count)  WER \(f(total, 2))%  median \(f(median(latency.flatMap { $0 }))) ms")
    if let out = options["--out"] { try report.joined(separator: "\n").write(toFile: out, atomically: true, encoding: .utf8) }

default:
    print("unknown command \(command)")
    exit(2)
}

/// Evenly spaced utterances from each duration bucket of the LibriSpeech manifest (deterministic).
func pickEvalSet(perBucket: Int) -> [(path: String, seconds: Double, reference: String, bucket: Int)] {
    let manifest = (try? String(contentsOfFile: options["--manifest"] ?? "bench/data/manifest.tsv", encoding: .utf8)) ?? ""
    let ranges: [Range<Double>] = [0..<3, 3..<6, 6..<12, 12..<60]
    let rows = manifest.split(separator: "\n").map { $0.split(separator: "\t", maxSplits: 2).map(String.init) }
    var picked: [(path: String, seconds: Double, reference: String, bucket: Int)] = []
    for (b, range) in ranges.enumerated() {
        let inBucket = rows.filter { $0.count == 3 && range.contains(Double($0[1]) ?? -1) }
        let step = max(1, inBucket.count / max(perBucket, 1))
        for index in stride(from: 0, to: inBucket.count, by: step).prefix(perBucket) {
            let row = inBucket[index]
            picked.append((row[0], Double(row[1])!, row[2], b))
        }
    }
    return picked
}

func normalizeForWER(_ text: String) -> [String] {
    let lowered = text.lowercased().replacingOccurrences(of: "-", with: " ")
    let kept = lowered.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) || $0 == "'" || $0 == " " ? Character($0) : " " }
    return String(kept).split(separator: " ").map(String.init)
}

func editDistance(_ a: [String], _ b: [String]) -> Int {
    if a.isEmpty { return b.count }
    if b.isEmpty { return a.count }
    var prev = Array(0...b.count)
    var cur = [Int](repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        cur[0] = i
        for j in 1...b.count {
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
        }
        swap(&prev, &cur)
    }
    return prev[b.count]
}
