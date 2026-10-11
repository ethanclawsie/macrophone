import CWhisper
import Foundation

public enum MacrophoneError: Error, CustomStringConvertible {
    case modelLoad(String)
    case noInputDevice
    case audioFile(String)

    public var description: String {
        switch self {
        case .modelLoad(let path): "could not load model at \(path)"
        case .noInputDevice: "no audio input device available"
        case .audioFile(let path): "could not read audio file \(path)"
        }
    }
}

/// Monotonic time helpers (nanoseconds, plus conversion from Core Audio host time).
public enum Clock {
    public static var now: UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

    public static func ms(since start: UInt64) -> Double { Double(now &- start) / 1e6 }

    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    /// Converts a mach host time (e.g. `AudioTimeStamp.mHostTime`) to the `now` time base.
    public static func nanoseconds(hostTime: UInt64) -> UInt64 {
        hostTime * UInt64(timebase.numer) / UInt64(timebase.denom)
    }
}

public enum Log {
    /// Set `MACROPHONE_VERBOSE=1` to see whisper.cpp / ggml info logs.
    nonisolated(unsafe) public static var verbose = ProcessInfo.processInfo.environment["MACROPHONE_VERBOSE"] != nil
    nonisolated(unsafe) private static var installed = false
    nonisolated(unsafe) fileprivate static var lastLevelShown = true

    /// whisper.cpp logs every VAD call and model detail at INFO level; only keep warnings and errors.
    static func installWhisperLogFilter() {
        guard !installed else { return }
        installed = true
        whisper_log_set({ level, text, _ in
            guard let text else { return }
            let show: Bool
            switch level {
            case GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR: show = true
            case GGML_LOG_LEVEL_CONT: show = Log.lastLevelShown
            default: show = Log.verbose
            }
            if level != GGML_LOG_LEVEL_CONT { Log.lastLevelShown = show }
            if show { fputs(String(cString: text), stderr) }
        }, nil)
    }
}

/// Default locations under ~/Library/Application Support/Macrophone.
public enum Paths {
    public static var support: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Macrophone", isDirectory: true)
    }
    public static var models: URL { support.appendingPathComponent("models", isDirectory: true) }
    public static var config: URL { support.appendingPathComponent("config.json") }
    public static var defaultModel: URL { models.appendingPathComponent("ggml-large-v3-turbo-q4_0.bin") }
    public static var defaultVADModel: URL { models.appendingPathComponent("ggml-silero-v6.2.0.bin") }
}

@inline(__always)
func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    guard let string else { return body(nil) }
    return string.withCString { body($0) }
}
