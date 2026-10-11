import Foundation

/// User settings, stored as plain JSON at ~/Library/Application Support/Macrophone/config.json.
/// Missing keys fall back to defaults, so the file only needs what you want to change.
public struct MacrophoneConfig: Codable, Sendable {
    /// rightOption | rightCommand | rightControl | fn
    public var hotkey = "rightOption"
    public var modelPath = Paths.defaultModel.path
    public var vadModelPath = Paths.defaultVADModel.path
    public var language = "en"
    public var threads = 4
    /// Names and jargon Whisper should spell your way; sent as prompt text.
    public var vocabulary: [String] = []
    /// Case-insensitive phrase replacements applied after transcription.
    public var replacements: [String: String] = [:]
    public var appendTrailingSpace = true
    /// Put the previous clipboard back after pasting.
    public var restoreClipboard = true
    public var restoreClipboardDelayMs = 600
    /// Holding Shift when releasing the hotkey also presses Return (submits the prompt).
    public var shiftSubmits = true
    /// Holds shorter than this are ignored (accidental taps).
    public var minHoldMs = 180
    public var micKeepAliveSeconds = 0.0
    /// Record from the built-in mic when the system input is a Bluetooth headset (avoids the slow,
    /// low-quality headset mode).
    public var preferBuiltInMic = true
    public var kickGPUOnPress = false
    public var releaseStrategy = ReleaseStrategy.adaptive
    public var speculateAfterMs = 160
    public var commitAfterSeconds = 6.0

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MacrophoneConfig()
        hotkey = try c.decodeIfPresent(String.self, forKey: .hotkey) ?? d.hotkey
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath) ?? d.modelPath
        vadModelPath = try c.decodeIfPresent(String.self, forKey: .vadModelPath) ?? d.vadModelPath
        language = try c.decodeIfPresent(String.self, forKey: .language) ?? d.language
        threads = try c.decodeIfPresent(Int.self, forKey: .threads) ?? d.threads
        vocabulary = try c.decodeIfPresent([String].self, forKey: .vocabulary) ?? d.vocabulary
        replacements = try c.decodeIfPresent([String: String].self, forKey: .replacements) ?? d.replacements
        appendTrailingSpace = try c.decodeIfPresent(Bool.self, forKey: .appendTrailingSpace) ?? d.appendTrailingSpace
        restoreClipboard = try c.decodeIfPresent(Bool.self, forKey: .restoreClipboard) ?? d.restoreClipboard
        restoreClipboardDelayMs = try c.decodeIfPresent(Int.self, forKey: .restoreClipboardDelayMs) ?? d.restoreClipboardDelayMs
        shiftSubmits = try c.decodeIfPresent(Bool.self, forKey: .shiftSubmits) ?? d.shiftSubmits
        minHoldMs = try c.decodeIfPresent(Int.self, forKey: .minHoldMs) ?? d.minHoldMs
        micKeepAliveSeconds = try c.decodeIfPresent(Double.self, forKey: .micKeepAliveSeconds) ?? d.micKeepAliveSeconds
        preferBuiltInMic = try c.decodeIfPresent(Bool.self, forKey: .preferBuiltInMic) ?? d.preferBuiltInMic
        kickGPUOnPress = try c.decodeIfPresent(Bool.self, forKey: .kickGPUOnPress) ?? d.kickGPUOnPress
        releaseStrategy = try c.decodeIfPresent(ReleaseStrategy.self, forKey: .releaseStrategy) ?? d.releaseStrategy
        speculateAfterMs = try c.decodeIfPresent(Int.self, forKey: .speculateAfterMs) ?? d.speculateAfterMs
        commitAfterSeconds = try c.decodeIfPresent(Double.self, forKey: .commitAfterSeconds) ?? d.commitAfterSeconds
    }

    /// Loads the config, writing a default file on first run.
    public static func load(from url: URL = Paths.config) -> MacrophoneConfig {
        if let data = try? Data(contentsOf: url), let config = try? JSONDecoder().decode(MacrophoneConfig.self, from: data) {
            return config
        }
        let config = MacrophoneConfig()
        if !FileManager.default.fileExists(atPath: url.path) { try? config.save(to: url) }
        return config
    }

    public func save(to url: URL = Paths.config) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    public var pipeline: PipelineConfig {
        var p = PipelineConfig()
        p.releaseStrategy = releaseStrategy
        p.speculateAfterMs = speculateAfterMs
        p.commitAfterSeconds = commitAfterSeconds
        let words = vocabulary.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        p.vocabularyPrompt = words.isEmpty ? nil : "Glossary: " + words.joined(separator: ", ") + "."
        return p
    }

    public var cleanup: TextCleanup {
        TextCleanup(replacements: replacements, appendTrailingSpace: appendTrailingSpace)
    }
}
