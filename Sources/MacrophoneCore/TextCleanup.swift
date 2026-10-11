import Foundation

/// Final touches before text is inserted. Deliberately light: coding agents cope with raw speech,
/// and every extra step here is latency.
public struct TextCleanup: Sendable {
    /// Case-insensitive whole-phrase replacements, e.g. "clod code" → "Claude Code".
    public var replacements: [String: String]
    public var appendTrailingSpace: Bool

    public init(replacements: [String: String] = [:], appendTrailingSpace: Bool = true) {
        self.replacements = replacements
        self.appendTrailingSpace = appendTrailingSpace
    }

    /// Phrases Whisper is known to hallucinate on near-silence (YouTube subtitle artifacts).
    static let hallucinations: Set<String> = [
        "you", "thank you", "thank you.", "thanks.", "thanks for watching!", "thanks for watching.",
        "thank you for watching.", "thank you for watching!", "bye.", "bye!", "okay.", "so.",
        "subtitles by the amara.org community", "please subscribe.", "...",
    ]

    public func apply(_ raw: String, speechSeconds: Double) -> String {
        var text = raw.replacingOccurrences(of: #"\[[^\]]*\]|\*[^*]*\*"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if speechSeconds < 1.0, Self.hallucinations.contains(text.lowercased()) { return "" }
        for (from, to) in replacements where !from.isEmpty {
            let pattern = "(?<![\\w])" + NSRegularExpression.escapedPattern(for: from) + "(?![\\w])"
            text = text.replacingOccurrences(of: pattern, with: NSRegularExpression.escapedTemplate(for: to),
                                             options: [.regularExpression, .caseInsensitive])
        }
        guard !text.isEmpty else { return "" }
        return appendTrailingSpace ? text + " " : text
    }
}
