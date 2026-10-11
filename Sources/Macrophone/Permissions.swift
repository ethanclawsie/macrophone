import AppKit
import ApplicationServices
import AVFoundation

/// The three permissions Macrophone needs: Microphone (record), Input Monitoring (see the hotkey) and
/// Accessibility (post ⌘V into the focused app).
enum Permissions {
    static var microphone: Bool { AVCaptureDevice.authorizationStatus(for: .audio) == .authorized }
    static var inputMonitoring: Bool { CGPreflightListenEventAccess() }
    static var accessibility: Bool { AXIsProcessTrusted() }

    static func requestMicrophone(_ done: @escaping @Sendable (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio, completionHandler: done)
    }

    static func requestInputMonitoring() { _ = CGRequestListenEventAccess() }

    static func requestAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }

    /// Privacy_Microphone, Privacy_ListenEvent or Privacy_Accessibility.
    static func openSettings(_ anchor: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") {
            NSWorkspace.shared.open(url)
        }
    }
}
