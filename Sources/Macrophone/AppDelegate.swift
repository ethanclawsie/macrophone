import AppKit
import MacrophoneCore

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    enum State: Equatable {
        case loading, ready, recording, transcribing
        case error(String)
    }

    private var statusItem: NSStatusItem!
    private var config = MacrophoneConfig.load()
    private var controller: DictationController?
    private var hotkey: HotkeyMonitor?
    private var holding = false
    private var pressedAt: UInt64 = 0
    private var lastText = ""
    private var lastReport = ""
    private var permissionTimer: Timer?

    private var state: State = .loading {
        didSet { updateIcon() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
        updateIcon()

        if !Permissions.microphone { Permissions.requestMicrophone { _ in } }
        if !Permissions.accessibility { Permissions.requestAccessibility() }
        loadEngine()
        startHotkey()
    }

    // MARK: Engine

    private func loadEngine() {
        let config = self.config
        state = .loading
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let engine = try WhisperEngine(modelPath: config.modelPath, threads: Int32(config.threads),
                                               language: config.language)
                engine.warmup()
                let vad = try StreamingVAD(modelPath: config.vadModelPath)
                let controller = DictationController(runner: InferenceRunner(engine: engine), vad: vad)
                controller.config = config.pipeline
                controller.micKeepAlive = config.micKeepAliveSeconds
                controller.kickGPUOnPress = config.kickGPUOnPress
                DispatchQueue.main.async {
                    self.controller = controller
                    if Permissions.microphone { controller.prepareMicrophone(preferBuiltIn: config.preferBuiltInMic) }
                    if self.state == .loading { self.state = .ready }
                }
            } catch {
                DispatchQueue.main.async { self.state = .error("\(error)") }
            }
        }
    }

    private func applyConfig() {
        config = MacrophoneConfig.load()
        controller?.config = config.pipeline
        controller?.micKeepAlive = config.micKeepAliveSeconds
        controller?.kickGPUOnPress = config.kickGPUOnPress
        if hotkey?.key.rawValue != config.hotkey { startHotkey() }
    }

    // MARK: Hotkey

    private func startHotkey() {
        hotkey?.stop()
        hotkey = nil
        let key = HotkeyMonitor.Key(rawValue: config.hotkey) ?? .rightOption
        let monitor = HotkeyMonitor(key: key)
        monitor.onPress = { at in DispatchQueue.main.async { self.pressed(at: at) } }
        monitor.onRelease = { at, shift in DispatchQueue.main.async { self.released(at: at, shift: shift) } }
        monitor.onCancel = { DispatchQueue.main.async { self.abandon() } }
        guard monitor.start() else {
            // No Input Monitoring permission yet: ask, then keep retrying until it's granted.
            Permissions.requestInputMonitoring()
            permissionTimer?.invalidate()
            permissionTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
                MainActor.assumeIsolated {
                    if Permissions.inputMonitoring {
                        self.permissionTimer?.invalidate()
                        self.startHotkey()
                    }
                }
            }
            return
        }
        hotkey = monitor
        permissionTimer?.invalidate()
        // Taps can silently stop after the app is re-signed; check once in a while.
        permissionTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in
            MainActor.assumeIsolated { self.hotkey?.reviveIfNeeded() }
        }
    }

    private func pressed(at: UInt64) {
        guard let controller, !holding else { return }
        holding = true
        pressedAt = at
        state = .recording
        controller.press { error in
            DispatchQueue.main.async {
                self.holding = false
                self.state = .error("Microphone: \(error)")
            }
        }
    }

    private func released(at: UInt64, shift: Bool) {
        guard let controller, holding else { return }
        holding = false
        if Double(at &- pressedAt) / 1e6 < Double(config.minHoldMs) {
            controller.cancel()
            state = .ready
            return
        }
        state = .transcribing
        controller.release { outcome in
            DispatchQueue.main.async { self.deliver(outcome, releasedAt: at, submit: shift && self.config.shiftSubmits) }
        }
    }

    private func abandon() {
        guard holding else { return }
        holding = false
        controller?.cancel()
        state = .ready
    }

    private func deliver(_ outcome: DictationController.Outcome, releasedAt: UInt64, submit: Bool) {
        let result = outcome.result
        let text = config.cleanup.apply(result.text, speechSeconds: result.stats.speechSeconds)
        if !text.isEmpty {
            TextInserter.shared.insert(text, submit: submit, restoreClipboard: config.restoreClipboard,
                                       fallbackDelay: Double(config.restoreClipboardDelayMs) / 1000)
            lastText = text
        }
        let total = Double(Clock.now &- releasedAt) / 1e6
        if state == .transcribing { state = .ready }

        let s = result.stats
        let mic = outcome.micStartMs.map { String(format: "%.0f", $0) } ?? "?"
        let how = s.answeredFromSpeculation ? "speculation"
            : s.finalPassMs.map { String(format: "final pass %.0f ms on %.1f s", $0, s.finalPassSeconds ?? 0) } ?? "no speech"
        lastReport = String(format: "release→paste %.0f ms (%@) · mic start %@ ms · %.1f s audio · %d passes",
                            total, how, mic, s.audioSeconds, s.passes)
        LatencyLog.append(totalMs: total, outcome: outcome, chars: text.count)
    }

    // MARK: Menu

    private func updateIcon() {
        guard let button = statusItem?.button else { return }
        let symbol: String
        switch state {
        case .loading: symbol = "hourglass"
        case .ready: symbol = "waveform"
        case .recording: symbol = "mic.fill"
        case .transcribing: symbol = "ellipsis"
        case .error: symbol = "exclamationmark.triangle"
        }
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Macrophone")
        image?.isTemplate = true
        button.image = image
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let key = HotkeyMonitor.Key(rawValue: config.hotkey) ?? .rightOption
        let status: String
        switch state {
        case .loading: status = "Loading model…"
        case .ready, .recording, .transcribing: status = "Hold \(key.label) to talk · ⇧ on release to submit"
        case .error(let message): status = "Error: \(message)"
        }
        menu.addItem(withTitle: status, action: nil, keyEquivalent: "")
        if !lastReport.isEmpty { menu.addItem(withTitle: lastReport, action: nil, keyEquivalent: "") }
        if !lastText.isEmpty {
            let preview = lastText.count > 60 ? String(lastText.prefix(60)) + "…" : lastText
            menu.addItem(item("Copy “\(preview.trimmingCharacters(in: .whitespaces))”", #selector(copyLast)))
        }

        var missing: [NSMenuItem] = []
        if !Permissions.microphone { missing.append(item("Grant Microphone Access…", #selector(grantMicrophone))) }
        if !Permissions.inputMonitoring { missing.append(item("Grant Input Monitoring…", #selector(grantInputMonitoring))) }
        if !Permissions.accessibility { missing.append(item("Grant Accessibility (to paste)…", #selector(grantAccessibility))) }
        if !missing.isEmpty {
            menu.addItem(.separator())
            missing.forEach(menu.addItem)
        }

        menu.addItem(.separator())
        menu.addItem(item("Open Config File", #selector(openConfig)))
        menu.addItem(item("Reload Config", #selector(reloadConfig)))
        menu.addItem(item("Open Latency Log", #selector(openLog)))
        menu.addItem(.separator())
        menu.addItem(item("Quit Macrophone", #selector(quit), key: "q"))
    }

    private func item(_ title: String, _ action: Selector, key: String = "") -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
        item.target = self
        return item
    }

    @objc private func copyLast() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(lastText, forType: .string)
    }

    @objc private func grantMicrophone() { Permissions.openSettings("Privacy_Microphone") }
    @objc private func grantInputMonitoring() {
        Permissions.requestInputMonitoring()
        Permissions.openSettings("Privacy_ListenEvent")
    }
    @objc private func grantAccessibility() {
        Permissions.requestAccessibility()
        Permissions.openSettings("Privacy_Accessibility")
    }

    @objc private func openConfig() {
        if !FileManager.default.fileExists(atPath: Paths.config.path) { try? config.save() }
        NSWorkspace.shared.open(Paths.config)
    }

    @objc private func reloadConfig() { applyConfig() }
    @objc private func openLog() { NSWorkspace.shared.open(LatencyLog.url) }
    @objc private func quit() { NSApp.terminate(nil) }
}

/// One tab-separated line per dictation in ~/Library/Logs/Macrophone/latency.tsv.
enum LatencyLog {
    static let url = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/Macrophone/latency.tsv")

    static func append(totalMs: Double, outcome: DictationController.Outcome, chars: Int) {
        let s = outcome.result.stats
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let header = "time\trelease_to_paste_ms\tpipeline_ms\tfinal_pass_ms\tfinal_pass_s\twaited_ms\tfrom_speculation\tpasses\taudio_s\tspeech_s\tmic_start_ms\tchars\n"
            fm.createFile(atPath: url.path, contents: header.data(using: .utf8))
        }
        let fields: [String] = [
            ISO8601DateFormatter().string(from: Date()),
            String(format: "%.1f", totalMs),
            String(format: "%.1f", s.releaseToTextMs),
            s.finalPassMs.map { String(format: "%.1f", $0) } ?? "",
            s.finalPassSeconds.map { String(format: "%.2f", $0) } ?? "",
            String(format: "%.1f", s.waitedForInflightMs),
            s.answeredFromSpeculation ? "1" : "0",
            String(s.passes),
            String(format: "%.2f", s.audioSeconds),
            String(format: "%.2f", s.speechSeconds),
            outcome.micStartMs.map { String(format: "%.1f", $0) } ?? "",
            String(chars),
        ]
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write((fields.joined(separator: "\t") + "\n").data(using: .utf8)!)
            try? handle.close()
        }
    }
}
