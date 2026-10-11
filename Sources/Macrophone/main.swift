import AppKit

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate() // NSApplication.delegate is weak; this lives until app.run() returns
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
}
