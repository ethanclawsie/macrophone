import CoreGraphics
import Foundation
import MacrophoneCore

/// Hold-to-talk on a single modifier key, via a listen-only event tap on its own thread.
///
/// Modifier-only keys are the only ones whose press *and* release reach a tap even while Secure
/// Input is on (password fields, Terminal's Secure Keyboard Entry). Any other key pressed while the
/// modifier is held means it's being used for a shortcut, so the take is abandoned.
final class HotkeyMonitor: @unchecked Sendable {
    enum Key: String, CaseIterable {
        case rightOption, rightCommand, rightControl, fn

        var keyCode: Int64 {
            switch self {
            case .rightOption: 61
            case .rightCommand: 54
            case .rightControl: 62
            case .fn: 63
            }
        }

        var label: String {
            switch self {
            case .rightOption: "Right ⌥"
            case .rightCommand: "Right ⌘"
            case .rightControl: "Right ⌃"
            case .fn: "fn"
            }
        }

        func isDown(_ flags: CGEventFlags) -> Bool {
            // Device-dependent bits from IOKit's NX_DEVICER*KEYMASK tell left and right apart.
            switch self {
            case .rightOption: flags.rawValue & 0x40 != 0
            case .rightCommand: flags.rawValue & 0x10 != 0
            case .rightControl: flags.rawValue & 0x2000 != 0
            case .fn: flags.contains(.maskSecondaryFn)
            }
        }
    }

    /// Called on the hotkey thread with `Clock.now` at the moment of the event.
    var onPress: ((UInt64) -> Void)?
    var onRelease: ((UInt64, _ shift: Bool) -> Void)?
    var onCancel: (() -> Void)?

    let key: Key
    private var tap: CFMachPort?
    private var runLoop: CFRunLoop?
    private var isDown = false
    private var abandoned = false

    init(key: Key) { self.key = key }

    var isEnabled: Bool { tap.map { CGEvent.tapIsEnabled(tap: $0) } ?? false }

    /// Returns false when the tap can't be created (Input Monitoring not granted yet).
    func start() -> Bool {
        let mask = CGEventMask((1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.keyDown.rawValue))
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly, eventsOfInterest: mask,
            callback: { _, type, event, refcon in
                if let refcon { Unmanaged<HotkeyMonitor>.fromOpaque(refcon).takeUnretainedValue().handle(type, event) }
                return Unmanaged.passUnretained(event)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        ) else { return false }
        self.tap = tap

        let ready = DispatchSemaphore(value: 0)
        let thread = Thread {
            // The tap only holds an unretained pointer to the monitor: keep it alive until the run loop exits.
            withExtendedLifetime(self) {
                let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
                CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
                CGEvent.tapEnable(tap: tap, enable: true)
                self.runLoop = CFRunLoopGetCurrent()
                ready.signal()
                CFRunLoopRun()
            }
        }
        thread.name = "macrophone.hotkey"
        thread.qualityOfService = .userInteractive
        thread.start()
        ready.wait()
        return true
    }

    /// Removes the tap and ends its thread. Call before replacing the monitor.
    func stop() {
        guard let tap, let runLoop else { return }
        CGEvent.tapEnable(tap: tap, enable: false)
        CFMachPortInvalidate(tap)
        CFRunLoopStop(runLoop)
        self.tap = nil
        self.runLoop = nil
    }

    /// Re-enables the tap if macOS disabled it (it can go quiet after the app is re-signed).
    func reviveIfNeeded() {
        if let tap, !CGEvent.tapIsEnabled(tap: tap) { CGEvent.tapEnable(tap: tap, enable: true) }
    }

    private func handle(_ type: CGEventType, _ event: CGEvent) {
        switch type {
        case .tapDisabledByTimeout, .tapDisabledByUserInput:
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
        case .flagsChanged:
            guard event.getIntegerValueField(.keyboardEventKeycode) == key.keyCode else { return }
            let down = key.isDown(event.flags)
            if down, !isDown {
                isDown = true
                abandoned = false
                onPress?(Clock.now)
            } else if !down, isDown {
                isDown = false
                if !abandoned { onRelease?(Clock.now, event.flags.contains(.maskShift)) }
            }
        case .keyDown:
            guard isDown, !abandoned else { return }
            abandoned = true
            onCancel?()
        default:
            break
        }
    }
}
