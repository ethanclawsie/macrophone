import AVFoundation
import CoreAudio
import CMacrophoneRT

/// Microphone capture with the lowest practical latency.
///
/// An `AVAudioSinkNode` receives every hardware I/O buffer (~10 ms) on the real-time thread and hands
/// it to a lock-free C ring; `drain16k()` pulls from the ring on the pipeline queue and resamples to
/// 16 kHz mono. The graph is built and prepared once and kept between holds, so starting the device
/// on key-down is all that's left (tens of ms); nothing runs while stopped.
public final class AudioCapture: @unchecked Sendable {
    private var engine = AVAudioEngine()
    private var sink: AVAudioSinkNode?
    private let ring: OpaquePointer
    private var converter: AVAudioConverter?
    private var inputFormat: AVAudioFormat?
    private let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(WhisperEngine.sampleRate), channels: 1, interleaved: false)!
    private var scratch = [Float](repeating: 0, count: 1 << 16)
    private var configChanged = false
    private var observer: NSObjectProtocol?

    /// Record from the built-in microphone when the system input is a Bluetooth headset: opening a
    /// headset mic switches it to the low-quality call profile, which takes 0.5–2 s and degrades playback.
    public var preferBuiltInMic = true

    public private(set) var isRunning = false
    public private(set) var isPrepared = false
    /// Name of the input device in use (for the menu).
    public private(set) var deviceName = ""

    public init() {
        ring = macrophone_ring_create(1 << 21) // ~43 s at 48 kHz before the consumer must catch up
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil
        ) { [weak self] _ in self?.configChanged = true }
    }

    deinit {
        if let observer { NotificationCenter.default.removeObserver(observer) }
        stop()
        macrophone_ring_destroy(ring)
    }

    /// Builds and prepares the graph without starting the device (no privacy indicator).
    public func prepare() throws {
        if configChanged {
            // Device or format changed (e.g. headset plugged in): rebuild from scratch.
            teardown()
            engine = AVAudioEngine()
            configChanged = false
        }
        guard !isPrepared else { return }

        let input = engine.inputNode
        if preferBuiltInMic, let device = AudioDevices.builtInInputIfDefaultIsBluetooth() {
            var id = device.id
            if let unit = input.audioUnit {
                AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                     &id, UInt32(MemoryLayout<AudioDeviceID>.size))
            }
            deviceName = device.name
        } else {
            deviceName = AudioDevices.defaultInputName() ?? ""
        }

        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw MacrophoneError.noInputDevice }

        let ring = self.ring
        let sink = AVAudioSinkNode { timestamp, frames, abl in
            macrophone_ring_write_abl(ring, abl, frames, timestamp.pointee.mHostTime)
            return noErr
        }
        engine.attach(sink)
        engine.connect(input, to: sink, format: format)
        self.sink = sink

        let mono = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate, channels: 1, interleaved: false)!
        inputFormat = mono
        converter = AVAudioConverter(from: mono, to: outputFormat)
        converter?.sampleRateConverterQuality = AVAudioQuality.high.rawValue

        engine.prepare()
        isPrepared = true
    }

    /// Starts the input device if it isn't running already.
    public func start() throws {
        if configChanged { stop() }
        if isRunning { return }
        try prepare()
        try engine.start()
        isRunning = true
    }

    /// Stops the device but keeps the prepared graph for the next start.
    public func stop() {
        guard isRunning else { return }
        engine.stop()
        isRunning = false
    }

    private func teardown() {
        stop()
        if let sink {
            engine.disconnectNodeInput(sink)
            engine.detach(sink)
        }
        sink = nil
        isPrepared = false
    }

    /// Starts a new take: drops anything buffered and arms the first-buffer timestamp.
    public func beginTake() {
        macrophone_ring_discard(ring)
        macrophone_ring_arm(ring)
        converter?.reset()
    }

    /// Time (in `Clock.now` units) at which the first buffer of the current take arrived, if it has.
    public var firstBufferTime: UInt64? {
        let host = macrophone_ring_first_host_time(ring)
        return host == 0 ? nil : Clock.nanoseconds(hostTime: host)
    }

    /// Everything captured since the last call, as 16 kHz mono. Pass `flush` at the end of a take
    /// to also drain the resampler's internal delay line.
    public func drain16k(flush: Bool = false) -> [Float] {
        guard let converter, let inputFormat else { return [] }
        var out: [Float] = []
        while true {
            let n = scratch.withUnsafeMutableBufferPointer { macrophone_ring_read(ring, $0.baseAddress, UInt32($0.count)) }
            if n == 0 && !flush { break }
            let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: max(n, 1))!
            input.frameLength = n
            if n > 0 {
                scratch.withUnsafeBufferPointer { src in
                    input.floatChannelData![0].update(from: src.baseAddress!, count: Int(n))
                }
            }
            let capacity = AVAudioFrameCount(Double(n) * outputFormat.sampleRate / inputFormat.sampleRate) + 64
            let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity)!
            var supplied = false
            let endOfStream = flush && n == 0
            var error: NSError?
            _ = converter.convert(to: output, error: &error) { _, status in
                if supplied || n == 0 {
                    status.pointee = endOfStream ? .endOfStream : .noDataNow
                    return nil
                }
                supplied = true
                status.pointee = .haveData
                return input
            }
            if output.frameLength > 0 {
                out.append(contentsOf: UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
            }
            if endOfStream { break }
            if n < UInt32(scratch.count) && !flush { break }
        }
        return out
    }
}

/// Minimal Core Audio device queries.
enum AudioDevices {
    struct Device {
        let id: AudioDeviceID
        let name: String
        let transport: UInt32
        let hasInput: Bool
    }

    static func defaultInputName() -> String? {
        guard let id = defaultInputID() else { return nil }
        return device(id)?.name
    }

    /// The built-in microphone, but only when the default input is a Bluetooth device.
    static func builtInInputIfDefaultIsBluetooth() -> Device? {
        guard let defaultID = defaultInputID(), let current = device(defaultID) else { return nil }
        let bluetooth = current.transport == kAudioDeviceTransportTypeBluetooth
            || current.transport == kAudioDeviceTransportTypeBluetoothLE
        guard bluetooth else { return nil }
        return allDevices().first { $0.hasInput && $0.transport == kAudioDeviceTransportTypeBuiltIn }
    }

    private static func defaultInputID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id)
        return status == noErr && id != 0 ? id : nil
    }

    private static func allDevices() -> [Device] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr
        else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr
        else { return [] }
        return ids.compactMap(device)
    }

    private static func device(_ id: AudioDeviceID) -> Device? {
        var transport: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        AudioObjectGetPropertyData(id, &address, 0, nil, &size, &transport)

        var name: CFString = "" as CFString
        size = UInt32(MemoryLayout<CFString>.size)
        address.mSelector = kAudioObjectPropertyName
        _ = withUnsafeMutablePointer(to: &name) { AudioObjectGetPropertyData(id, &address, 0, nil, &size, $0) }

        address.mSelector = kAudioDevicePropertyStreamConfiguration
        address.mScope = kAudioObjectPropertyScopeInput
        var configSize: UInt32 = 0
        var hasInput = false
        if AudioObjectGetPropertyDataSize(id, &address, 0, nil, &configSize) == noErr, configSize > 0 {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(configSize), alignment: 16)
            defer { raw.deallocate() }
            if AudioObjectGetPropertyData(id, &address, 0, nil, &configSize, raw) == noErr {
                let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
                hasInput = list.contains { $0.mNumberChannels > 0 }
            }
        }
        return Device(id: id, name: name as String, transport: transport, hasInput: hasInput)
    }
}
