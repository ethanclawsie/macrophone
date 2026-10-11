import Foundation
import Metal

/// Keeps the GPU's clocks up while the hotkey is held.
///
/// The GPU drops to low clocks within moments of going idle, and a Whisper pass that starts cold runs
/// noticeably slower. While you talk the GPU is mostly idle, so the pass on release would start cold;
/// a light heartbeat (a few hundred microseconds of ALU work every few ms) keeps it ready.
public final class GPUWarmer: @unchecked Sendable {
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let buffer: MTLBuffer
    private let timerQueue = DispatchQueue(label: "macrophone.gpu-warmer", qos: .userInteractive)
    private var timer: DispatchSourceTimer?

    /// Threads per heartbeat dispatch (each runs `iterations` dependent FMAs).
    public var threads = 1 << 16
    public var iterations: Int32 = 512

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void macrophone_heartbeat(device float * out [[buffer(0)]], constant int & n [[buffer(1)]],
                                    uint id [[thread_position_in_grid]]) {
            float x = float(id) * 1e-6f;
            for (int i = 0; i < n; ++i) { x = fma(x, 0.999f, 1e-3f); }
            if (x == -1.0f) { out[id & 1023] = x; } // never true; keeps the loop alive
        }
        """
        guard let library = try? device.makeLibrary(source: source, options: nil),
              let function = library.makeFunction(name: "macrophone_heartbeat"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: 4096, options: .storageModeShared) else { return nil }
        self.queue = queue
        self.pipeline = pipeline
        self.buffer = buffer
    }

    /// One heartbeat, synchronously (for measuring its cost).
    public func beat(wait: Bool = false) {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else { return }
        var n = iterations
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 1)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        if wait { commands.waitUntilCompleted() }
    }

    /// Beats every `intervalMs` until `stop()`.
    public func start(intervalMs: Int = 8) {
        timerQueue.async { [self] in
            guard timer == nil else { return }
            let t = DispatchSource.makeTimerSource(flags: .strict, queue: timerQueue)
            t.schedule(deadline: .now(), repeating: .milliseconds(intervalMs), leeway: .milliseconds(1))
            t.setEventHandler { [weak self] in self?.beat() }
            t.resume()
            timer = t
        }
    }

    public func stop() {
        timerQueue.async { [self] in
            timer?.cancel()
            timer = nil
        }
    }
}
