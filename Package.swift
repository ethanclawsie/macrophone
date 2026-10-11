// swift-tools-version: 6.0
import PackageDescription

// whisper.cpp is built separately as static libraries (scripts/build-whisper.sh → build/whisper)
// so the Metal backend can be configured with CMake; SwiftPM only links the result.
let whisperBuild = Context.packageDirectory + "/build/whisper"

let package = Package(
    name: "Macrophone",
    platforms: [.macOS("26.0")],
    targets: [
        // C headers of whisper.cpp + the static libraries it links against.
        .target(
            name: "CWhisper",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(whisperBuild)/src",
                    "-L\(whisperBuild)/ggml/src",
                    "-L\(whisperBuild)/ggml/src/ggml-metal",
                    "-L\(whisperBuild)/ggml/src/ggml-blas",
                ]),
                .linkedLibrary("whisper"),
                .linkedLibrary("ggml"),
                .linkedLibrary("ggml-base"),
                .linkedLibrary("ggml-cpu"),
                .linkedLibrary("ggml-metal"),
                .linkedLibrary("ggml-blas"),
                .linkedLibrary("c++"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        // Real-time-safe pieces that run on the Core Audio I/O thread (no Swift runtime there).
        .target(name: "CMacrophoneRT"),
        // Engine, VAD, audio capture and the dictation scheduler. No UI.
        .target(name: "MacrophoneCore", dependencies: ["CWhisper", "CMacrophoneRT"]),
        // The menu bar app.
        .executableTarget(name: "Macrophone", dependencies: ["MacrophoneCore"]),
        // Benchmark / evaluation CLI.
        .executableTarget(name: "macrophone-bench", dependencies: ["MacrophoneCore"]),
    ],
    swiftLanguageModes: [.v5]
)
