// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "whisperkit-bench",
    platforms: [.macOS(.v15)],
    dependencies: [
        // argmax-oss-swift (formerly argmaxinc/WhisperKit), latest tagged release.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "1.1.1"),
    ],
    targets: [
        .executableTarget(
            name: "whisperkit-bench",
            dependencies: [
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
            ]
        ),
    ],
    swiftLanguageModes: [.v5]
)
