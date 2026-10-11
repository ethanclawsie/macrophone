import Foundation

setvbuf(stdout, nil, _IOLBF, 0)  // line-buffered so progress shows through `tee`

// Run everything off the main actor at user-initiated QoS (how an app would call WhisperKit
// from a background task), then exit with the returned status.
let status = await Task.detached(priority: .userInitiated) { await benchMain() }.value
exit(status)
