| implementation | 1.1 s clip | 4.4 s | 13.1 s | 24.6 s |
|---|---:|---:|---:|---:|
| macrophone | 45 | 87 | 240 | 401 |
| macrophone-q8 | 50 | 102 | 283 | 473 |
| mlx | 299 | 349 | 472 | 583 |
| mlx-4bit | 304 | 327 | 381 | 425 |
| mlx-8bit | 319 | 351 | 429 | 498 |
| whispercpp-metal-q8_0 | 360 | 395 | 478 | 571 |
| whispercpp-metal-f16 | 366 | 417 | 541 | 664 |
| hf-mps | 368 | 418 | 536 | 654 |
| whispercpp-metal-q5_0 | 373 | 405 | 475 | 550 |
| whispercpp-metal-f16-cli-defaults | 378 | 456 | 614 | 788 |
| whisperkit-gpudec | 383 | 495 | 753 | 1008 |
| whisperkit | 394 | 538 | 866 | 1187 |
| whispercpp-coreml-f16 | 442 | 498 | 637 | 756 |
| fasterwhisper-cpu | 2111 | 2208 | 2605 | 2893 |
