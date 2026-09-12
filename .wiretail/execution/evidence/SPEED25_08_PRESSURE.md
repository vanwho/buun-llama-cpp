# SPEED25_08_PRESSURE

Real Qwen3.8-27B-UD-IQ4_XS proof on an RTX 4080 (driver 595.91.07), source
`fedc600558214f2577f8038b02723bf3131e4951`.

The pressure driver used context 8192, 4096 prompt tokens, 128 fixed decode
tokens, 256-token pages, H=8 pages (32 logical pages), GPU Turbo4 K/V and FA.
It selected known cold logical page 0 through the ordinary live policy path.
The transaction evicted one page, promoted page 0 into physical slot 7,
completed 96 H2D chunks (4,325,376 useful/aligned bytes), and left 8 GPU pages
plus 7 host pages. Host and device readbacks had the same FNV-1a checksum:
`3935911351812092203`.

Driver timing: 4096/9,964.52 ms = 411.39 prompt tok/s; 128/4,331.44 ms =
29.53 decode tok/s. The 300-token all-fit control produced no H2D/promotion.

The timed server pressure request used the same logical pressure, with batch and
ubatch 256, and completed HTTP 200 for 3579 prompt + 128 output tokens in
12.643624 s: 425.66 prompt tok/s and 30.03 decode tok/s. Server metrics recorded
8 resident pages, 7 host pages, 12,976,100 H2D useful bytes, and 288 submitted
and completed transfer events. The request used the explicit deterministic
promotion probe (`--kv-test-force-page 0`) with MTP off to isolate target-page
movement. Native GPU Turbo4 MTP startup and a complete 3579+128 request were
also verified on the managed service; that smoke request did not expose a cold
H2D and is not counted as movement proof.

Raw artifacts: `raw/SPEED25_08_PRESSURE_driver.json`,
`raw/SPEED25_08_PRESSURE_server.txt`, and
`raw/SPEED25_08_PRESSURE_allfit.json`.
