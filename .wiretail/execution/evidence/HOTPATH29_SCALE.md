# HOTPATH29_SCALE

Result: partial, stopped on a current-source CUDA runtime defect.

The current CUDA build admitted L32768 with H=64 pages (16,384 tokens), full-L
Turbo4 MTP, and B/U=128/64. Its first 1,200-token append completed at 648.66
pp tok/s and 65.61 tg tok/s, with C=1,200 (slot frontier 1,203), A observed
as 1,203 rows over five selected pages, four host pages, 17.3 MB of host seal
data, and zero useful H2D bytes or faults. The second append attempted 1,400
new tokens after 1,203 cached tokens, ran for 136.16 seconds without a valid
response, and crashed the server with an illegal device memory access reported
at `ggml_backend_cuda_synchronize` (`ggml/src/ggml-cuda/ggml-cuda.cu:2685`).
The service restarted automatically. This is a source/runtime failure, not a
capacity ceiling or a cold-promotion result.

After stopping the scale campaign, L131072 with H=256 pages (65,536 tokens)
was admitted with full-L native Turbo4 MTP. The measured target pool was
1.1073 GB, MTP was 131,072 rows / 138,543,104 bytes, charged memory was
15.4698 GB, reserved memory 1.57278 GB, and reported headroom 201.327 MB;
`nvidia-smi` showed 872 MiB free after startup. A 512-token smoke reached
C=512 (slot frontier 513), A observed as 515 rows over three selected pages,
and completed at 866.10 pp tok/s and 21.41 tg tok/s with 2 proposed / 0
accepted MTP tokens. It exported eight direct prefill routes and zero useful
H2D bytes/faults. This proves allocation plus smoke only, not full-context
population or operationally safe H.

The required stop-on-source-defect rule prevented C>H continuation, natural
cold promotion/use, no-fault decode, and a populated near-L q0. No 256K or
YaRN work was started. The previously successful L8192/H4096 native-MTP
bundle was restored and remains loaded on 8080; unrelated 8092 was not
touched. The raw append/smoke artifacts and crash journal are referenced in
`HOTPATH29_SCALE.json`.

## Verification

- `cmake --build build --target test-kv-attention-execution -j2 && build/bin/test-kv-attention-execution` — pass.
- `cmake --build build-cuda --target test-kv-attention-execution -j2 && build-cuda/bin/test-kv-attention-execution` — pass.
- `python3 tools/server/bench/test_resume_contract.py` — 12/12 pass.
- `python3 -m py_compile tools/server/bench/run-incremental-scale.py tools/server/bench/run-incremental-recall.py` — pass.
- `git diff --check` — pass.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — pass.

## Deferred verification

No hardware or authorization check was deferred: CUDA was available and
`sudo -n true` passed. Full-L scaling and cold movement remain deferred solely
until the owning continuation defect is repaired.
