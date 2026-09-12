# Interactive 26-03 frontier evidence

Repository root: `/srv/repos/vanwho/buun-llama-cpp`
Task: `26-03`
Cluster: `26h-scratch-frontier-v7`

The implementation carries structured prepare failures through the memory/cache
boundary, keeps an asynchronous mutable tail pinned until a later page takes
over, clears erased frontier indices, and keeps server context-limit wording
restricted to logical-capacity failures. Native speculative frontier behavior
was exercised by the deterministic model-free F5 probe; rejected suffixes did
not advance the canonical host frontier.

Commands and results:

```text
cmake --build build --target test-kv-pager test-kv-pager-model test-server-prompt-cache -j2
PASS
ctest --test-dir build --output-on-failure -R 'test-(kv-pager|server-prompt-cache)'
PASS: 5/5
build/bin/test-kv-pager-model
PASS: synthetic F1/F2/F3/F5; accepted=0..3 all frontier_valid=true
cmake --build build-cuda --target test-kv-pager test-kv-pager-model test-server-prompt-cache -j2
PASS
ctest --test-dir build-cuda --output-on-failure -R 'test-(kv-pager|server-prompt-cache)'
PASS: 5/5
build-cuda/bin/test-kv-pager-model
PASS: synthetic F1/F2/F3/F5; accepted=0..3 all frontier_valid=true
cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 -j2
build-cuda/bin/test-cuda-fattn-paged-turbo4
PASS: RTX 4080; 530 selected rows; query tile 19.360 ms; split-KV capacities 1/2/3
git diff --check
PASS
```

Live status: deferred. `sudo -n -v` failed with
`sudo: interactive authentication is required`; the existing Qwen service PID
768838 owns the configured port/VRAM. It was preserved, and unrelated 8092 was
not touched. No live eviction/promotion, native-MTP continuation, or
near-boundary generation claim is made here.

Next executable gate after authorization: rerun `sudo -n -v`, then use the
authorized Qwen service lifecycle for the packet's primary L8192/H4096/P256
proof, 512–1024-token continuation, and near-L-minus-reserve case. Record the
actual model/config identity, committed positions, retained page range,
eviction/promotion counts, checksums, target/draft scratch, and recurrent/MTP
checkpoint frontier before marking live proof available.
