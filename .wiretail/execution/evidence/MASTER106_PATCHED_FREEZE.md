# Patched master build freeze — 106-02 recovery

- Cohort: `master106-draftkv-20261013`
- Protocol: `master-baseline-patched-draftkv-20261013`
- Base master: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`
- Source worktree: `/srv/ai/paged-kv/worktrees/master106-draftkv-20261013`
- Patch SHA-256: `4caf807505c4a34cc6e0ce0d21fc8b81f79b4906359f7f08a961543e45f94937`
- Executable SHA-256: `5e6055febf59618df1d24692ab7392aff66bd36a1e71bcdec33d8ec76ac5ded8`
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Build log SHA-256: `8b2e903f1724619083a504fd6fbb9f416e55939a13201be309327b68f3c6d557`
- Current JSON freeze SHA-256: `e875ec283b18f8d43c08844cb77be83310ab5bbcc1dba7f6d546f8633874f772`

## Patch scope

- `common/arg.cpp`
- `common/common.cpp`
- `common/common.h`
- `common/speculative.cpp`
- `tests/test-arg-parser.cpp`
- `tools/server/server-context.cpp`

## Loaded project DSOs

| DSO | SHA-256 |
|---|---|
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libllama-server-impl.so` | `31810c6799530853430d928977272c8b5a15147551ed9eaae7f1b40b288a26b2` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-base.so.0` | `a439532b707c4523f827e9a860f79549dda35409184ec9571590816781cbebdf` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-base.so.0.26.0` | `a439532b707c4523f827e9a860f79549dda35409184ec9571590816781cbebdf` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-cpu.so.0` | `a3973694a7eb13fd37c1e5f4f308e2b686929ae794a77135139a05f7168cb9e6` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-cpu.so.0.26.0` | `a3973694a7eb13fd37c1e5f4f308e2b686929ae794a77135139a05f7168cb9e6` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-cuda.so.0` | `7693b95f1c5bd007643a85d587a7c41a31b4597688502f08f6280a04607f8903` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml-cuda.so.0.26.0` | `7693b95f1c5bd007643a85d587a7c41a31b4597688502f08f6280a04607f8903` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml.so.0` | `d8b351a4794b455f45fbab20e7096308c99b46cb39b3c3838efad1fe15196e76` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libggml.so.0.26.0` | `d8b351a4794b455f45fbab20e7096308c99b46cb39b3c3838efad1fe15196e76` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libllama-common.so.0` | `e3b8903a3ed66180cf4c94b1f2dd35d2ad2ac1ab2d178b10969e610fb61563f9` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libllama-common.so.0.6.0` | `e3b8903a3ed66180cf4c94b1f2dd35d2ad2ac1ab2d178b10969e610fb61563f9` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libllama.so.0` | `26ced2d8f30ba427cd6188d0043daa08a7e2c99e4e6113a1b31f5eabbd9902eb` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libllama.so.0.6.0` | `26ced2d8f30ba427cd6188d0043daa08a7e2c99e4e6113a1b31f5eabbd9902eb` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libmtmd.so.0` | `dbd436f1aa607396b3abe44a55fdc6cf855255e671ae34d7ce7c7c2a20a888f0` |
| `/srv/ai/paged-kv/builds/master106-draftkv-20261013/bin/libmtmd.so.0.6.0` | `dbd436f1aa607396b3abe44a55fdc6cf855255e671ae34d7ce7c7c2a20a888f0` |

Offline parser and helper regressions passed. The candidate logs target and native-MTP KV context allocation device/host/bytes at startup; exact build, test, helper and lifecycle hashes are in the JSON freeze.
Original `MASTER106_FREEZE` and `MASTER106_GPU` remain unchanged. Both placement matrices use this candidate.

## Bounded harness recovery

The first 128K setup reached C=61,438, then failed before canonical requests because the runner's tail-closure path only recognized the final row target, not the intermediate preserved C=61,440 prefix. The partial append artifacts and exact failure are archived under `cpu-131072-126976/raw/failed-fill-attempt-00` and `driver-failure-00.log`. The harness now applies the existing deterministic closure at any intermediate preserve-prefix frontier. Eight helper tests passed; the 128K row then reached exact C=126,976 and completed 48/48 requests. This correction changed the harness and its test only; the frozen server executable and libraries are unchanged. The previous JSON freeze is retained as `raw/MASTER106_PATCHED_FREEZE-before-prefix-closure.json`.
