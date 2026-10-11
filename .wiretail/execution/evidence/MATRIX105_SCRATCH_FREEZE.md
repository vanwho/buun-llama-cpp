# MATRIX105_SCRATCH_FREEZE — corrected candidate, revision 3

Created 2026-10-11 for task 105-15. This freezes the corrected build, model, helpers, exact managed argv, and exact C=61,440 A/B/A plus tracked-source context.

- Candidate source: `ac9d7e0d7e7a24864b0e0777d0985a3415fb102a`; required fixes `0247624e9`, `c6d71b44d`, `f5478ee12` are ancestors. No production sources changed; only Wiretail state/handoff metadata is dirty.
- Build: `cmake --build build-cuda --parallel 16 --target llama-server` passed. Executable SHA-256 `136142968618a512c81e481f780a2c5ae8709c72d1125ce2e26068253e045b4d`; build receipt SHA-256 `8ee0a94b8fc9c5bfc7f079ebe7a280f144a724e204a2d7a637cba78bfce4f026`. Actual loaded project DSO hashes match the build (see JSON).
- Model SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Exact geometry L/H/C `262144/32768/61440`, B/U `1024/256`, Turbo4 target/draft, GPU MTP n-max=2, ACO `probe-rerank`.
- Exact argv SHA-256 `f0ec993d52b1e4b04018b21ade7692738c80b01e33dd0d070b95986fb432f9f1`.
- Fresh context fill reports exactly C=61,440 from server usage and same-settings tokenization; 18 curve points, 17 source chunks. Base SHA-256 `6d559c030d7f1bc9af5e157a9c441c77db74f8a39d248fe1187553cbd9521062`.
- Frozen helpers, source manifest, seed, and curve hashes are in `MATRIX105_SCRATCH_FREEZE.json`.
- Prior cohort pointer: `/srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/evidence/MATRIX105_FREEZE.json` (SHA-256 `3a978d416a65b2092346c6a66872642c4deb02949709a492d231954fd21cda53`). This revision supersedes the earlier scratch-freeze attempt and fixes its obsolete C target.
- The canonical schedule is 48 requests: 4 reasoning modes × 3 prompts × (1 warmup + 3 measured).

Freeze before scoring; keep candidate, argv, model, helpers, and base fixed for the matrix.
