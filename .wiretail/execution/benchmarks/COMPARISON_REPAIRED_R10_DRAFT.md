# Comparison campaign R10 — pristine-master-first draft, not executable

The user has authorized separately versioned build investigation and Luna runner tests. Root owns artifact and command pinning. This outline uses pristine upstream master first; built binary/DSO identities remain unset. It does not launch runs or assert that a fix has been established.

## Source and build policy

Build master from pristine upstream commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286` at `/srv/ai/paged-kv/worktrees/master-pristine-958b444c677a`. Add no source patches, hooks, or overlays; approved host-toolchain compatibility is the only allowed build variation. The former master-repair worktree/build is retired. Fork finite-diagnostic edits were reverted to source commit `8a79008b5473676bb99a54cc1505d11a4e107357`; defer fork repair until pristine-master findings support a concrete source change. Do not change ACO code without a specifically demonstrated defect. New binary/DSO hashes are pending build and are intentionally absent.

## Ordered campaign

1. **Pristine master startup and short checks.** Root pins the just-built identity and argv. Check master CPU/MTP-on startup and one short C6144/medium/P1 request. For master GPU/MTP-off, use a fresh server, normal mapped fill through C6144, then off/P1 with warmup40 and the first measured 400-token request. These are validation checks, not speed samples. They replace retired diagnostic launches.
2. **Master missing controls.** After successful startup/short gates and root review, use one fresh normal cumulative mapped fill per family to recover only the 41 master CPU/MTP-on and 42 master GPU/MTP-off quality-audited gaps. Use R9 L=200704 primary/L=131072 alternative for master CPU/MTP-on; do not retry its historical clean-start L=262144 failure without a separately reviewed startup gate. Use R9 L=122880 for master GPU/MTP-off.
3. **Fork CPU/MTP-on deferred.** Its 33 gaps are listed below but are not currently launchable. Wait for master results to identify a concrete fork source fix. Only then create and pin a separately versioned fork build and re-review its R9 L=262144 primary/fallback profile. No ACO source changes without a demonstrated defect.

Keep the original R9 template for initial source-fault diagnosis. Any later template change is separate and must be reviewed and pinned once across all repaired full curves. Preserve ordinary cumulative Q/A history, one mapped fill per configuration, original prompts/modes, request budgets, actual C, and the whole-triplet response-quality gate.

## Quality-audited feasible gaps

| Family | Feasible cells | By nominal C | Status |
|---|---:|---|---|
| master-cpu-mtp-on | 41 | 6144: 3; 12288: 9; 28672: 9; 61440: 9; 126976: 9; 196608: 2 | first in master queue after gates |
| master-gpu-mtp-off | 42 | 6144: 12; 12288: 10; 28672: 10; 62464: 1; 88064: 6; 120832: 3 | first in master queue after gates |
| fork-cpu-mtp-on | 33 | 12288: 6; 28672: 9; 61440: 9; 126976: 8; 196608: 1 | deferred pending master findings |

**Total unchanged: 116 feasible missing cells.** The separate 24 GPU C=130048 capacity-N/A cells are excluded. The retained quality-audited baseline remains 805 complete cells / 2,415 repetitions until new results qualify; do not blanket-rerun complete cells. R8/R9 runtime faults and response anomalies are historical/provisional observations tied to their binaries, not a proven single cause.

The JSON lists the exact 116 family × nominal-C × mode × prompt gaps and the baseline quality-report hash. It is a staged outline, not an executable completion plan. Root build/DSO identities and literal argv remain pending.

## Provenance

Quality audit: `/srv/ai/paged-kv/results/comparison-final-20261104-quality-audit/comparison-final-20261104.json` (SHA-256 `f3f9bfef2b39d0945c87ad43e357638553e310fd52b5b7eb9343cab34b99af2b`). R9 reference: `COMPARISON_COMPLETION_20261104_R9.json` (SHA-256 `d94223ec4357e3d9de6fda0adfb18bec4de1a4aef3c5d3bc3793dabf7787fa6e`).
