# 105-03 occupied-context findings — compact numeric record

Scope: occupancy trajectory and primary held-out fact only. The validated
checker reports `scope_execution_status=complete`, frontier C=257445 against
requested C=250000 (overshoot7445), and a correct fact answer
`<ff fe>` / `invalid_utf8=true`. This does not claim full task completion:
post-load canonical remains0/12 unmeasured, and physical large-query rank /
promotion is unknown. All23 requests returned successfully; no runtime fault.

## Candidate and provenance

Build source was repo HEAD `503aec68e4c920a4586941fdb7c60446a6ce0d82` plus
the graph-arena repair delta now committed as
`a079452f56788049109cbef6404f233a41f75e08` (not built from a079 HEAD).
The repair adds checked structural graph-node allowance from the physical plan;
it does not change GPU KV, routing, B or U. Geometry: L262144/H51200, 200
Turbo4 hot pages ×256, B1024/U256, selective probe-rerank, GPU MTP.

Model SHA-256:
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Executable SHA-256:
`9c95e267c7daf768ecf77f2cb03c2a23e4e2f100b9db12773f66369f36d05b65`.

Loaded DSO SHA-256s:

| DSO | SHA-256 |
| --- | --- |
| libggml-base.so.0.25.3 | 990f7fa9e76ccb89cacd7945119cb78a927f946113553b6929a672a316b353d3 |
| libggml-cpu.so.0.25.3 | c12734653ac843921f49ea959e105314cbd6640751cec39fbb989c8f8c608f87 |
| libggml-cuda.so.0.25.3 | f8b66c3fa9b17cc77e08678dba8407b4c2998362278e6e796f9d5cf67de42867 |
| libggml.so.0.25.3 | e3606444f8f701685a0b055f36e72da0a9eb1766ea2005f1665715e1ce954798 |
| libllama-common.so.0.5.0 | ac0d1ce65fa95979d8d752a944b45c70804713a46d1c21c9ae909ad5ff8b7941 |
| libllama-server-impl.so | efa611a672ab73931f8d834cd6d7017fc4e702acc2307f9ccebb1c1384f6b1c0 |
| libllama.so.0.5.0 | 5a03a591c7d0125a9469274bc747cf819fa801b9d580bda4294fa885b841e4bb |
| libmtmd.so.0.5.0 | de1199a2939f366e0bf1845c5b2f7c4d60b2733b25164dc1753b139418fe816c |

## Per-request curve

`Fresh/cached` and `processed prompt` are token counts. Reported PP/s is
server prompt throughput over processed tokens; fresh PP/s is only genuinely
new input. Decode is server TG/s. Output is actual completion tokens; MTP is
accepted/drafted. Rows are taken from checked occupancy findings, with decode
joined by request index from the checkpoint. F2–F21 abbreviate the driver's
`repo_continuation_2` through `repo_continuation_21` stages.

| # | Stage | C before→after | Fresh/cached | Processed prompt | Reported PP/s | Fresh PP/s | Decode TG/s | Output | MTP acc/draft |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A1 | 0→4258 | 3890/0 | 3890 | 1352.3 | 1352.3 | 85.9 | 368 | 217/302 |
| 2 | B | 4258→16545 | 11888/4258 | 16146 | 1251.4 | 626 | 88.4 | 400 | 247/304 |
| 3 | F2 | 16545→28800 | 11856/16545 | 28401 | 941.6 | 471.1 | 77.3 | 400 | 235/327 |
| 4 | F3 | 28800→41102 | 11903/28800 | 40703 | 755.8 | 378.1 | 79.6 | 400 | 248/301 |
| 5 | F4 | 41102→53361 | 11899/41102 | 53001 | 783.7 | 392.1 | 42.3 | 359 | 206/308 |
| 6 | F5 | 53361→65614 | 11902/53361 | 65263 | 812.9 | 406.6 | 42.3 | 352 | 203/296 |
| 7 | F6 | 65614→77879 | 11893/65614 | 77507 | 792.4 | 396.4 | 42.4 | 371 | 218/308 |
| 8 | F7 | 77879→90102 | 11861/77879 | 89740 | 780 | 390.1 | 39.4 | 362 | 200/324 |
| 9 | F8 | 90102→102373 | 11884/90102 | 101986 | 766.9 | 383.6 | 47.6 | 388 | 244/286 |
| 10 | F9 | 102373→114556 | 11855/102373 | 114228 | 752.1 | 376.2 | 45.9 | 329 | 203/250 |
| 11 | F10 | 114556→126804 | 11904/114556 | 126460 | 743.5 | 371.9 | 42.4 | 344 | 206/276 |
| 12 | F11 | 126804→139034 | 11879/126804 | 138683 | 732 | 366.2 | 42.5 | 352 | 211/280 |
| 13 | F12 | 139034→151333 | 11900/139034 | 150934 | 722.4 | 361.4 | 38.7 | 400 | 226/344 |
| 14 | F13 | 151333→163539 | 11894/151333 | 163227 | 710.6 | 355.5 | 40.9 | 312 | 185/254 |
| 15 | F14 | 163539→175836 | 11898/163539 | 175437 | 701.6 | 350.9 | 39.4 | 400 | 232/333 |
| 16 | F15 | 175836→187935 | 11860/175836 | 187696 | 689.2 | 344.8 | 38.1 | 239 | 137/204 |
| 17 | F16 | 187935→200234 | 11900/187935 | 199835 | 683.2 | 341.8 | 42.3 | 400 | 246/305 |
| 18 | F17 | 200234→212347 | 11880/200234 | 212114 | 672.7 | 336.5 | 39.7 | 234 | 139/188 |
| 19 | F18 | 212347→224455 | 11884/212347 | 224231 | 663.3 | 331.8 | 39.1 | 224 | 134/180 |
| 20 | F19 | 224455→236680 | 11901/224455 | 236356 | 655.8 | 328 | 37.8 | 323 | 190/268 |
| 21 | F20 | 236680→248771 | 11838/236680 | 248518 | 645 | 322.6 | 39 | 254 | 152/202 |
| 22 | F21 | 248771→257248 | 8218/248771 | 256989 | 611.8 | 306.1 | 38.4 | 260 | 155/208 |
| 23 | A2 | 257248→257445 | 64/257248 | 257312 | 38.6 | 21.1 | 36.6 | 134 | 77/112 |

Fill aggregate MTP is4434/6048 (73.31%); A2 is77/112 (68.75%). The A2
processed prompt is257312 with257248 cached and64 fresh tokens; its 134-token
natural completion answered the fact. Final GPU observation was15357MiB used,
about590MiB free. The C target drifted+7322 to actual A2 prompt257312 because
prior assistant completions consumed7471 tokens. The final fresh chunk was
8218 tokens, as planned, not oversized.

Canonical12 was withheld by an obsolete11072-token reserve although rendered
prompt257477 plus the400-token output ceiling fit L262144; the guard was
corrected to exact rendered prompt + measured output ceiling/template + MTP
allowance +4096. Do not rerun this trajectory. `_answer_envelope` now budgets
the measured >=400-token ceiling and assistant template, not the old
`/400` compressed-character estimate. The helper's low-entropy ceiling
regression is part of 105-03a validation.

## Evidence references

External raw artifacts remain append-only under
`/srv/ai/paged-kv/results/forward/105-03/attempt-02/`. Per-request request/SSE
paths and SHA-256s remain in the checkpoint; raw bodies are not copied here.

| Artifact | SHA-256 |
| --- | --- |
| checked-occupancy-findings.json | 40ef952b1b78780d0d4932959f144e121351ccc4058895ccf707991c704511db |
| incremental-state.json | 5b223dd33d9a7b6d84b2815bbb0752a5c28c62e4ff430b9af03bb11657d205b2 |
| candidate-identity.json | 7e9cb56085845422a9a0811d370534dab4a00119af4418a2cdf569575efc7b81 |
| checker-occupancy.stdout.json | 40ef952b1b78780d0d4932959f144e121351ccc4058895ccf707991c704511db |
| driver-console.log | 01ad2909b24973b2175151fd56b65d7b972ae40a8351fc0c228733139be6c3f1 |

Attempt01's separate partial-run CPU metadata-arena finding is recorded in
`handoffs/105-03.md`; its 368-byte shortage was not VRAM OOM.
