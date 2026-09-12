# SPEED25_13_WHOLE_MODEL

## Outcome

Whole-model coverage is recorded using retained matched CUDA runs and the
existing Turbo4 paged-kernel fixture. No new kernel rewrite is justified by
the available evidence. The largest observed remaining cost is the selective
prefill/lifecycle path: the latest 2048-token native-MTP run took 4.447 s,
with 2,343 seal calls, 4,672 summary-build calls, 22 host D2H calls, 1.498 s
of waits, and 2.508 s of queue time. The earlier incremental run reduced
summary calls from 274,080 to 3,536 and host seal D2H calls from 616 to 40.

The all-GPU off control measured 1.728k prompt tok/s and 98.9 decode tok/s;
the selective native-MTP run measured 463 prompt tok/s and 43.56 decode tok/s
with 25 proposals and 18 accepted. Output lengths differ, so this is an
attribution diagnostic rather than a quality or rate gate.

## Coverage decisions

Target placement, Turbo4 target/MTP identity, effective B=256, paged attention
shapes, bounded host publication, graph descriptors, MTP acceptance, and
optional performance-profile settings are covered by existing receipts. B=512
was tested but did not improve prefill (453.91 vs 463.00 tok/s), so the
existing B=256/ubatch-256 configuration is retained.

No CPU fallback was observed in the native receipt (`target_backend=CUDA`,
`fallbacks_total=0`). Nsight kernel counters, pinned bandwidth under load,
and fresh post-25-12 MTP A/B timing remain explicitly unmeasured.

## Hardware snapshot

RTX 4080, driver 595.91.07; PCIe link 16x current width, generation 1 at the
snapshot (maximum generation 4); GPU NUMA affinity 0 and CPU affinity 0–31;
35 C, P8, no thermal or power slowdown. Xorg, GNOME, and the retained server
were present. These are host observations, not timed-run utilization claims.

## Deferred verification

The active 8080 service is currently a non-MTP `--spec-type none` profile and
was not changed. Run a fresh V6 native-MTP short A/B after a compatible bundle
is loaded, then attach Nsight Systems/Compute if permissions allow. Do not
claim MMQ/cuBLAS utilization or post-25-12 throughput from this receipt.
