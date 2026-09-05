# Cluster 17g — exact and concurrent correctness oracles

Task `17-14` repairs the exact/native and concurrent-response failures before
any new quality or speed claim is made. It first gates live work on a bounded
resolved-context Turbo4/native-MTP startup probe: the phase-17 pressure run
showed a repeatable post-listen SIGSEGV and a systemd `Restart=always` cleanup
loop. That crash must be isolated or converted into a deterministic refusal,
and restoration must quiesce the restart loop before waiting for health. The
task uses the existing V4 facts and declared tolerances, preserves failed raw
evidence, and does not rewrite prior manifests.
