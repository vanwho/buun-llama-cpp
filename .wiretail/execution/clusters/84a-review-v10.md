# 84a — diagnose MTP token/state parity

Revision: `hotpath-v10-20260914`. Task: `84-01`.

First repair and prove the local benchmark B/U control plane, then trace the
first native-MTP token/state divergence using the stable matrix. Use bounded
debug output and deterministic regression tests around MTP checkpoint,
draft/target handoff, and verification route state. Do not accept startup,
requested geometry, or positive draft counters as proof of correct generation.
