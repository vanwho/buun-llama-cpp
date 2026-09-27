# Narrow selector repair baseline

Revision: `hotpath-v10-20260914`.

93-11n attempt03 used the immutable CUDA candidate at
`/srv/ai/paged-kv/results/v10/93-11n/attempt-03/` with L16384/H4096,
P256/B128/U64 and target plus native draft Turbo4/GPU. Page20 was resident
after request1 and cold/host-backed before request3. Nineteen bounded snapshots
and final two-entry history all showed selector_not_run, query_row=UINT32_MAX,
no Q position and no raw shortlist. No promotion chain was observed.

93-11o must diagnose the real graph-registration/capture gate and repair it
with one focused fixture plus one short request. 93-11n then proves the full
current natural chain before phase94 changes production policy. This summary
avoids loading a later-task handoff into the prerequisite's context. The raw
historical receipt remains unchanged and is accessed only via targeted fields.
