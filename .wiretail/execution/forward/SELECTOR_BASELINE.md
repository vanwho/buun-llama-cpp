# Narrow selector repair baseline

Revision: `hotpath-v10-20260914`.

93-11n attempt03 used the immutable CUDA candidate at
`/srv/ai/paged-kv/results/v10/93-11n/attempt-03/` with L16384/H4096,
P256/B128/U64 and target plus native draft Turbo4/GPU. Page20 was resident
after request1 and cold/host-backed before request3. Nineteen bounded snapshots
and final two-entry history all showed selector_not_run, query_row=UINT32_MAX,
no Q position and no raw shortlist. No promotion chain was observed.

93-11o subsequently repaired graph registration. The later 93-11n attempt13
nominated and authenticated a cold page but did not prove admission or promotion.
93-11n is now explicitly deferred; phase94 implements the corrected architecture
without rerunning this older baseline. See POLICY_ADMISSION_BASELINE.md. This summary
avoids loading a later-task handoff into the prerequisite's context. The raw
historical receipt remains unchanged and is accessed only via targeted fields.
