# Comparison completion — live handoff (latest user steering wins)

## Required method — user override on 2026-11-05

User: “fuck a cache policy just fill context and prompt normally, if a new
prompt adds tokens whatever, consider it the same size context as original”.

KEEP the full normal conversation: each question follows prior questions and
answers, preserving raw content/reasoning. No benchmark rewind to a frozen
repo base, disk KV save/restore/erase, or further cache-policy tuning. Group
under original nominal context checkpoint; record actual per-request counts
and their drift separately. Existing valid old measurements remain retained,
with their protocol labeled. Only fresh disjoint mapped repo bytes are ingested;
never count model re-evaluated old tokens as net-new repo input.

## Active R8

Plan COMPARISON_COMPLETION_20261104.json revision8 is reviewed/frozen/ready;
output /srv/ai/paged-kv/results/comparison-completion-20261104-r8.
Source helpers commit647191820;35 tests pass. Root checked all10 server argv
against R7 unchanged,492maximum canonical requests. Client alone changes to
normal cumulative conversation. Current+future reply budget1752percell+1024
leaves room underL; none of the400-token output budgets shrink.

Runtime owner /root/comparison_runtime_prep (LunaHigh) must launch durablequeue
with nohupsetsid once root commits/pushes. Do not resume R7. Remaining120cells
+3outliercohorts=492requests: masterGPUoff L122880;forkCPUon;masterCPUon;
3independentC28KBashrechecks;forkGPUoffL122880ingest-only. Serverflags remain
R7(CP32, lastforkingestCP8), no cache tuning. Realerrors assess; stat/cacheflags
never trigger automatic replay/rerun. Keep strictL bound/no contextshift by
reserving all remainingoutputs before newfiles, not by shorteninganswers.

## Preserved data

Current ACO on/off curves complete264cells/792measured+264warmups. R7masterCPU
off allsevenrows complete, including108newrequests at128K/192K/final256Kallocated
checkpoints. NEVER rerun those completed series. Matrix beforeR7 had774cells;
27newCPUoffcells bring available count to801 before R8. ConditionalGPU130048
endpoints remain real-capacityN/A, not fabricated timings. ApprovedoldR4/R5/R6
partials retained; R4MTPon target-onlyslot-restored cohorts quarantined.

R7paused GPUoff atC6144 afterfill/warmup/firstmeasure passed, then CUDAillegal
on fixed-base checkpoint6147 restore. User overrode that request method; now
append Q/A. Archived_R5/_R6/_R7 are provenance only, never executabledirections.

## Agents and report work

- Runtime owner: /root/comparison_runtime_prep, waitingrootlaunchhandoff.
- Docs /root/aco_docs_refresh_20261104 implemented cumulativehistory, logical
  newinput accounting (prompt_n-priorcommittedfrontier, notcache_n), persisted
  conversation/sourcecursor, nominal/actualcounts, and whole-series headroom.
  It is idle; followup for reporting ONLY when modeltimingsidle.
- Audit /root/routing_prefix_engine_audit audited exact assistantconstructor:
  rawcontent+optionalrawreasoning_content; don'tstrip orinsert stopmarkers.
- Root reporter changes groupprimaryplots byoriginalnominalC (userdirection),
  retain actualbefore/afterperrepeat. Append diagnostics stillplot actualC.
  Newnormal Q/A history is a distinct protocol; oldfixedbase data remainsvalid
  underitsrecorded protocol. Do not pretend exact-input retest of oldoutlier.
- Preview-r6 is stale: /srv/ai/paged-kv/results/comparison-final-20261104-preview-r6.
  All10families mapped viahashverifiedadjacentsourcemetadata. Noexport/render
  duringCPUtiming. Finalreport mustrefresh,inspectimages,checkallgaps/outliers,
  commit/pushcompactCSV/JSON/MD/graphs; rawpayloads/traces/binaries external.

## Final extra diagnostic and safeguards

AfterR8 measurements, execute profile-comparison-dispatch.py --execute once.
It traces thetwo frozenR6GPU-MTPon configs ingestonly6144/12288/28672; serverargv
andbinariesunchanged, no canonicaldecode. SHA118259d142fa7e9da10424d6d32ff7a8e12d1212a72d5418e663afc51980c012.
Traceabsence is unmeasured, neverzero. This mayexplain systematicpureGPUingest
master1346.87 vsfork910.57tok/s; CUDAfused-vs-materialized dispatch remains a
hypothesis untiltrace. Notes engineering-notes/dense-turbo4-prefill-cohort-difference-20261104.md.

OneGPU/model, /tmp/ai-pager-benchmark.lock. ProtectJellyfin/ffmpeg and8091;
nevertouch8092. Restoremodel/gateway healthy. Allmodel/build/DSO/map/template
identities pinned; no inferencecode edits or rebuild during comparison.
