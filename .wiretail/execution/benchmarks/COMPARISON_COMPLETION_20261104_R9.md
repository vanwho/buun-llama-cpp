# R9 comparison completion — 2026-11-05

**Root reviewed; ready to launch after the idle GPU diagnostic.** All executable, source, model, binary/DSO, template, map, server/performance flags, and normal-prefix history settings are inherited from R8. R9 changes the output root and selected decode cells, and retains validated R8 cells. No run was started.

## Primary remaining work

| Family | Remaining decode cells | Nominal contexts / cells | Notes |
|---|---:|---|---|
| Fork CPU, MTP on | 7 | 126,976 (6); 196,608 (1) | Retain R8’s complete 30 cells. Keep the full target-fill sequence through 256,000, but decode only medium/xhigh × prompts 0–2 at 126,976 and off × prompt 3 at 196,608. |
| Master CPU, MTP on | 41 | 6,144 (3); 12,288 (9); 28,672 (9); 61,440 (9); 126,976 (9); 196,608 (2) | Primary L200,704; lower-L fallback is alternative. Clean L262,144 startup has now confirmed a 1,730 MiB GPU allocation failure; retain L200,704 primary. |
| Master GPU, MTP off | 42 | 6,144 (12); 12,288 (10); 28,672 (10); 62,464 (1); 88,064 (6); 120,832 (3) | R8 faulted on the first canonical request after a successful C6,144 fill; no R8 cell triplet qualifies. |
| Master CPU, MTP-on outlier checks | 3 independent series | Each: 28,672, off, prompt 3 | Preserve all three valid triplets independently. |
| Fork GPU, MTP off | 0 decode cells | 8 mapped fill targets | Ingest-only step; does not create decode gaps. |

That is 90 remaining unique comparison cells plus three independent outlier checks: 372 canonical requests including warmups. Fork CPU MTP-on remains one normal cumulative series; no per-mode refills or giant catch-up fill. The 32K source map is appended in its existing bounded target sequence, and actual occupied C is recorded rather than forced to equal nominal C.

The clean master CPU/MTP-on L262,144 startup confirmation has completed: before launch, GPU free memory was15,900MiB with no compute owner; the server failed a1,730MiB GPU allocation before readiness. No generation was sent. Keep L200,704 and do not schedule fabricated256K measurements. The JSON binds the new result and startup log hashes.

## R8 evidence retained

Fork CPU MTP-on has 30 valid cells and 120 successful records: 9 cells each at C12,288, C28,672, and C61,440, plus low prompts 0–2 at C126,976. The C126,976 row is partial; its three warmups and nine measured low-mode requests pass, while medium/xhigh cells remain unmeasured. Actual row endpoints after the questions were C17,372, C30,414, C63,182, and C127,834; these are actual occupied counts, not nominal labels. The R8 master GPU MTP-off fault is retained as runtime evidence; it contributes no complete cell.

R8 plan metadata labels several selected decode steps “ingest-only” despite nonempty selections and positive request counts. R9’s selected cells/request totals are authoritative; only fork-GPU/MTP-off is truly ingest-only.

## Prefix/append audit

At nominal C126,976 the normal append schedule shows two bounded appends: 32,349 new tokens (map’s 32,768-token full group; backend `prompt_n` 66,182, `cache_n` 29,349) and then 31,444 new tokens (`prompt_n` 31,445, `cache_n` 95,531). The first append reprocessed roughly 33,833 tokens beyond new input; this is a prefix/cache observation, not extra repository content. Keep the existing chunked sequence and report backend prompt tokens separately from fresh mapped input. Do not replace it with one large fill or separate per-mode fills.

See JSON for exact selected mode/prompt patterns, frozen argv templates, source paths, retained series, runtime finding, and confirmed capacity result.
