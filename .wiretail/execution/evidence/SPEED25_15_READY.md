# SPEED25_15_READY

## Outcome

The retained candidate is executable and native-MTP capable. V6 all-fit and
real-pressure requests completed on the same frozen CUDA bundle. The pressure
case committed 128 tokens with 102 MTP proposals and 75 accepted. Host Turbo4
backing and bounded H=8 storage were observed.

Natural attention-driven movement is not accepted: the pressure request ended
with zero attention samples and zero H2D bytes. The existing indexed 25-08
deterministic movement proof remains separate and is not relabeled as natural
recall. Therefore `ready_for_full_population=false` until 26-01 qualifies this
path after the routing sample issue is repaired or explained.

| Case | Prompt | Output | pp/s | tg/s | Wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-fit | 2048 | 1 | 508.60 | null | 4.040997 s |
| pressure | 4096 | 128 | 410.00 | 60.22 | 12.133511 s |

## Exact candidate invocation

```text
PAGER_BUNDLE_ROOT=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin
python3 tools/server/bench/run-final-curve.py --suite pressure --context 8192
  --prompt-tokens 4096 --question-index 0
  --output /srv/ai/paged-kv/results/25-15-v6-pressure-20260912
  --endpoint http://127.0.0.1:8080 --api-key-file /srv/ai/config/llama/api-keys
  --model qwen38-fast-turbo4-mtp --mode selective --prefill-policy runtime
  --warmups 0 --trials 1 --max-tokens 128 --reserve-context 512
  --cache-condition cold-prefill --hot-pages 8
  --server-bin build-cuda/bin/llama-server
  --startup-timeout 30 --progress-idle-timeout 120
  --decode-idle-timeout 120 --total-timeout 300
```

The all-fit command is identical except `--suite micro`, `--prompt-tokens
2048`, and output `25-15-v6-allfit-20260912`. The managed command line used
`-c 8192 -b 1024 -ub 256 -ctk turbo4 -ctv turbo4 --kv-pager selective
--kv-page-size 256 --kv-hot-pages 8 --kv-host-budget 4G
--spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2`.

## Remaining work

The top measured costs are host seal D2H/repeated sealing and the stale-drop
routing publication path. No ablation was promoted while natural sampling was
invalid. Full-L allocation and near-full occupied proof remain 26-01.
