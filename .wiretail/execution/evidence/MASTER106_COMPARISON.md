# Master GPU and CPU-KV comparison

> The GPU and CPU-F16 measurements below are paired rows from the same replacement build based on upstream master. The ACO references are a separate Phase-105 candidate/build cohort. No cross-cohort medians are combined and no same-binary causal claim is made.

## Identity and protocol

- Base upstream master SHA: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`.
- Paired candidate cohort: `master106-draftkv-20261013`; narrow draft-KV placement patch SHA-256 `4caf807505c4a34cc6e0ce0d21fc8b81f79b4906359f7f08a961543e45f94937`.
- Shared executable SHA-256: `5e6055febf59618df1d24692ab7392aff66bd36a1e71bcdec33d8ec76ac5ded8`; all loaded project DSOs match between paired reports.
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Every row keeps GPU Turbo4 draft MTP (`n_max=2`, `-b 1024 -ub 256`). Target K/V differs by placement: GPU Turbo4 versus host F16.
- Prompt text, reasoning modes, warmup/measured request counts, token ceilings, and seed/temperature follow the frozen protocol and identical canonical schedules.
- The original 106-01 freeze/GPU report are preserved as historical upstream-master evidence; direct GPU/CPU pairing here uses the replacement GPU replay from 106-02.

## GPU-only Turbo4 target-KV rows

GPU-only Turbo4 target KV: 5 rows; each completed 48 canonical requests (12 warmups, 36 measured).

| L | C | target K/V | draft MTP K/V | cases |
| --- | --- | --- | --- | --- |
| 77824 | empty | gpu-turbo4 | gpu-turbo4 | 48 |
| 8192 | 6144 | gpu-turbo4 | gpu-turbo4 | 48 |
| 16384 | 14336 | gpu-turbo4 | gpu-turbo4 | 48 |
| 32768 | 30000 | gpu-turbo4 | gpu-turbo4 | 48 |
| 65536 | 61440 | gpu-turbo4 | gpu-turbo4 | 48 |

### Per-prompt median tables

#### Decode tok/s — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 108.62990774990705 | 89.87771637366716 | 105.82970488630866 |
| L=8192 / C=6144 | 107.3258644573857 | 75.1957043401156 | 96.24160840712662 |
| L=16384 / C=14336 | 100.34779885449127 | 75.59111552422206 | 91.35998779853387 |
| L=32768 / C=30000 | 96.04849196037583 | 76.44768459075296 | 90.13502633569266 |
| L=65536 / C=61440 | 91.80035092770513 | 74.31852490186088 | 82.18807245044734 |

#### Decode tok/s — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 107.75216030928922 | 84.01137636979524 | 97.9636090972101 |
| L=8192 / C=6144 | 104.98642806262805 | 80.22253207644664 | 98.42660508137833 |
| L=16384 / C=14336 | 101.93464041214348 | 82.37984100098026 | 89.48202689574066 |
| L=32768 / C=30000 | 93.7740046487964 | 72.85574953290063 | 85.78484858974224 |
| L=65536 / C=61440 | 86.22040188850856 | 65.6715639333634 | 78.1968856473438 |

#### Decode tok/s — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 104.75875162225095 | 86.40798123885656 | 98.66173638226947 |
| L=8192 / C=6144 | 98.85019142005113 | 79.79462184248781 | 98.2785005804343 |
| L=16384 / C=14336 | 97.50241891624266 | 80.349769190757 | 89.3063981655257 |
| L=32768 / C=30000 | 92.17662209603795 | 74.3094834454024 | 86.81745487777016 |
| L=65536 / C=61440 | 86.52075258282912 | 69.10919460594941 | 81.0947589209316 |

#### Decode tok/s — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 104.07955106203998 | 81.2908147632457 | 98.66615231679995 |
| L=8192 / C=6144 | 101.14233165901372 | 80.33067704493526 | 91.62793912689827 |
| L=16384 / C=14336 | 98.95150635142258 | 78.40458484235307 | 84.2661746704749 |
| L=32768 / C=30000 | 94.48305564500826 | 71.13983805524009 | 72.36558002826065 |
| L=65536 / C=61440 | 89.14721071337405 | 69.05993675563687 | 77.77444242157256 |

#### Request-local MTP acceptance (%) — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 93.10344827586206 | 67.3913043478261 | 88.27160493827161 |
| L=8192 / C=6144 | 94.9090909090909 | 50.806451612903224 | 79.54545454545455 |
| L=16384 / C=14336 | 87.87878787878788 | 54.48717948717949 | 76.04166666666667 |
| L=32768 / C=30000 | 89.39393939393939 | 60.0 | 80.3921568627451 |
| L=65536 / C=61440 | 94.26229508196721 | 67.56756756756756 | 79.72027972027972 |

#### Request-local MTP acceptance (%) — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 91.48936170212765 | 60.38461538461539 | 78.70967741935483 |
| L=8192 / C=6144 | 92.14876033057851 | 58.743169398907106 | 83.55704697986577 |
| L=16384 / C=14336 | 92.1259842519685 | 65.2892561983471 | 74.6875 |
| L=32768 / C=30000 | 88.65546218487395 | 57.30769230769231 | 76.7515923566879 |
| L=65536 / C=61440 | 90.36697247706422 | 57.196969696969695 | 77.31629392971246 |

#### Request-local MTP acceptance (%) — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 87.58620689655173 | 63.53276353276353 | 79.80456026058631 |
| L=8192 / C=6144 | 83.55704697986577 | 57.567567567567565 | 82.66666666666667 |
| L=16384 / C=14336 | 84.73282442748092 | 61.1731843575419 | 73.83177570093459 |
| L=32768 / C=30000 | 83.46153846153847 | 57.5 | 76.19047619047619 |
| L=65536 / C=61440 | 86.66666666666667 | 59.01639344262295 | 78.13504823151125 |

#### Request-local MTP acceptance (%) — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 87.09677419354838 | 56.36363636363637 | 80.32786885245902 |
| L=8192 / C=6144 | 87.95180722891567 | 58.8235294117647 | 74.14330218068535 |
| L=16384 / C=14336 | 89.02439024390245 | 59.5 | 67.75147928994083 |
| L=32768 / C=30000 | 90.11627906976744 | 55.10204081632653 | 56.98924731182796 |
| L=65536 / C=61440 | 95.51282051282051 | 62.60623229461756 | 76.7515923566879 |

#### Fresh prompt prefill tok/s — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 249.85008994603237 | 231.15646724427245 | 238.5780746749374 |
| L=8192 / C=6144 | 120.52505094922608 | 109.3541543643243 | 113.36210049001683 |
| L=16384 / C=14336 | 117.22121775084453 | 107.59938596617074 | 110.78827645606174 |
| L=32768 / C=30000 | 110.08182749176889 | 101.00670011110736 | 103.78861941048062 |
| L=65536 / C=61440 | 99.05740686282097 | 90.90054226875216 | 94.16136747415271 |

#### Fresh prompt prefill tok/s — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 278.6385145684706 | 264.5375402818527 | 265.69624277045267 |
| L=8192 / C=6144 | 1742.3119257811622 | 1743.0249492577952 | 1743.7965625027234 |
| L=16384 / C=14336 | 1739.9949384602332 | 1742.038831708447 | 1737.5306376321014 |
| L=32768 / C=30000 | 1608.9990474785584 | 1607.194379086561 | 1610.9445835384818 |
| L=65536 / C=61440 | 1364.0901428916584 | 1364.3148938252777 | 1364.5025130516135 |

#### Fresh prompt prefill tok/s — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 234.71423541837814 | 215.74414470391275 | 221.9092732471301 |
| L=8192 / C=6144 | 117.33135510145377 | 104.71948268575554 | 107.90376472514308 |
| L=16384 / C=14336 | 109.6417544095438 | 98.7058952095914 | 103.62064837940878 |
| L=32768 / C=30000 | 104.86931929663133 | 138.2675081232161 | 143.77503668742315 |
| L=65536 / C=61440 | 95.2157270990307 | 86.48967249244016 | 88.5902133125779 |

#### Fresh prompt prefill tok/s — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 333.4889615153739 | 318.41685042962513 | 322.9145890911854 |
| L=8192 / C=6144 | 1743.8266237474575 | 1741.2317743695392 | 1742.1270712811681 |
| L=16384 / C=14336 | 1741.772996534949 | 1738.3902477344873 | 1739.9273039604814 |
| L=32768 / C=30000 | 1612.1341331194042 | 1611.203474155482 | 1612.1764328230781 |
| L=65536 / C=61440 | 1364.2634092075848 | 1364.328046311694 | 1364.6758092920245 |

### Per-context fresh append curves

#### L=77824 / C=empty

Empty baseline; no source fill append curve.

#### L=8192 / C=6144

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 1726.595203083548 | 2.267456775996834 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-8192-6144/raw/append-0000.sse |
| 1 | 6144 | 2254 | 1552.6153371393304 | 1.4684704010142013 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-8192-6144/raw/append-0001.sse |

#### L=16384 / C=14336

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 1722.464479151538 | 2.2728980949614197 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0000.sse |
| 1 | 7985 | 4095 | 1675.0548943508725 | 2.4567347670090385 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0001.sse |
| 2 | 9257 | 1272 | 1352.5252987092529 | 0.9539184540044516 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0002.sse |
| 3 | 9388 | 131 | 399.52423068712073 | 0.34769068501191214 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0003.sse |
| 4 | 9586 | 198 | 548.5402733836069 | 0.38049607403809205 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0004.sse |
| 5 | 11732 | 2146 | 1456.3913933105077 | 1.495105292997323 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0005.sse |
| 6 | 14336 | 2604 | 1469.7424394902407 | 1.7995091030024923 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-16384-14336/raw/append-0006.sse |

#### L=32768 / C=30000

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 1708.9924193665304 | 2.2907231430290267 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0000.sse |
| 1 | 7985 | 4095 | 1669.7268131214842 | 2.470634551020339 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0001.sse |
| 2 | 9257 | 1272 | 1343.469912399847 | 0.9658691969816573 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0002.sse |
| 3 | 9388 | 131 | 391.50172290643707 | 0.35642553999787197 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0003.sse |
| 4 | 9586 | 198 | 540.8476557805142 | 0.3860153410350904 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0004.sse |
| 5 | 11732 | 2146 | 1448.1125613980157 | 1.5041299069998786 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0005.sse |
| 6 | 14847 | 3115 | 1493.0817350808059 | 2.1107146939611994 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0006.sse |
| 7 | 18071 | 3224 | 1453.4909932248565 | 2.244916527008172 | 14847 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0007.sse |
| 8 | 18349 | 278 | 606.9524437477348 | 0.48658161802450195 | 18071 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0008.sse |
| 9 | 18461 | 112 | 327.6846737196892 | 0.3712303609936498 | 18349 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0009.sse |
| 10 | 22557 | 4096 | 1429.5765996180348 | 2.89819780603284 | 18461 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0010.sse |
| 11 | 26653 | 4096 | 1347.8831882834202 | 3.0683210539864376 | 22557 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0011.sse |
| 12 | 27984 | 1331 | 1076.080769927302 | 1.2668601190089248 | 26653 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0012.sse |
| 13 | 29302 | 1318 | 1056.087204918578 | 1.286056062032003 | 27984 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0013.sse |
| 14 | 30000 | 698 | 906.901023057114 | 0.8023751419968903 | 29302 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-32768-30000/raw/append-0014.sse |

#### L=65536 / C=61440

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 1686.9534747772268 | 2.32026589801535 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0000.sse |
| 1 | 7985 | 4095 | 1667.6168757126568 | 2.473309504974168 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0001.sse |
| 2 | 9257 | 1272 | 1341.8684040627386 | 0.9665663489722647 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0002.sse |
| 3 | 9388 | 131 | 395.67237119505137 | 0.3439855530159548 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0003.sse |
| 4 | 9586 | 198 | 534.4663299708742 | 0.39047580503392965 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0004.sse |
| 5 | 11732 | 2146 | 1441.8234968855672 | 1.504204417055007 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0005.sse |
| 6 | 14847 | 3115 | 1493.1955343147358 | 2.110925740969833 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0006.sse |
| 7 | 18071 | 3224 | 1448.7843063945134 | 2.253702692047227 | 14847 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0007.sse |
| 8 | 18349 | 278 | 609.4793774540863 | 0.48429850599495694 | 18071 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0008.sse |
| 9 | 18461 | 112 | 328.2032046698627 | 0.3699112129979767 | 18349 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0009.sse |
| 10 | 22557 | 4096 | 1430.26997393319 | 2.8961564330384135 | 18461 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0010.sse |
| 11 | 26653 | 4096 | 1348.3486350274377 | 3.067038035020232 | 22557 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0011.sse |
| 12 | 27984 | 1331 | 1072.4625302863901 | 1.2783073600148782 | 26653 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0012.sse |
| 13 | 29302 | 1318 | 1055.479967102339 | 1.2870971029624343 | 27984 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0013.sse |
| 14 | 30984 | 1682 | 1136.2584670508688 | 1.5248162090429105 | 29302 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0014.sse |
| 15 | 35080 | 4096 | 1232.6048045105633 | 3.3657431509927846 | 30984 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0015.sse |
| 16 | 39176 | 4096 | 1189.7666128715553 | 3.492356352042407 | 35080 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0016.sse |
| 17 | 41953 | 2777 | 1096.0519632039134 | 2.5754090509726666 | 39176 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0017.sse |
| 18 | 45243 | 3290 | 1089.2214748124984 | 3.0705752900103107 | 41953 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0018.sse |
| 19 | 49339 | 4096 | 1091.6649409512142 | 3.8104192960308865 | 45243 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0019.sse |
| 20 | 53435 | 4096 | 1054.4249033744172 | 3.9401511419564486 | 49339 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0020.sse |
| 21 | 55661 | 2226 | 956.3655017567738 | 2.3796303950366564 | 53435 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0021.sse |
| 22 | 59757 | 4096 | 1004.7327425622549 | 4.13872694602469 | 55661 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0022.sse |
| 23 | 60752 | 995 | 784.0146118260922 | 1.3337527730036527 | 59757 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0023.sse |
| 24 | 61438 | 686 | 672.3690620059533 | 1.0829136229585856 | 60752 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0024.sse |
| 25 | 61440 | 2 | 13.498646760662243 | 0.2119775169994682 | 61438 | /srv/ai/paged-kv/results/forward/106-master/106-02/paired-gpu/gpu-65536-61440/raw/append-0025.sse |

CPU-F16 target KV: 6 rows; each completed 48 canonical requests (12 warmups, 36 measured).

| L | C | target K/V | draft MTP K/V | cases |
| --- | --- | --- | --- | --- |
| 77824 | empty | cpu-f16 | gpu-turbo4 | 48 |
| 8192 | 6144 | cpu-f16 | gpu-turbo4 | 48 |
| 16384 | 14336 | cpu-f16 | gpu-turbo4 | 48 |
| 32768 | 30000 | cpu-f16 | gpu-turbo4 | 48 |
| 65536 | 61440 | cpu-f16 | gpu-turbo4 | 48 |
| 131072 | 126976 | cpu-f16 | gpu-turbo4 | 48 |

### Per-prompt median tables

#### Decode tok/s — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 31.623704329484948 | 26.23484518238251 | 32.144238308848955 |
| L=8192 / C=6144 | 26.940362536838098 | 18.679576648948608 | 22.538458575047475 |
| L=16384 / C=14336 | 20.251571249311638 | 14.114560504615083 | 18.112796433523297 |
| L=32768 / C=30000 | 14.319233298443171 | 10.518029497990996 | 12.57116809214471 |
| L=65536 / C=61440 | 9.484250754967913 | 7.403607729409703 | 8.772105175894819 |
| L=131072 / C=126976 | 5.422303228333995 | 4.109743508339055 | 4.979604918145795 |

#### Decode tok/s — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 32.84366907410487 | 24.968717825236773 | 28.31614730074019 |
| L=8192 / C=6144 | 25.333610137391386 | 20.760749840821244 | 23.132046650439065 |
| L=16384 / C=14336 | 20.389961995229278 | 15.815394075907015 | 18.03997604435863 |
| L=32768 / C=30000 | 14.228301882875733 | 10.533463552658343 | 12.61824545162851 |
| L=65536 / C=61440 | 8.890152027452121 | 7.170829175154402 | 8.607290927662968 |
| L=131072 / C=126976 | 5.2665322340655925 | 4.416095888117779 | 4.777815703602515 |

#### Decode tok/s — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 30.853517320306292 | 28.161625994408894 | 29.77235120092263 |
| L=8192 / C=6144 | 23.981941057437712 | 19.829192626840236 | 22.34976463009151 |
| L=16384 / C=14336 | 19.175699163374563 | 16.727575584430678 | 17.946921641895944 |
| L=32768 / C=30000 | 13.745316731540578 | 10.624162980814813 | 12.383758331607936 |
| L=65536 / C=61440 | 8.955157172943437 | 7.111973151178411 | 8.504777788061194 |
| L=131072 / C=126976 | 5.112319415373316 | 4.035986596316555 | 4.800077032213666 |

#### Decode tok/s — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 30.133856287310277 | 25.511010496671645 | 25.255746077842765 |
| L=8192 / C=6144 | 25.11867748840988 | 19.446976399967383 | 23.022481366503712 |
| L=16384 / C=14336 | 20.477224911918356 | 16.171062314070415 | 15.41708310178729 |
| L=32768 / C=30000 | 14.022387407960087 | 10.482491799698886 | 12.182426431578607 |
| L=65536 / C=61440 | 9.56048029325994 | 7.225947388257378 | 8.444156538916737 |
| L=131072 / C=126976 | 4.987156820228843 | 4.072095564061166 | 4.599179649282466 |

#### Request-local MTP acceptance (%) — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 91.08527131782945 | 64.66666666666667 | 89.50617283950618 |
| L=8192 / C=6144 | 93.54838709677419 | 57.534246575342465 | 77.88461538461539 |
| L=16384 / C=14336 | 90.0 | 49.24242424242424 | 77.55905511811024 |
| L=32768 / C=30000 | 89.39393939393939 | 53.48837209302326 | 73.37461300309597 |
| L=65536 / C=61440 | 94.26229508196721 | 62.295081967213115 | 83.21428571428571 |
| L=131072 / C=126976 | 93.44262295081967 | 58.641975308641975 | 81.05263157894737 |

#### Request-local MTP acceptance (%) — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 91.48936170212765 | 61.15384615384615 | 77.56410256410257 |
| L=8192 / C=6144 | 90.98360655737704 | 67.3529411764706 | 77.31629392971246 |
| L=16384 / C=14336 | 93.05555555555556 | 61.03448275862069 | 75.70977917981072 |
| L=32768 / C=30000 | 89.83050847457628 | 53.64963503649635 | 75.15723270440252 |
| L=65536 / C=61440 | 86.32478632478633 | 60.0 | 81.51815181518151 |
| L=131072 / C=126976 | 89.41605839416059 | 67.3529411764706 | 77.31629392971246 |

#### Request-local MTP acceptance (%) — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 86.3013698630137 | 73.14814814814815 | 85.08474576271186 |
| L=8192 / C=6144 | 81.51815181518151 | 60.833333333333336 | 75.9493670886076 |
| L=16384 / C=14336 | 83.71212121212122 | 66.86217008797654 | 75.70977917981072 |
| L=32768 / C=30000 | 84.765625 | 55.43478260869565 | 71.64634146341463 |
| L=65536 / C=61440 | 85.9504132231405 | 58.743169398907106 | 79.54545454545455 |
| L=131072 / C=126976 | 85.27131782945736 | 56.98924731182796 | 76.7515923566879 |

#### Request-local MTP acceptance (%) — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 86.92307692307692 | 61.29032258064516 | 62.429378531073446 |
| L=8192 / C=6144 | 89.63414634146342 | 59.905660377358494 | 78.96440129449839 |
| L=16384 / C=14336 | 94.23076923076923 | 61.904761904761905 | 57.41239892183288 |
| L=32768 / C=30000 | 88.42105263157895 | 54.10958904109589 | 69.46107784431138 |
| L=65536 / C=61440 | 95.51282051282051 | 60.22099447513812 | 79.22077922077922 |
| L=131072 / C=126976 | 82.30088495575221 | 58.9041095890411 | 72.8395061728395 |

#### Fresh prompt prefill tok/s — reasoning `off`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 114.08839568897982 | 102.7764877848241 | 108.17075526366621 |
| L=8192 / C=6144 | 61.83469151047161 | 54.34310059994783 | 55.87125820271174 |
| L=16384 / C=14336 | 52.46756550496057 | 47.82050096756814 | 49.12338526678752 |
| L=32768 / C=30000 | 42.11950452146499 | 37.82091042495575 | 39.453003200783975 |
| L=65536 / C=61440 | 28.392227982492642 | 25.87566830605384 | 26.783275094589598 |
| L=131072 / C=126976 | 18.276402340044097 | 16.60934596824736 | 17.229247926654534 |

#### Fresh prompt prefill tok/s — reasoning `low`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 136.12626884938098 | 128.43025536604958 | 130.4722629948044 |
| L=8192 / C=6144 | 896.0309511184429 | 895.1433780967567 | 899.0321526827045 |
| L=16384 / C=14336 | 868.810464685362 | 870.2058423131878 | 869.6352431417374 |
| L=32768 / C=30000 | 775.9584374761313 | 776.6276068816369 | 775.9073326224312 |
| L=65536 / C=61440 | 633.6760882142646 | 633.586266097392 | 634.0343489742665 |
| L=131072 / C=126976 | 451.0066226973108 | 450.9447130933572 | 450.8029626358351 |

#### Fresh prompt prefill tok/s — reasoning `medium`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 106.49424740895692 | 97.77503138578508 | 99.96001599360255 |
| L=8192 / C=6144 | 58.01256439862492 | 52.35102860422809 | 53.807332640641235 |
| L=16384 / C=14336 | 48.816975709617736 | 44.61738994642727 | 46.08954085281543 |
| L=32768 / C=30000 | 39.010885295413075 | 35.29764741180001 | 36.963817521104424 |
| L=65536 / C=61440 | 26.95066680441452 | 24.56278247199843 | 25.35141130401322 |
| L=131072 / C=126976 | 17.158877366610486 | 15.566754970937426 | 16.101736322407845 |

#### Fresh prompt prefill tok/s — reasoning `xhigh`

| Context | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| L=77824 / C=empty | 162.83765550996102 | 158.1662161683644 | 161.08514534381018 |
| L=8192 / C=6144 | 881.5316218451959 | 898.5197559392459 | 896.0107123378872 |
| L=16384 / C=14336 | 869.8443806205747 | 871.1978226222849 | 871.3851382594326 |
| L=32768 / C=30000 | 776.4661533774329 | 776.1930367043799 | 776.9144963884495 |
| L=65536 / C=61440 | 634.3275197609388 | 633.9383174482017 | 634.4553810666301 |
| L=131072 / C=126976 | 450.932697117481 | 450.8010989517815 | 450.9616038213198 |

### Per-context fresh append curves

#### L=77824 / C=empty

Empty baseline; no source fill append curve.

#### L=8192 / C=6144

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 912.7367366916413 | 4.276873970986344 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-8192-6144/raw/append-0000.sse |
| 1 | 6144 | 2254 | 806.8699210420149 | 2.809780039009638 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-8192-6144/raw/append-0001.sse |

#### L=16384 / C=14336

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 912.7084682452535 | 4.276818723999895 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0000.sse |
| 1 | 7985 | 4095 | 861.6097015358745 | 4.770709521020763 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0001.sse |
| 2 | 9257 | 1272 | 690.5245054769667 | 1.8612595080048777 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0002.sse |
| 3 | 9388 | 131 | 209.50312575465105 | 0.6446903279866092 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0003.sse |
| 4 | 9586 | 198 | 299.94909954674364 | 0.6797955760266632 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0004.sse |
| 5 | 11732 | 2146 | 720.8008480326099 | 2.9988987860269845 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0005.sse |
| 6 | 14336 | 2604 | 707.3603506594967 | 3.703790195984766 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-16384-14336/raw/append-0006.sse |

#### L=32768 / C=30000

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 911.0266544384563 | 4.284672411973588 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0000.sse |
| 1 | 7985 | 4095 | 863.0220651243405 | 4.763325947977137 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0001.sse |
| 2 | 9257 | 1272 | 693.9761560267901 | 1.8513590850052424 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0002.sse |
| 3 | 9388 | 131 | 210.57301579136876 | 0.6415999259916134 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0003.sse |
| 4 | 9586 | 198 | 298.4683148750277 | 0.683085165976081 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0004.sse |
| 5 | 11732 | 2146 | 721.4010374594045 | 2.9964077370241284 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0005.sse |
| 6 | 14847 | 3115 | 723.9369340538772 | 4.32710858801147 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0006.sse |
| 7 | 18071 | 3224 | 698.5561295483609 | 4.64294054399943 | 14847 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0007.sse |
| 8 | 18349 | 278 | 280.1476115904524 | 1.0202484049950726 | 18071 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0008.sse |
| 9 | 18461 | 112 | 153.4297017984153 | 0.7592921600444242 | 18349 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0009.sse |
| 10 | 22557 | 4096 | 679.9859354471555 | 6.056366084027104 | 18461 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0010.sse |
| 11 | 26653 | 4096 | 636.982494489386 | 6.466220019967295 | 22557 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0011.sse |
| 12 | 27984 | 1331 | 492.4848878181587 | 2.740799352992326 | 26653 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0012.sse |
| 13 | 29302 | 1318 | 481.6170640068347 | 2.7755353079992346 | 27984 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0013.sse |
| 14 | 30000 | 698 | 406.2781025997724 | 1.7569805170060135 | 29302 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-32768-30000/raw/append-0014.sse |

#### L=65536 / C=61440

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 903.8946240112055 | 4.318572962016333 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0000.sse |
| 1 | 7985 | 4095 | 856.0759651916795 | 4.801633560971823 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0001.sse |
| 2 | 9257 | 1272 | 686.7620142859456 | 1.870865370030515 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0002.sse |
| 3 | 9388 | 131 | 203.07212470798694 | 0.6648087609792128 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0003.sse |
| 4 | 9586 | 198 | 293.40853285360544 | 0.6936221159994602 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0004.sse |
| 5 | 11732 | 2146 | 715.021583922743 | 3.0223172190017067 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0005.sse |
| 6 | 14847 | 3115 | 719.5720359753676 | 4.347006777999923 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0006.sse |
| 7 | 18071 | 3224 | 695.3018599971748 | 4.658891412022058 | 14847 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0007.sse |
| 8 | 18349 | 278 | 276.44279772044075 | 1.0345612940145656 | 18071 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0008.sse |
| 9 | 18461 | 112 | 150.40219157479152 | 0.767799221968744 | 18349 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0009.sse |
| 10 | 22557 | 4096 | 676.0699144761654 | 6.090107508993242 | 18461 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0010.sse |
| 11 | 26653 | 4096 | 634.7394972385887 | 6.488361627969425 | 22557 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0011.sse |
| 12 | 27984 | 1331 | 488.8987656316289 | 2.76083586004097 | 26653 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0012.sse |
| 13 | 29302 | 1318 | 475.9163918171268 | 2.8073504469939508 | 27984 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0013.sse |
| 14 | 30984 | 1682 | 508.91031771797594 | 3.3450181360240094 | 29302 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0014.sse |
| 15 | 35080 | 4096 | 569.8631966598449 | 7.229639987985138 | 30984 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0015.sse |
| 16 | 39176 | 4096 | 545.0254056886495 | 7.55946259200573 | 35080 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0016.sse |
| 17 | 41953 | 2777 | 496.8224474293151 | 5.636212823970709 | 39176 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0017.sse |
| 18 | 45243 | 3290 | 491.8364125182981 | 6.738749314041343 | 41953 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0018.sse |
| 19 | 49339 | 4096 | 489.45643210399226 | 8.421315683983266 | 45243 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0019.sse |
| 20 | 53435 | 4096 | 468.21531320129424 | 8.80373070499627 | 49339 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0020.sse |
| 21 | 55661 | 2226 | 415.2877378739618 | 5.417640633997507 | 53435 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0021.sse |
| 22 | 59757 | 4096 | 442.34770862590995 | 9.321149171038996 | 55661 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0022.sse |
| 23 | 60752 | 995 | 331.45641281547074 | 3.167509413033258 | 59757 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0023.sse |
| 24 | 61438 | 686 | 279.2937857132679 | 2.518916027038358 | 60752 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0024.sse |
| 25 | 61440 | 2 | 4.915115947585203 | 0.4689788359683007 | 61438 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-65536-61440/raw/append-0025.sse |

#### L=131072 / C=126976

| append | cumulative C | fresh tokens | fresh tok/s | seconds | cached prefix | raw sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3890 | 3890 | 901.1843276400414 | 4.33160901296651 | 0 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0000.sse |
| 1 | 7985 | 4095 | 860.9743959460405 | 4.774138709006365 | 3890 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0001.sse |
| 2 | 9257 | 1272 | 691.9688635771417 | 1.8571367660188116 | 7985 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0002.sse |
| 3 | 9388 | 131 | 208.20386814178843 | 0.648498534981627 | 9257 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0003.sse |
| 4 | 9586 | 198 | 295.0142590225194 | 0.6912989839911461 | 9388 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0004.sse |
| 5 | 11732 | 2146 | 717.7235921363448 | 3.0101141409832053 | 9586 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0005.sse |
| 6 | 14847 | 3115 | 721.0815065163546 | 4.344125272997189 | 11732 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0006.sse |
| 7 | 18071 | 3224 | 696.1269732144418 | 4.6591272769728675 | 14847 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0007.sse |
| 8 | 18349 | 278 | 276.2897663463163 | 1.0340491769602522 | 18071 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0008.sse |
| 9 | 18461 | 112 | 152.2270684192714 | 0.7585820560343564 | 18349 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0009.sse |
| 10 | 22557 | 4096 | 678.7983082410523 | 6.065472910995595 | 18461 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0010.sse |
| 11 | 26653 | 4096 | 636.6039170413087 | 6.469677570043132 | 22557 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0011.sse |
| 12 | 27984 | 1331 | 491.0514995332981 | 2.7478032860090025 | 26653 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0012.sse |
| 13 | 29302 | 1318 | 479.55836967619257 | 2.78697488096077 | 27984 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0013.sse |
| 14 | 30984 | 1682 | 512.2811954730424 | 3.322678545024246 | 29302 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0014.sse |
| 15 | 35080 | 4096 | 571.3304537125391 | 7.215093567967415 | 30984 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0015.sse |
| 16 | 39176 | 4096 | 545.3946508182651 | 7.5628742320113815 | 35080 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0016.sse |
| 17 | 41953 | 2777 | 497.45860811188936 | 5.628518202982377 | 39176 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0017.sse |
| 18 | 45243 | 3290 | 492.6309743297337 | 6.726742074999493 | 41953 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0018.sse |
| 19 | 49339 | 4096 | 489.70261565278935 | 8.42237424600171 | 45243 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0019.sse |
| 20 | 53435 | 4096 | 470.6132690652088 | 8.7589807039476 | 49339 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0020.sse |
| 21 | 55661 | 2226 | 416.75544094260874 | 5.404585656011477 | 53435 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0021.sse |
| 22 | 59757 | 4096 | 442.6995376837713 | 9.316738291003276 | 55661 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0022.sse |
| 23 | 60752 | 995 | 332.167686260109 | 3.055967368010897 | 59757 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0023.sse |
| 24 | 61438 | 686 | 282.1843371238999 | 2.4925316349836066 | 60752 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0024.sse |
| 25 | 61440 | 2 | 5.004040762916055 | 0.461065242998302 | 61438 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0025.sse |
| 26 | 65535 | 4095 | 419.5946070565669 | 9.824906425026711 | 61440 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0026.sse |
| 27 | 66639 | 1104 | 307.3526607302688 | 3.6600328700151294 | 65535 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0027.sse |
| 28 | 68326 | 1687 | 349.0714121066325 | 4.901313022011891 | 66639 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0028.sse |
| 29 | 72422 | 4096 | 395.24345346669327 | 10.437912861001678 | 68326 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0029.sse |
| 30 | 76126 | 3704 | 372.03158170472426 | 10.033906072028913 | 72422 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0030.sse |
| 31 | 78252 | 2126 | 332.5727127374254 | 6.4719830310205 | 76126 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0031.sse |
| 32 | 78742 | 490 | 215.19533807846372 | 2.3592686649644747 | 78252 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0032.sse |
| 33 | 82838 | 4096 | 364.6628195451437 | 11.315735852986109 | 78742 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0033.sse |
| 34 | 83273 | 435 | 192.27700618955154 | 2.34606721799355 | 82838 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0034.sse |
| 35 | 86016 | 2743 | 334.2676905356302 | 8.293491370044649 | 83273 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0035.sse |
| 36 | 90112 | 4096 | 345.6490205921923 | 11.941218229010701 | 86016 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0036.sse |
| 37 | 94208 | 4096 | 335.56403677637655 | 12.3013586839661 | 90112 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0037.sse |
| 38 | 96164 | 1956 | 292.157541622741 | 6.792615902959369 | 94208 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0038.sse |
| 39 | 98348 | 2184 | 290.69353434489534 | 7.612742562952917 | 96164 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0039.sse |
| 40 | 101719 | 3371 | 301.0786224264186 | 11.300126363988966 | 98348 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0040.sse |
| 41 | 105815 | 4096 | 310.5467389142737 | 13.295039143005852 | 101719 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0041.sse |
| 42 | 109910 | 4095 | 301.5882987569775 | 13.686902520013973 | 105815 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0042.sse |
| 43 | 114006 | 4096 | 294.90189076318467 | 14.005914187990129 | 109910 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0043.sse |
| 44 | 115144 | 1138 | 221.64022727277154 | 5.248415768961422 | 114006 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0044.sse |
| 45 | 119240 | 4096 | 285.9976699848029 | 14.438345134956762 | 115144 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0045.sse |
| 46 | 123336 | 4096 | 279.03515420575326 | 14.796298757952172 | 119240 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0046.sse |
| 47 | 126976 | 3640 | 261.21376708501293 | 14.059155456954613 | 123336 | /srv/ai/paged-kv/results/forward/106-master/106-02/cpu/cpu-131072-126976/raw/append-0047.sse |

## Phase-105 ACO reference cohort (separate build/source)

- MATRIX105 original ACO source HEAD `7a7e4befe175d050348375b4e6a862b47c416872`, candidate SHA-256 `6b5b225b4791c1c5a9402304207c74e1a119e5c7f51a5300a891909dbdae34c7`, freeze SHA-256 `3a978d416a65b2092346c6a66872642c4deb02949709a492d231954fd21cda53`.
- The ACO candidate uses selective paging and GPU Turbo4 MTP; its target/pager/source/build differ from the replacement master cohort. Treat these values as context only.
- Only exact L/C matches are included below. Other master geometries are explicitly unavailable in the selected ACO report rather than matched by approximate occupancy.

| ACO task | L | C | H | status | cohort | candidate SHA |
| --- | --- | --- | --- | --- | --- | --- |
| 105-08 | 77824 | empty | 65536 | complete | original | 6b5b225b4791c1c5a9402304207c74e1a119e5c7f51a5300a891909dbdae34c7 |
| 105-09 | 8192 | 6144 | 4096 | complete | original | 6b5b225b4791c1c5a9402304207c74e1a119e5c7f51a5300a891909dbdae34c7 |

### ACO Decode tok/s — reasoning `off` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 102.03813486589952 | 73.5370323464033 | 99.60534150244698 |
| 105-09 L=8192 / C=6144 | 90.83140941054276 | 65.86595737788704 | 77.16454258402447 |

### ACO Decode tok/s — reasoning `low` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 103.31333617617196 | 77.47188724650387 | 88.81627664849115 |
| 105-09 L=8192 / C=6144 | 90.13759503882677 | 68.44158510711108 | 80.47612896943477 |

### ACO Decode tok/s — reasoning `medium` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 97.93858858741216 | 73.36497442405285 | 90.1593747267044 |
| 105-09 L=8192 / C=6144 | 84.84864168991342 | 62.35781443966905 | 78.05735537386839 |

### ACO Decode tok/s — reasoning `xhigh` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 99.16115582243226 | 84.79055320183261 | 78.48916211838325 |
| 105-09 L=8192 / C=6144 | 82.70298651359617 | 66.65215130927044 | 71.76469933910094 |

### ACO Request-local MTP acceptance (%) — reasoning `off` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 90.76923076923077 | 51.785714285714285 | 86.58536585365853 |
| 105-09 L=8192 / C=6144 | 91.13475177304964 | 52.247191011235955 | 69.66966966966967 |

### ACO Request-local MTP acceptance (%) — reasoning `low` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 92.831541218638 | 58.75 | 73.14814814814815 |
| 105-09 L=8192 / C=6144 | 89.16083916083916 | 56.29139072847682 | 75.39432176656152 |

### ACO Request-local MTP acceptance (%) — reasoning `medium` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 84.79729729729729 | 51.53061224489796 | 74.92163009404389 |
| 105-09 L=8192 / C=6144 | 81.78807947019868 | 46.95863746958638 | 71.12462006079028 |

### ACO Request-local MTP acceptance (%) — reasoning `xhigh` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 87.39837398373983 | 67.1875 | 58.743169398907106 |
| 105-09 L=8192 / C=6144 | 77.734375 | 52.89855072463768 | 61.34453781512605 |

### ACO Fresh prompt prefill tok/s — reasoning `off` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 312.7084723148766 | 282.9061799283304 | 292.69414507176236 |
| 105-09 L=8192 / C=6144 | 1435.7639265948178 | 1430.564516316867 | 1430.7957357443236 |

### ACO Fresh prompt prefill tok/s — reasoning `low` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 99.12802738765784 | 93.22966194924578 | 98.96166377394111 |
| 105-09 L=8192 / C=6144 | 1427.2911188566509 | 1426.0234723001297 | 1426.1145794798933 |

### ACO Fresh prompt prefill tok/s — reasoning `medium` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 295.02254815189445 | 264.1645005177624 | 275.71872448276224 |
| 105-09 L=8192 / C=6144 | 1438.943919115712 | 1430.056384747587 | 1433.303657584643 |

### ACO Fresh prompt prefill tok/s — reasoning `xhigh` (separate cohort)

| ACO config | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| 105-08 L=77824 / C=empty | 105.08972034874773 | 94.68943758261653 | 94.79122233281198 |
| 105-09 L=8192 / C=6144 | 1432.4810210154317 | 1430.097562171768 | 1434.0335300414256 |

### ACO exact-geometry coverage

| Master geometry (L/C) | ACO exact match | Availability |
| --- | --- | --- |
| 77824/empty | 105-08 | available |
| 8192/6144 | 105-09 | available |
| 16384/14336 | — | unavailable: MATRIX105_FINAL has no ACO row at exact L=16384, C=14336; nearby L/C configurations are not substituted |
| 32768/30000 | — | unavailable: MATRIX105_FINAL has no ACO row at exact L=32768, C=30000; nearby L/C configurations are not substituted |
| 65536/61440 | — | unavailable: MATRIX105_FINAL has no ACO row at exact L=65536, C=61440; nearby L/C configurations are not substituted |
| 131072/126976 | — | unavailable: MATRIX105_FINAL has no ACO row at exact L=131072, C=126976; nearby L/C configurations are not substituted |

### ACO actual fresh-ingestion checkpoints (separate cohort; no interpolation)

| C checkpoint | Task | Cohort | Fresh tokens | Fresh tok/s | Cached history tokens |
| --- | --- | --- | --- | --- | --- |
| 8192 | unavailable | — | — | — | — |
| 32768 | unavailable | — | — | — | — |
| 61440 | 105-15 | scratch | 10771 | 910.4431916752137 | 54190 |
| 61440 | 105-16 | scratch | 7250 | 602.0353111563454 | 54190 |
| 61440 | 105-17 | scratch | 7250 | 441.1893393563894 | 54190 |
| 61440 | 105-18 | scratch | 10771 | 511.5875943784356 | 54190 |
| 61440 | 105-19 | scratch | 10771 | 511.52605304151416 | 54190 |
| 65536 | unavailable | — | — | — | — |
| 131072 | 105-18 | scratch | 4387 | 551.516160290937 | 126926 |
| 131072 | 105-19 | scratch | 4387 | 553.0361964522734 | 126926 |
| 196608 | 105-19 | scratch | 7888 | 541.8498649977764 | 191335 |
| 256000 | 105-20 | scratch | 3640 | 441.07830072888186 | 252527 |

MATRIX105 overall: 15 configurations; 14 fully completed; 3 execution-incomplete requests in 105-20; recall `2/2` semantic correct and `2` physically verified promotions. Cross-cohort comparison is explicitly descriptive only. These measurements do not certify the ACO goal.

## Availability and interpretation

- All required master placement rows are available and passed identity, canonical schedule, and exact L/C validation.
- Fresh prompt prefill medians are separate request-level metrics; append curves report actual uncached source appends only.
- Null median cells remain unavailable; they are not rewritten as zero. Incomplete ACO configurations remain identified by their MATRIX105 status and are not used in exact-match rows unless geometries match.
