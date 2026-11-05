#!/usr/bin/env python3
"""Build a provenance-preserving comparison digest and plots from retained runs.

No benchmark, server, model, or build operations are performed. Output defaults
to /srv/ai/paged-kv/results/comparison-final-20261104. Raw request data remains
at its source; only validated summaries and source hashes are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / ".wiretail/execution/evidence"
BENCH = ROOT / ".wiretail/execution/benchmarks"
DEFAULT_OUT = Path("/srv/ai/paged-kv/results/comparison-final-20261104")
MODES = ("off", "low", "medium", "xhigh")
PROMPTS = ("P1", "P2", "P3")
REPS = (1, 2, 3)
_HASH_CACHE = {}
_RECORD_CACHE = {}
_IDENTITY_CACHE = {}
_IDENTITY_ARTIFACTS = {}
ALLOW_PILLOW_DRAFT = False
DATA_ONLY = False
PREFILL_GROUPS = []
_PREFILL_SEEN = set()
OUTLIER_SERIES_CELLS = []
RESPONSE_QUALITY_EXCLUSIONS = []
RESPONSE_QUALITY_UNASSESSED = []
_RESPONSE_QUALITY_CACHE = {}


def plotting_pyplot():
    # These are exported artifacts, never GUI windows. Avoid environment-
    # dependent Tk selection when running unattended on the benchmark host.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt
FAMILIES = (
    "master-gpu-mtp-off", "master-gpu-mtp-on", "fork-gpu-mtp-off", "fork-gpu-mtp-on",
    "master-cpu-mtp-off", "master-cpu-mtp-on", "fork-cpu-mtp-off", "fork-cpu-mtp-on",
    "current-aco-mtp-off", "current-aco-mtp-on",
)


def jread(p):
    return json.loads(Path(p).read_text())


def sha(p):
    p = Path(p)
    st = p.stat()
    cache_key = (str(p), st.st_mtime_ns, st.st_size)
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    value = h.hexdigest()
    _HASH_CACHE[cache_key] = value
    return value


def finite_pos(x):
    try:
        return math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


def finite_nonnegative(x):
    try:
        return math.isfinite(float(x)) and float(x) >= 0
    except (TypeError, ValueError):
        return False


def timing_close(derived, declared):
    return finite_pos(derived) and finite_pos(declared) and abs(float(derived)-float(declared)) <= max(0.02, abs(float(declared))*0.01)


def resolve_verified_chunk_map(identity_path, expected_sha, source_hashes):
    """Bind a mapped-prefill point only to an adjacent config/manifest and verified map file."""
    if not expected_sha or not identity_path:
        return None
    anchor=Path(identity_path)
    candidates=[]
    for parent in [anchor.parent,*list(anchor.parents)[:5]]:
        for name in ('row.json','run-config.json','source-manifest.json'):
            q=parent/name
            if q.is_file() and q not in candidates:
                candidates.append(q)
    for q in candidates:
        try:
            meta=jread(q)
        except Exception:
            continue
        sh=source_hashes.get(str(q),{}).get('sha256') or sha(q)
        source_hashes[str(q)]={"sha256":sh}
        maps=[]
        if isinstance(meta,dict):
            for obj,key in ((meta,'source_chunk_map'),(meta,'chunk_map'),(meta.get('identity',{}),'source_chunk_map')):
                val=obj.get(key) if isinstance(obj,dict) else None
                if isinstance(val,dict): maps.append(val)
        for item in maps:
            map_path=item.get('path')
            declared=item.get('sha256')
            if not map_path or declared!=expected_sha: continue
            mp=Path(map_path)
            if not mp.is_absolute(): mp=q.parent/mp
            if not mp.is_file(): continue
            actual=sha(mp)
            source_hashes[str(mp)]={"sha256":actual,"declared_sha256":declared,"hash_verified":True}
            if actual!=declared: continue
            return {"sha256":actual,"identity_path":str(q),"identity_sha256":sh,"map_path":str(mp)}
    return None


def read_records(path):
    p=Path(path); st=p.stat(); cache_key=(str(p),st.st_mtime_ns,st.st_size)
    if cache_key in _RECORD_CACHE: return _RECORD_CACHE[cache_key]
    out = []
    if p.suffix=='.json':
        payload=jread(p)
        source_rows=payload if isinstance(payload,list) else payload.get('rows',[])
    else:
        source_rows=[]
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip(): source_rows.append(json.loads(line))
    for r in source_rows:
        # Retain only measurement fields; response/output payloads are never exported.
        usage=r.get('usage') or {}
        mtp=r.get('mtp')
        if mtp is None and r.get('mtp_acceptance_percent') is not None:
            mtp={"acceptance_percent":r.get('mtp_acceptance_percent'),"status":r.get('mtp_status')}
        out.append({"kind":r.get('kind'),"status":r.get('status') or ("pass" if r.get('http_status')==200 else None),
                    "http_status":r.get('http_status'),"stream_complete":r.get('stream_complete'),
                    "repetition":r.get('repetition',r.get('rep')),"prompt_index":r.get('prompt_index'),
                    "reasoning":r.get('reasoning',r.get('mode')),
                    "completion_tokens":r.get('completion_tokens',usage.get('completion_tokens')),
                    "decode_tok_s":r.get('decode_tok_s'),"mtp":mtp,
                    "actual_C":r.get('actual_C_before',r.get('actual_C',r.get('actual_occupancy_tokens'))),
                    "actual_C_after":r.get('actual_C_after'),
                    "nominal_target_C":r.get('nominal_target_C',r.get('target_tokens')),
                    "raw_path":r.get('raw_path',r.get('sse_path')),
                    "raw_sha256":r.get('raw_sha256',r.get('sse_sha256')),
                    "conversation_protocol":r.get('conversation_protocol'),
                    "context_tokens":r.get('context_tokens'),"target_tokens":r.get('target_tokens')})
    _RECORD_CACHE[cache_key] = out
    return out


def validated_triplet(records, prompt_index, mode, allow_unlabeled_mode=False):
    selected = {}
    for r in records:
        if r.get("kind") not in ("measured", "measurement"):
            continue
        if r.get("status") not in ("pass", "complete", "ok") or r.get("http_status") != 200:
            continue
        if r.get("stream_complete") is not True and r.get("_completion_attested") is not True:
            continue
        if r.get("prompt_index") != prompt_index:
            continue
        if r.get("reasoning") != mode and not (allow_unlabeled_mode and mode == "off" and r.get("reasoning") is None):
            continue
        rep = r.get("repetition")
        if rep not in REPS or not finite_pos(r.get("decode_tok_s")):
            continue
        try:
            if int(r.get("completion_tokens", 0)) <= 3:
                continue
        except (TypeError, ValueError):
            continue
        selected[rep] = r
    if set(selected) != set(REPS):
        return None
    return [selected[i] for i in REPS]


def summary(vals):
    vals = [float(x) for x in vals]
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals), "repeats": vals}


def neighboring_identity(records_path, source_hashes):
    """Capture compact build/config identity artifacts without copying command payloads."""
    p = Path(records_path)
    result = {}
    scalar_keys = {
        "source_commit", "repo_context_commit", "commit", "branch", "dirty",
        "model_sha256", "binary_sha256", "executable_sha256", "build_sha256",
        "fixture_sha256", "template_sha256", "chunk_map_sha256", "sha256",
        "target_tokens", "requested_target_tokens", "actual_occupancy_tokens",
        "context_tokens", "C_tokens", "L_tokens", "H_tokens", "hot_pages",
        "batch", "ubatch", "threads", "n_threads", "thread_count",
        "mtp", "placement", "target_kv", "draft", "quantization",
        "pager", "aco", "turbo", "turbo_level", "template_id",
        "protocol", "repo_context_commit", "source_worktree_commit",
    }
    mapping_keys = {"geometry", "configuration", "source_chunk_map", "model", "target", "draft", "build", "git_identity", "identity", "loaded_project_dsos", "dso_sha256"}

    def compact(obj):
        if not isinstance(obj, dict):
            return {}
        out = {}
        for k, v in obj.items():
            if k not in scalar_keys and k not in mapping_keys:
                continue
            if isinstance(v, dict):
                nested = compact(v)
                if nested:
                    out[k] = nested
            elif isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
        return out

    for parent in list(p.parents)[:6]:
        for name in ("run-config.json", "source-manifest.json", "server-argv.json", "build-receipt.json"):
            q = parent / name
            if not q.is_file():
                continue
            digest = source_hashes.get(str(q),{}).get("sha256") or sha(q)
            source_hashes[str(q)] = {"sha256": digest}
            st=q.stat(); identity_key=(str(q),st.st_mtime_ns,st.st_size)
            if identity_key in _IDENTITY_CACHE:
                artifact = _IDENTITY_CACHE[identity_key]
                _IDENTITY_ARTIFACTS[str(q)] = artifact
                result[name] = {"path": str(q), "sha256": digest}
                continue
            try:
                data = jread(q)
            except Exception:
                continue
            # Keep only configuration/provenance scalars. Never copy inventories,
            # environment, argv, histories, source catalogs, or raw payloads.
            safe = compact(data)
            artifact = {"path": str(q), "sha256": digest, "identity_fields": safe}
            _IDENTITY_ARTIFACTS[str(q)] = artifact
            result[name] = {"path": str(q), "sha256": digest}
            _IDENTITY_CACHE[identity_key] = artifact
    return result


def neighboring_geometry(records_path, source_hashes):
    p=Path(records_path)
    for parent in list(p.parents)[:6]:
        rowp=parent/'row.json'
        if not rowp.is_file(): continue
        try:
            row=jread(rowp); geo=row.get('geometry',{}) if isinstance(row.get('geometry',{}),dict) else {}; base=row.get('base',{}) if isinstance(row.get('base',{}),dict) else {}
            source_hashes[str(rowp)]={"sha256":sha(rowp)}
            actual=geo.get('actual_C',row.get('actual_occupancy_tokens',base.get('occupied_tokens')))
            nominal=geo.get('target_tokens',base.get('requested_target_tokens'))
            return {"actual_C":actual,"nominal_target_C":nominal,"row_path":str(rowp),"row_sha256":source_hashes[str(rowp)]['sha256']}
        except Exception: continue
    return {}


def campaign_fallback_geometry(records, cfg, rowmeta):
    """Recover grouping geometry for a partial campaign without a final row.json."""
    geo = rowmeta.get("geometry", {}) if isinstance(rowmeta.get("geometry"), dict) else {}
    row_base = rowmeta.get("base", {}) if isinstance(rowmeta.get("base"), dict) else {}
    cfg_geo = cfg.get("geometry", {}) if isinstance(cfg.get("geometry"), dict) else {}
    cfg_base = cfg.get("base", {}) if isinstance(cfg.get("base"), dict) else {}
    identity = cfg.get("identity", {}) if isinstance(cfg.get("identity"), dict) else {}
    identity_geo = identity.get("geometry", {}) if isinstance(identity.get("geometry"), dict) else {}
    nominal = (geo.get("target_tokens") or cfg.get("requested_target_tokens") or
               cfg_geo.get("target_tokens") or identity_geo.get("target_tokens"))
    actual = (rowmeta.get("actual_occupancy_tokens") or rowmeta.get("actual_C") or
              geo.get("actual_C") or row_base.get("occupied_tokens") or
              cfg.get("actual_occupancy_tokens") or cfg.get("actual_C") or
              cfg_base.get("occupied_tokens"))
    if nominal is None:
        nominal = next((r.get("nominal_target_C") or r.get("target_tokens") for r in records
                        if r.get("nominal_target_C") is not None or r.get("target_tokens") is not None), None)
    if actual is None:
        values = [r.get("actual_C_after") for r in records if r.get("actual_C_after") is not None]
        if not values:
            values = [r.get("actual_C") for r in records if r.get("actual_C") is not None]
        if values:
            actual = max(int(v) for v in values)
    return actual, nominal


def append_schedule_groups(schedule_path, family, nominal_c, mtp, protocol, source_hashes, *, identity_path=None, expected_map_sha=None):
    """Import only successful append schedule entries; never retain request/output text."""
    p=Path(schedule_path)
    if not p.is_file(): return
    digest=sha(p); source_hashes[str(p)]={"sha256":digest}
    for line_no,line in enumerate(p.open(encoding='utf-8'),1):
        if not line.strip(): continue
        try: r=json.loads(line)
        except Exception: continue
        ix=r.get('append_index')
        raw=r.get('raw_path')
        rawsha=r.get('raw_sha256')
        if not isinstance(ix,int) or not raw or not rawsha: continue
        append_status=r.get('status')
        if append_status is not None and append_status not in ('pass','complete','ok','success'):
            continue
        rawpath=Path(raw)
        if not rawpath.is_absolute(): rawpath=p.parent/rawpath
        if not rawpath.is_file(): continue
        actual_rawsha=rawsha if DATA_ONLY else sha(rawpath)
        source_hashes[str(rawpath)]={"sha256":actual_rawsha,"declared_sha256":rawsha,"hash_verified":not DATA_ONLY}
        if actual_rawsha!=rawsha: continue
        key=(str(p),ix)
        if key in _PREFILL_SEEN: continue
        timings=r.get('timings') if isinstance(r.get('timings'),dict) else {}
        backend_tokens=r.get('backend_prefill_tokens')
        if backend_tokens is None: backend_tokens=timings.get('prompt_n',r.get('prompt_n'))
        net_tokens=r.get('fresh_tokens',r.get('fresh_prefill_tokens',r.get('fresh_tokens')))
        backend_seconds=r.get('backend_prompt_seconds')
        if backend_seconds is None and finite_pos(timings.get('prompt_ms')): backend_seconds=float(timings['prompt_ms'])/1000.0
        wall_seconds=r.get('wall_seconds',r.get('elapsed_seconds'))
        backend_rate=r.get('backend_prefill_tok_s')
        net_rate=r.get('net_new_tok_s')
        derived_backend=(float(backend_tokens)/float(backend_seconds)) if finite_pos(backend_tokens) and finite_pos(backend_seconds) else None
        native_backend=timings.get('prompt_per_second')
        backend_timing_valid=(derived_backend is not None and (native_backend is None or timing_close(derived_backend,native_backend)) and (backend_rate is None or timing_close(derived_backend,backend_rate)))
        if backend_timing_valid and backend_rate is None: backend_rate=derived_backend
        elif not backend_timing_valid and backend_seconds is not None: backend_rate=None
        if net_rate is None and finite_pos(net_tokens) and finite_pos(backend_seconds) and backend_timing_valid:
            net_rate=float(net_tokens)/float(backend_seconds)
        wall_rate=r.get('wall_prefill_tok_s',r.get('http_wall_tok_s'))
        if wall_rate is None and finite_pos(net_tokens) and finite_pos(wall_seconds): wall_rate=float(net_tokens)/float(wall_seconds)
        actual_c=r.get('cumulative_tokens')
        if actual_c is None and r.get('prior_occupied_tokens') is not None and net_tokens is not None:
            actual_c=int(r['prior_occupied_tokens'])+int(net_tokens)
        if actual_c is None: actual_c=nominal_c
        source={"schedule_path":str(p),"schedule_sha256":digest,"line":line_no,"raw_path":str(rawpath),"raw_sha256":rawsha}
        if identity_path and Path(identity_path).is_file():
            source["identity_path"]=str(identity_path); source["identity_sha256"]=sha(identity_path); source_hashes[str(identity_path)]={"sha256":source["identity_sha256"]}
        map_identity=resolve_verified_chunk_map(identity_path or p,expected_map_sha,source_hashes)
        if map_identity:
            source['chunk_map_sha256']=map_identity['sha256']
            source['chunk_map_identity_path']=map_identity['identity_path']
            source['chunk_map_path']=map_identity['map_path']
        PREFILL_GROUPS.append({"family":family,"nominal_target_C":int(nominal_c) if nominal_c is not None else None,"actual_C":int(actual_c),"append_index":ix,"chunk_kind":r.get('chunk_kind') or 'legacy_append',"protocol":protocol,"mtp":"on" if mtp else "off","backend_tokens":backend_tokens,"net_new_tokens":net_tokens,"backend_prompt_seconds":backend_seconds,"http_wall_seconds":wall_seconds,"backend_tok_s":backend_rate,"net_new_tok_s":net_rate,"http_wall_tok_s":wall_rate,"chunk_map_sha256":source.get('chunk_map_sha256'),"source":source})
        _PREFILL_SEEN.add(key)


def row_is_complete(row):
    if row.get('status')=='complete': return True
    if row.get('failed_cases') not in (None,[]): return False
    expected=row.get('expected_case_count')
    if expected is None or row.get('successful_case_count')!=expected: return False
    valid=row.get('valid_repeat_counts')
    modes=row.get('reasoning_modes',[])
    if not isinstance(valid,dict) or not modes: return False
    prompts=row.get('prompt_indices',[0,1,2])
    return all(all(int(valid.get(mode,{}).get(str(pi),0))==3 for pi in prompts) for mode in modes)


def timing_from_sse(path, expected_sha):
    """Read only final SSE metadata frames; discard all response content fields."""
    p=Path(path)
    if not p.is_file(): return None
    if sha(p)!=expected_sha: return None
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-65536)); tail=f.read().decode('utf-8','replace')
    found=[]
    for line in tail.splitlines():
        if not line.startswith('data:'): continue
        try: d=json.loads(line[5:].strip())
        except Exception: continue
        if not isinstance(d,dict): continue
        t=d.get('timings')
        if isinstance(t,dict):
            found.append({k:t.get(k) for k in ('prompt_n','prompt_ms','prompt_per_second','cache_n')})
    return found[-1] if found else None


def response_quality_from_text(text):
    """Identify long repeated-punctuation responses without rejecting short EOS answers."""
    compact=''.join(ch for ch in text if not ch.isspace())
    if len(compact)<64 or any(ch.isalnum() for ch in compact):
        return {"classification":"no_repeated_punctuation_failure","nonspace_chars":len(compact),"alphanumeric_chars":sum(ch.isalnum() for ch in compact),"dominant_character_fraction":None}
    counts=defaultdict(int)
    for ch in compact: counts[ch]+=1
    dominant=max(counts.values(),default=0)/len(compact)
    failed=dominant>=0.80
    return {"classification":"repeated_punctuation_only" if failed else "no_repeated_punctuation_failure","nonspace_chars":len(compact),"alphanumeric_chars":0,"dominant_character_fraction":dominant}


def response_quality_for_record(record, source_hashes):
    """Read and hash-check one SSE response, returning metrics but never its text."""
    path=record.get('raw_path'); expected=record.get('raw_sha256')
    if not path or not expected: return {"classification":"raw_response_unavailable","path":path,"sha256":expected,"hash_verified":False}
    p=Path(path)
    if not p.is_file(): return {"classification":"raw_response_unavailable","path":str(p),"sha256":expected,"hash_verified":False}
    digest=sha(p)
    source_hashes[str(p)]={"sha256":digest,"declared_sha256":expected,"hash_verified":digest==expected}
    if digest!=expected: return {"classification":"raw_hash_mismatch","path":str(p),"sha256":digest,"hash_verified":False}
    st=p.stat(); key=(str(p),st.st_mtime_ns,st.st_size,digest)
    if key in _RESPONSE_QUALITY_CACHE: return dict(_RESPONSE_QUALITY_CACHE[key])
    parts=[]
    def add_text(value):
        if isinstance(value,str): parts.append(value)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,str): parts.append(item)
                elif isinstance(item,dict) and isinstance(item.get('text'),str): parts.append(item['text'])
    with p.open('r',encoding='utf-8',errors='replace') as stream:
        for line in stream:
            if not line.startswith('data:'): continue
            try: frame=json.loads(line[5:].strip())
            except Exception: continue
            if not isinstance(frame,dict): continue
            for choice in frame.get('choices',[]) or []:
                if not isinstance(choice,dict): continue
                for block in ('delta','message'):
                    obj=choice.get(block) or {}
                    if not isinstance(obj,dict): continue
                    add_text(obj.get('content')); add_text(obj.get('reasoning_content'))
    result=response_quality_from_text(''.join(parts))
    result.update({"path":str(p),"sha256":digest,"hash_verified":True})
    _RESPONSE_QUALITY_CACHE[key]=result
    return dict(result)


def cell(family, c, mode, pi, triplet, source, identity, variant, mtp, nominal_c=None):
    rates = [float(r["decode_tok_s"]) for r in triplet]
    actuals=[int(r['actual_C']) for r in triplet if r.get('actual_C') is not None]
    actual_values=actuals if len(actuals)==3 else None
    actual_median=int(statistics.median(actual_values)) if actual_values else int(c)
    accept = []
    if mtp:
        for r in triplet:
            m = r.get("mtp")
            a = m.get("acceptance_percent") if isinstance(m, dict) else None
            if a is None and isinstance(m, dict):
                accepted, drafted = m.get("accepted_tokens"), m.get("draft_tokens")
                if isinstance(accepted,(int,float)) and isinstance(drafted,(int,float)) and drafted > 0:
                    a = 100.0 * accepted / drafted
            if not finite_nonnegative(a):
                return None
            accept.append(float(a))
    return {
        "family": family, "variant": variant, "mtp": "on" if mtp else "off", "placement": "GPU" if "gpu" in family else ("CPU Q4_0 target" if "cpu" in family else "GPU target; ACO"),
        "actual_C": actual_median, "actual_C_repeats": actual_values,
        "actual_C_after_repeats": [r.get("actual_C_after") for r in triplet],
        "conversation_protocols": sorted({r.get("conversation_protocol") for r in triplet if r.get("conversation_protocol")}),
        "actual_C_min": min(actual_values) if actual_values else None,
        "actual_C_median": actual_median if actual_values else int(c),
        "actual_C_max": max(actual_values) if actual_values else None,
        "nominal_target_C": int(nominal_c if nominal_c is not None else c), "reasoning_mode": mode, "prompt": f"P{pi+1}",
        "decode_tok_s": summary(rates),
        "mtp_acceptance_percent": summary(accept) if mtp else None,
        "completion_tokens": [r["completion_tokens"] for r in triplet],
        "source": source, "source_identity_ref": family,
    }


def add_raw_family(cells, family, rows, identity, variant, mtp, source_hashes):
    """rows: C, path, declared hash, provenance, optional requested mode/unlabeled flag."""
    for row in rows:
        c, path, declared, provenance = row[:4]
        requested_mode = row[4] if len(row) > 4 else None
        allow_unlabeled = bool(row[5]) if len(row) > 5 else False
        attested_complete = bool(row[6]) if len(row) > 6 else False
        nominal_override = row[7] if len(row) > 7 else None
        p = Path(path)
        if not p.is_file():
            continue
        actual_hash = sha(p)
        source_hashes[str(p)] = {"sha256": actual_hash, "declared_sha256": declared}
        if declared and actual_hash != declared:
            continue
        records = [dict(r, _completion_attested=attested_complete) for r in read_records(p)]
        raw_identity = neighboring_identity(p, source_hashes)
        geometry_meta=neighboring_geometry(p,source_hashes)
        for parent in list(p.parents)[:5]:
            schedule=parent/'schedule.jsonl'; rowp=parent/'row.json'
            if not schedule.is_file() or not rowp.is_file(): continue
            try:
                rowmeta=jread(rowp)
                if not row_is_complete(rowmeta): continue
                geo=rowmeta.get('geometry',{}) if isinstance(rowmeta.get('geometry',{}),dict) else {}
                actual=rowmeta.get('base',{}).get('occupied_tokens',geo.get('actual_C',c))
                nominal=geo.get('target_tokens',c)
                protocol=rowmeta.get('filled_prefix_method','historical schedule')
                source_hashes[str(rowp)]={"sha256":sha(rowp)}
                append_schedule_groups(schedule,family,nominal,mtp,str(protocol),source_hashes,identity_path=rowp)
            except Exception: pass
            break
        for pi in range(3):
            for mode in MODES:
                if requested_mode and mode != requested_mode:
                    continue
                t = validated_triplet(records, pi, mode, allow_unlabeled)
                if not t:
                    continue
                response_audits=[response_quality_for_record(r,source_hashes) for r in t]
                bad=[(r,q) for r,q in zip(t,response_audits) if q and q.get('classification')=='repeated_punctuation_only' and q.get('hash_verified')]
                unverified=[(r,q) for r,q in zip(t,response_audits) if q and q.get('classification')=='raw_hash_mismatch']
                unassessed=[(r,q) for r,q in zip(t,response_audits) if q and q.get('classification')=='raw_response_unavailable']
                if unassessed:
                    RESPONSE_QUALITY_UNASSESSED.append({"family":family,"nominal_target_C":int(nominal_override if nominal_override is not None else c),"reasoning_mode":mode,"prompt":f"P{pi+1}","missing_raw_repetitions":[int(r['repetition']) for r,_ in unassessed],"records_path":str(p)})
                if bad or unverified:
                    exclusion_status=('whole_triplet_excluded_due_to_repeated_punctuation_only_output' if bad else 'whole_triplet_excluded_due_to_unverified_raw_response')
                    RESPONSE_QUALITY_EXCLUSIONS.append({
                            "family":family,
                            "nominal_target_C":int(nominal_override if nominal_override is not None else c),
                            "reasoning_mode":mode,
                            "prompt":f"P{pi+1}",
                            "status":exclusion_status,
                            "bad_repetitions":[int(r['repetition']) for r,_ in bad],
                            "unverified_repetitions":[int(r['repetition']) for r,_ in unverified],
                            "responses":[{"repetition":int(r['repetition']),"path":q['path'],"sha256":q['sha256'],"hash_verified":q.get('hash_verified'),"classification":q.get('classification'),"nonspace_chars":q.get('nonspace_chars'),"alphanumeric_chars":q.get('alphanumeric_chars'),"dominant_character_fraction":q.get('dominant_character_fraction')} for r,q in [*bad,*unverified]],
                            "records_path":str(p),
                            "records_sha256":actual_hash,
                        })
                    continue
                # Prefer newer current campaign by caller ordering; whole cell is replaced.
                observed_cs={int(r['actual_C']) for r in t if r.get('actual_C') is not None}
                actual_c=next(iter(observed_cs)) if len(observed_cs)==1 else (geometry_meta.get('actual_C') if geometry_meta.get('actual_C') is not None else int(c))
                nominal_c=nominal_override if nominal_override is not None else (geometry_meta.get('nominal_target_C') if geometry_meta.get('nominal_target_C') is not None else c)
                item = cell(family, actual_c, mode, pi, t, {"path": str(p), "sha256": actual_hash, "context_identity_artifacts":raw_identity,"observed_geometry":geometry_meta, **provenance}, identity, variant, mtp, nominal_c=nominal_c)
                if item:
                    independent_series=provenance.get('independent_series_id')
                    if independent_series:
                        item['independent_series_id']=independent_series
                        OUTLIER_SERIES_CELLS.append(item)
                    else:
                        cells[(family, int(nominal_c), mode, pi)] = item


def discover_current_completion(plan, source_hashes):
    rows = defaultdict(list)
    # Explicitly approved retained series are the only imports outside the active
    # plan root. Keep this allowlist narrow so quarantined R4 variants cannot leak
    # into primary coverage through broad directory discovery.
    for spec in plan.get("retained_completed_series", []):
        if not isinstance(spec, dict) or spec.get("family") not in FAMILIES[:8]:
            continue
        series = Path(spec.get("path", ""))
        if not series.is_dir() or series.parent.name != spec["family"] or not series.name.startswith("attempt-"):
            continue
        for p in sorted(series.glob("cell/C*/campaign/records.jsonl")):
            try:
                cfgp = p.parent / "run-config.json"
                cfg = jread(cfgp) if cfgp.exists() else {}
                rowp = p.parent / "row.json"
                rowmeta = jread(rowp) if rowp.exists() else {}
                if not spec.get("allow_partial_triplets") and not row_is_complete(rowmeta): continue
                actual, nominal = campaign_fallback_geometry(read_records(p), cfg, rowmeta)
                if actual is None or nominal is None: continue
                rows[spec["family"]].append((int(actual), str(p), None,
                    {"approved_retained_series":str(series),"retention_reason":spec.get("reason"),"allow_partial_triplets":bool(spec.get("allow_partial_triplets")),"run_config":str(cfgp) if cfgp.exists() else None,"row_metadata":str(rowp) if rowp.exists() else None},
                    None, False, False, int(nominal)))
            except Exception:
                continue
    root = Path(plan.get("output_root", ""))
    if not root.is_dir():
        return rows
    # Locate only compact campaign record files. In-progress/failed records naturally fail
    # triplet validation; the path and file hash remain the cell provenance.
    step_by_id={s.get("id"):s for s in plan.get("steps",[]) if isinstance(s,dict)}
    for step_id,step in step_by_id.items():
        family=step.get("family",step_id)
        if family not in FAMILIES[:8]:
            continue
        for p in root.glob(f"{step_id}/**/records.jsonl"):
            try:
                cfgp = p.parent / "run-config.json"
                cfg = jread(cfgp) if cfgp.exists() else {}
                rowp = p.parent / "row.json"
                rowmeta = jread(rowp) if rowp.exists() else {}
                c, nominal = campaign_fallback_geometry(read_records(p), cfg, rowmeta)
                if nominal is None or c is None: continue
                provenance={"run_config":str(cfgp) if cfgp.exists() else None,"row_metadata":str(rowp) if rowp.exists() else None,"execution_step_id":step_id}
                if step.get("outlier_series_id"):
                    provenance["independent_series_id"]=step["outlier_series_id"]
                rows[family].append((int(c), str(p), None, provenance,None,False,False,int(nominal)))
            except Exception:
                continue
    return rows


def discover_current_prefill(plan, source_hashes):
    roots=[]
    root=Path(plan.get('output_root',''))
    if root.is_dir(): roots.append((root, None))
    for spec in plan.get("retained_completed_series", []):
        if isinstance(spec,dict) and spec.get("family") in FAMILIES[:8]:
            series=Path(spec.get("path",""))
            if series.is_dir() and series.parent.name==spec["family"] and series.name.startswith("attempt-"):
                roots.append((series,spec))
    steps={s.get('id'):s for s in plan.get('steps',[]) if isinstance(s,dict)}
    for root_entry in roots:
      root,spec=root_entry
      families=[(spec["family"], spec.get("series_id", spec["family"]))] if spec else [(s.get('family',sid),sid) for sid,s in steps.items() if s.get('family',sid) in FAMILIES[:8]]
      for family,step_id in families:
        step=steps.get(step_id,{})
        mtp=step.get('mtp')=='on' if step else family.endswith('-mtp-on')
        if spec: mtp=family.endswith('-mtp-on')
        schedules=root.glob('cell/C*/campaign/schedule.jsonl') if spec else root.glob(f'{step_id}/**/schedule.jsonl')
        for schedule in schedules:
            rowp=schedule.parent/'row.json'
            try:
                row=jread(rowp) if rowp.is_file() else {}
                # Partial series may contribute recorded append groups, but not imply series completion.
                if spec and not spec.get('allow_partial_triplets') and not row_is_complete(row): continue
                geo=row.get('geometry',{}) if isinstance(row.get('geometry',{}),dict) else {}
                base=row.get('base',{}) if isinstance(row.get('base',{}),dict) else {}
                cfgp=schedule.parent/'run-config.json'
                cfg=jread(cfgp) if cfgp.is_file() else {}
                record_path=schedule.parent/'records.jsonl'
                compact_records=read_records(record_path) if record_path.is_file() else []
                actual,nominal=campaign_fallback_geometry(compact_records,cfg,row)
                if actual is None:
                    entries=[json.loads(line) for line in schedule.read_text().splitlines() if line.strip()]
                    actual=entries[-1].get('cumulative_tokens') if entries else None
                if nominal is None or actual is None: continue
                identity=row.get('filled_prefix_method') or cfg.get('identity',{}).get('filled_prefix_method') or 'current-plan schedule'
                if rowp.is_file(): source_hashes[str(rowp)]={"sha256":sha(rowp)}
                if cfgp.is_file(): source_hashes[str(cfgp)]={"sha256":sha(cfgp)}
                first=len(PREFILL_GROUPS)
                expected=(plan.get('chunk_map') or {}).get('sha256')
                append_schedule_groups(schedule,family,nominal,mtp,str(identity),source_hashes,identity_path=rowp if rowp.is_file() else cfgp,expected_map_sha=expected)
                for group in PREFILL_GROUPS[first:]:
                    group['execution_step_id']=step_id
                    group['source']['execution_step_id']=step_id
                    if spec:
                        group['retained_series_path']=str(root)
                        group['retained_series_allow_partial']=bool(spec.get('allow_partial_triplets'))
                        group['retention_reason']=spec.get('reason')
            except Exception: continue


def collect(out, plan_path=None):
    PREFILL_GROUPS.clear(); _PREFILL_SEEN.clear(); OUTLIER_SERIES_CELLS.clear(); RESPONSE_QUALITY_EXCLUSIONS.clear(); RESPONSE_QUALITY_UNASSESSED.clear()
    _IDENTITY_ARTIFACTS.clear()
    cells, source_hashes, identities = {}, {}, {}
    empty_context_controls=[]
    plan_path = Path(plan_path) if plan_path else BENCH / "COMPARISON_COMPLETION_20261104.json"
    plan = jread(plan_path)
    source_hashes[str(plan_path)] = {"sha256":sha(plan_path)}
    coverage = jread(BENCH / "COMPARISON_COVERAGE_20261104.json")
    coverage_path = BENCH / "COMPARISON_COVERAGE_20261104.json"
    source_hashes[str(coverage_path)] = {"sha256": sha(coverage_path)}
    for key, entries in coverage.items():
        family = key
        identities[family] = {"source": str(coverage_path), "declared_family": key, "protocol": "retained historical comparison records; per-mode fresh-prefill schedule where source records indicate it"}
        for e in entries:
            legacy = e.get("legacy_request_mode_verified") is True and e.get("reasoning") == "off"
            add_raw_family(cells, family, [(e["target_tokens"], e["records_path"], e.get("records_sha256"), {"index": str(coverage_path), "legacy_request_mode_verified":e.get("legacy_request_mode_verified")}, e.get("reasoning"), legacy, True)],
                           identities[family], "historical", "mtp-on" in family, source_hashes)

    # Master GPU MTP-on controls, validated from retained raw records.
    mp = EVIDENCE / "MASTER106_GPU.json"; master = jread(mp)
    source_hashes[str(mp)] = {"sha256": sha(mp)}
    for row in master.get("rows", []):
        raw = row.get("raw", {})
        rec = raw.get("records")
        C = row.get("geometry", {}).get("C_tokens")
        if rec and C is None and row.get("geometry", {}).get("placement", "").lower() == "gpu":
            source_hashes[rec]={"sha256":raw.get('records_sha256'),"declared_sha256":raw.get('records_sha256'),"hash_verified":False}
            empty_context_controls.append({"family":"master-gpu-mtp-on","cohort":"separate empty-prefix baseline","actual_C":0,"configured_L_tokens":row.get('geometry',{}).get('L_tokens'),"label":row.get('label'),"medians":row.get('medians'),"report_path":str(mp),"report_sha256":source_hashes[str(mp)]['sha256'],"records_path":rec,"records_sha256":raw.get('records_sha256'),"master_build_commit":master.get('master_build_commit'),"binary_sha256":master.get('build',{}).get('executable_sha256'),"model_sha256":master.get('build',{}).get('model',{}).get('sha256')})
        if rec and C is not None and row.get("geometry", {}).get("placement", "").lower() == "gpu":
            L = row["geometry"].get("L_tokens")
            identities["master-gpu-mtp-on"] = {"report": str(mp), "report_sha256": source_hashes[str(mp)]["sha256"], "build_commit": master.get("master_build_commit"), "binary_sha256": master.get("build", {}).get("executable_sha256"), "protocol": "GPU Turbo4 MTP filled-context control"}
            add_raw_family(cells, "master-gpu-mtp-on", [(C, rec, raw.get("records_sha256"), {"label": row.get("label"), "L_tokens": L}, None, False, True)], identities["master-gpu-mtp-on"], "master 106", True, source_hashes)
            for append in row.get('fill_curve',[]):
                idx=append.get('append_index')
                if idx is None: continue
                timing=timing_from_sse(append.get('raw_path',''),append.get('raw_sha256')) if append.get('raw_path') and append.get('raw_sha256') else None
                if append.get('raw_path') and append.get('raw_sha256'):
                    source_hashes[str(append['raw_path'])]={"sha256":append['raw_sha256'],"declared_sha256":append['raw_sha256'],"hash_verified":timing is not None}
                backend_tokens=timing.get('prompt_n') if timing else None
                backend_seconds=(float(timing['prompt_ms'])/1000.0) if timing and finite_pos(timing.get('prompt_ms')) else None
                derived_backend=(float(backend_tokens)/backend_seconds) if finite_pos(backend_tokens) and finite_pos(backend_seconds) else None
                backend_rate=derived_backend if (timing is None or timing.get('prompt_per_second') is None or timing_close(derived_backend,timing.get('prompt_per_second'))) else None
                fresh=append.get('frontier_fresh_tokens')
                net_rate=(float(fresh)/backend_seconds) if finite_pos(fresh) and finite_pos(backend_seconds) and backend_rate is not None else None
                wall_rate=(float(fresh)/float(append['elapsed_seconds'])) if finite_pos(fresh) and finite_pos(append.get('elapsed_seconds')) else None
                PREFILL_GROUPS.append({"family":"master-gpu-mtp-on","nominal_target_C":int(C),"actual_C":append.get('cumulative_C_tokens'),"append_index":idx,"chunk_kind":"checkpoint_tail" if append.get('tail_adjustment') else "legacy_append","protocol":"MASTER106 GPU fresh-fill curve; legacy protocol","mtp":"on","backend_tok_s":backend_rate,"net_new_tok_s":net_rate,"http_wall_tok_s":wall_rate,"backend_tokens":backend_tokens,"net_new_tokens":fresh,"backend_prompt_seconds":backend_seconds,"http_wall_seconds":append.get('elapsed_seconds'),"source":{"report":str(mp),"report_sha256":source_hashes[str(mp)]['sha256'],"records_path":raw.get('records'),"records_sha256":raw.get('records_sha256'),"append":{k:append.get(k) for k in ('append_index','cumulative_C_tokens','frontier_fresh_tokens','evaluated_fresh_tokens','elapsed_seconds','fresh_prefill_tok_s','request_path','request_sha256','raw_path','raw_sha256')},"sse_prompt_timing":timing}})
    upstream_on = Path("/srv/ai/paged-kv/results/forward/benchmark-reruns-20261101/upstream-gpu-mtp-on")
    if upstream_on.exists():
        rows=[]
        for p in sorted(upstream_on.glob("C*/**/campaign/records.jsonl")):
            try:
                c=int(p.parents[3].name[1:])
                rows.append((c,str(p),None,{"campaign":str(p.parent),"protocol":"retained upstream GPU MTP-on control"}))
            except Exception: pass
        add_raw_family(cells,"master-gpu-mtp-on",rows,identities.get("master-gpu-mtp-on",{"protocol":"upstream GPU MTP-on"}),"upstream retained controls",True,source_hashes)

    # Fork GPU MTP-on retained controls: the 105 matrix Turbo4 GPU-only points plus
    # the valid November rerun campaigns; complete later cells replace earlier whole cells.
    matrix_path = Path("/srv/ai/paged-kv/results/forward/105-matrix/105-22/attempt-01/MATRIX105_FINAL.json")
    if matrix_path.exists():
        matrix = jread(matrix_path); source_hashes[str(matrix_path)] = {"sha256": sha(matrix_path)}
        identities["fork-gpu-mtp-on"] = {"report": str(matrix_path), "report_sha256": source_hashes[str(matrix_path)]["sha256"], "protocol": "fork native MTP with Turbo4 GPU target; legacy matrix controls"}
        for item in matrix.get("configurations", []):
            cfg = item.get("configuration", {})
            if cfg.get("pager") != "off" or cfg.get("aco") is not False or cfg.get("target_kv") != "turbo4":
                continue
            if not str(cfg.get("label", "")).startswith(("105-10", "105-13", "105-16")):
                if cfg.get('C') is None and cfg.get('pager')=='off' and cfg.get('aco') is False and cfg.get('target_kv')=='turbo4':
                    if item.get('path') and item.get('sha256'): source_hashes[item['path']]={"sha256":item['sha256'],"declared_sha256":item['sha256'],"hash_verified":False}
                    empty_context_controls.append({"family":"fork-gpu-mtp-on","cohort":"separate empty-prefix baseline","actual_C":0,"configured_L_tokens":cfg.get('L'),"label":cfg.get('label'),"medians":item.get('medians'),"report_path":item.get('path'),"report_sha256":item.get('sha256'),"matrix_summary_path":str(matrix_path),"matrix_summary_sha256":source_hashes[str(matrix_path)]['sha256'],"configuration":{k:cfg.get(k) for k in ('target_kv','draft','batch','ubatch','mtp','pager','aco')}})
                continue
            reportp = Path(item.get("path", ""))
            if not reportp.is_file(): continue
            rep = jread(reportp)
            for idx,append in enumerate(rep.get('prefill_curve',[])):
                key=(str(reportp),"prefill",idx)
                if key in _PREFILL_SEEN: continue
                rate=append.get('fresh_prefill_tok_s',append.get('fresh_prefill_tps'))
                fresh=append.get('fresh_prefill_tokens')
                timing=append.get('timings') if isinstance(append.get('timings'),dict) else {}
                ms=timing.get('prompt_ms',append.get('fresh_prefill_ms'))
                backend_tokens=timing.get('prompt_n')
                derived_backend=(float(backend_tokens)*1000.0/float(ms)) if finite_pos(backend_tokens) and finite_pos(ms) else None
                native_backend=timing.get('prompt_per_second')
                backend_rate=derived_backend if (derived_backend is not None and (native_backend is None or timing_close(derived_backend,native_backend))) else None
                net_rate=(float(fresh)*1000.0/float(ms)) if finite_pos(fresh) and finite_pos(ms) and backend_rate is not None else None
                wall_seconds=append.get('elapsed_s')
                wall_rate=(float(fresh)/float(wall_seconds)) if finite_pos(fresh) and finite_pos(wall_seconds) else None
                raw_refs={}
                for pathkey,hashkey in [('request_path','request_sha256'),('sse_path','sse_sha256')]:
                    ref=append.get(pathkey); declared=append.get(hashkey)
                    if not ref: continue
                    resolved=Path(ref) if Path(ref).is_absolute() else reportp.parent/ref
                    raw_refs[pathkey]={"path":str(resolved),"sha256":declared}
                    if declared: source_hashes[str(resolved)]={"sha256":declared,"declared_sha256":declared,"hash_verified":False}
                PREFILL_GROUPS.append({"family":"fork-gpu-mtp-on","nominal_target_C":cfg.get('C'),"actual_C":append.get('occupied_C',cfg.get('C')),"append_index":append.get('append_index',idx),"chunk_kind":"fresh_full_prompt","protocol":"MATRIX105 fresh-prefill request; isolated legacy cohort","mtp":"on","backend_tok_s":backend_rate,"net_new_tok_s":net_rate,"http_wall_tok_s":wall_rate,"backend_tokens":backend_tokens,"net_new_tokens":fresh,"backend_prompt_seconds":float(ms)/1000.0 if finite_pos(ms) else None,"http_wall_seconds":wall_seconds,"source":{"report":str(reportp),"report_sha256":sha(reportp),"append_index":idx,"raw_refs":raw_refs}})
                _PREFILL_SEEN.add(key)
            rec = rep.get("raw", {}).get("records") or rep.get("records_path")
            if rec and not Path(rec).is_absolute(): rec=str(reportp.parent / rec)
            if not rec: continue
            C = cfg.get("C")
            if C is None: continue
            add_raw_family(cells, "fork-gpu-mtp-on", [(C, rec, rep.get("raw", {}).get("records_sha256", item.get("records_sha256")), {"task": item.get("task"), "label": cfg.get("label"), "L_tokens": cfg.get("L")}, None, False, rep.get("status") == "complete" and item.get("status") == "complete")], identities["fork-gpu-mtp-on"], "fork 105 matrix", True, source_hashes)
    # Add retained November fork GPU-on records, with paths/hashes from each run itself.
    onroot = Path("/srv/ai/paged-kv/results/forward/benchmark-reruns-20261101/fork-gpu-mtp-on")
    if onroot.exists():
        rows = []
        for p in sorted(onroot.glob("C*/**/campaign/records.jsonl")):
            try:
                cp = p.parent / "run-config.json"; cfg = jread(cp) if cp.exists() else {}
                c = int(next(a.name[1:] for a in p.parents if a.name.startswith("C") and a.name[1:].isdigit()))
                rows.append((c, str(p), None, {"run_config": str(cp) if cp.exists() else None, "campaign": str(p.parent)}))
            except Exception: pass
        add_raw_family(cells, "fork-gpu-mtp-on", rows, identities.get("fork-gpu-mtp-on", {"protocol":"historical fork GPU MTP-on"}), "fork November rerun", True, source_hashes)

    # Current ACO rows are already certified 3-repeat measured cells in the report.
    aco_path = EVIDENCE / "ACO_FULL_CURVES_20261104.json"; aco = jread(aco_path)
    source_hashes[str(aco_path)] = {"sha256": sha(aco_path)}
    for run in aco.get("runs", []):
        if run.get("status") != "complete": continue
        mtp = run.get("mtp") == "on"
        family = f"current-aco-mtp-{'on' if mtp else 'off'}"
        geo = run.get("geometry", {}) if isinstance(run.get("geometry"), dict) else {}
        identity_geometry = {k: geo[k] for k in ("binary_sha256", "repo_context_commit", "source_commit", "template_sha256", "chunk_map_sha256", "L_tokens", "C_tokens", "H_tokens", "batch", "ubatch", "threads", "mtp", "placement", "target_kv", "draft", "quantization", "turbo_level", "hot_pages") if k in geo and isinstance(geo[k], (str, int, float, bool, type(None)))}
        dsos = geo.get("loaded_project_dsos")
        if isinstance(dsos, dict):
            identity_geometry["loaded_project_dsos"] = {k:v for k,v in dsos.items() if isinstance(v, (str, int, float, bool))}
        identities[family] = {"report": str(aco_path), "report_sha256": source_hashes[str(aco_path)]["sha256"], "build_label": aco.get("compiled_build_label"), "binary_sha256": geo.get("binary_sha256"), "repo_commit": geo.get("repo_context_commit"), "geometry": identity_geometry, "prefill_protocol": "persistent 32K-map fill with checkpoint tails; distinct from historical per-mode fresh-prefill controls"}
        for ck, x in run.get("cells", {}).items():
            if x.get("status") != "complete" or x.get("valid_measured_repeats") != 3: continue
            rs = x.get("decode_tok_s_repeats", [])
            if len(rs) != 3 or not all(finite_pos(v) for v in rs): continue
            completions=x.get('actual_completion_tokens',[])
            if len(completions)!=3 or not all(isinstance(v,(int,float)) and v>3 for v in completions): continue
            if mtp:
                ac = x.get("mtp_acceptance_percent_repeats", [])
                if len(ac) != 3 or not all(finite_nonnegative(v) for v in ac): continue
            prompt = x.get("prompt", "P1"); mode = x.get("reasoning_mode")
            pi = int(prompt[-1]) - 1
            raw_refs = x.get("raw_refs", [])
            sources = []
            for raw in raw_refs:
                sources.append({"path": raw.get("path"), "sha256": raw.get("sha256"), "request_path": raw.get("request_path"), "request_sha256": raw.get("request_sha256")})
            audits=[response_quality_for_record({'raw_path':r.get('path'),'raw_sha256':r.get('sha256')},source_hashes) for r in raw_refs]
            bad=[(i,q) for i,q in enumerate(audits,1) if q and q.get('classification')=='repeated_punctuation_only' and q.get('hash_verified')]
            unverified=[(i,q) for i,q in enumerate(audits,1) if q and q.get('classification')=='raw_hash_mismatch']
            unassessed=[(i,q) for i,q in enumerate(audits,1) if q and q.get('classification')=='raw_response_unavailable']
            if unassessed:
                RESPONSE_QUALITY_UNASSESSED.append({'family':family,'nominal_target_C':int(x.get('target_C',run.get('target_C',x['actual_C']))),'reasoning_mode':x.get('reasoning_mode'),'prompt':x.get('prompt'),'missing_raw_repetitions':[i for i,_ in unassessed],'source_report':str(aco_path)})
            if bad or unverified:
                RESPONSE_QUALITY_EXCLUSIONS.append({'family':family,'nominal_target_C':int(x.get('target_C',run.get('target_C',x['actual_C']))),'reasoning_mode':x.get('reasoning_mode'),'prompt':x.get('prompt'),'status':'whole_triplet_excluded_due_to_repeated_punctuation_only_output' if bad else 'whole_triplet_excluded_due_to_unverified_raw_response','bad_repetitions':[i for i,_ in bad],'unverified_repetitions':[i for i,_ in unverified],'responses':[{'repetition':i,'path':q.get('path'),'sha256':q.get('sha256'),'hash_verified':q.get('hash_verified'),'classification':q.get('classification'),'nonspace_chars':q.get('nonspace_chars'),'alphanumeric_chars':q.get('alphanumeric_chars'),'dominant_character_fraction':q.get('dominant_character_fraction')} for i,q in [*bad,*unverified]],'source_report':str(aco_path),'source_report_sha256':source_hashes[str(aco_path)]['sha256']})
                continue
            source = {"report": str(aco_path), "report_sha256": source_hashes[str(aco_path)]["sha256"], "cell_key": ck, "raw_refs": sources, "row_path": run.get("row_path"), "row_sha256": run.get("row_sha256"), "config_path": run.get("config_path"), "config_sha256": run.get("config_sha256")}
            nominal_c=int(x.get("target_C",run.get("target_C",x["actual_C"])))
            d = {"family":family,"variant":"current ACO","mtp":"on" if mtp else "off","placement":"GPU target; ACO","actual_C":int(x["actual_C"]),"nominal_target_C":nominal_c,"reasoning_mode":mode,"prompt":prompt,
                 "decode_tok_s":{"min":x["decode_tok_s_min"],"median":x["decode_tok_s_median"],"max":x["decode_tok_s_max"],"repeats":rs},
                 "mtp_acceptance_percent":({"min":x["mtp_acceptance_percent_min"],"median":x["mtp_acceptance_percent_median"],"max":x["mtp_acceptance_percent_max"],"repeats":x["mtp_acceptance_percent_repeats"]} if mtp else None),
                 "completion_tokens":x.get("actual_completion_tokens"),"source":source,"source_identity_ref":family}
            cells[(family,nominal_c,mode,pi)] = d

    identities["current_completion_plan"] = {"path":str(plan_path),"sha256":source_hashes[str(plan_path)]["sha256"],"revision":plan.get("revision"),"status":plan.get("status"),"output_root":plan.get("output_root")}
    # Include only raw cells from the current plan with complete measured triplets and observed C.
    r3rows = discover_current_completion(plan, source_hashes)
    for fam, rows in r3rows.items():
        if not rows: continue
        identities[fam] = {"plan":str(plan_path),"plan_sha256":source_hashes[str(plan_path)]["sha256"],"revision":plan.get("revision"),"protocol":"current fixed comparison protocol; only complete measured triplets retained"}
        add_raw_family(cells, fam, rows, identities[fam], "current completion", "mtp-on" in fam, source_hashes)
    discover_current_prefill(plan,source_hashes)

    # Construct a compact count table against plan-declared checkpoints, without asserting
    # completion while the queue is running or in assessment.
    required = {}
    for step in plan.get("steps", []):
        fam = step.get("family",step.get("id"))
        if fam not in FAMILIES[:8]: continue
        points = set()
        required_cells = set()
        for prof in step.get("profiles", []):
            for c in prof.get("targets", []):
                selection=prof.get("checkpoint_selection", {}).get(str(c), {})
                # Prefill-only targets intentionally carry empty decode selections.
                if not any(isinstance(pis,list) and pis for pis in selection.values()):
                    continue
                points.add(int(c))
                for mode,pis in selection.items():
                    for pi in pis: required_cells.add((int(c),mode,int(pi)))
        entry=required.setdefault(fam,{"checkpoints":[] ,"cells":[]})
        entry["checkpoints"]=sorted(set(entry["checkpoints"])|points)
        entry["cells"]=sorted(set(tuple(x) for x in entry["cells"])|required_cells)
    # Preserve original scientific decode requirements even when R6 executes only
    # prefill at those checkpoints or holds a previously observed fault.
    for req in plan.get("requested_decode_requirements",[]):
        if not isinstance(req,dict): continue
        fam=req.get("family")
        if fam not in FAMILIES[:8]: continue
        entry=required.setdefault(fam,{"checkpoints":[],"cells":[]})
        for prof in req.get("profiles",[]):
            for c,selection in prof.get("checkpoint_selection",{}).items():
                if not isinstance(selection,dict): continue
                has_required=False
                for mode,pis in selection.items():
                    if mode not in MODES or not isinstance(pis,list): continue
                    for pi in pis:
                        if pi in (0,1,2):
                            has_required=True
                            entry["cells"].append((int(c),mode,int(pi)))
                if has_required: entry["checkpoints"].append(int(c))
        entry["checkpoints"]=sorted(set(entry["checkpoints"]))
        entry["cells"]=sorted(set(tuple(x) for x in entry["cells"]))
    return cells, source_hashes, identities, aco, plan, required, empty_context_controls


def write_svg_png(pathbase, mode, subset_families, cells, metric="decode_tok_s", title=None):
    """Three prompt panels using measured medians and asymmetric min/max error bars."""
    try:
        plt = plotting_pyplot()
    except ImportError:
        if not ALLOW_PILLOW_DRAFT:
            raise RuntimeError("Matplotlib is required for final charts; use --allow-pillow-draft only for an interim preview")
    else:
        title = title or f"{mode} · {metric.replace('_',' ')} · min / median / max"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharey=True)
        palette=plt.get_cmap('tab10')
        family_index={family:i for i,family in enumerate(FAMILIES)}
        for pi, (ax,prompt) in enumerate(zip(axes,PROMPTS)):
            for fi,fam in enumerate(subset_families):
                pts=sorted([(d['nominal_target_C'],d) for (f,c,m,p),d in cells.items() if f==fam and m==mode and p==pi and d.get(metric)],key=lambda z:z[0])
                if not pts: continue
                xs=[c for c,d in pts]; ys=[d[metric]['median'] for c,d in pts]
                err=[[d[metric]['median']-d[metric]['min'] for c,d in pts],[d[metric]['max']-d[metric]['median'] for c,d in pts]]
                # Plot only observed points: connecting across sparse/missing
                # checkpoints can imply an unmeasured continuous curve.
                ax.errorbar(xs,ys,yerr=err,fmt='o',linestyle='none',markersize=3.5,capsize=2,linewidth=1.25,color=palette(family_index[fam]%10),label=fam)
            if any(f.startswith('current-aco') for f in subset_families):
                ax.axvline(65536,color='#666666',linestyle='--',linewidth=1.0,label='ACO H = 65,536 tokens' if pi==0 else None)
            ax.set_title(prompt); ax.set_xlabel('Requested context checkpoint C (tokens; actual counts in data)'); ax.grid(True,alpha=.25)
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[0].set_ylabel('Decode rate (tokens/s)' if metric=='decode_tok_s' else 'MTP acceptance (%)')
        fig.suptitle(title); handles,labels=axes[0].get_legend_handles_labels()
        # Gather labels from any prompt panel where a family has points.
        if not handles:
            for ax in axes:
                handles,labels=ax.get_legend_handles_labels()
                if handles: break
        if handles: fig.legend(handles,labels,loc='lower center',ncol=min(5,len(labels)),frameon=False,bbox_to_anchor=(.5,.01))
        fig.tight_layout(rect=(0,.14,1,.94))
        png=pathbase.with_suffix('.png'); svg=pathbase.with_suffix('.svg')
        fig.savefig(png,dpi=180,bbox_inches='tight'); fig.savefig(svg,bbox_inches='tight'); plt.close(fig)
        return str(svg),str(png)
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1700, 1120
    im = Image.new("RGB", (W,H), "white"); dr=ImageDraw.Draw(im)
    font=ImageFont.load_default()
    title = title or (f"{mode} · {metric.replace('_',' ')} · min / median / max" )
    dr.text((45,25),title,fill="#152238",font=font)
    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    plot_top=70; plot_h=970; panel_w=525; gap=32; left=45
    maxy=0.0
    selected=[]
    for fam in subset_families:
        for (f,c,m,pi),d in cells.items():
            if f==fam and m==mode and metric in d and d[metric]:
                selected.append((fam,d['nominal_target_C'],pi,d)); maxy=max(maxy,float(d[metric]["max"]))
    maxy=max(10, math.ceil(maxy/10)*10)
    for pi,prompt in enumerate(PROMPTS):
        x0=left+pi*(panel_w+gap); x1=x0+panel_w-35; y0=plot_top+plot_h; y1=plot_top+25
        dr.text((x0,y1-18),prompt,fill="#222",font=font)
        dr.line((x0,y0,x1,y0),fill="#555",width=2); dr.line((x0,y0,x0,y1),fill="#555",width=2)
        for t in range(6):
            yy=y0-(y0-y1)*t/5; val=maxy*t/5
            dr.line((x0,yy,x1,yy),fill="#e4e8ed",width=1); dr.text((x0-32,yy-6),f"{val:.0f}",fill="#555",font=font)
        points=[(fam,c,d) for fam,c,p,d in selected if p==pi]
        cs=sorted({c for _,c,_ in points})
        if len(cs)>1: xmin,xmax=min(cs),max(cs)
        else: xmin,xmax=(cs[0]-1,cs[0]+1) if cs else (0,1)
        for fi,fam in enumerate(subset_families):
            fp=sorted([(c,d) for ff,c,p,d in selected if ff==fam and p==pi],key=lambda z:z[0])
            for c,d in fp:
                x=x0+(c-xmin)/(xmax-xmin)*(x1-x0)
                mid=d[metric]["median"]; lo=d[metric]["min"]; hi=d[metric]["max"]
                y=y0-mid/maxy*(y0-y1); yl=y0-hi/maxy*(y0-y1); yh=y0-lo/maxy*(y0-y1)
                color=colors[FAMILIES.index(fam)%len(colors)]
                dr.line((x,yl,x,yh),fill=color,width=2); dr.ellipse((x-4,y-4,x+4,y+4),fill=color)
        dr.text((x0,y0+7),f"C checkpoints: {min(cs) if cs else 'none'} … {max(cs) if cs else 'none'} (actual observed points only)",fill="#555",font=font)
    # Legend in two columns
    lx=45; ly=1070
    for i,f in enumerate(subset_families):
        x=lx+(i%5)*320; y=ly+(i//5)*18
        dr.line((x,y+5,x+18,y+5),fill=colors[FAMILIES.index(f)%len(colors)],width=3); dr.text((x+23,y),f,fill="#222",font=font)
    png=pathbase.with_suffix(".png"); svg=pathbase.with_suffix(".svg")
    im.save(png)
    # SVG mirrors the PNG via embedded pixels only when Pillow supplied it. Also provide a
    # scalable SVG rendition of a metadata/axis chart for publication-friendly viewers.
    rows=[]
    rows.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="45" y="42" font-family="sans-serif" font-size="20" fill="#152238">{title}</text>')
    for pi,prompt in enumerate(PROMPTS):
        x0=left+pi*(panel_w+gap); x1=x0+panel_w-35; y0=plot_top+plot_h; y1=plot_top+25
        rows.append(f'<text x="{x0}" y="{y1-18}" font-family="sans-serif" font-size="15">{prompt}</text><path d="M{x0},{y0}H{x1} M{x0},{y0}V{y1}" stroke="#555" fill="none"/>')
        for t in range(6):
            yy=y0-(y0-y1)*t/5; val=maxy*t/5
            rows.append(f'<path d="M{x0},{yy:.1f}H{x1}" stroke="#e4e8ed"/><text x="{x0-35}" y="{yy+4:.1f}" font-size="12" fill="#555">{val:.0f}</text>')
        cs=sorted({c for f,c,p,d in selected if p==pi})
        xmin,xmax=(min(cs),max(cs)) if len(cs)>1 else ((cs[0]-1,cs[0]+1) if cs else (0,1))
        for fi,fam in enumerate(subset_families):
            fp=sorted([(c,d) for ff,c,p,d in selected if ff==fam and p==pi],key=lambda z:z[0]); color=colors[FAMILIES.index(fam)%len(colors)]
            for c,d in fp:
                x=x0+(c-xmin)/(xmax-xmin)*(x1-x0); mid=d[metric]['median']; lo=d[metric]['min']; hi=d[metric]['max']; y=y0-mid/maxy*(y0-y1); yl=y0-hi/maxy*(y0-y1); yh=y0-lo/maxy*(y0-y1)
                rows.append(f'<path d="M{x:.1f},{yl:.1f}V{yh:.1f}" stroke="{color}" stroke-width="2"/><circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    for i,f in enumerate(subset_families):
        x=45+(i%5)*320; y=1080+(i//5)*18; color=colors[i%len(colors)]
        rows.append(f'<path d="M{x},{y-4}h18" stroke="{color}" stroke-width="3"/><text x="{x+23}" y="{y}" font-size="12">{f}</text>')
    rows.append('</svg>'); svg.write_text('\n'.join(rows))
    return str(svg), str(png)


def write_prefill_chart(out, rows, metric, label):
    """Plot mapped regular fills, mapped tails, and historical appends separately."""
    try:
        plt = plotting_pyplot()
    except ImportError:
        raise RuntimeError("Matplotlib is required for final prefill charts")
    data=[r for r in rows if finite_pos(r.get(metric)) and r.get('actual_C') is not None]
    if not data: return []
    palette=plt.get_cmap('tab10')
    family_index={family:i for i,family in enumerate(FAMILIES)}
    markers={'checkpoint_tail':'s','mapped_bulk':'o','legacy_append':'^','fresh_full_prompt':'v'}
    marker_labels={'checkpoint_tail':'checkpoint tail','mapped_bulk':'regular mapped bulk','legacy_append':'historical append','fresh_full_prompt':'historical full prompt'}
    cohorts=(
        ('mapped-bulk',lambda r:bool(r.get('chunk_map_sha256')) and r.get('chunk_kind')!='checkpoint_tail',
         'Mapped chunk-map fills · regular bulk groups'),
        ('mapped-tails',lambda r:bool(r.get('chunk_map_sha256')) and r.get('chunk_kind')=='checkpoint_tail',
         'Mapped chunk-map fills · checkpoint tails'),
        ('historical-variable',lambda r:not r.get('chunk_map_sha256'),
         'Unmapped / historical variable append protocols'),
    )
    outputs=[]
    for cohort,include,title in cohorts:
        subset=[r for r in data if include(r)]
        if not subset: continue
        fig,ax=plt.subplots(figsize=(13,7.2))
        # Plot independent observed groups only; never draw through absent points.
        families=sorted({r['family'] for r in subset},key=lambda f:family_index.get(f,99))
        def marker_key(r):
            if r.get('chunk_kind')=='checkpoint_tail': return 'checkpoint_tail'
            if r.get('chunk_map_sha256'): return 'mapped_bulk'
            return r.get('chunk_kind','legacy_append')
        kinds=sorted({marker_key(r) for r in subset})
        for family in families:
            color=palette(family_index.get(family,0)%10)
            for kind in kinds:
                pts=[r for r in subset if r['family']==family and marker_key(r)==kind]
                if not pts: continue
                ax.scatter([r['actual_C'] for r in pts],[r[metric] for r in pts],color=color,marker=markers.get(kind,'^'),s=22,alpha=.72,zorder=2)
        ax.set_xlabel('Actual occupied context C (tokens)'); ax.set_ylabel(label); ax.grid(True,alpha=.25)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.axvline(65536,color='#555555',linestyle='--',linewidth=1.2,label='ACO H = 65,536 tokens')
        ax.set_title(f'{title}\n{label}; observed append groups only; protocol details are in the data table')
        from matplotlib.lines import Line2D
        family_handles=[Line2D([0],[0],color=palette(family_index[f]%10),marker='o',linewidth=1.2,markersize=4,label=f) for f in families]
        marker_handles=[Line2D([0],[0],color='#333333',marker=markers.get(k,'^'),linestyle='None',markersize=5,label=marker_labels.get(k,k)) for k in kinds]
        if marker_handles:
            ax.legend(handles=marker_handles,loc='upper right',frameon=False,title='Append kind',fontsize='small')
        fig.legend(handles=family_handles,loc='lower center',ncol=min(5,len(family_handles)),frameon=False,bbox_to_anchor=(.5,.01),fontsize='small',title='Family (colors match decode plots)')
        fig.tight_layout(rect=(0,.12,1,.94))
        base=out/f'prefill-{metric}-{cohort}'
        fig.savefig(base.with_suffix('.png'),dpi=180,bbox_inches='tight')
        fig.savefig(base.with_suffix('.svg'),bbox_inches='tight')
        plt.close(fig); outputs.append(str(base.with_suffix('.svg')))
    return outputs


def source_reference(d):
    src=d.get('source',{})
    if src.get('raw_refs'):
        return '; '.join(f"{x.get('path')}@{x.get('sha256')}" for x in src['raw_refs'])
    path=src.get('path') or src.get('report')
    digest=src.get('sha256') or src.get('report_sha256')
    return f"{path}@{digest}" if path else "source unavailable"


def prefill_series_id(r):
    src=r.get('source',{}) if isinstance(r.get('source'),dict) else {}
    sched=src.get('schedule_path')
    if sched:
        parts=Path(sched).parts
        for i,part in enumerate(parts):
            if part.startswith('attempt-') and 'cell' in parts[i+1:]: return str(Path(*parts[:i+1]))
        return str(Path(sched))
    path=src.get('path')
    if path and r.get('family','').startswith('current-aco'):
        parts=Path(path).parts
        if 'cell' in parts: return str(Path(*parts[:parts.index('cell')]))
    if src.get('report'): return src['report']
    return f"{r.get('family')}|{r.get('protocol','unknown')}|unknown-source"


def resource_series_id(path, family):
    p=Path(path) if path else None
    if p is None: return f"{family}|source cohort"
    parts=p.parts
    for i,part in enumerate(parts):
        if (part.startswith('attempt-') or part.startswith('session-')) and 'cell' in parts[i+1:]:
            return str(Path(*parts[:i+1]))
    if 'cell' in parts:
        return str(Path(*parts[:parts.index('cell')]))
    if 'campaign' in parts:
        return str(Path(*parts[:parts.index('campaign')+1]))
    return str(p.parent)


def _resource_scope(metric):
    key=metric.lower().replace('-','_').replace('.','_')
    if 'vmhwm' in key: return 'server_process_VmHWM'
    if 'vmrss' in key: return 'server_process_VmRSS'
    if 'vmswap' in key or ('swap' in key and 'host' not in key): return 'server_process_VmSwap'
    if 'memavailable' in key or 'host_memory_available' in key: return 'whole_host_MemAvailable'
    if ('gpu' in key or 'cuda' in key or 'vram' in key) and ('process' in key or 'vram' in key or 'allocated' in key or 'reserved' in key): return 'process_VRAM'
    if 'gpu' in key and any(x in key for x in ('used','free','total')): return 'GPU_global_used_free'
    if 'memory' in key or 'mem' in key: return 'other_memory_measure'
    return None


def extract_resource_measurements(obj, prefix=''):
    """Extract only scalar values from explicitly named resource sections."""
    found=[]
    resource_names={'resource_peaks','captured_resources','resources','resource_summary','memory_peaks','memory_observations','captured_resource_peaks'}
    def visit(value,path,inside=False):
        if isinstance(value,dict):
            for k,v in value.items():
                next_path=f'{path}.{k}' if path else str(k)
                next_inside=inside or k.lower() in resource_names
                if next_inside:
                    visit(v,next_path,next_inside)
        elif isinstance(value,list):
            for i,v in enumerate(value): visit(v,f'{path}[{i}]',inside)
        elif inside and isinstance(value,(int,float)) and math.isfinite(float(value)):
            scope=_resource_scope(path)
            if scope:
                unit='bytes' if any(t in path.lower() for t in ('_bytes','_byte','rss','hwm','swap','memavailable')) else ('MiB' if 'mib' in path.lower() or '_mb' in path.lower() else 'reported')
                found.append({'metric':path,'scope':scope,'value':value,'unit':unit})
    visit(obj,prefix,False)
    return found


def retained_resource_receipt(path, source_hashes):
    p=Path(path) if path else None
    if p is None or not p.exists(): return {'source_files':[],'L':None,'H':None,'actual_C':None,'measurements':[]}
    files=[p] if p.is_file() else sorted([*p.glob('**/row.json'),*p.glob('**/cell.json')])
    lvals=[]; hvals=[]; cvals=[]; measurements=[]; refs=[]
    for q in files:
        try: data=jread(q)
        except Exception: continue
        digest=sha(q); source_hashes[str(q)]={'sha256':digest}; refs.append({'path':str(q),'sha256':digest})
        geo=data.get('geometry',{}) if isinstance(data.get('geometry',{}),dict) else {}
        base=data.get('base',{}) if isinstance(data.get('base',{}),dict) else {}
        lv=geo.get('L_tokens',data.get('L_tokens',data.get('context_tokens')))
        hv=geo.get('H_tokens',data.get('H_tokens'))
        cv=geo.get('actual_C',data.get('actual_C',data.get('actual_occupancy_tokens',base.get('occupied_tokens'))))
        if lv is not None: lvals.append(lv)
        if hv is not None: hvals.append(hv)
        if cv is not None: cvals.append(cv)
        measurements.extend(extract_resource_measurements(data))
    return {'source_files':refs,'L':max(lvals) if lvals else None,'H':max(hvals) if hvals else None,'actual_C':max(cvals) if cvals else None,'measurements':measurements}


def summarize_resource_measurements(measurements):
    grouped=defaultdict(lambda:defaultdict(list))
    for m in measurements:
        grouped[m['scope']][(m['metric'],m['unit'])].append(m['value'])
    return [{'scope':scope,'measurements':[{'metric':metric,'unit':unit,'peak':max(values),'samples':len(values)} for (metric,unit),values in sorted(metrics.items())]} for scope,metrics in sorted(grouped.items())]


def collect_memory_and_failure_rows(celllist, plan, aco_report, source_hashes):
    """Reuse retained resource/failure metadata; this function launches no telemetry."""
    grouped={}
    resource_file_cache={}
    for d in celllist:
        src=d.get('source',{}) if isinstance(d.get('source'),dict) else {}
        record_path=src.get('path')
        series=resource_series_id(record_path,d['family'])
        key=(d['family'],series)
        row=grouped.setdefault(key,{'family':d['family'],'series_id':series,'configured_L_tokens':None,'H_reference_tokens':65536 if d['family'].startswith('current-aco') else None,'actual_final_C':None,'series_status':'valid complete cells retained','measurement_scopes':defaultdict(dict),'source_files':set(),'resource_source_files':set()})
        if d.get('actual_C') is not None:
            row['actual_final_C']=max(row['actual_final_C'] or 0,int(d['actual_C']))
        if record_path and src.get('sha256'): row['source_files'].add((record_path,src['sha256']))
        candidates=[]
        if record_path:
            rp=Path(record_path)
            for parent in list(rp.parents)[:7]:
                for name in ('cell.json','row.json','resources.json','resource-peaks.json'):
                    q=parent/name
                    if q.is_file(): candidates.append(q)
        for q in candidates:
            cached=resource_file_cache.get(str(q))
            if cached is None:
                try: data=jread(q)
                except Exception: continue
                digest=sha(q)
                geo=data.get('geometry',{}) if isinstance(data.get('geometry',{}),dict) else {}
                cached={'sha256':digest,'L':geo.get('L_tokens',data.get('L_tokens',data.get('context_tokens'))),'H':geo.get('H_tokens',data.get('H_tokens')),'measurements':extract_resource_measurements(data)}
                resource_file_cache[str(q)]=cached
            digest=cached['sha256']; source_hashes[str(q)]={'sha256':digest}; row['source_files'].add((str(q),digest))
            if row['configured_L_tokens'] is None:
                row['configured_L_tokens']=cached.get('L')
            if row['H_reference_tokens'] is None:
                row['H_reference_tokens']=cached.get('H')
            for metric in cached['measurements']:
                scope=row['measurement_scopes'][metric['scope']]
                values=scope.setdefault((metric['metric'],metric['unit']),[])
                values.append(metric['value'])
                row['resource_source_files'].add((str(q),digest))
        if record_path:
            # Read adjacent run configuration only for L/H geometry; never copy argv/env.
            rp=Path(record_path)
            for parent in list(rp.parents)[:6]:
                q=parent/'run-config.json'
                if not q.is_file(): continue
                try: cfg=jread(q)
                except Exception: break
                digest=sha(q); source_hashes[str(q)]={'sha256':digest}; row['source_files'].add((str(q),digest))
                base=cfg.get('base',{}) if isinstance(cfg.get('base',{}),dict) else {}
                geo=cfg.get('geometry',{}) if isinstance(cfg.get('geometry',{}),dict) else {}
                if row['configured_L_tokens'] is None:
                    row['configured_L_tokens']=geo.get('L_tokens',base.get('context_tokens',cfg.get('context_tokens')))
                break
    # ACO run summaries can hold memory peaks even when a neighboring cell receipt does not.
    for run in aco_report.get('runs',[]):
        family=f"current-aco-mtp-{'on' if run.get('mtp')=='on' else 'off'}"
        path=run.get('row_path') or str(run.get('source_root') or '')
        series=resource_series_id(path,family)
        key=(family,series)
        geo=run.get('geometry',{}) if isinstance(run.get('geometry',{}),dict) else {}
        row=grouped.setdefault(key,{'family':family,'series_id':series,'configured_L_tokens':geo.get('L_tokens'),'H_reference_tokens':geo.get('H_tokens',65536),'actual_final_C':run.get('actual_final_C'),'series_status':run.get('status','ACO source report'),'measurement_scopes':defaultdict(dict),'source_files':set(),'resource_source_files':set()})
        if row.get('configured_L_tokens') is None: row['configured_L_tokens']=geo.get('L_tokens')
        if row.get('H_reference_tokens') is None: row['H_reference_tokens']=geo.get('H_tokens',65536)
        if row.get('actual_final_C') is None and run.get('actual_final_C') is not None: row['actual_final_C']=run['actual_final_C']
        if run.get('row_path'):
            rp=run['row_path']; sh=run.get('row_sha256') or source_hashes.get(rp,{}).get('sha256')
            if sh: row['source_files'].add((rp,sh))
        run_metrics=extract_resource_measurements(run)
        cellrows=run.get('cells',{})
        if isinstance(cellrows,dict):
            for cellrow in cellrows.values():
                run_metrics.extend(extract_resource_measurements(cellrow))
        for metric in run_metrics:
            vals=row['measurement_scopes'][metric['scope']].setdefault((metric['metric'],metric['unit']),[]); vals.append(metric['value'])
        if run_metrics and run.get('row_path'):
            sh=run.get('row_sha256') or source_hashes.get(run['row_path'],{}).get('sha256')
            if sh: row['resource_source_files'].add((run['row_path'],sh))
        elif run_metrics:
            for report_path,meta in source_hashes.items():
                if Path(report_path).name=='ACO_FULL_CURVES_20261104.json':
                    row['resource_source_files'].add((report_path,meta.get('sha256')))
    result=[]
    for row in grouped.values():
        measurements=[]
        for scope,metrics in row['measurement_scopes'].items():
            for (metric,unit),samples in metrics.items():
                measurements.extend({'scope':scope,'metric':metric,'unit':unit,'value':v} for v in samples)
        scopes=summarize_resource_measurements(measurements)
        row['measurement_scopes']=scopes or 'No resource telemetry found in retained source files'
        sources=row['resource_source_files'] or set(sorted(row['source_files'])[:2])
        row['source_files']=[{'path':p,'sha256':h} for p,h in sorted(sources)]
        row.pop('resource_source_files',None)
        result.append(row)
    # Recorded plan limits/failures are explicit outcomes, not missing measurements.
    known=plan.get('known_capacity_result_reused')
    if isinstance(known,dict):
        path=known.get('receipt_path'); sh=sha(path) if path and Path(path).is_file() else None
        if path and sh: source_hashes[path]={'sha256':sh}
        receipt=retained_resource_receipt(path,source_hashes)
        result.append({'family':known.get('family'),'series_id':'known capacity result','configured_L_tokens':known.get('allocation_tokens',receipt['L']),'H_reference_tokens':receipt['H'],'actual_final_C':receipt['actual_C'],'series_status':f"known {known.get('status')} (not a missing mandatory decode cell)",'measurement_scopes':summarize_resource_measurements(receipt['measurements']) or 'No resource telemetry found in retained failure receipt','source_files':receipt['source_files']})
    limit=plan.get('master_gpu_131072_runtime_limit')
    if isinstance(limit,dict):
        path=limit.get('path'); sh=sha(path) if path and Path(path).is_file() else None
        if path and sh: source_hashes[path]={'sha256':sh}
        receipt=retained_resource_receipt(path,source_hashes)
        result.append({'family':'master-gpu-mtp-off','series_id':'known runtime-limit result','configured_L_tokens':131072,'H_reference_tokens':receipt['H'],'actual_final_C':receipt['actual_C'],'series_status':f"known {limit.get('status')}; not classified as allocation OOM",'measurement_scopes':summarize_resource_measurements(receipt['measurements']) or 'No resource telemetry found in retained failure receipt','source_files':receipt['source_files']})
    for item in plan.get('quarantined_series',[]):
        if not isinstance(item,dict): continue
        receipt=retained_resource_receipt(item.get('path'),source_hashes)
        result.append({'family':item.get('family'),'series_id':item.get('path'),'configured_L_tokens':receipt['L'],'H_reference_tokens':receipt['H'],'actual_final_C':receipt['actual_C'],'series_status':'quarantined; excluded from primary comparison','measurement_scopes':summarize_resource_measurements(receipt['measurements']) or item.get('reason'),'source_files':receipt['source_files']})
    return result


def prefill_cohort_summaries(rows, plan, aco_report):
    groups=defaultdict(list)
    for r in rows: groups[(r['family'],prefill_series_id(r),r.get('protocol','unknown'))].append(r)
    plan_targets={}
    for step in plan.get('steps',[]):
        targets=set()
        for profile in step.get('profiles',[]): targets.update(int(c) for c in profile.get('targets',[]))
        plan_targets[step.get('id')]=targets
    result=[]
    for (family,series,protocol),xs in sorted(groups.items()):
        def validpair(r,t,s): return finite_pos(r.get(t)) and finite_pos(r.get(s))
        b=[r for r in xs if validpair(r,'backend_tokens','backend_prompt_seconds')]
        n=[r for r in xs if validpair(r,'net_new_tokens','backend_prompt_seconds')]
        w=[r for r in xs if validpair(r,'net_new_tokens','http_wall_seconds')]
        bt=sum(float(r['backend_tokens']) for r in b); bs=sum(float(r['backend_prompt_seconds']) for r in b)
        nt=sum(float(r['net_new_tokens']) for r in n); ns=sum(float(r['backend_prompt_seconds']) for r in n)
        wt=sum(float(r['net_new_tokens']) for r in w); ws=sum(float(r['http_wall_seconds']) for r in w)
        observed=sorted({int(r['nominal_target_C']) for r in xs if r.get('nominal_target_C') is not None})
        actuals=[int(r['actual_C']) for r in xs if r.get('actual_C') is not None]
        maps=sorted({r.get('chunk_map_sha256') for r in xs if r.get('chunk_map_sha256')})
        execution_ids={r.get('execution_step_id') or r.get('source',{}).get('execution_step_id') for r in xs}
        execution_ids.discard(None)
        retained_specs={r.get('retained_series_path') for r in xs if r.get('retained_series_path')}
        if family.startswith('current-aco'):
            status='complete ACO fill curve' if aco_report.get('status')=='complete' else 'partial ACO fill curve'
        elif retained_specs:
            partial=any(r.get('retained_series_allow_partial') for r in xs)
            if partial:
                status='retained partial series; source-level completion not asserted'
            elif family=='master-cpu-mtp-off' and any('/r5/' in str(p) for p in retained_specs):
                status='R5 operator-held series (intentional exit 143); not a runtime failure; historical coverage retained'
            else:
                status='retained series approved by plan; historical completeness kept separate from active step'
        elif execution_ids:
            execution_id=next(iter(execution_ids)) if len(execution_ids)==1 else None
            expected=plan_targets.get(execution_id,set())
            status='complete planned fill series' if expected and expected.issubset(set(observed)) else 'partial planned fill series'
        else: status='complete source fill record(s); historical cohort only'
        result.append({"family":family,"series_id":series,"protocol":protocol,"status":status,"append_groups":len(xs),"observed_nominal_targets_C":observed,"final_actual_C":max(actuals) if actuals else None,"total_backend_tokens":bt if b else None,"backend_seconds":bs if b else None,"aggregate_backend_tok_s":bt/bs if b and bs else None,"backend_eligible_groups":len(b),"total_net_new_tokens":nt if n else None,"net_new_backend_seconds":ns if n else None,"aggregate_net_new_tok_s":nt/ns if n and ns else None,"net_new_eligible_groups":len(n),"http_wall_seconds":ws if w else None,"aggregate_http_wall_tok_s":wt/ws if w and ws else None,"http_wall_eligible_groups":len(w),"chunk_map_sha256":maps if maps else None,"source_files":sorted({(r.get('source',{}).get('schedule_path') or r.get('source',{}).get('report') or r.get('source',{}).get('path') or '') for r in xs})})
    return result


def prefill_checkpoint_summaries(rows):
    """Interval-weight each complete fill checkpoint over every recorded append."""
    groups=defaultdict(list)
    for r in rows:
        if not r.get('chunk_map_sha256'): continue
        if r.get('nominal_target_C') is None: continue
        groups[(r['family'],prefill_series_id(r),r.get('protocol','unknown'),r.get('chunk_map_sha256'),int(r['nominal_target_C']))].append(r)
    result=[]
    for (family,series,protocol,chunk_map,target),xs in sorted(groups.items(),key=lambda x:(x[0][0],x[0][1],x[0][4])):
        def validpair(r,t,s): return finite_pos(r.get(t)) and finite_pos(r.get(s))
        b=[r for r in xs if validpair(r,'backend_tokens','backend_prompt_seconds')]
        n=[r for r in xs if validpair(r,'net_new_tokens','backend_prompt_seconds')]
        w=[r for r in xs if validpair(r,'net_new_tokens','http_wall_seconds')]
        bt=sum(float(r['backend_tokens']) for r in b); bs=sum(float(r['backend_prompt_seconds']) for r in b)
        nt=sum(float(r['net_new_tokens']) for r in n); ns=sum(float(r['backend_prompt_seconds']) for r in n)
        wt=sum(float(r['net_new_tokens']) for r in w); ws=sum(float(r['http_wall_seconds']) for r in w)
        starts=[int(r['actual_C'])-int(r['net_new_tokens']) for r in xs if r.get('actual_C') is not None and finite_nonnegative(r.get('net_new_tokens'))]
        sources=sorted({r.get('source',{}).get('schedule_path') or r.get('source',{}).get('report') or r.get('source',{}).get('path') or '' for r in xs})
        result.append({"family":family,"series_id":series,"protocol":protocol,"nominal_target_C":target,"actual_C_start":min(starts) if starts else None,"actual_C":max(int(r['actual_C']) for r in xs if r.get('actual_C') is not None) if any(r.get('actual_C') is not None for r in xs) else None,"chunk_map_sha256":chunk_map,"cohort":"mapped chunk-map","append_groups":len(xs),"backend_tokens":bt if b else None,"backend_seconds":bs if b else None,"backend_tok_s":bt/bs if b and bs else None,"backend_eligible_groups":len(b),"net_new_tokens":nt if n else None,"net_new_backend_seconds":ns if n else None,"net_new_tok_s":nt/ns if n and ns else None,"net_new_eligible_groups":len(n),"http_wall_seconds":ws if w else None,"http_wall_tok_s":wt/ws if w and ws else None,"http_wall_eligible_groups":len(w),"source_files":sources})
    return result


def write_prefill_checkpoint_chart(out, rows, metric, label):
    """Primary map-matched ingest comparison, weighted over all chunks to each C."""
    plt = plotting_pyplot()
    from matplotlib.lines import Line2D
    data=[r for r in rows if r.get('chunk_map_sha256') and finite_pos(r.get(metric)) and r.get('nominal_target_C') is not None]
    if not data: return None
    palette=plt.get_cmap('tab10'); family_index={family:i for i,family in enumerate(FAMILIES)}
    fig,ax=plt.subplots(figsize=(12,7))
    family_index_map={family:i for i,family in enumerate(FAMILIES)}
    for family in sorted({r['family'] for r in data},key=lambda f:family_index_map.get(f,99)):
        pts=[r for r in data if r['family']==family]
        color=palette(family_index_map.get(family,0)%10)
        ax.scatter([r['nominal_target_C'] for r in pts],[r[metric] for r in pts],marker='o',s=28,color=color,label=family)
    ax.axvline(65536,color='#555555',linestyle='--',linewidth=1.2)
    ax.set_xlabel('Requested context checkpoint C (tokens; actual counts in data)'); ax.set_ylabel(label); ax.grid(True,alpha=.25)
    ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    ax.set_title(f'Mapped-fill interval rate by checkpoint · {label}\nNew input since the previous checkpoint; observed points only; tails included')
    from matplotlib.lines import Line2D
    families=sorted({r['family'] for r in data},key=lambda f:family_index_map.get(f,99))
    handles=[Line2D([0],[0],color=palette(family_index_map[f]%10),marker='o',linestyle='None',markersize=4,label=f) for f in families]
    handles.append(Line2D([0],[0],color='#555555',linestyle='--',label='ACO H = 65,536 tokens'))
    ax.legend(handles=handles,frameon=False,ncol=4,fontsize='small',loc='upper center',bbox_to_anchor=(0.5,-0.15))
    fig.tight_layout()
    base=out/f'prefill-{metric}-mapped-checkpoints'
    fig.savefig(base.with_suffix('.png'),dpi=180,bbox_inches='tight'); fig.savefig(base.with_suffix('.svg'),bbox_inches='tight'); plt.close(fig)
    return str(base.with_suffix('.svg'))


def matched_decode_comparisons(celllist):
    idx={(d['family'],d['nominal_target_C'],d['reasoning_mode'],d['prompt']):d for d in celllist}
    result=[]
    for aco in ('current-aco-mtp-off','current-aco-mtp-on'):
        mtp=aco.rsplit('-',1)[-1]
        controls=[f for f in FAMILIES[:8] if f.endswith(f'mtp-{mtp}')]
        acells=[d for d in celllist if d['family']==aco]
        for a in acells:
            for control in controls:
                c=idx.get((control,a['nominal_target_C'],a['reasoning_mode'],a['prompt']))
                if c is None: continue
                am=a['decode_tok_s']['median']; cm=c['decode_tok_s']['median']
                result.append({"mtp":mtp,"nominal_target_C":a['nominal_target_C'],"aco_family":aco,"aco_actual_C":a['actual_C'],"control_family":control,"control_actual_C":c['actual_C'],"mode":a['reasoning_mode'],"prompt":a['prompt'],"aco_median_tok_s":am,"control_median_tok_s":cm,"percent_delta_aco_over_control":100*(am/cm-1),"aco_source":a['source'],"control_source":c['source']})
    return result


def write_tables_appendix(path, celllist, prefill, cohorts, checkpoint_ingest, memory_rows, aco_report, matched):
    lines=["# Comparison result tables — 2026-11-04","","This appendix lists each validated cell as min / median / max across three measured repeats. Nominal target C is the requested checkpoint; actual C is the observed occupied context. Decode and MTP acceptance are separate tables. MTP-off acceptance is explicitly N/A.","","## Decode and MTP acceptance by variant","","Source paths and SHA-256 values are retained in the corresponding JSON/CSV rows; source references below point to the measurement report or raw record set.",""]
    for family in FAMILIES:
        rows=sorted((d for d in celllist if d['family']==family),key=lambda d:(d['nominal_target_C'],MODES.index(d['reasoning_mode']),PROMPTS.index(d['prompt'])))
        lines += [f"### {family}","","#### Decode rate (tokens/s)","","| Nominal target C | Actual C | Reasoning | Prompt | Min / median / max | Source |","|---:|---:|---|---|---:|---|"]
        for d in rows:
            v=d['decode_tok_s']; src=source_reference(d)
            lines.append(f"| {d['nominal_target_C']} | {d['actual_C']} | {d['reasoning_mode']} | {d['prompt']} | {v['min']:.2f} / {v['median']:.2f} / {v['max']:.2f} | `{src}` |")
        lines += ["","#### MTP acceptance (%)","","| Nominal target C | Actual C | Reasoning | Prompt | Min / median / max | Source |","|---:|---:|---|---|---:|---|"]
        for d in rows:
            v=d.get('mtp_acceptance_percent'); val="N/A (MTP off)" if d['mtp'] == 'off' else (f"{v['min']:.2f} / {v['median']:.2f} / {v['max']:.2f}" if v else "N/A (not measured)")
            lines.append(f"| {d['nominal_target_C']} | {d['actual_C']} | {d['reasoning_mode']} | {d['prompt']} | {val} | `{source_reference(d)}` |")
        lines.append("")
    lines += ["## Exact nominal-C ACO/control comparisons","","Decode median delta is `(ACO / control - 1) × 100` for the same nominal C, MTP state, reasoning mode, and prompt. Actual C and source refs are shown for both cells. No interpolation is used; differing targets such as 61440 and 62464 are not paired.","","| MTP | Nominal C | Mode | Prompt | ACO family (actual C) | ACO median | Control family (actual C) | Control median | Delta % | ACO source | Control source |","|---|---:|---|---|---|---:|---|---:|---:|---|---|"]
    if matched:
        for x in matched:
            lines.append(f"| {x['mtp']} | {x['nominal_target_C']} | {x['mode']} | {x['prompt']} | {x['aco_family']} ({x['aco_actual_C']}) | {x['aco_median_tok_s']:.2f} | {x['control_family']} ({x['control_actual_C']}) | {x['control_median_tok_s']:.2f} | {x['percent_delta_aco_over_control']:+.2f} | `{source_reference({'source':x['aco_source']})}` | `{source_reference({'source':x['control_source']})}` |")
    else: lines.append("| N/A | N/A | N/A | N/A | No exact nominal-C/mode/prompt matches are available | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines += ["","## Mapped-fill interval rates by checkpoint","","Each checkpoint rate sums valid append chunks carrying new input since the previous checkpoint, including a final checkpoint tail when present. The actual-C interval start is estimated as the minimum `(actual C after append − net-new tokens)` across available groups; the end is the largest observed actual C. Rates divide the interval's summed tokens by summed matching seconds. They are not cumulative-from-zero averages or the rate of the final small tail. Historical/unmapped rates remain available in append-level diagnostics and weighted cohort tables.","","| Family | Fill series | Nominal target C | Actual C interval (start → end) | Append groups | Backend tok/s | Net-new tok/s | HTTP-wall tok/s | Chunk map | Sources |","|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in checkpoint_ingest:
        fmt=lambda x:f"{float(x):.2f}" if finite_pos(x) else "N/A"
        lines.append(f"| {r['family']} | `{r['series_id']}` | {r['nominal_target_C']} | {r.get('actual_C_start','N/A')} → {r.get('actual_C','N/A')} | {r['append_groups']} | {fmt(r.get('backend_tok_s'))} | {fmt(r.get('net_new_tok_s'))} | {fmt(r.get('http_wall_tok_s'))} | `{r.get('chunk_map_sha256')}` | `{'; '.join(r.get('source_files',[]))}` |")
    lines += ["","## Shared context-chunking / bulk-ingest append diagnostics","","These rows are context-fill append measurements, not question-prefill measurements. Current ACO uses one persistent-map fill shared across prompts and reasoning modes; historical fill cohorts remain source/protocol-labeled, including per-mode schedules where applicable. Backend, net-new, and HTTP-wall rates are separate; no rate is repeated as a per-prompt value. N/A means that the corresponding timer or token count is not recorded/validated.","","| Family | Protocol cohort | Nominal target C | Actual append C | Append | Chunk | Backend tok/s | Net-new tok/s | HTTP-wall tok/s | Source |","|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for r in sorted(prefill,key=lambda x:(x['family'],str(x.get('protocol')),x.get('nominal_target_C') or -1,x.get('actual_C') or -1,x.get('append_index') or -1)):
        fmt=lambda x:f"{float(x):.2f}" if finite_pos(x) else "N/A"
        src=r.get('source',{}); paths=[]
        if isinstance(src,dict):
            if src.get('schedule_path'): paths.append(f"{src['schedule_path']}@{src.get('schedule_sha256')}")
            if src.get('raw_path'): paths.append(f"{src['raw_path']}@{src.get('raw_sha256')}")
            if src.get('report'): paths.append(f"{src['report']}@{src.get('report_sha256')}")
            if src.get('identity_path'): paths.append(f"{src['identity_path']}@{src.get('identity_sha256')}")
            if src.get('path'): paths.append(f"{src['path']}@{src.get('sha256')}")
            if src.get('request_path'): paths.append(f"{src['request_path']}@{src.get('request_sha256')}")
            for ref in src.get('raw_refs',{}).values() if isinstance(src.get('raw_refs'),dict) else []:
                if isinstance(ref,dict) and ref.get('path'): paths.append(f"{ref['path']}@{ref.get('sha256')}")
            if isinstance(src.get('append'),dict) and src['append'].get('raw_path'):
                paths.append(f"{src['append']['raw_path']}@{src['append'].get('raw_sha256')}")
            if src.get('path'): paths.append(f"{src['path']}@{src.get('sha256')}")
            for ref in src.get('raw_refs',{}).values() if isinstance(src.get('raw_refs'),dict) else []:
                if isinstance(ref,dict) and ref.get('path'): paths.append(f"{ref['path']}@{ref.get('sha256')}")
            if isinstance(src.get('append'),dict):
                app=src['append']
                if app.get('raw_path'): paths.append(f"{app['raw_path']}@{app.get('raw_sha256')}")
        elif src: paths.append(json.dumps(src,sort_keys=True))
        lines.append(f"| {r['family']} | {r.get('protocol','unknown')} | {r.get('nominal_target_C','N/A')} | {r.get('actual_C','N/A')} | {r.get('append_index','N/A')} | {r.get('chunk_kind','N/A')} | {fmt(r.get('backend_tok_s'))} | {fmt(r.get('net_new_tok_s'))} | {fmt(r.get('http_wall_tok_s'))} | `{'; '.join(paths) or 'source unavailable'}` |")
    lines += ["","### Weighted throughput by independent fill series/cohort","","Rates use summed token counts divided by summed matching durations; append rates are not averaged. Each independent historical per-mode/per-context fill remains its own cohort. Current-plan series are marked partial until their declared context targets have completed fill records.","","| Family | Series / source-root ID | Protocol | Status | Append groups | Nominal targets C | Final actual C | Total net-new tokens | Chunk-map SHA-256 | Backend tok/s | Net-new tok/s | HTTP-wall tok/s |","|---|---|---|---|---:|---|---:|---:|---|---:|---:|---:|"]
    for c in cohorts:
        fmt=lambda x:f"{float(x):.2f}" if finite_pos(x) else "N/A"
        lines.append(f"| {c['family']} | `{c['series_id']}` | {c['protocol']} | {c['status']} | {c['append_groups']} | {', '.join(map(str,c['observed_nominal_targets_C'])) or 'N/A'} | {c.get('final_actual_C','N/A')} | {c.get('total_net_new_tokens','N/A')} | `{', '.join(c.get('chunk_map_sha256') or []) or 'N/A'}` | {fmt(c.get('aggregate_backend_tok_s'))} | {fmt(c.get('aggregate_net_new_tok_s'))} | {fmt(c.get('aggregate_http_wall_tok_s'))} |")
    lines += ["","## Retained memory observations and known resource outcomes","","This table reuses resource peaks/captures already present in cell receipts, row metadata, and the ACO report. It does not infer memory at each C when the source is only cohort-level. GPU-global used/free values are separate from process VRAM; server VmHWM/VmRSS and VmSwap are separate from whole-host MemAvailable. VmSwap is reported as observed and does not establish which allocations were swapped or page-in I/O. Known startup failures and quarantined target-only MTP restore cohorts are explicit statuses; they are not presented as general master-default failures.","","| Family / cohort | Configured L | H reference | Actual final C | Series status | Retained memory/resource metrics | Source path @ SHA-256 |","|---|---:|---:|---:|---|---|---|"]
    for r in memory_rows:
        scopes=r.get('measurement_scopes')
        if isinstance(scopes,list):
            metrics='; '.join(f"{s['scope']}: "+', '.join(f"{m['metric']} peak {m['peak']} {m['unit']} (n={m['samples']})" for m in s['measurements']) for s in scopes)
        else: metrics=str(scopes or 'No resource telemetry found')
        sources='; '.join(f"{x.get('path')}@{x.get('sha256') or 'SHA unavailable'}" for x in r.get('source_files',[]))
        lines.append(f"| {r.get('family','N/A')} | {r.get('configured_L_tokens','N/A')} | {r.get('H_reference_tokens','N/A')} | {r.get('actual_final_C','N/A')} | {r.get('series_status','N/A')} | {metrics} | `{sources}` |")
    weighted=aco_report.get('prefill_weighted_by_mtp',{})
    lines += ["","### Current ACO weighted ingest summary","","One persistent-map context fill per MTP variant is shared across all prompts and reasoning modes. Rates are token totals divided by their corresponding summed durations.","","| MTP | Append groups | Backend tokens | Net-new tokens | Backend seconds | HTTP-wall seconds | Backend tok/s | Net-new tok/s | HTTP-wall tok/s |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mtp in ('on','off'):
        w=weighted.get(mtp,{})
        lines.append(f"| {mtp} | {w.get('append_groups','N/A')} | {w.get('backend_tokens','N/A')} | {w.get('net_new_tokens','N/A')} | {w.get('backend_prompt_seconds_weighted','N/A')} | {w.get('http_wall_seconds_weighted','N/A')} | {w.get('backend_tok_s_token_weighted','N/A')} | {w.get('net_new_tok_s_backend_time_weighted','N/A')} | {w.get('http_wall_tok_s_token_weighted','N/A')} |")
    path.write_text('\n'.join(lines)+'\n')


def main():
    global ALLOW_PILLOW_DRAFT, DATA_ONLY
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=DEFAULT_OUT); ap.add_argument('--plan',type=Path,help='Use a specific frozen/active comparison plan'); ap.add_argument('--allow-pillow-draft',action='store_true',help='Allow interim Pillow charts when Matplotlib is unavailable; not for final report output'); ap.add_argument('--data-only',action='store_true',help='Write JSON/CSV/Markdown only; skip all chart renderers'); args=ap.parse_args()
    ALLOW_PILLOW_DRAFT=args.allow_pillow_draft
    DATA_ONLY=args.data_only
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    cells, hashes, ids, aco, plan, required, empty_context_controls = collect(out,args.plan)
    celllist=sorted(cells.values(),key=lambda x:(x['family'],x['reasoning_mode'],x['actual_C'],x['prompt']))
    # Spread diagnostics are flags for review only; they do not discard a cell.
    flags=[]
    for d in celllist:
        v=d['decode_tok_s']['repeats']; spread=(max(v)-min(v))/statistics.median(v)
        if spread>0.30: flags.append({"type":"repeat_spread_over_30_percent","family":d['family'],"C":d['actual_C'],"mode":d['reasoning_mode'],"prompt":d['prompt'],"relative_range":spread})
    # Neighbor discontinuity flags on actual adjacent observed C points.
    groups=defaultdict(list)
    for d in celllist: groups[(d['family'],d['reasoning_mode'],d['prompt'])].append(d)
    for k,arr in groups.items():
        arr.sort(key=lambda x:x['actual_C'])
        for a,b in zip(arr,arr[1:]):
            if b['actual_C']>a['actual_C'] and abs(b['decode_tok_s']['median']/a['decode_tok_s']['median']-1)>.30:
                flags.append({"type":"adjacent_observed_checkpoint_change_over_30_percent","family":k[0],"mode":k[1],"prompt":k[2],"from_C":a['actual_C'],"to_C":b['actual_C'],"relative_change":b['decode_tok_s']['median']/a['decode_tok_s']['median']-1})
    counts={}
    capacity_excluded=[]
    capacity_keys=defaultdict(set)
    for outcome in plan.get('capacity_n_a_outcomes',[]):
        if not isinstance(outcome,dict) or outcome.get('family') not in FAMILIES[:8] or outcome.get('nominal_C') is None: continue
        fam=outcome['family']; c=int(outcome['nominal_C'])
        for mode in MODES:
            for pi in range(3):
                key=(c,mode,pi); capacity_keys[fam].add(key)
                capacity_excluded.append({'family':fam,'nominal_target_C':c,'reasoning_mode':mode,'prompt':f'P{pi+1}','status':outcome.get('status'),'source_path':outcome.get('source_path')})
    for f in FAMILIES:
        got=sorted({d['actual_C'] for d in celllist if d['family']==f})
        nominal=sorted({d['nominal_target_C'] for d in celllist if d['family']==f})
        want=required.get(f,{"checkpoints":[],"cells":[]})
        observed={(d['nominal_target_C'],d['reasoning_mode'],int(d['prompt'][-1])-1) for d in celllist if d['family']==f}
        needed={tuple(x) for x in want["cells"]}
        missing_cells=[{"nominal_target_C":c,"mode":m,"prompt":f"P{pi+1}"} for c,m,pi in sorted(needed-observed)]
        feasible_missing=needed-observed-capacity_keys.get(f,set())
        quality_missing={(x['nominal_target_C'],x['reasoning_mode'],int(x['prompt'][-1])-1) for x in RESPONSE_QUALITY_EXCLUSIONS if x['family']==f} & (needed-observed)
        unassessed_quality_count=sum(x['family']==f for x in RESPONSE_QUALITY_UNASSESSED)
        if needed:
            if not missing_cells: coverage_status="complete declared coverage"
            else: coverage_status=f"incomplete: {len(needed & observed)}/{len(needed)} usable; {len(quality_missing)} quality-rejected; {len(needed & capacity_keys.get(f,set()))} capacity-N/A; {len(feasible_missing)} feasible gaps"
        elif f.startswith('current-aco-'):
            run=next((r for r in aco.get('runs',[]) if r.get('mtp')==f.rsplit('-',1)[-1]),None)
            coverage_status="complete source curve" if run and run.get('status')=='complete' and sum(d['family']==f for d in celllist)==132 else "incomplete source curve"
        else:
            coverage_status="no plan-declared required cells"
        counts[f]={"observed_actual_C":got,"observed_nominal_target_C":nominal,"complete_cells":sum(d['family']==f for d in celllist),"complete_repeats":3*sum(d['family']==f for d in celllist),"declared_required_checkpoints":want["checkpoints"],"missing_required_checkpoints":sorted(set(want["checkpoints"])-set(nominal)),"declared_required_cell_count":len(needed),"available_declared_required_cell_count":len(needed & observed),"missing_declared_required_cells":missing_cells,"missing_feasible_cells":[{"nominal_target_C":c,"mode":m,"prompt":f"P{pi+1}"} for c,m,pi in sorted(feasible_missing)],"quality_rejected_required_cells":[{"nominal_target_C":c,"mode":m,"prompt":f"P{pi+1}"} for c,m,pi in sorted(quality_missing)],"capacity_excluded_required_cells":[{"nominal_target_C":c,"mode":m,"prompt":f"P{pi+1}"} for c,m,pi in sorted(needed & capacity_keys.get(f,set()))],"response_quality_unassessed_source_triplet_count":unassessed_quality_count,"coverage_status":coverage_status}
    # CSV contains sanitized per-cell triplet summaries only.
    csvpath=out/'comparison-cells-20261104.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['family','variant','placement','mtp','nominal_target_C','actual_C','reasoning_mode','prompt','decode_min','decode_median','decode_max','decode_repeats','acceptance_min','acceptance_median','acceptance_max','source_path','source_sha256']); w.writeheader()
        for d in celllist:
            src=d['source']; path=src.get('path') or src.get('report'); sh=src.get('sha256') or src.get('report_sha256')
            ac=d.get('mtp_acceptance_percent') or {}
            w.writerow({'family':d['family'],'variant':d['variant'],'placement':d['placement'],'mtp':d['mtp'],'nominal_target_C':d['nominal_target_C'],'actual_C':d['actual_C'],'reasoning_mode':d['reasoning_mode'],'prompt':d['prompt'],'decode_min':d['decode_tok_s']['min'],'decode_median':d['decode_tok_s']['median'],'decode_max':d['decode_tok_s']['max'],'decode_repeats':json.dumps(d['decode_tok_s']['repeats']),'acceptance_min':ac.get('min'),'acceptance_median':ac.get('median'),'acceptance_max':ac.get('max'),'source_path':path,'source_sha256':sh})
    report={"schema":"comparison-final-digest-v1","generated":"2026-11-04","data_only_export":DATA_ONLY,"charts_rendered":not DATA_ONLY,"queue_status":None,"queue_progress_path":str(Path(plan.get('output_root',''))/'queue-progress.json'),"queue_progress":None,"families":FAMILIES,"source_identities":ids,"source_identity_artifacts":_IDENTITY_ARTIFACTS,"source_files":hashes,"coverage":counts,"cell_count":len(celllist),"measured_repeat_count":3*len(celllist),"cells":celllist,"empty_context_controls_separate_cohort":empty_context_controls,"diagnostic_flags_review_only":flags,"limitations":["Only observed actual C values are plotted; no interpolation or manufactured grid points.","At a matching family/nominal target C/mode/prompt, later valid complete evidence replaces the whole earlier cell; repetitions are never cherry-picked.","MTP-on acceptance is shown only when measured for all three valid repetitions; MTP-off is N/A.","Historical fresh-prefill and current persistent 32K-map prefill are separate protocols and are not pooled.","Rate summaries do not establish causality or statistical significance.","A nonzero VmSwap observation exists for current ACO MTP-on; memory residency and page-in I/O are not certified."]}
    report['independent_outlier_rechecks']={"declared":plan.get('outlier_rechecks'),"complete_cells":sorted(OUTLIER_SERIES_CELLS,key=lambda d:(d.get('independent_series_id',''),d['reasoning_mode'],d['prompt'])),"replacement_policy":"Independent series remain separate robustness evidence; they do not overwrite the main-family cell."}
    report['capacity_excluded_cells']=capacity_excluded
    report['reused_decode_outcomes']=plan.get('reused_decode_outcomes',[])
    report["excluded_series"]=[{k:s.get(k) for k in ("family","path","reason")} for s in plan.get("quarantined_series",[]) if isinstance(s,dict)]
    report['response_quality_audit']={
        'scope':'Hash-verified saved SSE responses available from imported benchmark cells; full answer text is never exported.',
        'classifier':'Flag a whole measured triplet if any repetition has at least 64 nonspace output characters, no alphanumeric characters across visible content plus reasoning, and one punctuation character accounts for at least 80% of the output.',
        'excluded_source_triplets':RESPONSE_QUALITY_EXCLUSIONS,
        'unassessed_source_triplets':RESPONSE_QUALITY_UNASSESSED,
        'limits':['This narrow artifact check catches repeated-punctuation failures; it is not a semantic-accuracy or general answer-quality validator.','Short EOS answers and reasoning-only answers with alphanumeric text are preserved.','Missing raw SSE is marked unassessed, not treated as pathological.']}
    quality_csv=out/'comparison-response-quality-failures-20261104.csv'
    with quality_csv.open('w',newline='') as f:
        fields=['family','nominal_target_C','reasoning_mode','prompt','status','bad_repetitions','unverified_repetitions','records_path','records_sha256','response_sources']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in RESPONSE_QUALITY_EXCLUSIONS:
            w.writerow({'family':x.get('family'),'nominal_target_C':x.get('nominal_target_C'),'reasoning_mode':x.get('reasoning_mode'),'prompt':x.get('prompt'),'status':x.get('status'),'bad_repetitions':json.dumps(x.get('bad_repetitions',[])),'unverified_repetitions':json.dumps(x.get('unverified_repetitions',[])),'records_path':x.get('records_path') or x.get('source_report'),'records_sha256':x.get('records_sha256') or x.get('source_report_sha256'),'response_sources':json.dumps(x.get('responses',[]),sort_keys=True)})
    execution_findings=[]
    for finding in plan.get('runtime_findings',[]):
        if not isinstance(finding,dict): continue
        item={k:finding.get(k) for k in ('family','status','reason','complete_cells_imported','source_path') if k in finding}
        source_path=finding.get('source_path')
        if source_path and Path(source_path).is_file():
            item['source_sha256']=sha(source_path)
            hashes[source_path]={'sha256':item['source_sha256']}
        execution_findings.append(item)
    capacity_receipt=plan.get('known_capacity_result_reused')
    if isinstance(capacity_receipt,dict):
        receipt=capacity_receipt.get('receipt') or {}
        startup=capacity_receipt.get('startup_log') or {}
        for artifact in (receipt,startup):
            path=artifact.get('path'); declared=artifact.get('sha256')
            if path and Path(path).is_file(): hashes[path]={'sha256':sha(path),'declared_sha256':declared,'hash_verified':sha(path)==declared}
        execution_findings.append({'family':capacity_receipt.get('family'),'status':capacity_receipt.get('status'),'reason':capacity_receipt.get('note'),'source_path':receipt.get('path'),'source_sha256':hashes.get(receipt.get('path'),{}).get('sha256'),'startup_log_path':startup.get('path'),'startup_log_sha256':hashes.get(startup.get('path'),{}).get('sha256')})
    # Preserve an operator assessment only when an approved retained series has
    # a source-side assessment artifact. Do not infer the disposition from an
    # exit code or bake a cell count into the report generator.
    for spec in plan.get('retained_completed_series',[]):
        if not isinstance(spec,dict): continue
        series_path=Path(spec.get('path',''))
        assessment=series_path/'operator-assessment.json'
        if not assessment.is_file(): continue
        try: disposition=jread(assessment)
        except Exception: continue
        classification=disposition.get('classification')
        if classification!='operator_stopped_after_valid_measurements': continue
        digest=sha(assessment); hashes[str(assessment)]={'sha256':digest}
        item={'family':spec.get('family'),'series_id':spec.get('series_id'),'status':classification,'series_status_preserved':disposition.get('series_status_preserved'),'reason':disposition.get('note'),'source_path':str(assessment),'source_sha256':digest}
        execution_findings.append(item)
    for step in plan.get('steps',[]):
        if not isinstance(step,dict) or not step.get('outlier_series_id'): continue
        for profile in step.get('profiles',[]):
            profile_root=profile.get('profile_run_root')
            if not profile_root: continue
            assessment=Path(profile_root)/'operator-assessment.json'
            if not assessment.is_file(): continue
            try: disposition=jread(assessment)
            except Exception: continue
            digest=sha(assessment); hashes[str(assessment)]={'sha256':digest}
            completed=disposition.get('completed_valid_requests',[])
            fault=disposition.get('fault') or {}
            execution_findings.append({'family':step.get('family'),'series_id':step.get('outlier_series_id'),'status':disposition.get('classification'),'reason':disposition.get('note'),'valid_measured_repetitions':sum(r.get('kind')=='measured' and r.get('status')=='pass' for r in completed if isinstance(r,dict)),'runtime_fault_status':fault.get('status'),'runtime_fault_error':fault.get('error'),'source_path':str(assessment),'source_sha256':digest})
    report['execution_findings']=execution_findings
    report['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    qp=Path(plan.get('output_root',''))/'queue-progress.json'
    if qp.exists():
        try:
            q=jread(qp); report['queue_status']=q.get('status')
            steps=q.get('steps',[])
            report['queue_progress']={k:q.get(k) for k in ('status','next_step','updated_at','plan_sha256') if k in q}
            report['queue_progress']['step_count']=len(steps) if isinstance(steps,list) else None
            if isinstance(steps,list):
                report['queue_progress']['steps']=[{k:s.get(k) for k in ('id','status','attempt_count','completed_count') if k in s} for s in steps if isinstance(s,dict)]
        except Exception: pass
    if not DATA_ONLY:
        for name in ('gpu-decode','cpu-decode','all-decode','mtp-acceptance'):
            fams=FAMILIES if name=='all-decode' else ([x for x in FAMILIES if 'gpu' in x or x.startswith('current-aco')] if name=='gpu-decode' else ([x for x in FAMILIES if 'cpu' in x or x.startswith('current-aco')] if name=='cpu-decode' else [x for x in FAMILIES if x.endswith('mtp-on')]))
            metrics=['decode_tok_s'] if name!='mtp-acceptance' else ['mtp_acceptance_percent']
            for mode in MODES:
                for metric in metrics:
                    chart_fams=[x for x in fams if any(d['family']==x and d['reasoning_mode']==mode and d.get(metric) for d in celllist)]
                    if chart_fams:
                        write_svg_png(out/f'{name}-{mode}-{metric}',mode,chart_fams,cells,metric=metric,title=f'{name}: reasoning {mode} — {metric.replace("_"," ")} (min/median/max)')
    # Prefill data is a separate current-protocol export; append classes are preserved.
    prefill=list(PREFILL_GROUPS)
    for run in aco.get('runs',[]):
        row_path=run.get('row_path'); rowmeta={}
        if row_path and Path(row_path).is_file():
            try:
                rowmeta=jread(row_path); source_hashes[row_path]={"sha256":run.get('row_sha256') or sha(row_path),"declared_sha256":run.get('row_sha256'),"hash_verified":False}
            except Exception: rowmeta={}
        map_identity=resolve_verified_chunk_map(row_path,(plan.get('chunk_map') or {}).get('sha256'),hashes)
        cmap={"sha256":map_identity['sha256']} if map_identity else {}
        for x in run.get('prefill_appends',[]):
            backend_sec=x.get('backend_prompt_seconds'); wall_sec=x.get('elapsed_wall_seconds',x.get('wall_seconds'))
            bcalc=float(x['backend_tokens'])/float(backend_sec) if finite_pos(x.get('backend_tokens')) and finite_pos(backend_sec) else None
            ncalc=float(x['net_new_tokens'])/float(backend_sec) if finite_pos(x.get('net_new_tokens')) and finite_pos(backend_sec) else None
            wcalc=float(x['net_new_tokens'])/float(wall_sec) if finite_pos(x.get('net_new_tokens')) and finite_pos(wall_sec) else None
            br=x.get('backend_tok_s') if bcalc is not None and timing_close(bcalc,x.get('backend_tok_s')) else None
            nr=x.get('net_new_tok_s') if ncalc is not None and timing_close(ncalc,x.get('net_new_tok_s')) else None
            wr=x.get('http_wall_tok_s') if wcalc is not None and timing_close(wcalc,x.get('http_wall_tok_s')) else None
            source={"path":x.get('raw_ref',{}).get('path'),"sha256":x.get('raw_ref',{}).get('sha256'),"request_path":x.get('raw_ref',{}).get('request_path'),"request_sha256":x.get('raw_ref',{}).get('request_sha256'),"row_path":row_path,"row_sha256":run.get('row_sha256')}
            if map_identity:
                source['chunk_map_identity_path']=map_identity['identity_path']; source['chunk_map_path']=map_identity['map_path']
            prefill.append({"family":f"current-aco-mtp-{run.get('mtp')}","nominal_target_C":int(run.get('target_C',x.get('actual_C',0))),"actual_C":x.get('actual_C'),"prior_actual_C":x.get('prior_actual_C'),"append_index":x.get('append_index'),"chunk_kind":x.get('chunk_kind'),"protocol":"persistent_32k_map_with_checkpoint_tails","mtp":run.get('mtp'),"backend_tokens":x.get('backend_tokens'),"net_new_tokens":x.get('net_new_tokens'),"backend_prompt_seconds":backend_sec,"http_wall_seconds":wall_sec,"backend_tok_s":br,"net_new_tok_s":nr,"http_wall_tok_s":wr,"chunk_map_sha256":cmap.get('sha256'),"source":source})
    prefill_coverage={}
    for family in FAMILIES:
        group=[r for r in prefill if r['family']==family]
        prefill_coverage[family]={"append_groups":len(group),"protocols":sorted({r.get('protocol','unknown') for r in group}),"backend_rate_groups":sum(finite_pos(r.get('backend_tok_s')) for r in group),"net_new_rate_groups":sum(finite_pos(r.get('net_new_tok_s')) for r in group),"http_wall_rate_groups":sum(finite_pos(r.get('http_wall_tok_s')) for r in group),"status":"available" if group else "no eligible append schedule/rate source located"}
    # Source path/hash pairs are indexed once in source_files. Keep append rows
    # compact by dropping repeated hash strings and repeated per-row config refs.
    for group in prefill:
        src=group.get('source')
        if not isinstance(src,dict): continue
        src.pop('identity_path',None); src.pop('identity_sha256',None)
        for key in list(src):
            if key.endswith('_sha256') and key != 'chunk_map_sha256': src.pop(key,None)
        nested=src.get('append')
        if isinstance(nested,dict):
            for key in list(nested):
                if key.endswith('_sha256'): nested.pop(key,None)
    report['prefill_comparison']={"definition":"backend=backend prompt tokens/backend prompt seconds; net-new=net-new tokens/backend prompt seconds; HTTP wall=net-new tokens/HTTP wall seconds","protocol_policy":"Historical fresh-prefill/replay, current ACO persistent-map, and current-plan append groups remain separately labeled; rates are not pooled across protocols. Source paths resolve to SHA-256 entries in source_files.","family_coverage":prefill_coverage,"rows":len(prefill),"append_groups":prefill}
    bulk_cohorts=prefill_cohort_summaries(prefill,plan,aco)
    checkpoint_ingest=prefill_checkpoint_summaries(prefill)
    memory_rows=collect_memory_and_failure_rows(celllist,plan,aco,hashes)
    checkpoint_csv=out/'comparison-prefill-mapped-checkpoints-20261104.csv'
    with checkpoint_csv.open('w',newline='') as f:
        fields=['family','series_id','protocol','nominal_target_C','actual_C_start','actual_C','append_groups','backend_tokens','backend_seconds','backend_tok_s','net_new_tokens','net_new_backend_seconds','net_new_tok_s','http_wall_seconds','http_wall_tok_s','chunk_map_sha256','source_files']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in checkpoint_ingest: w.writerow({k:(json.dumps(r.get(k),sort_keys=True) if k=='source_files' else r.get(k)) for k in fields})
    report['bulk_ingest_cohorts']={"aggregation":"sum tokens / sum matching durations, grouped by independent fill series/cohort","cohorts":bulk_cohorts}
    report['mapped_prefill_interval_by_checkpoint']={"aggregation":"within each family/source series/requested target, sum all append tokens / sum matching durations; includes tails; mapped chunk-map groups only","rows":checkpoint_ingest}
    report['memory_and_failure_summary']={"scope":"existing retained cell/row/ACO resource observations and declared plan outcomes only; no new telemetry","rows":memory_rows,"scope_notes":["GPU-global used/free and process VRAM are distinct scopes.","Server VmHWM/VmRSS/VmSwap and whole-host MemAvailable are distinct scopes.","VmSwap does not identify swapped allocations or page-in I/O.","A series-level resource peak is not assigned to an individual context checkpoint."]}
    prefill_csv=out/'comparison-prefill-append-groups-20261104.csv'
    with prefill_csv.open('w',newline='') as f:
        fields=['family','protocol','mtp','nominal_target_C','actual_C','append_index','chunk_kind','backend_tokens','net_new_tokens','backend_prompt_seconds','http_wall_seconds','backend_tok_s','net_new_tok_s','http_wall_tok_s','source']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in prefill: w.writerow({k:(json.dumps(r.get(k),sort_keys=True) if k=='source' else r.get(k)) for k in fields})
    if not DATA_ONLY:
        for metric,label in [("backend_tok_s","backend prompt tokens / backend prompt seconds"),("net_new_tok_s","net-new tokens / backend prompt seconds"),("http_wall_tok_s","net-new tokens / HTTP wall seconds")]:
            write_prefill_checkpoint_chart(out,checkpoint_ingest,metric,label)
            write_prefill_chart(out,prefill,metric,label)
    audit_path=Path('/srv/ai/paged-kv/results/comparison-outlier-audit-20261104.json')
    if audit_path.exists():
        audit=jread(audit_path); report['outlier_audit']={"path":str(audit_path),"sha256":sha(audit_path),"conclusion":audit.get('conclusion'),"current_aco":audit.get('current_aco'),"issues":[{k:i.get(k) for k in ('family','actual_context_tokens','reasoning','prompt_index','repeat_spread_over_median','cause_assessment','recheck')} for i in audit.get('issues',[])]}; hashes[str(audit_path)]={"sha256":report['outlier_audit']['sha256']}
    matched=matched_decode_comparisons(celllist)
    report['matched_nominal_context_decode_comparisons']={"formula":"100 * (ACO median / control median - 1)","matching":"same nominal target C, MTP state, reasoning mode, and prompt; no interpolation","rows":matched,"matched_pair_count":len(matched)}
    report['tables_appendix_path']=str(out/'comparison-tables-20261104.md')
    write_tables_appendix(out/'comparison-tables-20261104.md',celllist,prefill,bulk_cohorts,checkpoint_ingest,memory_rows,aco,matched)
    (out/'comparison-final-20261104.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=["# Comparison report digest — 2026-11-04","",f"Validated complete cells: **{len(celllist)}**; measured repeats: **{3*len(celllist)}**.",f"Queue status at generation: **{report['queue_status'] or 'not available'}**. This is not a completeness claim while the completion queue is running or awaiting assessment.","","Each primary plotted point groups the original requested context checkpoint, with min/median/max of three valid measured HTTP 200 complete-stream repetitions. New normal-conversation runs retain earlier questions and answers, so actual occupancy may grow; per-repetition actual counts and ranges remain in the data. Append-level prefill diagnostics still use actual occupancy. Nominal requested C and observed actual C are both retained; coverage matching and primary plots use nominal C; actual counts are retained separately, and append diagnostics use actual C. No interpolation or missing checkpoint substitution is used. If multiple sources cover the same family/nominal-C/mode/prompt, the complete cell from the later source replaces the prior cell as a whole.","","## Family coverage","","A complete streaming HTTP response is necessary but does not certify semantic quality. Hash-verified responses matching the narrow repeated-punctuation rule are excluded as whole triplets; short EOS and reasoning-only responses remain eligible. Missing raw response files are marked unassessed. Capacity N/A outcomes are listed separately from measured cells and runnable gaps.","","| Family | Status | Complete cells | Nominal target C | Observed actual C | Missing required nominal C |","|---|---|---:|---|---|---|"]
    for f in FAMILIES:
        c=counts[f]; md.append(f"| {f} | {c['coverage_status']} | {c['complete_cells']} | {', '.join(map(str,c['observed_nominal_target_C'])) or '—'} | {', '.join(map(str,c['observed_actual_C'])) or '—'} | {', '.join(map(str,c['missing_required_checkpoints'])) or '—'} |")
    if RESPONSE_QUALITY_EXCLUSIONS:
        md += ["", "## Response-quality exclusions", "", "Whole triplets with hash-verified repeated-punctuation-only output were removed from the normal comparison cells. The JSON/CSV inventory lists source paths, hashes, repetitions, and text-shape counts without copying response text.", "", "| Family | Nominal C | Mode | Prompt | Bad reps | Status |", "|---|---:|---|---|---|---|"]
        for x in RESPONSE_QUALITY_EXCLUSIONS:
            md.append(f"| {x.get('family')} | {x.get('nominal_target_C')} | {x.get('reasoning_mode')} | {x.get('prompt')} | {', '.join(map(str,x.get('bad_repetitions',[]))) or '—'} | {x.get('status')} |")
    if RESPONSE_QUALITY_UNASSESSED:
        md += ["", f"Raw response audit unavailable for **{len(RESPONSE_QUALITY_UNASSESSED)} source triplets**; those cells are marked unassessed and are not excluded by this audit."]
    if capacity_excluded:
        md += ["", "## Capacity-excluded points", "", "These are known allocation/capacity limits, not completed benchmark cells and not currently scheduled runnable gaps.", "", "| Family | Nominal C | Modes / prompts | Status | Source |", "|---|---:|---|---|---|"]
        for outcome in plan.get('capacity_n_a_outcomes',[]):
            md.append(f"| {outcome.get('family')} | {outcome.get('nominal_C')} | all 4 modes × 3 prompts | {outcome.get('status')} | `{outcome.get('source_path')}` |")
    if execution_findings:
        md += ["", "## Runtime and operator dispositions", "", "These dispositions explain incomplete families; they are not measurement cells. Only complete validated triplets appear in the decode tables.", "", "| Family / series | Status | Finding | Source |", "|---|---|---|---|"]
        for finding in execution_findings:
            md.append(f"| {finding.get('family','—')} / {finding.get('series_id','—')} | {finding.get('status','—')} | {finding.get('reason','—')} | `{finding.get('source_path','—')}` |")
    if report.get("excluded_series"):
        md += ["", "## Excluded series", "", "These R4 attempts are quarantined because their target-only slot restore did not preserve the native MTP draft history. They are excluded from primary cells and prefill summaries.", "", "| Family | Series path | Reason |", "|---|---|---|"]
        for s in report["excluded_series"]:
            md.append(f"| {s.get('family','—')} | `{s.get('path','—')}` | {s.get('reason','—')} |")
    md += ["","## Separate empty-prefix baselines","","These original unfilled-context controls retain their configured L and source table. They are not points in the stepped filled-context curves.","","| Family | Configured L | Control | Source table |","|---|---:|---|---|"]
    for e in empty_context_controls:
        md.append(f"| {e['family']} | {e.get('configured_L_tokens','—')} | {e.get('label','empty-prefix baseline')} | `{e.get('report_path')}` |")
    md += ["","## Charts","","Charts are separated by placement group and reasoning mode. Each has three prompt panels; bars show the full three-repeat min-to-max range around the median. GPU and CPU target families are separated; current ACO variants are included in both comparison groupings as the shared candidate.",""]
    md += ["Full per-variant decode and acceptance tables, exact nominal-C ACO/control median deltas, shared context-chunking tables, and retained memory/resource outcomes are in the [comparison tables appendix](comparison-tables-20261104.md). The main tables retain actual C separately from the requested nominal target.",""]
    if DATA_ONLY: md.append("Charts were intentionally skipped in this data-only readiness export.")
    else:
        for p in sorted(out.glob('*.svg')): md.append(f"- [{p.name}]({p.name})")
    md += ["","## Prefill","","Mapped-fill interval charts sum chunks containing new input since the previous checkpoint; they do not plot cumulative-from-zero averages. Per-checkpoint rates divide the interval's summed tokens by summed matching durations, including any checkpoint tail. Actual-C interval start and end are provided where derivable. Separate append-level diagnostics show mapped regular groups, mapped checkpoint tails, and unmapped historical variable groups. Family colors match the decode charts, and markers identify append kind. Protocol values remain in the JSON/CSV data; MTP acceptance scope is not treated as a prefill protocol. No prefill rows are removed from the data exports.","","## Provenance and caveats","","Full sanitized cell data, source paths, SHA-256 digests, family identities, queue state, exclusions, and review flags are in `comparison-final-20261104.json`; decode cells are in `comparison-cells-20261104.csv`, response-quality failures in `comparison-response-quality-failures-20261104.csv`, append diagnostics in `comparison-prefill-append-groups-20261104.csv`, and mapped checkpoint ingest summaries in `comparison-prefill-mapped-checkpoints-20261104.csv`. Raw request/output payloads remain external and are not copied.","","Repeat-spread and neighboring-point flags are review cues only; they do not automatically invalidate otherwise complete evidence. Rates are observations, not causal claims or significance tests. Current ACO MTP-on includes a documented VmSwap peak; the report does not certify RAM-only residency or page-in I/O.",""]
    (out/'comparison-final-20261104.md').write_text('\n'.join(md))
    print(json.dumps({"out":str(out),"cells":len(celllist),"repeats":3*len(celllist),"queue_status":report['queue_status'],"families":{k:v['complete_cells'] for k,v in counts.items()},"files":sorted(p.name for p in out.iterdir())},indent=2))


if __name__=='__main__': main()
