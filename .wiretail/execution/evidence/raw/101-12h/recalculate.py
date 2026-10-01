#!/usr/bin/env python3
import hashlib, json, statistics
from pathlib import Path

root=Path('/srv/ai/paged-kv/results/forward/101-18/attempt-01')
routes={'selected':'selected','pager_off_all_gpu':'pager-off-all-gpu','cpu_main_kv_gpu_mtp':'cpu-main-kv-gpu-mtp'}
rows={}
for key,dirname in routes.items():
    path=root/dirname/'records.jsonl'
    data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    measured=[r for r in data if r.get('phase')=='measured']
    assert len(measured)==9, (key,len(measured))
    assert all(r.get('http_code')==200 for r in measured)
    assert all(r.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')==0 for r in measured)
    grouped={}
    for r in measured:
        grouped.setdefault(r['prompt_index'],[]).append(r)
    assert sorted(grouped)==[0,1,2]
    rows[key]={}
    for ix,items in sorted(grouped.items()):
        assert len(items)==3
        pref=statistics.median(r['timings']['prompt_per_second'] for r in items)
        dec=statistics.median(r['timings']['predicted_per_second'] for r in items)
        acc=[r['mtp']['accepted_tokens'] for r in items]
        draft=[r['mtp']['draft_tokens'] for r in items]
        rates=[100*a/d for a,d in zip(acc,draft)]
        rows[key][f'prompt_{ix+1}']={
            'prefill_tok_s_median':pref,'decode_tok_s_median':dec,
            'mtp_acceptance_percent_median':statistics.median(rates),
            'mtp_row_percent':rates,'accepted':acc,'drafted':draft,
            'input_tokens':[r['usage']['prompt_tokens'] for r in items],
            'cached_tokens':[r['usage']['prompt_tokens_details']['cached_tokens'] for r in items],
            'output_tokens':[r['usage']['completion_tokens'] for r in items],
            'http_codes':[r['http_code'] for r in items],
        }
configs={}
for key,dirname in routes.items():
    c=json.loads((root/dirname/'run-config.json').read_text())
    configs[key]={
      'model_sha256':c['model']['sha256'],
      'binary':c['backend']['binary'],
      'binary_sha256':c['runtime_identity']['candidate'].get('binary_sha256') or c['runtime_identity']['candidate'].get('sha256'),
      'version':c['backend']['version'],
      'exact_command':c['backend']['exact_command'],
    }
selected=rows['selected']; cpu=rows['cpu_main_kv_gpu_mtp']; dense=rows['pager_off_all_gpu']
comparison={}
for p in selected:
    s=selected[p]; c=cpu[p]; g=dense[p]
    comparison[p]={
      'selected_to_cpu_prefill':s['prefill_tok_s_median']/c['prefill_tok_s_median'],
      'selected_to_cpu_decode':s['decode_tok_s_median']/c['decode_tok_s_median'],
      'selected_to_dense_gpu_prefill':s['prefill_tok_s_median']/g['prefill_tok_s_median'],
      'selected_to_dense_gpu_decode':s['decode_tok_s_median']/g['decode_tok_s_median'],
      'mtp_gate':{'prompt_1':75,'prompt_2':40,'prompt_3':60}[p],
      'mtp_margin_pp':s['mtp_acceptance_percent_median']-{'prompt_1':75,'prompt_2':40,'prompt_3':60}[p],
    }
result={
 'task':'101-12h','source_campaign':'101-18 attempt-01 canonical measured raw rows',
 'raw_root':str(root),'raw_records_sha256':{k:hashlib.sha256((root/v/'records.jsonl').read_bytes()).hexdigest() for k,v in routes.items()},
 'route_configs':configs,'measured_rows_per_route':9,'rows':rows,'comparisons':comparison,
 'gates':{'prefill_minimum_tok_s':500,'prefill_preferred_tok_s':750,'mtp_floor_percent':{'prompt_1':75,'prompt_2':40,'prompt_3':60},'all_prefill_minimum_pass':all(x['prefill_tok_s_median']>=500 for x in selected.values()),'all_prefill_preferred_pass':all(x['prefill_tok_s_median']>=750 for x in selected.values()),'mtp_pass':all(selected[p]['mtp_acceptance_percent_median']>=comparison[p]['mtp_gate'] for p in selected),'all_decode_beats_cpu':all(selected[p]['decode_tok_s_median']>cpu[p]['decode_tok_s_median'] for p in selected)},
}
out=Path(__file__).with_name('recalculation.json')
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(out),'gates':result['gates'],'selected':{p:{k:round(v,5) for k,v in d.items() if k.endswith('_median')} for p,d in selected.items()},'comparisons':{p:{k:round(v,5) for k,v in d.items() if isinstance(v,(int,float))} for p,d in comparison.items()}},indent=2))
