#!/usr/bin/env python3
"""Execute a frozen comparison plan, without choosing server settings at runtime."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = Path(__file__).with_name('COMPARISON_COMPLETION_20261104.json')
MATRIX_FILE = ROOT / 'tools/server/bench/run-master-context-matrix.py'
SPEC = importlib.util.spec_from_file_location('planned_comparison_matrix', MATRIX_FILE)
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def service_state(name):
    return subprocess.run(['sudo', '-n', 'systemctl', 'is-active', name],
                          capture_output=True, text=True).stdout.strip()


def port_state(port):
    return subprocess.run(['sudo', '-n', 'ss', '-tlnpH', f'sport = :{port}'],
                          check=True, capture_output=True, text=True).stdout.strip()


def protected_state():
    found = subprocess.run(['pgrep', '-af', '/jellyfin/jellyfin|jellyfin-ffmpeg'],
                           capture_output=True, text=True)
    return {'jellyfin_and_ffmpeg': found.stdout.strip(),
            'port8091': port_state(8091), 'port8092': port_state(8092)}


def verify_plan(plan, build):
    expected = {ROOT / p: digest for p, digest in plan['driver_sha256'].items()}
    expected.update({Path(build['path']): build['sha256'],
                     Path(plan['chunk_map']['path']): plan['chunk_map']['sha256'],
                     Path(plan['template']['path']): plan['template']['sha256']})
    expected.update({Path(p): digest for p, digest in build['loaded_project_dso_sha256'].items()})
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError(f'frozen input changed: {path}')
    # Streaming avoids a model-sized Python allocation before the run.
    if sha(plan['model']['path']) != plan['model']['sha256']:
        raise RuntimeError('frozen model changed')


def stop_owned_server(run_root, expected_binary, slot_state):
    owner_file = run_root / 'cell/server-process.json'
    if not owner_file.exists():
        return
    owner = json.loads(owner_file.read_text())
    proc = Path('/proc') / str(owner['pid'])
    try:
        argv = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
        if proc.joinpath('exe').resolve() != expected_binary.resolve():
            return
        if '--slot-save-path' not in argv or argv[argv.index('--slot-save-path') + 1] != str(slot_state):
            return
        os.killpg(owner['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while proc.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        if proc.exists():
            os.killpg(owner['pid'], signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass


def run_one(plan, step, profile_index, attempt, full_decode):
    profile = step['profiles'][profile_index]
    run_root = Path(plan['output_root']) / step['id'] / f'attempt-{attempt:03d}-{profile["id"]}'
    run_root.mkdir(parents=True, exist_ok=False)
    build = plan['builds'][step['build']]
    verify_plan(plan, build)
    slot = run_root / 'slot-state'
    argv = [part.replace('{run_root}', str(run_root)) for part in profile['server_argv_template']]
    generated = matrix.argv_for_row(profile['context_tokens'], step['placement'],
        mtp=step['mtp'] == 'on', pager=profile['pager'], draft_device=profile['draft_device'],
        server_bin=Path(build['path']), slot_state=slot,
        checkpoint_free_control=profile.get('checkpoint_free_control', False))
    if argv != generated:
        raise RuntimeError('frozen command differs from driver command; fix before touching services')
    selections = profile['checkpoint_selection']
    if full_decode:
        selections = {str(c): {m: [0, 1, 2] for m in matrix.MODES} for c in profile['targets']}
    save(run_root / 'checkpoint-selection.json', selections)
    save(run_root / 'planned-server-argv.json', argv)
    command = [sys.executable, '-u', str(ROOT / 'tools/server/bench/run-context-matrix-cell.py'),
        '--context-tokens', str(profile['context_tokens']), '--target-sequence', *map(str, profile['targets']),
        '--placement', step['placement'], '--mtp', step['mtp'], '--pager', profile['pager'],
        '--server-binary', build['path'], '--source-worktree', plan['source_worktree'],
        '--source-chunk-map', plan['chunk_map']['path'], '--slot-state', str(slot),
        '--output', str(run_root / 'cell'), '--api-key-file', '/srv/ai/config/llama/api-keys',
        '--reasoning-modes', *matrix.MODES, '--prompt-indices', '0', '1', '2',
        '--snapshot-format', profile.get('snapshot_format', 'resume'),
        '--checkpoint-selection', str(run_root / 'checkpoint-selection.json')]
    if profile['draft_device'] is not None:
        command.extend(['--draft-device', profile['draft_device']])
    if profile.get('checkpoint_free_control', False):
        command.append('--checkpoint-free-controls')
    result = {'step': step['number'], 'id': step['id'], 'profile': profile['id'],
              'attempt': attempt, 'run_root': str(run_root), 'full_decode': full_decode,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'starting'}
    child = None
    prior = {}
    lock = open(plan['lifecycle']['lock'], 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        prior = {name: service_state(name) for name in plan['lifecycle']['managed_services']}
        save(run_root / 'prior-state.json', {'services': prior, 'protected': protected_state()})
        for name, state in prior.items():
            if state == 'active':
                subprocess.run(['sudo', '-n', 'systemctl', 'stop', name], check=True)
        deadline = time.monotonic() + 40
        while port_state(8080) and time.monotonic() < deadline:
            time.sleep(0.5)
        if port_state(8080):
            raise RuntimeError('8080 remains owned after stopping managed server; no second model will start')
        result.update(status='running', driver_argv=command)
        save(run_root / 'series-result.json', result)
        print(f"START step={step['number']} {step['id']} {profile['id']} full_decode={full_decode} root={run_root}", flush=True)
        with (run_root / 'driver.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        result['exit_code'] = code
        cell_path = run_root / 'cell/cell.json'
        cell_report = json.loads(cell_path.read_text()) if cell_path.exists() else {}
        result['status'] = cell_report.get('status', 'driver_failure')
        result['stages'] = [{'target': s['target_tokens'], 'status': s['status']}
                            for s in cell_report.get('stages', [])]
        result['candidate_pid'] = cell_report.get('server_pid')
        result['error'] = cell_report.get('error')
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'setup_failure',
                      error=f'{type(error).__name__}: {error}')
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        stop_owned_server(run_root, Path(build['path']), slot)
        restore_errors = []
        for name in reversed(list(prior)):
            if prior[name] == 'active':
                restored = subprocess.run(['sudo', '-n', 'systemctl', 'start', name], capture_output=True, text=True)
                if restored.returncode:
                    restore_errors.append(name + ': ' + restored.stderr[-400:])
        if prior.get('llama-server.service') == 'active':
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                restore_errors.append('8080 health did not return')
        result['restoration'] = {'services': {n: service_state(n) for n in prior},
                                 'protected': protected_state(), 'errors': restore_errors}
        if restore_errors:
            result['status'] = 'restoration_failure'
        result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(run_root / 'series-result.json', result)
        lock.close()
    print(f"END {step['id']} {profile['id']} status={result['status']} root={run_root}", flush=True)
    return result


def sparse_anomalies(run_root, step_id, baselines):
    flags = []
    for path in (run_root / 'cell').glob('C*/campaign/row.json'):
        row = json.loads(path.read_text())
        if row.get('failed_cases'):
            flags.append({'checkpoint': row['geometry']['target_tokens'], 'reason': 'failed canonical cases'})
        records_path = path.with_name('records.jsonl')
        if not records_path.exists():
            continue
        cells = {}
        for line in records_path.read_text().splitlines():
            r = json.loads(line)
            if r.get('kind') == 'measured' and r.get('status') == 'pass':
                cells.setdefault((r['reasoning'], r['prompt_index']), []).append(r['decode_tok_s'])
        for key, values in cells.items():
            if len(values) == 3 and (max(values) - min(values)) > 0.30 * statistics.median(values):
                flags.append({'checkpoint': row['geometry']['target_tokens'], 'cell': key, 'reason': 'repeat spread >30% of median'})
            target = row['geometry']['target_tokens']
            candidates = [anchor for anchor in baselines.get(step_id, [])
                          if anchor['reasoning'] == key[0] and anchor['prompt_index'] == key[1]
                          and abs(anchor['target_tokens'] - target) <= 0.25 * target]
            if len(values) == 3 and candidates:
                anchor = min(candidates, key=lambda value: abs(value['target_tokens'] - target))
                baseline = anchor['decode_median']
                median = statistics.median(values)
                if baseline > 0 and abs(median / baseline - 1) > 0.25:
                    flags.append({'checkpoint': target, 'cell': key,
                                  'reason': 'median differs >25% from nearby historical anchor',
                                  'current_median': median, 'baseline': anchor})
    return flags


def run_queue(plan, plan_path, resume=False, assessed_step=None,
              assessment_action=None, assessment_note=None):
    output = Path(plan['output_root'])
    progress_path = output / 'queue-progress.json'
    if progress_path.exists():
        if not resume:
            raise RuntimeError('queue already exists; use --resume-queue after assessment')
        progress = json.loads(progress_path.read_text())
        if progress['plan_sha256'] != sha(plan_path):
            raise RuntimeError('resume refused: frozen plan changed')
    else:
        progress = {'status': 'running', 'plan': str(plan_path), 'plan_sha256': sha(plan_path), 'steps': []}
    baseline_ref = plan.get('spot_baselines')
    baselines = {}
    if baseline_ref:
        if sha(baseline_ref['path']) != baseline_ref['sha256']:
            raise RuntimeError('historical spot baselines changed')
        baselines = json.loads(Path(baseline_ref['path']).read_text())
    progress['status'] = 'running'
    progress.pop('next_step', None)
    progress.pop('note', None)
    save(progress_path, progress)

    def next_attempt(step):
        used = [int(match.group(1)) for path in (output / step['id']).glob('attempt-*')
                if (match := re.match(r'attempt-(\d+)-', path.name))]
        return max(used, default=0) + 1

    def execute_attempt(step, index, full):
        attempt = next_attempt(step)
        try:
            return run_one(plan, step, index, attempt, full)
        except Exception as error:
            # Preflight/hash/lock failures happen before a model is started.
            # Keep an explicit receipt so the queue never appears to be
            # running forever after a missing-input traceback.
            run_root = output / step['id'] / f'attempt-{attempt:03d}-{step["profiles"][index]["id"]}'
            receipt = run_root / 'series-result.json'
            result = {'id': step['id'], 'step': step['number'], 'attempt': attempt,
                      'profile': step['profiles'][index]['id'], 'run_root': str(run_root),
                      'full_decode': full, 'status': 'setup_failure',
                      'error': f'{type(error).__name__}: {error}', 'stages': []}
            if not receipt.exists():
                save(receipt, result)
            return result

    for step in plan['steps']:
        item = next((entry for entry in progress['steps'] if entry['id'] == step['id']), None)
        if item is not None and item.get('status') in {'complete', 'capacity_limit', 'finding_recorded'}:
            continue
        if item is None:
            item = {'number': step['number'], 'id': step['id'], 'attempts': []}
            progress['steps'].append(item)
        start_profile = 0
        retry_full = False
        if item['attempts']:
            last = item['attempts'][-1]
            start_profile = next(i for i, value in enumerate(step['profiles']) if value['id'] == last['profile'])
            if assessed_step == step['number']:
                item.setdefault('assessments', []).append({'action': assessment_action,
                    'note': assessment_note, 'epoch': time.time()})
                if assessment_action == 'continue':
                    if last['status'] not in {'complete', 'capacity_limit'}:
                        raise RuntimeError('cannot accept incomplete measurements as an explained anomaly')
                    item['status'] = last['status']
                    item['measurement_gaps_remain'] = last['status'] != 'complete'
                    save(progress_path, progress)
                    continue
                if assessment_action == 'record-finding':
                    item['status'] = 'finding_recorded'
                    item['measurement_gaps_remain'] = last['status'] not in {'complete', 'capacity_limit'}
                    save(progress_path, progress)
                    continue
                retry_full = assessment_action == 'full-curve' or bool(last.get('full_decode'))
            elif last.get('full_decode'):
                progress.update(status='needs_assessment', next_step=step['number'],
                    note='Full-curve escalation already attempted. Use an explicit assessed retry or record-finding action; do not edit progress JSON or blindly repeat it.')
                save(progress_path, progress)
                return 1
        for index in range(start_profile, len(step['profiles'])):
            profile = step['profiles'][index]
            result = execute_attempt(step, index, retry_full)
            item['attempts'].append(result)
            save(progress_path, progress)
            if result['status'] == 'capacity_limit' and not result.get('stages') and index + 1 < len(step['profiles']):
                continue
            if result['status'] not in {'complete', 'capacity_limit', 'incomplete'}:
                progress.update(status='needs_assessment', next_step=step['number'])
                save(progress_path, progress)
                return 1
            # A later capacity limit does not erase an earlier completed spot
            # checkpoint's anomaly. Inspect every available canonical row.
            if step.get('full_curve_escalation', False) and not retry_full:
                flags = sparse_anomalies(Path(result['run_root']), step['id'], baselines)
                if flags:
                    save(Path(result['run_root']) / 'anomaly-assessment.json', flags)
                    item['status'] = 'needs_assessment'
                    progress.update(status='needs_assessment', next_step=step['number'],
                        note='Classify the flags before rerunning: continue for an evidenced explanation, full-curve for an unexplained anomaly, or record-finding for a real limitation.')
                    save(progress_path, progress)
                    return 1
            item['status'] = item['attempts'][-1]['status']
            save(progress_path, progress)
            break
    progress['status'] = ('finished' if all(s.get('status') == 'complete' for s in progress['steps'])
                          else 'finished_with_findings')
    save(progress_path, progress)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--step', type=int)
    group.add_argument('--queue', action='store_true')
    parser.add_argument('--profile', type=int, default=0)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--full-decode', action='store_true')
    parser.add_argument('--resume-queue', action='store_true',
                        help='resume an assessed stopped queue without overwriting completed attempts')
    parser.add_argument('--assessed-step', type=int)
    parser.add_argument('--assessment-action', choices=('retry', 'record-finding', 'continue', 'full-curve'))
    parser.add_argument('--assessment-note')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get('status') != 'ready':
        parser.error('the comparison plan is still a draft; do not start model runs')
    assessment = (args.assessed_step, args.assessment_action, args.assessment_note)
    if any(value is not None for value in assessment) and not (
            args.queue and args.resume_queue and all(value is not None for value in assessment)):
        parser.error('assessment requires --queue --resume-queue plus step, action, and note')
    if args.queue:
        output = Path(plan['output_root'])
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'queue-owner.lock').open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                return run_queue(plan, args.plan, args.resume_queue, *assessment)
            except BaseException as error:
                progress_path = output / 'queue-progress.json'
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    if progress.get('status') == 'running':
                        progress.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit))
                                        else 'needs_assessment', error=f'{type(error).__name__}: {error}')
                        save(progress_path, progress)
                raise
    step = next(s for s in plan['steps'] if s['number'] == args.step)
    result = run_one(plan, step, args.profile, args.attempt, args.full_decode)
    return 0 if result['status'] in {'complete', 'capacity_limit'} else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    raise SystemExit(main())
