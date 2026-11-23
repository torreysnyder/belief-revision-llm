#!/usr/bin/env python3
"""Stage 1: prepare a vignette-disjoint SAE corpus from behavioral JSONL.

No model calls, activation extraction, or SAE fitting occur in this script.

Example:
  python prepare_sae_corpus.py --input dialogues_full_crossed_3v_qwen2-5-7b-base-ndif_openrouter_qc_v4.jsonl \
      --split-manifest data/caa_qwen7b_openrouter_qc_v4/caa_split_manifest_by_vignette.json

Outputs (under --out-dir):
  sae_corpus_{train,validation,test}.jsonl  one full conversation per line
  sae_turns_{train,validation,test}.jsonl   one user+assistant exchange per line
  sae_corpus_manifest.json                provenance and quality report

The target is Qwen/Qwen2.5-7B BASE, with plain role-labeled transcript
formatting and add_special_tokens=False, matching the CAA pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B'
MODEL_LABEL = 'qwen2.5-7b-base-ndif_openrouter_qc_v4'
PROMPT_FORMAT = 'qwen2.5-7b-base_plain_role_transcript'
PHASES = ('intro','core_challenge_1','elaboration_1','core_challenge_2','elaboration_2','closing')
SPLITS = ('train','validation','test')
DEFAULT_MANIFEST = 'data/caa_qwen7b_openrouter_qc_v4/caa_split_manifest_by_vignette.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fail(message):
    raise ValueError(message)


def load_assignments(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('split_unit') != 'vignette_id':
        fail('Expected split_unit=vignette_id in the existing CAA manifest')
    assignments = data.get('assignments')
    if not isinstance(assignments, dict) or not assignments:
        fail('Split manifest must contain a nonempty assignments mapping')
    normalized = {}
    for vignette, value in assignments.items():
        if isinstance(value, dict):
            value = value.get('split')
        if value not in SPLITS:
            fail(f'Unexpected split assignment for {vignette}: {value!r}')
        normalized[str(vignette)] = value
    return normalized


def read_jsonl(paths):
    for path in paths:
        with path.open('r', encoding='utf-8-sig') as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        fail(f'{path}:{lineno}: invalid JSON: {exc}')
                    if not isinstance(obj, dict):
                        fail(f'{path}:{lineno}: expected JSON object')
                    yield path, lineno, obj


def valid_score(v):
    return isinstance(v, (int,float)) and not isinstance(v, bool) and 0 <= v <= 100


def get_claims(obj):
    vignette = (obj.get('primary_cell') or {}).get('vignette') or {}
    auxiliaries = vignette.get('auxiliaries') or []
    return {
        'core_claim': vignette.get('core_claim'),
        'auxiliary_claims': {str(x.get('id')): x.get('text') for x in auxiliaries if isinstance(x, dict)},
        'display_A1_text': (obj.get('aux_mapping') or {}).get('display_A1_text'),
        'display_A2_text': (obj.get('aux_mapping') or {}).get('display_A2_text'),
    }


def build_conversation(obj, path, lineno, split, strict):
    vid = str(obj.get('vignette_id') or '')
    cid = str(obj.get('cell_id') or '')
    if not vid or not cid:
        fail(f'{path}:{lineno}: missing vignette_id or cell_id')
    target = str(obj.get('target_model') or '')
    if target != MODEL_LABEL:
        fail(f'{path}:{lineno}: unexpected target_model={target!r}; expected {MODEL_LABEL!r}')
    history = obj.get('dialogue_history')
    rows = obj.get('rows')
    if not isinstance(history, list) or not isinstance(rows, list):
        fail(f'{path}:{lineno}: missing dialogue_history or rows')
    if len(history) != 2*len(rows):
        fail(f'{path}:{lineno}: expected alternating user/assistant history matching rows')
    if strict and len(rows) != 15:
        fail(f'{path}:{lineno}: expected 15 exchanges, got {len(rows)}')
    plan = obj.get('run_plan') or {}
    style = obj.get('style_profile') or {}
    claims = get_claims(obj)
    turns = []
    messages = []
    for i, row in enumerate(rows):
        u, a = history[2*i:2*i+2]
        if u.get('speaker') != 'user' or a.get('speaker') != 'assistant':
            fail(f'{path}:{lineno}: invalid role order at exchange {i+1}')
        if u.get('text') != row.get('user_turn') or a.get('text') != row.get('assistant_turn'):
            fail(f'{path}:{lineno}: row/history mismatch at exchange {i+1}')
        phase = row.get('phase')
        if phase not in PHASES or u.get('phase') != phase or a.get('phase') != phase:
            fail(f'{path}:{lineno}: invalid phase at exchange {i+1}')
        if strict and row.get('global_turn_index') != i+1:
            fail(f'{path}:{lineno}: nonsequential global_turn_index')
        if not isinstance(u.get('text'), str) or not isinstance(a.get('text'), str):
            fail(f'{path}:{lineno}: non-text message at exchange {i+1}')
        if row.get('inject_evidence') and not row.get('evidence_id'):
            fail(f'{path}:{lineno}: evidence injection without evidence_id')
        messages.extend(({'role':'user','content':u['text']}, {'role':'assistant','content':a['text']}))
        judge = {k: row.get(k) for k in ('belief_core','belief_A1','belief_A2',
                'confidence_belief_core','confidence_belief_A1','confidence_belief_A2',
                'assistant_epistemic_confidence','assistant_stance','main_change','changed_node')}
        turn = {
            'conversation_id': cid, 'vignette_id':vid, 'split':split,
            'exchange_index':i+1, 'phase':phase, 'turn_in_phase':row.get('turn_in_phase'),
            'domain':row.get('domain'), 'conversation_goal':row.get('conversation_goal') or plan.get('conversation_goal'),
            'style_id':row.get('style_id') or style.get('style_id'),
            'evidence_condition':row.get('evidence_role') if row.get('inject_evidence') else 'none',
            'inject_evidence':bool(row.get('inject_evidence')),
            'evidence_id':row.get('evidence_id'), 'evidence_role':row.get('evidence_role'),
            'evidence_strength':row.get('evidence_strength'),
            'evidence_credibility':row.get('evidence_credibility'),
            'evidence_targets':row.get('evidence_targets'),
            'user_message':u['text'], 'assistant_message':a['text'],
            'judge_probe_after_turn':bool(row.get('probe_ran_after_turn')),
            'judge_scores':judge if row.get('probe_ran_after_turn') else None,
            'user_simulated_belief_core':row.get('user_belief_core'),
            'user_simulated_confidence_core':row.get('user_confidence_core'),
            'core_claim':claims['core_claim'],
            'display_A1_text':row.get('display_A1_text') or claims['display_A1_text'],
            'display_A2_text':row.get('display_A2_text') or claims['display_A2_text'],
        }
        if turn['judge_probe_after_turn']:
            for key in ('belief_core','belief_A1','belief_A2'):
                if not valid_score(judge[key]):
                    fail(f'{path}:{lineno}: missing/invalid {key} at probe exchange {i+1}')
        turns.append(turn)
    probe_outputs = obj.get('probe_outputs') or []
    expected_probes = [t['phase'] for t in turns if t['judge_probe_after_turn']]
    if strict and (len(probe_outputs) != 5 or expected_probes !=
                   ['intro','core_challenge_1','core_challenge_2','elaboration_2','closing']):
        fail(f'{path}:{lineno}: expected five QCv4 checkpoints, got {expected_probes}')
    return {
        'conversation_id':cid,'vignette_id':vid,'split':split,
        'replicate_index':obj.get('replicate_index'),'seed':obj.get('seed'),
        'model':MODEL,'target_model_label':MODEL_LABEL,
        'prompt_format':PROMPT_FORMAT,'tokenization':'add_special_tokens=False',
        'domain':(rows[0] or {}).get('domain') if rows else None,
        'conversation_goal':plan.get('conversation_goal'),
        'style_id':style.get('style_id'), 'belief_anchor_level':plan.get('belief_anchor_level'),
        'trust_in_institutions':plan.get('trust_in_institutions'),
        'claims':claims,'messages':messages,'turns':turns,
        'probe_outputs':probe_outputs,
        'source_file':str(path.resolve()),'source_line':lineno,
    }


def dump_line(f, obj):
    f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',nargs='+',required=True,help='One or more QCv4 behavioral conversation JSONL files')
    ap.add_argument('--split-manifest',default=DEFAULT_MANIFEST)
    ap.add_argument('--out-dir',default='data/sae_qwen7b_base_qcv4')
    ap.add_argument('--allow-incomplete',action='store_true',help='Permit fewer than 15 exchanges / missing checkpoints (not recommended for final corpus)')
    ap.add_argument('--allow-missing-vignettes',action='store_true',help='Permit a pilot subset of the manifest; still forbid unknown vignette IDs')
    ap.add_argument('--expected-conversations',type=int,default=200,help='Expected number of conversations; set 0 to disable check')
    args=ap.parse_args()
    inputs=[Path(x) for x in args.input]
    manifest=Path(args.split_manifest)
    for path in [*inputs,manifest]:
        if not path.is_file():
            ap.error(f'Input does not exist: {path}')
    assignments=load_assignments(manifest)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    counts=Counter(); vignette_counts=defaultdict(set); phases=Counter(); evidence=Counter(); domains=Counter()
    seen=set()
    # Write to temporary files and publish only after every validation succeeds.
    handles={}
    paths={}
    try:
        for split in SPLITS:
            for kind in ('corpus','turns'):
                dest=out/f'sae_{kind}_{split}.jsonl'
                tmp=out/f'.{dest.name}.partial'
                paths[(split,kind)]=(tmp,dest)
                handles[(split,kind)]=tmp.open('w',encoding='utf-8',newline='\n')
        for path,lineno,obj in read_jsonl(inputs):
            vid=str(obj.get('vignette_id') or '')
            if vid not in assignments:
                fail(f'{path}:{lineno}: vignette {vid!r} missing from frozen CAA split manifest')
            split=assignments[vid]
            conv=build_conversation(obj,path,lineno,split,strict=not args.allow_incomplete)
            key=(conv['conversation_id'],conv['replicate_index'])
            if key in seen:
                fail(f'Duplicate conversation/replicate: {key}')
            seen.add(key)
            dump_line(handles[(split,'corpus')],conv)
            for turn in conv['turns']:
                dump_line(handles[(split,'turns')],turn)
                phases[(split,turn['phase'])]+=1
                evidence[(split,str(turn['evidence_condition']))]+=1
            counts[split]+=1
            domains[(split,str(conv['domain']))]+=1
            vignette_counts[split].add(vid)
        total=sum(counts.values())
        if args.expected_conversations and total != args.expected_conversations:
            fail(f'Expected {args.expected_conversations} conversations; found {total}. Use --expected-conversations 0 for pilot subsets.')
        if not args.allow_missing_vignettes:
            missing=set(assignments)-set().union(*vignette_counts.values())
            if missing:
                fail(f'{len(missing)} manifest vignettes absent from corpus (examples: {sorted(missing)[:8]}). Use --allow-missing-vignettes for pilot subsets.')
        for f in handles.values():
            f.close()
        for tmp,dest in paths.values():
            os.replace(tmp,dest)
    except Exception:
        for f in handles.values():
            if not f.closed: f.close()
        for tmp,_ in paths.values():
            tmp.unlink(missing_ok=True)
        raise
    report={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'prepare_sae_corpus','target_model':MODEL,
        'target_model_label':MODEL_LABEL,'prompt_format':PROMPT_FORMAT,
        'tokenization':'add_special_tokens=False',
        'split_unit':'vignette_id','split_manifest':str(manifest.resolve()),
        'split_manifest_sha256':sha256(manifest),
        'input_files':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in inputs],
        'expected_conversations':args.expected_conversations,
        'conversation_counts':dict(counts),'vignette_counts':{s:len(vignette_counts[s]) for s in SPLITS},
        'turn_counts':{s:sum(n for (sp,_),n in phases.items() if sp==s) for s in SPLITS},
        'phase_counts':{s:{p:phases[(s,p)] for p in PHASES} for s in SPLITS},
        'evidence_condition_counts':{s:{k:n for (sp,k),n in evidence.items() if sp==s} for s in SPLITS},
        'domain_counts':{s:{k:n for (sp,k),n in domains.items() if sp==s} for s in SPLITS},
        'outputs':{f'{kind}_{split}':{'path':str(dest.resolve()),'sha256':sha256(dest)}
                   for (split,kind),(_,dest) in paths.items()},
        'note':'Judge scores are observational metadata, not ground truth or SAE training targets.',
        'stage2_prompt_rule':'System: <system prompt>\\n\\nUser: ...\\n\\nAssistant: ...; add_special_tokens=False',
    }
    report_path=out/'sae_corpus_manifest.json'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'conversations':dict(counts),
                      'vignettes':report['vignette_counts'],'turns':report['turn_counts'],
                      'manifest':str(report_path)},indent=2))

if __name__=='__main__':
    try: main()
    except (ValueError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
