#!/usr/bin/env python3
"""Stage 2: extract token-level residual activations for SAE training.

Reads Stage 1 sae_corpus_{split}.jsonl and saves one resumable tensor shard
per conversation/layer/split. Uses Qwen/Qwen2.5-7B BASE via remote NDIF/NNsight.
No OpenRouter calls and no SAE fitting.

Example (small validation smoke test):
  python -u extract_sae_activations.py --split validation --layer 18 \
      --max-conversations 1 --max-exchanges 2 --positions-per-message 2

Full train extraction:
  python -u extract_sae_activations.py --split train --layer 18

IMPORTANT: This script has been syntax/unit checked, not tested against a live
NDIF endpoint. Verify a smoke test before launching the full extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B'
DIM = 3584
N_LAYERS = 28
SPLITS = ('train', 'validation', 'test')
ROLE_NAMES = {'system': 'System', 'user': 'User', 'assistant': 'Assistant'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-dir', default='data/sae_qwen7b_base_qcv4')
    p.add_argument('--out-dir', default='data/sae_qwen7b_base_qcv4/activations')
    p.add_argument('--split', choices=SPLITS, default='train')
    p.add_argument('--layer', type=int, default=18)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--model-revision', default=None)
    p.add_argument('--positions-per-message', type=int, default=4,
                   help='Sample up to this many positions from each user/assistant message in the current exchange')
    p.add_argument('--sample-mode', choices=('uniform', 'last'), default='uniform')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-conversations', type=int, default=0, help='0 means all')
    p.add_argument('--max-exchanges', type=int, default=0, help='0 means all 15 exchanges')
    p.add_argument('--every-n-exchanges', type=int, default=1,
                   help='1 extracts at each exchange; 3 extracts at exchanges 3,6,...')
    p.add_argument('--max-seq-tokens', type=int, default=8192,
                   help='Skip overlength conversation prefixes rather than silently truncate')
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--retry-delay', type=float, default=8.0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate corpus and token sampling; do not call NDIF')
    return p.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{i}: {e}') from e
                require(isinstance(item, dict), f'{path}:{i}: expected JSON object')
                yield item


def render_messages(messages):
    """Exactly the QCv4 plain role-labeled message convention (no ChatML)."""
    chunks, spans, cursor = [], [], 0
    for i, message in enumerate(messages):
        role = message['role']
        content = message['content']
        require(role in ROLE_NAMES and isinstance(content, str), f'Invalid message {i}')
        if i:
            chunks.append('\n\n')
            cursor += 2
        prefix = ROLE_NAMES[role] + ': '
        chunks.append(prefix)
        cursor += len(prefix)
        start = cursor
        chunks.append(content)
        cursor += len(content)
        spans.append({'message_index': i, 'role': role, 'start': start, 'end': cursor})
    return ''.join(chunks), spans


def choose_positions(tokenizer, messages, exchange_index, per_message, mode, seed):
    """Sample token indices from both messages of the CURRENT exchange.

    Fast tokenizer character offsets ensure the token indices are aligned to
    the *whole* prompt, including role prefixes and message separators.
    """
    prompt, spans = render_messages(messages)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    require(len(ids) == len(offsets), 'Tokenizer offset mismatch')
    rng = random.Random(seed)
    chosen, metadata = [], []
    for span in spans[-2:]:
        require(span['message_index'] in (2*exchange_index-2, 2*exchange_index-1),
                'Unexpected exchange message indices')
        candidates = [i for i, (a, b) in enumerate(offsets)
                      if b > span['start'] and a < span['end'] and b > a]
        if not candidates:
            continue
        count = min(per_message, len(candidates))
        selected = (candidates[-count:] if mode == 'last'
                    else sorted(rng.sample(candidates, count)))
        for pos in selected:
            a, b = offsets[pos]
            chosen.append(pos)
            metadata.append({
                'token_position': pos,
                'token_id': ids[pos],
                'token_text': tokenizer.decode([ids[pos]], skip_special_tokens=False),
                'role': span['role'],
                'message_index': span['message_index'],
                'char_start': a,
                'char_end': b,
            })
    require(chosen, f'No sampleable content tokens at exchange {exchange_index}')
    require(len(set(chosen)) == len(chosen), 'Duplicate sampled positions')
    return prompt, ids, chosen, metadata


def check_conversation(conv, split):
    require(conv.get('model') == MODEL, 'Corpus model is not Qwen2.5-7B Base')
    require(conv.get('split') == split, 'Corpus split mismatch')
    require(conv.get('tokenization') == 'add_special_tokens=False', 'Tokenization mismatch')
    messages = conv.get('messages')
    turns = conv.get('turns')
    require(isinstance(messages, list) and isinstance(turns, list), 'Missing messages/turns')
    require(len(messages) == 2 * len(turns), 'Message/turn count mismatch')
    require(conv.get('conversation_id') and conv.get('vignette_id'), 'Missing IDs')
    for i, turn in enumerate(turns):
        require(messages[2*i]['role'] == 'user' and messages[2*i+1]['role'] == 'assistant',
                f'Invalid role sequence at exchange {i+1}')
        require(turn.get('exchange_index') == i+1, f'Invalid turn index {i+1}')
        require(messages[2*i]['content'] == turn.get('user_message') and
                messages[2*i+1]['content'] == turn.get('assistant_message'),
                f'Turn/message mismatch at exchange {i+1}')
    return messages, turns


def remote_extract(wrapper, ids, positions, layer):
    """Save a small [n_positions, 3584] slice on the remote NDIF worker.

    Qwen decoder block .output is a tensor [batch, seq, hidden], NOT a tuple.
    nnsight.save(container) matches the working QCv4 NDIF extraction pattern.
    """
    import torch
    import nnsight
    input_ids = torch.tensor([ids], dtype=torch.long)
    with wrapper.trace(input_ids, remote=True):
        saved = nnsight.save(
            wrapper.model.layers[layer].output[0, positions, :].detach().float().cpu()
        )
    tensor = saved.value if hasattr(saved, 'value') else saved
    if isinstance(tensor, (list, tuple)) and len(tensor) == 1:
        tensor = tensor[0]
    tensor = torch.as_tensor(tensor).detach().float().cpu()
    require(tuple(tensor.shape) == (len(positions), DIM),
            f'Unexpected NDIF shape {tuple(tensor.shape)}, expected {(len(positions), DIM)}')
    require(torch.isfinite(tensor).all().item(), 'Nonfinite activations returned')
    return tensor


def is_transient(exc):
    message = f'{type(exc).__name__}: {exc}'.lower()
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(x in message for x in
        ('429', '502', '503', '504', 'timeout', 'timed out', 'connection reset',
         'websocket', 'service unavailable', 'bad gateway', 'temporarily unavailable'))


def extract_with_retries(wrapper, ids, positions, layer, max_retries, delay):
    for attempt in range(max_retries+1):
        try:
            return remote_extract(wrapper, ids, positions, layer)
        except Exception as exc:
            if attempt == max_retries or not is_transient(exc):
                raise
            pause = delay * (2 ** attempt)
            print(f'  NDIF transient error: {exc}; retrying in {pause:.0f}s', flush=True)
            time.sleep(pause)
    raise AssertionError('unreachable')


def safe_name(value):
    """Stable hash avoids collisions, unsafe filenames and sensitive IDs in paths."""
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:24]


def main():
    args = parse_args()
    require(args.model == MODEL, f'Target must be {MODEL} (Base), not {args.model}')
    require(0 <= args.layer < N_LAYERS, f'Layer must be 0..{N_LAYERS-1}')
    require(args.positions_per_message > 0 and args.every_n_exchanges > 0,
            'Sampling counts must be positive')
    require(args.max_seq_tokens > 0 and args.max_retries >= 0, 'Invalid limits')
    corpus_path = Path(args.corpus_dir)/f'sae_corpus_{args.split}.jsonl'
    require(corpus_path.is_file(), f'Missing Stage 1 output: {corpus_path}')
    manifest_path = Path(args.corpus_dir)/'sae_corpus_manifest.json'
    require(manifest_path.is_file(), f'Missing Stage 1 manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('target_model') == MODEL, 'Stage 1 manifest model mismatch')
    expected_hash = (manifest.get('outputs', {}).get(f'corpus_{args.split}') or {}).get('sha256')
    if expected_hash:
        require(sha256(corpus_path) == expected_hash, 'Stage 1 corpus hash mismatch')

    # Tokenizer weights are small relative to the model; Qwen weights remain remote.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    require(tokenizer.is_fast, 'Fast tokenizer with offset_mapping is required')

    wrapper = None
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv()
        from nnsight import LanguageModel, CONFIG
        require(os.environ.get('NDIF_API_KEY'), 'NDIF_API_KEY must be configured')
        CONFIG.set_default_api_key(os.environ['NDIF_API_KEY'])
        wrapper = LanguageModel(args.model, dispatch=False)
        require(not getattr(wrapper, 'dispatched', False), 'Unexpected local model dispatch')

    out = Path(args.out_dir)/args.split/f'layer_{args.layer:02d}'
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    seen = set()
    source_sha = sha256(corpus_path)
    config = {
        'stage': 'extract_sae_activations', 'model': MODEL,
        'model_revision': args.model_revision, 'split': args.split,
        'layer': args.layer, 'hook': f'model.layers.{args.layer}.output',
        'hook_semantics': 'decoder block output (post-block residual stream)',
        'hidden_dim': DIM, 'tokenization': 'add_special_tokens=False',
        'prompt_format': 'plain role-labeled transcript, no trailing Assistant: cue',
        'context_policy': 'causal prefix ending at the current assistant message',
        'positions_per_message': args.positions_per_message,
        'sample_mode': args.sample_mode, 'seed': args.seed,
        'every_n_exchanges': args.every_n_exchanges,
        'max_exchanges': args.max_exchanges,
        'max_seq_tokens': args.max_seq_tokens,
        'corpus_sha256': source_sha,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for conv in jsonl(corpus_path):
        if args.max_conversations and stats['conversations'] >= args.max_conversations:
            break
        messages, turns = check_conversation(conv, args.split)
        cid = conv['conversation_id']
        require(cid not in seen, f'Duplicate conversation_id: {cid}')
        seen.add(cid)
        stats['conversations'] += 1
        limit = min(len(turns), args.max_exchanges or len(turns))
        for exchange in range(1, limit+1):
            if exchange % args.every_n_exchanges:
                continue
            prefix = messages[:2*exchange]
            seed = int(hashlib.sha256(f'{args.seed}:{cid}:{exchange}'.encode()).hexdigest()[:12],16)
            prompt, ids, positions, token_meta = choose_positions(
                tokenizer, prefix, exchange, args.positions_per_message, args.sample_mode, seed)
            stats['candidate_exchanges'] += 1
            if len(ids) > args.max_seq_tokens:
                stats['overlength_exchanges'] += 1
                print(f'SKIP overlength: conversation={cid} exchange={exchange} tokens={len(ids)}', flush=True)
                continue
            shard_path = out/f'{safe_name(cid)}_exchange_{exchange:02d}.pt'
            if shard_path.exists() and not args.overwrite and not args.dry_run:
                import torch
                existing = torch.load(shard_path, map_location='cpu', weights_only=True)
                require(existing.get('config_hash') == config_hash and
                        existing.get('conversation_id') == cid and
                        existing.get('exchange_index') == exchange,
                        f'Shard mismatch: {shard_path}; use a separate out-dir or --overwrite')
                stats['resumed_exchanges'] += 1
                continue
            if args.dry_run:
                stats['dry_run_exchanges'] += 1
                stats['dry_run_vectors'] += len(positions)
                continue
            activations = extract_with_retries(
                wrapper, ids, positions, args.layer, args.max_retries, args.retry_delay)
            turn = turns[exchange-1]
            rows = []
            for i, tok in enumerate(token_meta):
                rows.append({
                    **tok, 'conversation_id': cid, 'vignette_id': conv['vignette_id'],
                    'split': args.split, 'layer': args.layer,
                    'exchange_index': exchange, 'phase': turn.get('phase'),
                    'domain': turn.get('domain') or conv.get('domain'),
                    'conversation_goal': turn.get('conversation_goal') or conv.get('conversation_goal'),
                    'evidence_condition': turn.get('evidence_condition'),
                    'evidence_id': turn.get('evidence_id'),
                    'inject_evidence': turn.get('inject_evidence'),
                    'judge_probe_after_turn': turn.get('judge_probe_after_turn'),
                    'judge_scores': turn.get('judge_scores'),
                    'user_simulated_belief_core': turn.get('user_simulated_belief_core'),
                    'token_sample_index': i,
                })
            payload = {
                'config_hash': config_hash, 'config': config,
                'conversation_id': cid, 'vignette_id': conv['vignette_id'],
                'exchange_index': exchange, 'num_prompt_tokens': len(ids),
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'activations': activations.contiguous(), 'token_metadata': rows,
            }
            import torch
            tmp = shard_path.with_suffix('.pt.partial')
            try:
                torch.save(payload, tmp)
                os.replace(tmp, shard_path)
            finally:
                tmp.unlink(missing_ok=True)
            stats['extracted_exchanges'] += 1
            stats['extracted_vectors'] += len(rows)
            print(f'[{stats["conversations"]}] {cid} exchange={exchange} '
                  f'tokens={len(ids)} vectors={len(rows)} saved={shard_path.name}', flush=True)
    if not args.dry_run:
        # Do not overwrite the shared manifest for a subset run; use a distinct run report.
        report_path = out/f'extraction_report_{config_hash[:12]}.json'
        report = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'config_hash': config_hash, 'config': config, 'counts': dict(stats),
                  'output_dir': str(out.resolve())}
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(f'Report: {report_path}', flush=True)
    print(json.dumps({'dry_run': args.dry_run, 'counts': dict(stats),
                      'layer': args.layer, 'split': args.split,
                      'output_dir': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
