"""Actual completion-only LoRA training. Selection reads validation, not test outputs."""
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
from runtime import ROOT, load_base, amp_context
import torch
from peft import LoraConfig, TaskType, get_peft_model
from data import load_splits, audit
from task import MODEL_ID, REVISION, digest, messages


def encode(tokenizer, row):
    prompt = tokenizer.apply_chat_template(messages(row['text']), tokenize=True, add_generation_prompt=True)
    answer = tokenizer.encode(json.dumps(row['target'], separators=(',', ':')) + tokenizer.eos_token, add_special_tokens=False)
    if len(prompt) + len(answer) > 384:
        raise ValueError('Training example exceeds token cap; no truncation')
    return {'input_ids': prompt + answer, 'labels': [-100] * len(prompt) + answer}


def collate(rows, pad_id, device):
    width = max(len(r['input_ids']) for r in rows)
    return {key: torch.tensor([r[key] + ([pad_id] if key == 'input_ids' else [-100]) * (width - len(r[key])) for r in rows], device=device)
            for key in ('input_ids', 'labels')} | {'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (width - len(r['input_ids'])) for r in rows], device=device)}


def validation_loss(model, examples, pad_id, device, batch_size):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], pad_id, device)
            count = int((batch['labels'][:, 1:] != -100).sum())
            with amp_context(model):
                loss = model(**batch).loss
            total += float(loss) * count
            tokens += count
    return total / tokens


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = load_splits()
    manifest = audit(datasets)
    challenge = [json.loads(line) for line in (ROOT / 'data/challenge.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    tokenizer, base, device = load_base(args.device)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=.05,
                        target_modules=['q_proj', 'v_proj'], bias='none')
    model = get_peft_model(base, config)
    model.peft_config['default'].base_model_name_or_path = MODEL_ID
    model.peft_config['default'].revision = REVISION
    model.config.use_cache = False
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('Only LoRA parameters may be trained')
    initial = {name: p.detach().cpu().clone() for name, p in trainable}
    frozen_before = digest(model.get_input_embeddings().weight.detach().flatten()[:2048].cpu().tolist())
    training = [encode(tokenizer, row) for row in datasets['train']]
    validation = [encode(tokenizer, row) for row in datasets['validation']]
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output directory already exists. Choose a new version to preserve earlier artifacts.')
    output.mkdir(parents=True)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate, weight_decay=.01)
    started = time.perf_counter()
    best = math.inf
    history, steps = [], 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(training)
        total, tokens = 0.0, 0
        for group_start in range(0, len(training), args.batch_size * args.accumulation):
            group = training[group_start:group_start + args.batch_size * args.accumulation]
            denominator = sum(sum(label != -100 for label in row['labels'][1:]) for row in group)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(group), args.batch_size):
                batch = collate(group[start:start + args.batch_size], tokenizer.pad_token_id, device)
                count = int((batch['labels'][:, 1:] != -100).sum())
                with amp_context(model):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss; stop without publishing a checkpoint')
                (loss * count / denominator).backward()
                total += float(loss.detach()) * count
                tokens += count
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()
            steps += 1
        val_loss = validation_loss(model, validation, tokenizer.pad_token_id, device, args.batch_size)
        if not math.isfinite(val_loss):
            raise ValueError('Non-finite validation loss')
        row = {'epoch': epoch + 1, 'train_completion_loss': total / tokens, 'validation_completion_loss': val_loss, 'optimizer_steps': steps}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(output / 'best', safe_serialization=True)
            tokenizer.save_pretrained(output / 'best')
            best_epoch = epoch + 1
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    frozen_after = digest(model.get_input_embeddings().weight.detach().flatten()[:2048].cpu().tolist())
    if frozen_after != frozen_before:
        raise ValueError('Frozen embedding sample changed unexpectedly')
    delta = sum(float((p.detach().cpu() - initial[name]).abs().sum()) for name, p in trainable)
    metadata = {'base_model': MODEL_ID, 'base_revision': REVISION, 'dataset': manifest, 'challenge_hash': digest(challenge), 'device': device,
                'hardware': torch.cuda.get_device_name() if device == 'cuda' else platform.processor(),
                'autocast_dtype': 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float32',
                'torch_version': torch.__version__, 'seed': args.seed, 'config': vars(args), 'history': history,
                'best_epoch': best_epoch, 'selection': 'minimum validation completion-token loss; test generations not inspected',
                'trainable_parameters': sum(p.numel() for _, p in trainable),
                'all_parameters': sum(p.numel() for p in model.parameters()), 'lora_absolute_weight_change': delta,
                'frozen_embedding_sample_unchanged': frozen_after == frozen_before,
                'seconds': time.perf_counter() - started,
                'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
    (output / 'training.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved real trained adapter:', output / 'best', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='private/run-v1')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 1 <= args.batch_size <= 16 or not 1 <= args.accumulation <= 16 or not 0 < args.learning_rate <= .01:
        parser.error('Invalid bounded training configuration')
    train(args)
