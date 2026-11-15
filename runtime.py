"""Pinned local base model and adapter inference, never executes remote model code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'private/hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import json
import time
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from task import MODEL_ID, REVISION, messages, parse_output


def amp_context(model):
    return torch.autocast('cuda', dtype=torch.bfloat16) if model.device.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()


def device_name(requested='auto'):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install the CUDA PyTorch build or select cpu')
    if requested not in ('auto', 'cuda', 'cpu'):
        raise ValueError('Use auto, cuda or cpu')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def load_base(device='auto', offline=False):
    device = device_name(device)
    torch.set_num_threads(4)
    from download_model import BASE_PATH, sha256, FILES
    if (BASE_PATH / 'source-manifest.json').exists():
        manifest = json.loads((BASE_PATH / 'source-manifest.json').read_text(encoding='utf-8'))
        if manifest['revision'] != REVISION or manifest['model_id'] != MODEL_ID or not {'config.json', 'tokenizer.json', 'model.safetensors'} <= set(manifest['files']) or not set(manifest['files']) <= FILES:
            raise ValueError('Local model checksum/revision mismatch')
        for name, info in manifest['files'].items():
            if sha256(BASE_PATH / name) != info['sha256']:
                raise ValueError('Local model file checksum mismatch')
        source = str(BASE_PATH)
    elif not offline:
        from download_model import download
        source = str(download())
    else:
        raise ValueError('Run download_model.py before offline inference')
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, local_files_only=True).to(device)
    model.eval()
    return tokenizer, model, device


def load_adapter(path, device='auto', offline=True):
    from peft import PeftModel
    path = Path(path)
    config = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path') != MODEL_ID or config.get('revision') != REVISION:
        raise ValueError('Adapter is not for the pinned base model')
    tokenizer, model, device = load_base(device, offline)
    model = PeftModel.from_pretrained(model, path, is_trainable=False, local_files_only=True)
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, texts, max_new_tokens=96):
    prompts = [tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True) for text in texts]
    batch = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False)
    if batch['input_ids'].shape[1] > 320:
        raise ValueError('Prompt exceeds 320 tokens; shorten the request. No silent truncation.')
    batch = {key: value.to(model.device) for key, value in batch.items()}
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), amp_context(model):
        output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    completion = output[:, batch['input_ids'].shape[1]:]
    return tokenizer.batch_decode(completion, skip_special_tokens=True), elapsed


def infer(tokenizer, model, text):
    outputs, elapsed = generate(tokenizer, model, [text])
    return {'raw_output': outputs[0], 'validation': parse_output(outputs[0], text), 'seconds': elapsed,
            'scope': 'Model output is not repaired. Invalid schema or unsupported order IDs require human review.'}
