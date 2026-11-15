import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from train import encode, collate


class TrainingTests(unittest.TestCase):
    def test_completion_only_labels_and_padding(self):
        class Tokenizer:
            eos_token = '<eos>'
            def apply_chat_template(self, *_args, **_kwargs):
                return [1, 2, 3]
            def encode(self, *_args, **_kwargs):
                return [4, 5]
        row = encode(Tokenizer(), {'text': 'Refund please', 'target': {'intent': 'refund'}})
        self.assertEqual(row['labels'], [-100, -100, -100, 4, 5])
        short = {'input_ids': [1, 2], 'labels': [-100, 2]}
        batch = collate([row, short], 0, 'cpu')
        self.assertEqual(batch['labels'][1].tolist(), [-100, 2, -100, -100, -100])
        self.assertEqual(batch['attention_mask'][1].tolist(), [1, 1, 0, 0, 0])

    def test_real_lora_optimizer_step_on_tiny_random_model(self):
        # Architectural integration only; this is not the pretrained-model benchmark.
        torch.manual_seed(27)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        model = get_peft_model(LlamaForCausalLM(config), LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj']))
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all('lora_' in n for n, _ in trainable))
        old = [p.detach().clone() for _, p in trainable]
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=.01)
        ids = torch.randint(0, 64, (2, 16))
        labels = ids.clone()
        labels[:, :8] = -100
        model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before, p) for before, (_, p) in zip(old, trainable)))
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(frozen[name], p))
