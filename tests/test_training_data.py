import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import train_lora_local
from train_lora_local import TaraShareGPTDataset


class Tokenizer:
    pad_token = "<pad>"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ''.join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        return text + ('<assistant>' if add_generation_prompt else '')

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(text.encode('utf-8'))}


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'dataset.json'
        self.tokenizer = Tokenizer()

    def row(self, ident='short', answer='answer-END', prompt='question'):
        return {'id': ident, 'task_type': 'tara_rampup', 'system': 'rules',
                'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]}

    def dataset(self, rows, **kwargs):
        self.path.write_text(json.dumps(rows), encoding='utf-8')
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return TaraShareGPTDataset(str(self.path), self.tokenizer, **kwargs)
        finally:
            self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_complete_response_and_chat_markers_are_preserved(self):
        data = self.dataset([self.row()])[0]
        prefix = b'<system>rules</end><user>question</end><assistant>'
        full = prefix + b'answer-END</end>'
        self.assertEqual(data['input_ids'], list(full))
        self.assertEqual(data['labels'], [-100] * len(prefix) + list(full[len(prefix):]))
        self.assertEqual(len(data['attention_mask']), len(full))

    def test_overlong_answer_blocks_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'max_length'):
            self.dataset([self.row(answer='X' * 500)], max_length=100)

    def test_overlong_prompt_cannot_be_relabelled_as_answer(self):
        with self.assertRaisesRegex(ValueError, 'max_length'):
            self.dataset([self.row(prompt='X' * 500)], max_length=100)

    def test_explicit_skip_only_filters_in_memory(self):
        dataset = self.dataset([self.row(), self.row('long', answer='X' * 500)], max_length=100, overlength='skip')
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.report['overlength'], 1)
        self.assertEqual(dataset.report['by_task']['tara_rampup']['overlength'], 1)
        self.assertEqual(dataset[0]['input_ids'][-len(b'answer-END</end>'):], list(b'answer-END</end>'))

    def test_no_usable_data_is_an_error(self):
        with self.assertRaises(ValueError):
            self.dataset([self.row(answer='X' * 500)], max_length=100, overlength='skip')

    def test_empty_answer_and_multi_turn_are_not_silently_accepted(self):
        for row in [self.row(answer=''), {**self.row(), 'conversations': self.row()['conversations'] * 2}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.dataset([row])

    def test_template_prefix_mismatch_blocks_training(self):
        original = self.tokenizer.apply_chat_template
        self.tokenizer.apply_chat_template = lambda messages, **kw: original(messages, **kw) + ('wrong' if kw.get('add_generation_prompt') else '')
        with self.assertRaises(ValueError):
            self.dataset([self.row()])

    def test_exact_length_is_accepted(self):
        size = len(self.dataset([self.row()])[0]['input_ids'])
        self.assertEqual(len(self.dataset([self.row()], max_length=size)), 1)

    def test_preflight_never_loads_model_and_bad_data_blocks_before_loading(self):
        for check_only, answer in [(True, 'answer-END'), (False, 'X' * 500)]:
            self.path.write_text(json.dumps([self.row(answer=answer)]), encoding='utf-8')
            args = ['train', '--dataset_path', str(self.path), '--max_length', '100']
            if check_only:
                args.append('--check-only')
            with self.subTest(check_only=check_only), patch.object(sys, 'argv', args), \
                    patch.object(train_lora_local.AutoTokenizer, 'from_pretrained', return_value=self.tokenizer), \
                    patch.object(train_lora_local.AutoModelForCausalLM, 'from_pretrained') as model, \
                    contextlib.redirect_stdout(io.StringIO()):
                if check_only:
                    train_lora_local.train()
                else:
                    with self.assertRaises(ValueError):
                        train_lora_local.train()
                model.assert_not_called()

    def test_collator_mutation_cannot_change_cached_samples(self):
        dataset = self.dataset([self.row()])
        expected = dataset[0]
        changed = dataset[0]
        changed['input_ids'].clear()
        changed['labels'].clear()
        self.assertEqual(dataset[0], expected)


if __name__ == '__main__':
    unittest.main()
