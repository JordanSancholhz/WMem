"""CPU checks for the known-state objective, quality gates and label-service failures."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import torch
from tensordict import TensorDict

from recurrent.future_prediction import build_prediction_tensors, backward_prediction_minibatch, validate_prediction_config
from recurrent.future_state_labels import FrozenStateLabeler, memory_units, parse_labels

ROOT = Path(__file__).resolve().parents[1]
CURRENT = "The user enjoys science fiction films.\nThe user prefers short films only while commuting.\nThe user likes cooking elaborate dinners."
FOLLOWING = "The user enjoys science fiction films.\nThe user prefers short films in all situations.\nFUTURE_ONLY_NEW_FACT: the user has a peanut allergy."


def config(**kwargs):
    result = dict(enabled=True, mode="known_state", coefficient=0.002, warmup_steps=20,
                  micro_batch_size_per_gpu=1, max_prompt_tokens=4096, max_target_tokens=1024,
                  quality_gate=True, min_final_reward=1.0, min_memory_quality=0.7,
                  max_pairs_per_step=8, max_units_per_pair=3, max_unit_chars=400,
                  max_label_prompt_tokens=4096, label_max_tokens=768, label_concurrency=4,
                  label_timeout=30, audit_pairs_per_step=2)
    result.update(kwargs)
    return result


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0

    def __init__(self):
        self.prompts = []

    def decode(self, ids, **kwargs):
        words = {11: CURRENT, 12: FOLLOWING, 21: CURRENT, 22: FOLLOWING,
                 31: "FINAL_ANSWER_A", 32: "FINAL_ANSWER_B", 101: "CURRENT_CONTEXT",
                 102: "FUTURE_INPUT_MUST_NOT_LEAK"}
        return " ".join(words.get(i, "") for i in ids if i not in (0, 2)).strip()

    def encode(self, text, **kwargs):
        return [{"A": 4, "B": 5, "C": 6}[text]]

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return [7, 8, 9]


def rollout():
    responses = torch.tensor([[11,2],[21,2],[12,2],[22,2],[31,2],[32,2]])
    ids = torch.cat([torch.tensor([[101],[101],[102],[102],[102],[102]]), responses],dim=1)
    return dict(input_ids=ids, responses=responses, attention_mask=torch.ones_like(ids),
                intermediate_rewards=torch.tensor([0.8,0.8,0.8,0.8,0.,0.]))


class Labeler:
    def __init__(self, error=None, uncertain=False):
        self.records, self.error, self.uncertain = [], error, uncertain

    def label_batch(self, records):
        self.records = records
        out = []
        for record in records:
            if self.error:
                out.append({"error": self.error})
                continue
            labels = []
            for i, unit in enumerate(record["units"]):
                if self.uncertain:
                    status, quote = "uncertain", ""
                elif "science" in unit:
                    status, quote = "preserved", "The user enjoys science fiction films."
                elif "commuting" in unit:
                    status, quote = "revised", "The user prefers short films in all situations."
                else:
                    status, quote = "absent", ""
                labels.append(dict(id=i,status=status,evidence=quote))
            out.append({"labels": labels})
        return out


class Actor:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(0.3))
        self.config = SimpleNamespace(future_prediction=SimpleNamespace(micro_batch_size_per_gpu=1))
        self.calls = 0

    def _forward_micro_batch(self, micro, **kwargs):
        self.calls += 1
        return None, torch.nn.functional.logsigmoid(self.parameter * micro['responses'].float())


class KnownStateTests(unittest.TestCase):
    def build(self, *, data=None, scores=None, cfg=None, labeler=None, audit=None):
        self.tokenizer = Tokenizer()
        self.labeler = labeler or Labeler()
        return build_prediction_tensors(
            data if data is not None else rollout(), [0,0,0,0,1,1], [0,1,0,1,0,1],
            self.tokenizer, cfg or config(), world_size=8, step=20,
            final_scores=torch.tensor([1.,0.]) if scores is None else scores,
            labeler=self.labeler, audit_path=audit)

    def test_only_state_labels_are_targets_and_future_evidence_never_enters_prompt(self):
        tensors, meta, stats = self.build()
        self.assertEqual(stats['future_prediction/used_pairs'],1)
        self.assertEqual(stats['future_prediction/training_examples'],3)
        self.assertEqual(stats['future_prediction/dropped_quality'],1)
        self.assertEqual(sorted(tensors['responses'][:3,0].tolist()),[4,5,6])
        self.assertEqual(meta['prediction_valid_counts'],[3])
        self.assertEqual(meta['prediction_coefficient'],0.002)
        for prompt in self.tokenizer.prompts:
            self.assertIn('CURRENT_CONTEXT',prompt)
            self.assertIn('only while commuting',prompt)
            for forbidden in ['FUTURE_ONLY_NEW_FACT','FUTURE_INPUT_MUST_NOT_LEAK','FINAL_ANSWER',
                              'in all situations','following_memory','evidence','scores']:
                self.assertNotIn(forbidden,prompt)

    def test_raw_answer_and_both_memory_quality_gates(self):
        for field,row in [('quality',0),('quality',2),('answer',0)]:
            data, scores = rollout(), torch.tensor([1.,0.])
            if field=='quality':data['intermediate_rewards'][row]=0.4
            else:scores[row]=0.0
            _, meta, stats = self.build(data=data,scores=scores)
            self.assertEqual(meta['prediction_valid_counts'],[0])
            self.assertEqual(self.labeler.records,[])
        _, _, stats = self.build(cfg=config(quality_gate=False))
        self.assertEqual(stats['future_prediction/used_pairs'],2)
        # Gate must be wired before the final score is mixed with guideline reward.
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        build=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='build_prediction_tensors')
        combine=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='_combine_recurrent_intermediate_reward')
        self.assertLess(build.lineno,combine.lineno)
        score_arg=next(k.value for k in build.keywords if k.arg=='final_scores')
        self.assertEqual(ast.unparse(score_arg),'reward_tensor.sum(-1)')

    def test_failed_or_uncertain_labels_skip_globally_without_rank_collectives(self):
        for lab in [Labeler(error='label_service_error'),Labeler(uncertain=True)]:
            tensors, meta, stats = self.build(labeler=lab)
            self.assertEqual(meta['prediction_valid_counts'],[0])
            self.assertEqual(tensors['prediction_loss_mask'].sum().item(),0)
            for chunk in TensorDict(tensors,batch_size=[8]).chunk(8):
                actor=Actor()
                backward_prediction_minibatch(actor,SimpleNamespace(batch=chunk,meta_info=meta),
                                             minibatch_index=0,device='cpu')
                self.assertEqual(actor.calls,0)
                self.assertIsNone(actor.parameter.grad)

    def test_known_state_eight_rank_gradients_equal_global_reference(self):
        tensors, meta, _ = self.build()
        batch=TensorDict(tensors,batch_size=[8])
        grads, counts=[],[]
        for shard in batch.chunk(8):
            actor=Actor()
            backward_prediction_minibatch(actor,SimpleNamespace(batch=shard,meta_info=meta),
                                         minibatch_index=0,device='cpu')
            grads.append(actor.parameter.grad);counts.append(actor.calls)
        self.assertEqual(len(set(counts)),1)
        reference=Actor()
        _, lp=reference._forward_micro_batch(batch)
        mask=batch['prediction_loss_mask']
        loss=-(lp*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        (0.002*loss.sum()/3).backward()
        torch.testing.assert_close(torch.stack(grads).mean(),reference.parameter.grad)

    def test_current_and_following_truncation_not_used_as_labels(self):
        for row,key in [(0,'dropped_current_unterminated'),(2,'dropped_unterminated')]:
            data=rollout();data['responses'][row,-1]=99
            _,meta,stats=self.build(data=data)
            self.assertEqual(stats['future_prediction/'+key],1)
            self.assertEqual(meta['prediction_valid_counts'],[0])

    def test_label_quotes_ids_and_paraphrase_contract(self):
        record=dict(units=['User likes science fiction.'],following_memory='Science fiction appeals to this user.')
        good={'labels':[dict(id=0,status='preserved',evidence=record['following_memory'])]}
        self.assertEqual(parse_labels(json.dumps(good),record)[0]['status'],'preserved')
        for bad in [{'labels':[dict(id=0,status='preserved',evidence='fabricated')]},
                    {'labels':[dict(id=0,status='absent',evidence='fabricated')]},
                    {'labels':[dict(id=True,status='absent',evidence='')]},
                    {'labels':[]}, {'labels':[dict(id=0,status='invented',evidence='')]}]:
            with self.assertRaises(ValueError):parse_labels(json.dumps(bad),record)

    def test_unit_selection_preserves_conditions_is_deterministic_and_bounded(self):
        text='# Header\n- The user likes short films; only when commuting.\n'+('x'*500)
        first=memory_units(text,3,400,'seed')
        self.assertEqual(first,['The user likes short films; only when commuting.'])
        self.assertEqual(first,memory_units(text,3,400,'seed'))
        _,_,stats=self.build(cfg=config(max_pairs_per_step=1),scores=torch.tensor([1.,1.]))
        self.assertEqual(len(self.labeler.records),1)
        self.assertEqual(stats['future_prediction/budget_skipped_pairs'],1)

    def test_audit_contains_labels_without_changing_training_inputs(self):
        # Test serialized audit contents independently of OS temporary-dir ACLs.
        writer=mock_open()
        with patch.object(Path,'mkdir'), patch.object(Path,'open',writer):
            self.build(audit=ROOT/'tmp'/'audit.jsonl')
            audit=json.loads(writer().write.call_args.args[0])
            self.assertEqual(audit['step'],20)
            self.assertIn('FUTURE_ONLY_NEW_FACT',audit['pairs'][0]['following_memory'])
            self.assertEqual(len(audit['pairs'][0]['labels']),3)
            self.assertTrue(all('FUTURE_ONLY_NEW_FACT' not in s for s in self.tokenizer.prompts))

    def test_http_payload_and_truncated_invalid_or_unavailable_service(self):
        memory=SimpleNamespace(intermediate_reward_model='local-qwen',
                               intermediate_reward_base_url='http://127.0.0.1:6025/v1',
                               intermediate_reward_api_key_env='TEST_STATE_KEY')
        labeler=FrozenStateLabeler(config(),memory,Tokenizer())
        record=dict(current_memory=CURRENT,following_memory=FOLLOWING,
                    units=['The user enjoys science fiction films.'])
        content=json.dumps({'labels':[dict(id=0,status='preserved',evidence=record['units'][0])]})
        def reply(reason='stop',text=content):
            return io.BytesIO(json.dumps({'choices':[{'finish_reason':reason,'message':{'content':text}}]}).encode())
        with patch.dict('os.environ',{'TEST_STATE_KEY':'test'}):
            with patch('urllib.request.urlopen',return_value=reply()) as http:
                self.assertIn('labels',labeler.label_one(record))
                payload=json.loads(http.call_args.args[0].data)
                self.assertEqual(payload['model'],'local-qwen')
                self.assertEqual(payload['temperature'],0)
                self.assertEqual(http.call_args.kwargs['timeout'],30)
            with patch('urllib.request.urlopen',return_value=reply('length')):
                self.assertEqual(labeler.label_one(record),{'error':'label_truncated'})
            with patch('urllib.request.urlopen',return_value=reply(text='not json')):
                self.assertEqual(labeler.label_one(record),{'error':'label_invalid'})
            with patch('urllib.request.urlopen',side_effect=TimeoutError):
                self.assertEqual(labeler.label_one(record),{'error':'label_service_error'})

    def test_invalid_settings_and_missing_gate_inputs_fail_explicitly(self):
        kwargs=dict(recurrent='memory',strategy='fsdp',sequence_parallel=1,train_batch=8,mini_batch=8)
        validate_prediction_config(config(),**kwargs)
        for values in [dict(mode='bad'),dict(min_final_reward=2),dict(label_timeout=float('nan')),
                       dict(label_concurrency=0),dict(max_pairs_per_step=0)]:
            with self.assertRaises(ValueError):validate_prediction_config(config(**values),**kwargs)
        data=rollout();data.pop('intermediate_rewards')
        with self.assertRaises(ValueError):self.build(data=data)


if __name__=='__main__':unittest.main()
