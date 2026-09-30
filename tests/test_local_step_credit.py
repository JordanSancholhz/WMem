"""CPU checks of local credit math, real driver hooks and sharded scoring plan."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from recurrent.future_prediction import build_prediction_tensors
from recurrent.local_step_credit import (
    compute_local_credit, apply_local_credit, score_state_nll, validate_local_credit_config,
)

ROOT = Path(__file__).resolve().parents[1]


def examples(records, dummy=0):
    records = records + [(-1, -1, -1, -1)] * dummy
    keys = ('prediction_current_row', 'prediction_following_row',
            'prediction_sample_index', 'prediction_target_class')
    out = {key: torch.tensor([r[j] for r in records], dtype=torch.long) for j, key in enumerate(keys)}
    out['prediction_loss_mask'] = torch.tensor([[float(r[0] >= 0)] * 2 for r in records]).reshape(-1, 2)
    return out


class CreditMathTests(unittest.TestCase):
    def test_hand_calculated_pair_and_missing_weight_one(self):
        # Mean guideline=.5; w=[1,1.08,1], d=[.4,-.324,-.1], mean d=-.008.
        g = torch.tensor([.9, .2, .4, float('nan')], requires_grad=True)
        data = examples([(0, 1, 0, 0)])
        delta, metrics, audit = compute_local_credit(
            g, [0, 0, 0, 1], [0]*4, prediction_tensors=data,
            state_nll=torch.tensor([-math.log(.8)], requires_grad=True), coefficient=.05, state_strength=.1)
        torch.testing.assert_close(delta, torch.tensor([.0204, -.0158, -.0046, 0.]))
        self.assertFalse(delta.requires_grad)
        self.assertIsNone(audit[0]['state_nll'])
        self.assertEqual(audit[0]['weight'], 1.)
        self.assertAlmostEqual(audit[1]['weight'], 1.08, places=6)
        self.assertEqual(metrics['local_credit/scored_transitions'], 1)
        self.assertLess(metrics['local_credit/center_residual_max'], 1e-7)

    def test_interleaved_unequal_trajectories_and_duplicate_units(self):
        # Same question's two rollouts still have separate baselines.
        final = [0, 0, 0, 1, 0, 1]
        samples = [0, 1, 0, 1, 0, 0]
        data = examples([(0, 2, 0, 0), (0, 2, 0, 1)], dummy=2)
        delta, _, audit = compute_local_credit([.9, .3, .2, 0, .4, 0], final, samples,
            prediction_tensors=data, state_nll=torch.tensor([-math.log(.9), -math.log(.4), float('nan'), 0.]))
        # exp(mean log p)=sqrt(.9*.4)=.6, not arithmetic mean .65.
        self.assertAlmostEqual(next(r['weight'] for r in audit if r['row'] == 2), 1.06, places=6)
        self.assertEqual(delta[1], 0.)
        self.assertEqual(delta[3], 0.)
        self.assertEqual(delta[5], 0.)
        self.assertAlmostEqual(float(delta[[0,2,4]].sum()), 0., places=7)
        perm = torch.tensor([3, 1, 2, 0])
        other, _, _ = compute_local_credit([.9, .3, .2, 0, .4, 0], final, samples,
            prediction_tensors={k:v[perm] for k,v in data.items()},
            state_nll=torch.tensor([-math.log(.9), -math.log(.4), float('nan'), 0.])[perm])
        torch.testing.assert_close(other, delta)

    def test_eta_zero_and_empty_labels_reduce_to_local_guideline(self):
        expected = torch.tensor([.015, -.015, 0.])
        for kwargs in ({'state_strength':0., 'prediction_tensors':examples([(0,1,0,0)])},
                       {'prediction_tensors':examples([], dummy=8)}, {}):
            result, _, _ = compute_local_credit([.9,.3,0.], [0,0,1], [0,0,0], **kwargs)
            torch.testing.assert_close(result, expected)

    def test_invalid_loss_abstains_and_identical_guidelines_give_zero(self):
        data = examples([(0,1,0,0), (0,1,0,1), (1,2,0,2)])
        result, metrics, _ = compute_local_credit([.8,.8,.8,0.], [0,0,0,1], [0]*4,
            prediction_tensors=data, state_nll=torch.tensor([float('nan'), -1., .4]))
        self.assertTrue(torch.equal(result, torch.zeros(4)))
        self.assertEqual(metrics['local_credit/invalid_units'], 2)
        self.assertEqual(metrics['local_credit/scored_transitions'], 1)

    def test_zero_lambda_is_identity_even_with_valid_labels_and_no_rpc(self):
        delta, _, _ = compute_local_credit([.9,.3,0.], [0,0,1], [0]*3,
            prediction_tensors=examples([(0,1,0,0)]), coefficient=0.)
        base = torch.tensor([.2,.2,.2])
        result, _ = apply_local_credit(base, delta, torch.tensor([0,0,1]))
        self.assertTrue(torch.equal(result, base))

    def test_bad_mapping_or_missing_scores_fails(self):
        for records in ([(0,2,0,0)], [(0,1,4,0)], [(0,1,0,4)]):
            with self.assertRaises(ValueError):
                compute_local_credit([.9,.3,0.], [0,0,1], [0]*3,
                    prediction_tensors=examples(records), state_nll=torch.tensor([.3]))
        with self.assertRaises(ValueError):
            compute_local_credit([.9,.3,0.], [0,0,1], [0]*3,
                prediction_tensors=examples([(0,1,0,0)]))
        with self.assertRaises(ValueError):
            apply_local_credit(torch.ones(3), torch.ones(3), torch.tensor([0,0,1]))

    def test_config_guards_and_method8_compatibility(self):
        cfg = dict(enabled=True, mode='known_state', gradient_alignment=dict(enabled=True),
                   local_credit=dict(enabled=True))
        validate_local_credit_config(cfg)
        for bad in (dict(enabled=False), dict(mode='full_memory'),
                    dict(credit_weighting=dict(enabled=True)),
                    dict(local_credit=dict(enabled=True, coefficient=.2)),
                    dict(local_credit=dict(enabled=True, state_strength=float('nan')))):
            with self.assertRaises(ValueError):
                validate_local_credit_config({**cfg, **bad})


class PackingAndScoringTests(unittest.TestCase):
    def fixture(self):
        spec = importlib.util.spec_from_file_location('local_fixture', ROOT/'tests/test_future_state_labels.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_method8_prompts_targets_and_labels_unchanged(self):
        f = self.fixture()
        outputs, prompts = [], []
        for enabled in (False, True):
            tokenizer = f.Tokenizer()
            outputs.append(build_prediction_tensors(f.rollout(), [0,0,0,0,1,1], [0,1,0,1,0,1],
                tokenizer, f.config(local_credit=dict(enabled=enabled)), world_size=8, step=20,
                final_scores=torch.tensor([1.,0.]), labeler=f.Labeler()))
            prompts.append(tokenizer.prompts)
        (base, meta, stats), (local, local_meta, local_stats) = outputs
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(stats, local_stats)
        for key, value in base.items():
            self.assertTrue(torch.equal(value, local[key]), key)
        self.assertEqual(local['prediction_following_row'].tolist(), [2]*3+[-1]*5)
        for key, value in meta.items():
            self.assertEqual(value, local_meta[key])

    class Actor:
        ulysses_sequence_parallel_size = 1
        def __init__(self):
            self.actor_module = torch.nn.Linear(1, 1)
            self.config = SimpleNamespace(future_prediction=SimpleNamespace(micro_batch_size_per_gpu=1))
            self.calls = 0
        def _forward_micro_batch(self, micro, **kwargs):
            self.calls += 1
            assert not torch.is_grad_enabled()
            assert kwargs == dict(temperature=1., calculate_entropy=False)
            # Second column simulates a very different EOS loss.
            return None, -micro['responses'].float()

    def test_eight_rank_plan_has_equal_forwards_and_retains_original_score_order(self):
        f = self.fixture()
        tensors, meta, _ = build_prediction_tensors(f.rollout(), [0,0,0,0,1,1], [0,1,0,1,0,1],
            f.Tokenizer(), f.config(local_credit=dict(enabled=True)), world_size=8, step=20,
            final_scores=torch.tensor([1.,0.]), labeler=f.Labeler())
        batch = TensorDict(tensors, batch_size=[8])
        scores, calls = [], []
        for shard in batch.chunk(8):
            actor = self.Actor()
            score = score_state_nll(actor, SimpleNamespace(batch=shard, meta_info=meta))
            scores.append(score)
            calls.append(actor.calls)
            self.assertTrue(actor.actor_module.training)
            self.assertFalse(score.requires_grad)
        self.assertEqual(calls, [1]*8) # Five dummy-only ranks participate.
        torch.testing.assert_close(torch.cat(scores), batch['responses'][:,0].float())
        actor = self.Actor()
        result = score_state_nll(actor, SimpleNamespace(batch=batch[:1], meta_info={'prediction_valid_counts':[0]}))
        self.assertEqual(actor.calls, 0)
        self.assertEqual(result.item(), 0.)

    def test_real_actor_forward_scores_causal_state_token_not_eos_or_abc_softmax(self):
        tree = ast.parse((ROOT/'verl/workers/actor/dp_actor.py').read_text(encoding='utf-8'))
        method = deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                               and n.name=='_forward_micro_batch'))
        method.returns = None
        def unpad(values, mask):
            indices = mask.flatten().nonzero().squeeze(-1)
            return values.reshape(-1, values.shape[-1])[indices], indices, None, None
        def pad(hidden_states, indices, batch, seqlen):
            out = hidden_states.new_zeros((batch*seqlen,hidden_states.shape[-1]))
            out[indices] = hidden_states
            return out.reshape(batch,seqlen,-1)
        def logprobs(logits, labels, **kwargs):
            return logits.float().log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        ns = dict(torch=torch,unpad_input=unpad,pad_input=pad,logprobs_from_logits=logprobs,
                  rearrange=lambda x,pattern:x.reshape(-1,x.shape[-1]),index_first_axis=lambda x,i:x[i])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])), 'actual_forward','exec'),ns)
        class Model(torch.nn.Module):
            def forward(self,input_ids,**kwargs):
                x = input_ids.float()
                # Non-ABC vocabulary mass changes full-vocabulary NLL.
                return SimpleNamespace(logits=torch.stack([x*0, x, -x, 2*x, x*0+9],dim=-1))
        actor_type = type('RealForwardActor',(),{'_forward_micro_batch':ns['_forward_micro_batch']})
        ids = torch.tensor([[0,1,2,1,4],[0,0,3,2,4]])
        mask = (ids!=0).long()
        batch = TensorDict(dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),
                                responses=ids[:,-2:].clone()),batch_size=[2])
        expected = -torch.log_softmax(torch.tensor([[0.,2.,-2.,4.,9.],[0.,3.,-3.,6.,9.]]),-1)[[0,1],[1,2]]
        for remove in (False,True):
            actor = actor_type()
            actor.actor_module=Model()
            actor.use_remove_padding=remove
            actor.use_ulysses_sp=False
            actor.ulysses_sequence_parallel_size=1
            actor.config=SimpleNamespace(future_prediction=SimpleNamespace(micro_batch_size_per_gpu=2))
            data=SimpleNamespace(batch=batch.clone(),meta_info={'prediction_valid_counts':[2]})
            with patch.object(torch,'autocast',side_effect=lambda **kwargs:nullcontext()):
                actual=score_state_nll(actor,data)
                data.batch['input_ids'][:,-1]=0
                data.batch['responses'][:,-1]=0
                changed_eos=score_state_nll(actor,data)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(changed_eos,expected)


class DriverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
        cls.fit = next(n for n in ast.walk(cls.tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit')

    def execute(self, nodes, namespace):
        program = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
        exec(compile(program, 'real_local_credit_driver_hooks', 'exec'), namespace)

    def test_real_prepare_hook_skips_scoring_for_ablation_and_empty_labels(self):
        block = next(n for n in ast.walk(self.fit) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'local_credit_enabled' and 'compute_local_credit' in ast.unparse(n))
        for coefficient, eta, count, expected_calls in ((.05,.1,1,1), (.05,0.,1,0), (0.,.1,1,0), (.05,.1,0,0)):
            events = []
            def rpc(data):
                events.append('score')
                self.assertEqual(data.meta_info['prediction_score_kind'], 'state_nll')
                return SimpleNamespace(batch={'prediction_state_nll':torch.tensor([.2])})
            prediction = examples([(0,1,0,0)]) if count else examples([],dummy=1)
            batch = SimpleNamespace(batch={'intermediate_rewards':torch.tensor([.9,.3,0.]),
                                          'responses':torch.ones(3,2,dtype=torch.long)})
            ns = dict(local_credit_enabled=True, local_credit_config=dict(coefficient=coefficient,state_strength=eta),
                      prediction_meta={'prediction_valid_counts':[count]},
                      prediction_batch=SimpleNamespace(batch=prediction, meta_info={}), batch=batch,
                      final_mask=torch.tensor([0,0,1],dtype=torch.bool), sample_index=torch.tensor([0]*3),
                      _timer=lambda *args:nullcontext(), timing_raw={}, metrics={}, os=os,
                      self=SimpleNamespace(actor_rollout_wg=SimpleNamespace(compute_prediction_scores=rpc),
                                           global_steps=1, config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir='unused'))))
            with patch('recurrent.local_step_credit.append_local_audit') as audit:
                self.execute([block],ns)
            self.assertEqual(len(events),expected_calls)
            self.assertEqual(audit.call_count,1)
            self.assertEqual(batch.batch['local_advantage_delta'][-1],0.)
        # Score/attach happen before padding, and adjustment follows unpadding.
        calls = [n for n in ast.walk(self.fit) if isinstance(n,ast.Call)]
        unpad = min(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='unpad_dataproto')
        apply_line = next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='apply_local_credit')
        self.assertLess(block.lineno,unpad)
        self.assertLess(unpad,apply_line)

    def test_real_advantage_hook_and_mask_keep_final_reward_unchanged(self):
        block = next(n for n in ast.walk(self.fit) if isinstance(n,ast.If)
                     and ast.unparse(n.test)=='local_credit_enabled' and 'apply_local_credit' in ast.unparse(n))
        assignments = {ast.unparse(n.targets[0]):n for n in ast.walk(self.fit) if isinstance(n,ast.Assign)}
        delta = torch.tensor([.015,-.015,0.])
        reward = torch.tensor([.8,.8,.8])
        batch = SimpleNamespace(batch={'local_advantage_delta':delta, 'responses':torch.ones(3,3),
                                      'response_mask':torch.tensor([[1,0,0],[1,1,0],[1,1,1]]),
                                      'token_level_rewards':reward.clone()})
        ns = dict(local_credit_enabled=True,advantage_scalar=torch.tensor([.2,.2,.2]),
                  batch=batch,final_mask=torch.tensor([0,0,1]),metrics={})
        nodes = [block]+[assignments[key] for key in ('response_length','eos_mask','advantages',"batch.batch['advantages']")]
        self.execute(nodes,ns)
        torch.testing.assert_close(batch.batch['advantages'],torch.tensor([[.215,0,0],[.185,.185,0],[.2,.2,.2]]))
        self.assertTrue(torch.equal(batch.batch['token_level_rewards'],reward))

    def test_correction_survives_actual_padding_and_unpadding(self):
        class Proto:
            def __init__(self,batch): self.batch=batch
            def __len__(self): return len(self.batch)
            def __getitem__(self,index): return Proto(self.batch[index])
            @staticmethod
            def concat(items): return Proto(torch.cat([item.batch for item in items],dim=0))
        tree=ast.parse((ROOT/'verl/protocol.py').read_text(encoding='utf-8'))
        helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in ('pad_dataproto_to_divisor','unpad_dataproto')]
        ns={'DataProto':Proto,'torch':torch}
        self.execute(helpers,ns)
        source=Proto(TensorDict({'original_row':torch.arange(11),
                                'local_advantage_delta':torch.arange(11).float()/100},batch_size=[11]))
        padded,size=ns['pad_dataproto_to_divisor'](source,8)
        # Simulate existing rank-split / gathered row order.
        gathered=Proto.concat([Proto(shard) for shard in padded.batch.chunk(8)])
        restored=ns['unpad_dataproto'](gathered,size)
        for key in source.batch.keys():
            self.assertTrue(torch.equal(restored.batch[key],source.batch[key]))


if __name__ == '__main__':
    unittest.main()
