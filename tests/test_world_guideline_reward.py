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

from recurrent.world_guideline_reward import compute_world_reward, apply_world_reward, validate_world_reward_config
from recurrent.training_diagnostics import format_training_diagnostics
from recurrent.future_prediction import build_prediction_tensors

ROOT=Path(__file__).resolve().parents[1]


def labels(records, dummy=0):
    records=list(records)+[(-1,-1,-1,-1)]*dummy
    names=('prediction_current_row','prediction_following_row','prediction_sample_index','prediction_target_class')
    out={k:torch.tensor([r[j] for r in records],dtype=torch.long) for j,k in enumerate(names)}
    out['prediction_loss_mask']=torch.tensor([[float(r[0]>=0)]*2 for r in records]).reshape(-1,2)
    return out


class WorldRewardTests(unittest.TestCase):
    def test_normalized_weighted_reward_matches_hand_calculation(self):
        # g=[.9,.2], w=[1,1.08]; low-quality, predictable update reduces R.
        delta, metrics, audit=compute_world_reward([.9,.2,99.],[0,0,1],[0]*3,num_samples=1,
            prediction_tensors=labels([(0,1,0,0)]),state_nll=torch.tensor([-math.log(.8)]))
        weighted=(.9+1.08*.2)/2.08
        self.assertAlmostEqual(float(delta[0]),.5*(weighted-.55),places=7)
        self.assertAlmostEqual(audit['trajectories'][0]['weighted_guideline'],weighted,places=7)
        self.assertEqual(audit['memory_turns'][0]['weight'],1.)
        self.assertIsNone(audit['memory_turns'][0]['state_nll'])
        self.assertLess(metrics['world_reward/world_reward_mean'],0.)
        self.assertFalse(delta.requires_grad)

    def test_predictable_good_step_increases_reward_and_uniform_guidelines_do_not(self):
        args=dict(num_samples=1,prediction_tensors=labels([(0,1,0,0)]),state_nll=torch.tensor([0.]))
        delta,_,_=compute_world_reward([.2,.9,0.],[0,0,1],[0]*3,**args)
        self.assertGreater(float(delta[0]),0.)
        zero,_,_=compute_world_reward([.9,.9,0.],[0,0,1],[0]*3,**args)
        self.assertEqual(float(zero[0]),0.)

    def test_eta_zero_gamma_zero_or_missing_labels_are_exact_baseline(self):
        for kw in (dict(state_strength=0.,prediction_tensors=labels([(0,1,0,0)])),
                   dict(coefficient=0.,prediction_tensors=labels([(0,1,0,0)])),
                   dict(prediction_tensors=labels([],dummy=8)),
                   dict(prediction_tensors=labels([(0,1,0,0)]),state_nll=torch.tensor([float('nan')]))):
            delta,_,_=compute_world_reward([.2,.9,0.],[0,0,1],[0]*3,num_samples=1,**kw)
            self.assertTrue(torch.equal(delta,torch.zeros(1)))
            old=torch.tensor([[0.,.775,0.]])
            new,_=apply_world_reward(old,delta,torch.tensor([1]))
            self.assertTrue(torch.equal(old,new))

    def test_interleaved_samples_unit_mean_and_dummy_scores_preserve_alignment(self):
        # One prediction has two units; another trajectory has no prediction.
        kwargs=dict(num_samples=2,prediction_tensors=labels([(0,2,0,0),(0,2,0,1)],dummy=2),
                    state_nll=torch.tensor([-math.log(.9),-math.log(.4),float('nan'),-99.]))
        delta,metrics,audit=compute_world_reward([.2,.5,.9,0.,0.],[0,0,0,1,1],[0,1,0,1,0],**kwargs)
        self.assertEqual(float(delta[1]),0.)
        self.assertAlmostEqual(audit['memory_turns'][-1]['weight'],1.,places=6) # sample 1
        row=next(r for r in audit['memory_turns'] if r['row']==2)
        self.assertAlmostEqual(row['weight'],1.06,places=6)
        self.assertEqual(metrics['world_reward/scored_units'],2)
        perm=torch.tensor([3,1,0,2])
        kwargs['prediction_tensors']={k:v[perm] for k,v in kwargs['prediction_tensors'].items()}
        kwargs['state_nll']=kwargs['state_nll'][perm]
        shuffled,_,_=compute_world_reward([.2,.5,.9,0.,0.],[0,0,0,1,1],[0,1,0,1,0],**kwargs)
        torch.testing.assert_close(shuffled,delta)

    def test_zero_original_reward_uses_actual_final_token_not_argmax(self):
        old=torch.zeros(2,4)
        result,metrics=apply_world_reward(old,torch.tensor([.01,-.02]),torch.tensor([2,0]))
        torch.testing.assert_close(result,torch.tensor([[0.,0.,.01,0.],[-.02,0.,0.,0.]]))
        self.assertTrue(torch.equal(old,torch.zeros_like(old)))
        self.assertAlmostEqual(metrics['train/mixed_reward_mean'],-.005,places=7)

    def test_ambiguous_mappings_and_combined_objectives_fail_early(self):
        cfg=dict(enabled=True,mode='known_state',world_reward=dict(enabled=True))
        validate_world_reward_config(cfg,guideline_weight=.5)
        for change in (dict(local_credit=dict(enabled=True)),dict(credit_weighting=dict(enabled=True)),
                       dict(enabled=False),dict(world_reward=dict(enabled=True,coefficient=.7)),
                       dict(world_reward=dict(enabled=True,state_strength=float('nan')))):
            with self.assertRaises(ValueError): validate_world_reward_config({**cfg,**change})
        with self.assertRaises(ValueError): validate_world_reward_config(cfg,guideline_weight=.2)
        with self.assertRaises(ValueError):
            compute_world_reward([.5,0.],[0,1],[0,0],num_samples=2)
        with self.assertRaises(ValueError):
            compute_world_reward([.5,.7,0.],[0,0,1],[0]*3,num_samples=1,
                                 prediction_tensors=labels([(0,1,0,0)]))

    def test_selected_prompts_labels_and_shard_plan_stay_method8(self):
        spec=importlib.util.spec_from_file_location('world_fixture',ROOT/'tests/test_future_state_labels.py')
        f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
        outputs=[]
        for enabled in (False,True):
            outputs.append(build_prediction_tensors(f.rollout(),[0,0,0,0,1,1],[0,1,0,1,0,1],f.Tokenizer(),
                f.config(world_reward=dict(enabled=enabled)),world_size=8,step=20,
                final_scores=torch.tensor([1.,0.]),labeler=f.Labeler()))
        (base,meta,stats),(new,newmeta,newstats)=outputs
        self.assertEqual(stats,newstats)
        for key in base: self.assertTrue(torch.equal(base[key],new[key]),key)
        for key in meta: self.assertEqual(meta[key],newmeta[key])
        self.assertEqual(new['prediction_following_row'].tolist(),[2]*3+[-1]*5)

    def test_logging_separates_raw_guideline_new_reward_and_missing_prediction(self):
        delta,metrics,_=compute_world_reward([.5,0.],[0,1],[0,0],num_samples=1)
        _,shaped=apply_world_reward(torch.tensor([[.75]]),delta,torch.tensor([0]))
        text=format_training_diagnostics(1,185,{**metrics,**shaped})
        self.assertIn('state-only NLL=n/a',text)
        self.assertIn('G[original,weighted]=[0.5,0.5]',text)
        self.assertIn('mixed[base,new]=[0.75,0.75]',text)


class WorldDriverTests(unittest.TestCase):
    def test_real_reward_and_grpo_hooks_change_all_rows_without_local_advantage(self):
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
        fit=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
        blocks=[n for n in ast.walk(fit) if isinstance(n,ast.If) and ast.unparse(n.test)=='world_reward_enabled']
        prepare=next(n for n in blocks if 'compute_world_reward' in ast.unparse(n))
        apply=next(n for n in blocks if 'apply_world_reward' in ast.unparse(n))
        adv_stats=next(n for n in blocks if 'world_base_advantage' in ast.unparse(n))
        combine=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_combine_recurrent_intermediate_reward'))
        combine.returns=None
        for arg in combine.args.args: arg.annotation=None
        grpo=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_1D_grpo_advantage')
        ns=dict(torch=torch,os=os,_timer=lambda *a:nullcontext(),timing_raw={},metrics={},damage_reward_enabled=False)
        def execute(nodes):
            exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes),type_ignores=[])), 'real_world_reward_hooks','exec'),ns)
        execute([combine,grpo])
        # Two correct trajectories tied on raw guideline average; the state
        # weighting should separate their rewards BEFORE group demeaning.
        raw=torch.tensor([[0.,1.,0.],[1.,0.,0.]])
        final=torch.tensor([0,0,0,0,1,1],dtype=torch.bool)
        samples=torch.tensor([0,1,0,1,0,1])
        batch=SimpleNamespace(batch={'intermediate_rewards':torch.tensor([.2,.9,.9,.2,0.,0.])})
        reward_batch=SimpleNamespace(batch={'prompts':torch.ones(2,2),
            'attention_mask':torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])},non_tensor_batch={'uid':['same','same']})
        pred=SimpleNamespace(batch=labels([(0,2,0,0),(1,3,1,0)]),meta_info={})
        calls=[]
        def rpc(data):
            calls.append(1)
            return SimpleNamespace(batch={'prediction_state_nll':torch.zeros(2)})
        trainer=SimpleNamespace(recurrent_config=SimpleNamespace(intermediate_reward_enable=True,intermediate_reward_weight=.5),
            actor_rollout_wg=SimpleNamespace(compute_prediction_scores=rpc),global_steps=1,
            config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir='unused'),algorithm=SimpleNamespace(grpo_use_adv=False)))
        for gamma,eta,expected in ((.5,.1,1),(0.,.1,0),(.5,0.,0)):
            calls.clear()
            ns.update(self=trainer,world_reward_enabled=True,world_reward_config=dict(coefficient=gamma,state_strength=eta),
                prediction_meta={'prediction_valid_counts':[2]},prediction_batch=pred, reward_tensor=raw.clone(),
                batch=batch,reward_batch=reward_batch,final_mask=final,sample_index=samples)
            execute([prepare])
            self.assertEqual(len(calls),expected)
            mixed,m=ns['_combine_recurrent_intermediate_reward'](trainer,batch,reward_batch,ns['reward_tensor'],final,samples)
            ns['reward_tensor']=mixed
            with patch('recurrent.world_guideline_reward.append_world_reward_audit') as audit:
                execute([apply])
            self.assertEqual(audit.call_count,1)
            shaped=ns['reward_tensor']
            self.assertEqual(ns['world_audit']['trajectories'][0]['raw_answer_reward'],1.)
            ns['advantage_scalar']=ns['compute_1D_grpo_advantage'](shaped,['same','same'],use_adv=False)
            execute([adv_stats])
            actual=ns['advantage_scalar'][samples]
            if expected:
                # weighted G0=(.2+1.1*.9)/2.1; weighted G1 is reversed.
                value=.5*((.2+1.1*.9)/2.1-.55)
                torch.testing.assert_close(actual,torch.tensor([value,-value]*3),atol=1e-7,rtol=1e-5)
                self.assertGreater(float(actual[4]),0.) # final answer gets NEW trajectory A
            else: self.assertTrue(torch.equal(actual,torch.zeros(6)))
            self.assertNotIn('local_advantage_delta',batch.batch)
            self.assertTrue(torch.equal(raw,torch.tensor([[0.,1.,0.],[1.,0.,0.]])))
        self.assertLess(prepare.lineno,apply.lineno)
        self.assertLess(apply.lineno,adv_stats.lineno)


if __name__=='__main__': unittest.main()
