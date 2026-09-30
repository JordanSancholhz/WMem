"""Method2 labels with Method4 time decay only; no sparse-label scaling."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from tensordict import TensorDict
from recurrent.future_prediction import (_pack_examples, prediction_coefficient,
    backward_prediction_minibatch, build_prediction_tensors, validate_prediction_config)
from test_future_state_labels import Actor, Tokenizer, Labeler, config, rollout


class DecayTests(unittest.TestCase):
    def test_warmup_endpoints_monotonicity_and_resume(self):
        cfg=config(coefficient_schedule='cosine_decay')
        for step,value in [(0,0),(10,.001),(20,.002),(100,.002),(150,.00125),(200,.0005),(220,.0005)]:
            self.assertAlmostEqual(prediction_coefficient(cfg,step,200),value)
        values=[prediction_coefficient(cfg,s,185) for s in range(1,186)]
        self.assertEqual(values[80:],[prediction_coefficient(cfg,s,185) for s in range(81,186)])
        self.assertTrue(all(a>=b for a,b in zip(values[20:],values[21:])))
        self.assertEqual(prediction_coefficient(config(),185),.002)
        with self.assertRaises(ValueError): prediction_coefficient(cfg,10)
        for change in [dict(coefficient_schedule='bad'),dict(decay_start_fraction=1),dict(final_coefficient_ratio=-1)]:
            with self.assertRaises(ValueError):
                validate_prediction_config(config(**change),recurrent='memory',strategy='fsdp',sequence_parallel=1,train_batch=8,mini_batch=8)

    def test_eight_rank_gradient_uses_mean_without_sparse_scaling(self):
        for count in (0,1,3,8,13,24):
            cfg=config(coefficient_schedule='cosine_decay')
            examples=[([7,8],[4+i%3,2]) for i in range(count)]
            tensors,meta,_=_pack_examples(examples,{},Tokenizer(),cfg,8,150,1,200)
            batch=TensorDict(tensors,batch_size=[len(tensors['input_ids'])])
            grads,calls=[],[]
            for shard in batch.chunk(8):
                actor=Actor()
                backward_prediction_minibatch(actor,SimpleNamespace(batch=shard,meta_info=meta),minibatch_index=0,device='cpu')
                grads.append(actor.parameter.grad if actor.parameter.grad is not None else torch.tensor(0.))
                calls.append(actor.calls)
            self.assertEqual(len(set(calls)),1)
            if count==0:
                self.assertTrue(all(c==0 for c in calls));continue
            reference=Actor()
            _,lp=reference._forward_micro_batch(batch)
            mask=batch['prediction_loss_mask']
            nll=-(lp*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            (.00125*nll.sum()/count).backward()
            torch.testing.assert_close(torch.stack(grads).mean(),reference.parameter.grad)

    def test_schedule_changes_only_coefficient_not_method2_labels_or_tensors(self):
        results=[]
        for schedule in ('constant','cosine_decay'):
            results.append(build_prediction_tensors(rollout(),[0,0,0,0,1,1],[0,1,0,1,0,1],Tokenizer(),
                config(coefficient_schedule=schedule),world_size=8,step=150,total_steps=200,
                final_scores=torch.tensor([1.,0.]),labeler=Labeler()))
        (left,lmeta,_),(right,rmeta,stats)=results
        for key in left: torch.testing.assert_close(left[key],right[key])
        self.assertEqual(lmeta['prediction_valid_counts'],rmeta['prediction_valid_counts'])
        self.assertEqual(lmeta['prediction_coefficient'],.002)
        self.assertAlmostEqual(rmeta['prediction_coefficient'],.00125)
        self.assertAlmostEqual(stats['future_prediction/scheduled_coefficient'],.00125)

    def test_trainer_passes_actual_planned_steps(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_prediction_tensors')
        self.assertEqual(ast.unparse(next(k.value for k in call.keywords if k.arg=='total_steps')),'self.total_training_steps')


if __name__=='__main__': unittest.main()
