import ast
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from recurrent.evidence_damage import (
    apply_damage_reward, build_damage_context, parse_damage_checks,
    summarize_damage_audit, validate_damage_config,
)
from recurrent.intermediate_reward import OpenAIMemoryRewardJudge
from recurrent.training_diagnostics import format_training_diagnostics

ROOT = Path(__file__).resolve().parents[1]
OLD = "The user prefers short shows only during their commute."
NEW = "The user prefers short shows."
SOURCE = "User: I prefer short shows only during my commute."


def context(old=OLD, new=NEW):
    return build_damage_context(old, new, SOURCE, "User: I enjoy science fiction movies too.")


def check(status="lost_condition", **kwargs):
    result = dict(id=0, status=status, source_id="prior:0", source_quote=SOURCE,
                  updated_quote=NEW, reason="The commute-only condition is lost.")
    result.update(kwargs)
    return result


class EvidenceTests(unittest.TestCase):
    def test_source_retrieval_uses_intact_bounded_past_lines(self):
        ctx = context()
        self.assertEqual(ctx["units"], [{"id": 0, "text": OLD}])
        self.assertEqual(ctx["sources"][0], dict(id="prior:0", text=SOURCE))
        limited = build_damage_context(OLD, NEW, SOURCE, "new", max_source_chars=10)
        self.assertEqual(limited["sources"], [dict(id="section", text="new")])
        long_line = "User: " + SOURCE * 20
        self.assertEqual(len(build_damage_context(OLD, NEW, long_line, "new")["sources"]), 1)
        # Candidate sampling must not select units based on the generated action.
        self.assertEqual(context(new="different output")["units"], ctx["units"])
        self.assertEqual(build_damage_context("# Heading\nNo previous memory", NEW, SOURCE, "new")["units"], [])

    def test_verified_condition_loss_and_rewrite_are_penalties(self):
        for status in ("lost_condition", "unsupported_change"):
            audit = parse_damage_checks([check(status)], context())
            self.assertEqual(audit["damage_sum"], 1)
            self.assertEqual(audit["valid"], 1)

    def test_drop_requires_absence_and_cannot_punish_preserved_unit(self):
        dropped = parse_damage_checks([check("unsupported_drop", updated_quote="")], context(new="Other facts."))
        self.assertEqual(dropped["damage_sum"], 1)
        kept = parse_damage_checks([check("unsupported_drop", updated_quote="")], context(new=OLD))
        self.assertEqual(kept["damage_sum"], 0)
        self.assertEqual(kept["checks"][0]["rejection"], "unit_still_present")
        invalid = parse_damage_checks([check("unsupported_drop")], context())
        self.assertEqual(invalid["checks"][0]["rejection"], "drop_has_updated_quote")

    def test_justified_revision_paraphrase_and_uncertainty_do_not_penalize(self):
        for status in ("no_damage", "uncertain"):
            audit = parse_damage_checks([check(status, source_quote="", source_id="", updated_quote="")], context())
            self.assertEqual(audit["damage_sum"], 0)
            self.assertEqual(audit["invalid"], 0)
            self.assertEqual(audit["uncertain"], int(status == "uncertain"))

    def test_missing_fabricated_wrong_id_and_partial_schema_abstain(self):
        for items in (None, [], [check(source_quote="invented evidence")], [check(source_id="unread:4")],
                      [check(updated_quote="hallucinated updated quote")], [check(id=True)],
                      [check(id=99)], [dict(id=0)], [check(), check()], [check(reason="")]):
            audit = parse_damage_checks(items, context())
            self.assertEqual(audit["selected"], 1)
            self.assertEqual(audit["damage_sum"], 0)
            self.assertEqual(audit["invalid"], 1)

    def test_one_invalid_unit_does_not_discard_valid_check_or_shrink_denominator(self):
        ctx = context()
        ctx["units"].append(dict(id=1, text="The user likes science fiction movies."))
        audit = parse_damage_checks([check(), dict(id=1)], ctx)
        self.assertEqual((audit["selected"], audit["damage_sum"], audit["invalid"]), (2, 1, 1))
        metrics, rows = summarize_damage_audit([audit, None], [False, True], [7, 7])
        self.assertEqual(metrics["damage_reward/valid_coverage"], .5)
        self.assertEqual(rows[0]["sample"], 7)
        self.assertEqual(rows[0]["row"], 0)

    def test_config_bounds(self):
        for value in (-.1, .1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_damage_config(SimpleNamespace(damage_reward_coefficient=value))
        with self.assertRaises(ValueError):
            validate_damage_config(SimpleNamespace(damage_reward_enable=True))


class JudgeTests(unittest.TestCase):
    def request(self, response, ctx):
        judge = OpenAIMemoryRewardJudge(max_completion_tokens=512)
        encoded = json.dumps({"choices": [{"message": {"content": response}, "finish_reason": "stop"}]}).encode()
        with patch.dict(os.environ, MEMCOE_API_KEY="local-test"), patch(
            "urllib.request.urlopen", return_value=io.BytesIO(encoded)) as http:
            result = judge.score("Question", OLD, "Current section", NEW, ctx)
            payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(http.call_count, 1)
        return result, payload

    def test_disabled_keeps_original_prompt_and_call_budget(self):
        result, payload = self.request(json.dumps(dict(score=.7, reason="baseline")), None)
        original = OpenAIMemoryRewardJudge(max_completion_tokens=512)._build_payload("Question", OLD, "Current section", NEW)
        self.assertEqual(payload, original)
        self.assertIsNone(result.damage)

    def test_combined_call_keeps_guideline_score_and_evidence_audit(self):
        result, payload = self.request(json.dumps(dict(score=1., reason="ok", damage_checks=[check()])), context())
        self.assertEqual(result.score, 1.)
        self.assertEqual(result.damage["damage_sum"], 1.)
        self.assertIn('damage_checks', payload["messages"][1]["content"])
        self.assertNotIn("exactly two fields", payload["messages"][1]["content"])
        self.assertEqual(payload["max_tokens"], 768)

    def test_bad_optional_checks_do_not_retry_or_change_original_score(self):
        for checks in (None, [dict(id=0)], "bad"):
            result, _ = self.request(json.dumps(dict(score=.7, reason="valid score", damage_checks=checks)), context())
            self.assertEqual(result.score, .7)
            self.assertEqual(result.damage["damage_sum"], 0)

    def test_truncated_suffix_preserves_only_complete_score_and_reason(self):
        response = '{"score":0.7,"reason":"valid explanation","damage_checks":[{"id":0'
        result, _ = self.request(response, context())
        self.assertEqual(result.score, .7)
        self.assertTrue(result.damage["score_prefix_salvaged"])
        self.assertEqual(result.damage["invalid"], 1)
        for incomplete in ('{"score":0.', '{"score":0.7,"reason":"unfinished',
                           '{"score":true,"reason":"bad",'):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                OpenAIMemoryRewardJudge._parse_complete_score_prefix(incomplete)


class RewardTests(unittest.TestCase):
    def test_interleaved_rows_counts_and_final_token_mapping(self):
        old = torch.tensor([[0., .8, 0.], [.6, 0., 0.]], requires_grad=True)
        # Two updates each, followed by final Answers in REVERSED row order.
        result, metrics, rows = apply_damage_reward(old, [1,0,1,0,0,0], [2,2,2,1,0,0],
            [False]*4+[True]*2, [0,1,0,1,1,0], [1,0], coefficient=.01)
        torch.testing.assert_close(result, torch.tensor([[0.,.795,0.],[.6,0.,0.]]))
        self.assertEqual(rows[0]["selected"], 4)
        self.assertEqual(metrics["damage_reward/changed_trajectories"], 1)
        self.assertFalse(result.requires_grad)
        self.assertAlmostEqual(float(old[0,1].detach()), .8)

    def test_zero_coefficient_and_all_abstain_are_exact_reward_noops(self):
        old = torch.tensor([[0.,.8,0.]])
        for counts, coefficient in (([1,0], 0.), ([0,0], .01)):
            new,_,_ = apply_damage_reward(old, counts, [2,0], [False,True], [0,0], [1], coefficient=coefficient)
            self.assertTrue(torch.equal(old,new))

    def test_incorrect_answer_zero_reward_and_bound(self):
        result, _, _ = apply_damage_reward(torch.zeros(1,3), [2,0], [2,0], [False,True], [0,0], [2])
        torch.testing.assert_close(result, torch.tensor([[0.,0.,-.01]]))

    def test_final_rows_and_invalid_counts_are_rejected(self):
        for damaged,selected in (([1,1],[2,1]), ([3,0],[2,0]), ([-1,0],[2,0]),
                                 ([float("nan"),0],[2,0]), ([.5,0],[2,0])):
            with self.assertRaises(ValueError):
                apply_damage_reward(torch.zeros(1,3), damaged, selected, [False,True], [0,0], [2])

    def test_diagnostic_names_are_separate_from_answer_accuracy(self):
        text = format_training_diagnostics(10,185,dict({"damage_reward/enabled":1.,
            "damage_reward/damaged_units":2.,"damage_reward/coefficient":.01,
            "train/answer_correct":3,"train/answer_count":8,"train/answer_accuracy":3/8}))
        self.assertIn("rollout QA=3/8",text)
        self.assertIn("damaged=2",text)


class DriverTests(unittest.TestCase):
    def test_real_driver_hook_precedes_grpo_and_keeps_world_advantage_stats_separate(self):
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
        fit=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
        blocks=[n for n in ast.walk(fit) if isinstance(n,ast.If) and ast.unparse(n.test)=='damage_reward_enabled']
        apply=next(n for n in blocks if 'apply_damage_reward' in ast.unparse(n))
        adv=next(n for n in blocks if 'damage_base_advantage' in ast.unparse(n))
        grpo=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_1D_grpo_advantage')
        ns=dict(torch=torch,os=os,metrics={},damage_reward_enabled=True,
                reward_tensor=torch.tensor([[0.,.75,0.],[.75,0.,0.]]),
                final_mask=torch.tensor([False,False,True,True]),sample_index=torch.tensor([0,1,0,1]))
        audit=parse_damage_checks([check()],context())
        ns['batch']=SimpleNamespace(batch={'damage_sum':torch.tensor([1.,0.,0.,0.]),
             'damage_selected':torch.tensor([1.,1.,0.,0.])},
             non_tensor_batch={'damage_audit':np.array([audit,None,None,None],dtype=object)})
        ns['reward_batch']=SimpleNamespace(batch={'prompts':torch.ones(2,2),
            'attention_mask':torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])},non_tensor_batch={'uid':['same','same']})
        ns['self']=SimpleNamespace(recurrent_config=SimpleNamespace(damage_reward_coefficient=.01),global_steps=1,
             config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir='unused'),algorithm=SimpleNamespace(grpo_use_adv=False)))
        def execute(nodes):
            exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes),type_ignores=[])), 'damage_driver','exec'),ns)
        execute([grpo])
        with patch('recurrent.world_guideline_reward.append_world_reward_audit') as writer:
            execute([apply])
        self.assertEqual(writer.call_count,1)
        ns['advantage_scalar']=ns['compute_1D_grpo_advantage'](ns['reward_tensor'],['same','same'],use_adv=False)
        execute([adv])
        torch.testing.assert_close(ns['advantage_scalar'],torch.tensor([-.005,.005]),rtol=1e-4,atol=1e-7)
        self.assertAlmostEqual(ns['metrics']['train/mixed_reward_mean'], .745, places=6)
        self.assertNotIn('damage_audit', ns['batch'].non_tensor_batch)
        self.assertLess(apply.lineno,adv.lineno)


class AgentIntegrationTests(unittest.TestCase):
    def make_agent(self, validate=False):
        from test_validation_dispatch import load_function
        scope = dict(torch=torch, np=np)
        class Output:
            def __init__(self):
                self.batch, self.non_tensor_batch = {}, {}
            def __len__(self):
                return 2
        agent = SimpleNamespace(
            config=SimpleNamespace(damage_reward_enable=True, damage_reward_max_units=2,
                damage_reward_source_chars=1800, chunk_size=2), bsz=3, step=1,
            active_mask=torch.tensor([True,False,True]),
            tokenizer=SimpleNamespace(pad_token_id=0),
            memory=np.array([OLD,"finished",OLD], dtype=object),
            gen_batch=SimpleNamespace(meta_info={"validate":validate},
                non_tensor_batch={"prompt_ids":np.array(["q0","q1","q2"],dtype=object)},
                batch={"context_ids":torch.tensor([[11,12,21,22,91,92], [31,32,0,0,0,0], [41,42,51,52,81,82]])}))
        agent._decode_tokens=lambda tokens: tokens if isinstance(tokens,str) else ",".join(map(str,tokens.tolist()))
        agent._decode_optional_memory=lambda memory: memory
        for name in ("_attach_intermediate_rewards","_attach_empty_intermediate_rewards"):
            method=load_function('recurrent/impls/memory.py',name,scope)
            setattr(agent,name,MethodType(method,agent))
        return agent, Output()

    def test_only_consumed_context_active_samples_and_damaging_row_are_attached(self):
        agent, output=self.make_agent()
        calls=[]
        real_build=build_damage_context
        def build(old,new,prior,section,**kwargs):
            calls.append((prior,section))
            return real_build(old,new,prior,section,**kwargs)
        def judge(**kwargs):
            self.assertEqual(kwargs['questions'],['q0','q2'])
            return [SimpleNamespace(score=1.,reason='ok',damage=parse_damage_checks(None,ctx))
                    for ctx in kwargs['damage_contexts']]
        agent.intermediate_reward_judge=SimpleNamespace(score_batch=judge)
        with patch('recurrent.evidence_damage.build_damage_context',side_effect=build):
            agent._attach_intermediate_rewards(output,np.array([NEW,NEW],dtype=object))
        self.assertEqual(calls,[('11,12','21,22'),('41,42','51,52')])
        self.assertTrue(torch.equal(output.batch['damage_selected'],torch.tensor([1.,1.])))
        self.assertTrue(all(row['memory_update_index']==1 for row in output.non_tensor_batch['damage_audit']))
        self.assertEqual([row['question'] for row in output.non_tensor_batch['damage_audit']], ['q0','q2'])
        agent._attach_empty_intermediate_rewards(output)
        self.assertTrue(torch.equal(output.batch['damage_selected'],torch.zeros(2)))
        self.assertEqual(output.non_tensor_batch['damage_audit'].tolist(),[None,None])

    def test_validation_does_not_request_damage_checks(self):
        agent, output=self.make_agent(validate=True)
        def judge(**kwargs):
            self.assertNotIn('damage_contexts',kwargs)
            return [SimpleNamespace(score=1.,reason='ok',damage=None)]*2
        agent.intermediate_reward_judge=SimpleNamespace(score_batch=judge)
        with patch('recurrent.evidence_damage.build_damage_context') as builder:
            agent._attach_intermediate_rewards(output,np.array([NEW,NEW],dtype=object))
        builder.assert_not_called()
        self.assertTrue(torch.equal(output.batch['damage_sum'],torch.zeros(2)))


if __name__ == '__main__':
    unittest.main()
