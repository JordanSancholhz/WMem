import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    guideline_path: Optional[str] = None
    intermediate_reward_enable: bool = False
    intermediate_reward_weight: float = 0.0
    intermediate_reward_model: str = "Qwen2.5-7B-Instruct"
    intermediate_reward_base_url: str = "http://127.0.0.1:6025/v1"
    intermediate_reward_api_key_env: str = "MEMCOE_API_KEY"
    intermediate_reward_timeout: float = 60.0
    intermediate_reward_max_retries: int = 3
    intermediate_reward_concurrency: int = 4
    intermediate_reward_max_completion_tokens: int = 256
    intermediate_reward_reasoning_effort: Optional[str] = None
    intermediate_reward_temperature: Optional[float] = None
    intermediate_reward_fail_score: Optional[float] = None
    damage_reward_enable: bool = False
    damage_reward_coefficient: float = 0.01
    damage_reward_max_units: int = 2
    damage_reward_source_chars: int = 1800
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]

# TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

# <problem> 
# {prompt}
# </problem>

# <memory>
# {memory}
# </memory>

# <section>
# {chunk}
# </section>

# Updated memory:
# """

# TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

# <problem> 
# {prompt}
# </problem>

# <memory>
# {memory}
# </memory>

# Your answer:
# """

TEMPLATE = """
You are given a question with options, some new memory, and previous user memory. Read the new section and update
the user memory by prioritizing recent and relevant information that aligns with the user’s preferences and experiences.

<memory> {memory} </memory>

<section> {chunk} </section>

Update rules:
- Evidence-bound: Extract candidate memory items from chunk. Every stored item must be directly supported by chunk.
- Relevance & stability: Store long-term preferences, stable facts, recurring habits, long-term goals, and interaction preferences. Do not store one-off details (e.g., transient locations, momentary moods) unless explicitly stated as long-term.
- Conflict handling: If new info contradicts existing memory, prefer the most recent supported info as active, and mark the older one as deprecated with a short reason.
- Privacy: Do NOT store highly sensitive or uniquely identifying data (exact address, account credentials, financial/medical specifics, etc.).
- No domain bias: Do not assume any specific hobbies or interests unless stated in chunk.
- If chunk contains explicit user corrections/ratings about previous outputs, store them under "interaction_feedback". Otherwise, do not create feedback entries.
- Merge into a structured memory profile. Keep it concise, non-redundant, and internally consistent.
"""

TEMPLATE_FINAL_BOXED = """You are presented with a question along with its corresponding options, Find the most appropriate option to the question based on user memory and give your final answer (a), (b), (c), or (d).  Put the answer in \\boxed{{}}.

{prompt}

<memory>
{memory}
</memory>

Your answer:
"""

class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        from recurrent.guideline import load_guideline
        self.guideline = load_guideline(self.config.guideline_path, TEMPLATE)
        self.token_message_template = TokenTemplate(self.chat_template.format(message=self.guideline), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length 
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = tokenizer.encode("No previous memory", add_special_tokens=False)
        self.intermediate_reward_judge = None
        from recurrent.evidence_damage import validate_damage_config
        validate_damage_config(self.config)
        if self.config.intermediate_reward_enable:
            if not 0.0 <= self.config.intermediate_reward_weight <= 1.0:
                raise ValueError(f"intermediate_reward_weight must be in [0, 1], got {self.config.intermediate_reward_weight}")
            from recurrent.intermediate_reward import OpenAIMemoryRewardJudge

            self.intermediate_reward_judge = OpenAIMemoryRewardJudge(
                guideline=self.guideline,
                model=self.config.intermediate_reward_model,
                base_url=self.config.intermediate_reward_base_url,
                api_key_env=self.config.intermediate_reward_api_key_env,
                timeout=self.config.intermediate_reward_timeout,
                max_retries=self.config.intermediate_reward_max_retries,
                concurrency=self.config.intermediate_reward_concurrency,
                max_completion_tokens=self.config.intermediate_reward_max_completion_tokens,
                reasoning_effort=self.config.intermediate_reward_reasoning_effort,
                temperature=self.config.intermediate_reward_temperature,
                fail_score=self.config.intermediate_reward_fail_score,
            )
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.memory[:] = None
        self.is_final = False
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_sieze = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_mask.sum().item() == 0:
            self.is_final = True
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
            memory_i = self.memory[active_mask]
            
            # format: we use our token_template to avoid decoding & formatting with str function & encoding back.
            self.messages = [
                self.token_message_template.format(
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS, # use pre-tokenized "No previous memory" for first round
                        chunk=chunk[chunk != self.tokenizer.pad_token_id], # unpadding needed here
                )
                for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done')
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            updated_memory = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            self._attach_intermediate_rewards(gen_output, updated_memory)
            self.memory[self.active_mask] = updated_memory
        elif self.intermediate_reward_judge is not None:
            self._attach_empty_intermediate_rewards(gen_output)
        self.log_step(gen_output)
        self.step += 1
        return gen_output

    def _attach_empty_intermediate_rewards(self, gen_output: DataProto):
        gen_output.batch["intermediate_rewards"] = torch.zeros(len(gen_output), dtype=torch.float32)
        gen_output.non_tensor_batch["intermediate_reward_reason"] = np.array([""] * len(gen_output), dtype=object)
        if getattr(self.config, "damage_reward_enable", False):
            gen_output.batch["damage_sum"] = torch.zeros(len(gen_output), dtype=torch.float32)
            gen_output.batch["damage_selected"] = torch.zeros(len(gen_output), dtype=torch.float32)
            gen_output.non_tensor_batch["damage_audit"] = np.array([None] * len(gen_output), dtype=object)

    def _attach_intermediate_rewards(self, gen_output: DataProto, updated_memory: np.ndarray):
        if self.intermediate_reward_judge is None:
            return

        active_indices = torch.arange(self.bsz, dtype=torch.long)[self.active_mask]
        prompt_i = self.gen_batch.non_tensor_batch["prompt_ids"][self.active_mask]
        chunk_i = self.gen_batch.batch["context_ids"][
            self.active_mask, self.config.chunk_size * self.step : self.config.chunk_size * (self.step + 1)
        ]
        previous_memory_i = self.memory[self.active_mask]

        questions = [self._decode_tokens(prompt) for prompt in prompt_i]
        previous_memories = [self._decode_optional_memory(memory) for memory in previous_memory_i]
        sections = [self._decode_tokens(chunk[chunk != self.tokenizer.pad_token_id]) for chunk in chunk_i]
        updated_memories = [self._decode_tokens(memory) for memory in updated_memory]
        damage_kwargs = {}
        damage_enabled = getattr(self.config, "damage_reward_enable", False)
        if damage_enabled and not self.gen_batch.meta_info.get("validate", False):
            from recurrent.evidence_damage import build_damage_context
            # Only already-consumed raw tokens; no future sections or QA labels.
            prior_i = self.gen_batch.batch["context_ids"][
                self.active_mask, :self.config.chunk_size * self.step]
            prior_dialogues = [self._decode_tokens(tokens) for tokens in prior_i]
            damage_kwargs["damage_contexts"] = [
                build_damage_context(old, new, prior, section,
                                     max_units=self.config.damage_reward_max_units,
                                     max_source_chars=self.config.damage_reward_source_chars)
                for old, new, prior, section in zip(previous_memories, updated_memories, prior_dialogues, sections)
            ]
            for context, question in zip(damage_kwargs["damage_contexts"], questions):
                context["question"] = question  # prompt only; never the ground-truth answer
        results = self.intermediate_reward_judge.score_batch(
            questions=questions,
            previous_memories=previous_memories,
            sections=sections,
            updated_memories=updated_memories,
            **damage_kwargs,
        )
        scores = torch.tensor([result.score for result in results], dtype=torch.float32)
        reasons = np.array([result.reason for result in results], dtype=object)

        if len(scores) != len(active_indices):
            raise RuntimeError(f"intermediate reward size mismatch: {len(scores)=}, {len(active_indices)=}")
        gen_output.batch["intermediate_rewards"] = scores
        gen_output.non_tensor_batch["intermediate_reward_reason"] = reasons
        if damage_enabled:
            audits = [result.damage for result in results]
            gen_output.batch["damage_sum"] = torch.tensor(
                [a["damage_sum"] if a else 0. for a in audits], dtype=torch.float32)
            gen_output.batch["damage_selected"] = torch.tensor(
                [a["selected"] if a else 0. for a in audits], dtype=torch.float32)
            for audit in audits:
                if audit is not None:
                    audit["memory_update_index"] = self.step
            gen_output.non_tensor_batch["damage_audit"] = np.array(audits, dtype=object)

    def _decode_optional_memory(self, memory) -> str:
        if memory is None:
            return "No previous memory"
        return self._decode_tokens(memory)

    def _decode_tokens(self, tokens) -> str:
        if isinstance(tokens, str):
            return tokens
        if not isinstance(tokens, torch.Tensor):
            # Variable-length prompt/memory batches use dtype=object. Even a
            # single row can retain that dtype after NumPy batching/repeating;
            # PyTorch cannot ingest it directly. Unbox the integer IDs first.
            if isinstance(tokens, np.ndarray) and tokens.dtype == object:
                tokens = tokens.tolist()
            tokens = torch.as_tensor(tokens, dtype=torch.long)
        mask = tokens != self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            mask &= tokens != self.tokenizer.eos_token_id
        return self.tokenizer.decode(tokens[mask], skip_special_tokens=True)
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] STEP{step}{'='*30}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
