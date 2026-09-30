# WMem: Learning to Remember through World Models of Future Memory States

This repository contains the training implementation of **WMem**, a recurrent memory framework that learns to retain and update information through future memory-state prediction and predictive credit assignment.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hardware](#hardware)
- [Acknowledgements](#acknowledgements)

---

## Overview

Long interaction histories require language agents to continually update their memories while retaining information needed for future decisions. **WMem** trains a recurrent memory policy with **future memory-state prediction**, using predicted memory transitions to guide credit assignment during reinforcement learning.

WMem combines three components:

1. **Memory World Model**: predict whether information in the current memory will be preserved, revised, or absent in the next memory, with evidence-checked supervision from a frozen labeler.
2. **Step-Level Predictive Feedback**: use state-prediction likelihoods to reweight memory-quality rewards.
3. **Memory Damage Check**: apply a bounded penalty for evidence-supported memory loss or contradiction.

The policy processes dialogue in consecutive chunks, updates its memory, and answers questions using the resulting memory. **GRPO** provides the optimization backbone, while a frozen **Qwen2.5-7B-Instruct** service supplies memory-quality scores and state labels.

This repository provides the training implementation on **PersonaMem-32K**, with validation of the final trained model.

---

## Installation

### Python Environment

Use Linux with Python 3.11:

```bash
conda create -n wmem python=3.11 -y
conda activate wmem

git clone https://github.com/JordanSancholhz/WMem.git
cd WMem
```

Install a compatible CUDA build of **PyTorch**, **vLLM**, and **FlashAttention 2**, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The training code uses vLLM's **V0 worker interface** (`VLLM_USE_V1=0`). The original CUDA environment is not provided as a version-locked dependency file; use a compatible V0-capable stack. The launch scripts load the bundled, modified `verl` and `recurrent` packages directly.

### Models

Prepare a local copy of **Qwen2.5-7B-Instruct**. The default directory layout is:

```text
models/
└── Qwen2.5-7B-Instruct/
```

Each model directory must contain its weights, tokenizer, and configuration files. Training uses the local assets in Hugging Face offline mode.

To use models or prepared data stored elsewhere, set these variables in the service and training terminals:

```bash
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export DATASET_ROOT=/path/to/PersonaMem
```

---

## Datasets

Download the following datasets and place them in the `MGI_module/datasets/` directory:

| Dataset | Source |
|---------|--------|
| PersonaMem | [HuggingFace](https://huggingface.co/datasets/bowen-upenn/PersonaMem) |
| PrefEval | [HuggingFace](https://huggingface.co/datasets/siyanzhao/prefeval_explicit) |
| PersonaBench | [GitHub](https://github.com/SalesforceAIResearch/personabench) |

```bash
mkdir -p MGI_module/datasets
# Download and extract datasets to MGI_module/datasets/
```

---

## Training

Following [MemCoE](https://github.com/Applied-Machine-Learning-Lab/ACL2026_MemCoE), we use memory guidelines to construct memory-update prompts and assess the quality of memory updates. We then train the memory policy with:

```bash
bash train.sh
```

The training configuration is defined in `run_memory_7B.sh`.

---

## Evaluation

#### 1. Convert Checkpoint for vLLM

Run the merger script to convert the checkpoint:

```bash
bash scripts/merger.sh
```

> **Note**: Update `CKPT=checkpoints/wmem-qwen2.5-7b/global_step_xxx` in the script with the final checkpoint step number.

#### 2. Deploy the Trained Model

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve checkpoints/wmem-qwen2.5-7b/global_step_xxx/huggingface \
    --tensor_parallel_size 4 \
    --port 6025 \
    --host 0.0.0.0 \
    --served-model-name WMem-7B
```

#### 3. Run Inference

We follow the **Run Inference** section under **GMPO Inference** in [MemCoE](https://github.com/Applied-Machine-Learning-Lab/ACL2026_MemCoE#3-run-inference), using `WMem-7B` as the model name.

---

## Hardware

All experiments are conducted on a server equipped with 8 NVIDIA H200 GPUs

---

## Acknowledgements

This project builds on [verl](https://github.com/volcengine/verl), [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent), and [MemCoE](https://github.com/Applied-Machine-Learning-Lab/ACL2026_MemCoE). We thank the authors of these projects for releasing their code.


