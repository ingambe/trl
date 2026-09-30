# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from trl.experimental.async_distillation import AsyncDistillationTrainer
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.iw_opd import IWOPDTrainer
from trl.models.utils import _ForwardRedirection


@pytest.fixture
def iw_opd_loss_trainer(request, tiny_llama):
    """Build loss-only IWOPD trainers with mocked server transport."""
    use_teacher_server, objective, top_k, beta = request.param
    trainer = object.__new__(IWOPDTrainer)
    trainer.model, _ = tiny_llama
    teacher = deepcopy(trainer.model)
    with torch.no_grad():
        teacher.lm_head.weight.add_(0.1 * torch.randn_like(teacher.lm_head.weight))
    trainer.model.eval()
    teacher.eval()
    trainer.teacher_model = None if use_teacher_server else teacher
    trainer._local_teacher_tokenizer_matches_student = True
    trainer.use_teacher_server = use_teacher_server
    trainer.distillation_objective = objective
    trainer.loss_top_k = top_k
    trainer.loss_add_tail = True
    trainer.beta = beta
    trainer.temperature = 1.0
    trainer.reverse_kl_top_1_mode = "argmax"
    trainer.iw_opd_gamma = 0.5
    trainer.iw_opd_epsilon = 1e-8
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.args = SimpleNamespace(average_tokens_across_devices=True)
    trainer.accelerator = SimpleNamespace(num_processes=1)
    inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 1], [2, 3, 6, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 4, 5, 1], [-100, -100, 6, 1, -100]]),
    }
    if use_teacher_server:
        with torch.no_grad():
            log_probs = (
                teacher(**{k: inputs[k] for k in ("input_ids", "attention_mask")}).logits[:, 1:-1].log_softmax(-1)
            )
        actual = log_probs.gather(-1, inputs["input_ids"][:, 2:].unsqueeze(-1))
        top_probs, top_ids = log_probs.topk(top_k, dim=-1)
        result = {
            key: [values[0, :3].tolist(), values[1, :2].tolist()]
            for key, values in (("actual_logprobs", actual), ("logprobs", top_probs), ("logprob_token_ids", top_ids))
        }
        trainer.teacher_client = SimpleNamespace(get_sequence_logprobs=lambda **kwargs: result)
    return trainer, inputs


@pytest.fixture
def async_distillation_loss_trainer(tiny_llama):
    """Provide a single-rank AsyncDistillationTrainer with a real model and loss, without rollout setup."""
    trainer = object.__new__(AsyncDistillationTrainer)
    trainer.model, _ = tiny_llama
    trainer.accelerator = Accelerator(cpu=True)
    trainer.args = SimpleNamespace(beta=0.0, teacher_temperature=1.0, add_tail_bucket=True)
    trainer._forward_redirection = _ForwardRedirection()
    trainer._teacher_ids = ["default"]
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = 0.0
    trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = 0.0
    trainer._step_samples = 0.0
    trainer._step_forward_s = 0.0
    return trainer


@pytest.fixture
def async_grpo_loss_trainer():
    """Provide a single-rank AsyncGRPOTrainer whose model returns the scalar log-prob `theta` at every position."""
    theta = torch.zeros((), requires_grad=True)

    def model(input_ids, **kwargs):
        shape = (1, input_ids.size(1) - 1)
        return SimpleNamespace(log_probs=theta.expand(shape), entropy=torch.zeros(shape))

    trainer = object.__new__(AsyncGRPOTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(
        num_processes=1, reduce=lambda tensor, reduction: tensor, gather=lambda tensor: tensor
    )
    trainer.aux_loss_enabled = False
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = 0.0
    trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = 0.0
    trainer._step_samples = 0.0
    trainer._step_forward_s = 0.0
    return trainer, theta
