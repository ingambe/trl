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
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from trl.experimental.async_distillation import AsyncDistillationTrainer
from trl.models.utils import _ForwardRedirection


@pytest.fixture
def async_distillation_loss_trainer(tiny_llama):
    """Provide a single-rank AsyncDistillationTrainer with a real model and loss, without rollout setup."""
    trainer = object.__new__(AsyncDistillationTrainer)
    trainer.model, _ = tiny_llama
    trainer.accelerator = Accelerator(cpu=True)
    trainer.args = SimpleNamespace(beta=0.0, teacher_temperature=1.0, add_tail_bucket=True)
    trainer._forward_redirection = _ForwardRedirection()
    trainer._teacher_ids = ["default"]
    trainer.aux_loss_enabled = False
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = 0.0
    trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = 0.0
    trainer._step_samples = 0.0
    trainer._step_forward_s = 0.0
    return trainer
