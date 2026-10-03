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

import importlib
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
replay = importlib.import_module("replay")


def test_compare_logps_counts_flipped_clip_decisions():
    old = torch.zeros(1, 4)
    ref = torch.log(torch.tensor([[1.0, 1.1, 1.3, 1.3]]))
    cand = torch.log(torch.tensor([[1.0, 1.3, 1.1, 1.3]]))  # tokens 1 and 2 cross the clip boundary
    mask = torch.tensor([[1, 1, 1, 0]])

    metrics = replay.compare_logps(ref, cand, old, mask, epsilon_low=0.2, epsilon_high=0.2)

    assert metrics["clip_disagreement"] == pytest.approx(2 / 3)
    torch.testing.assert_close(
        metrics["logp_max"], (torch.log(torch.tensor(1.3)) - torch.log(torch.tensor(1.1))).item()
    )


def test_tensor_stats_reports_tensorwise_underflow():
    # One outlier sets the tensorwise scale, so the small values flush to zero in E4M3
    x = torch.tensor([448.0 * 2**10] + [1.0] * 9)

    stats = replay.tensor_stats(x)

    assert stats["max"] == 448.0 * 2**10
    assert stats["underflow"] == pytest.approx(0.9)
