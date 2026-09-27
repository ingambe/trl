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

import threading
from types import SimpleNamespace

import pytest
import torch

from trl.generation import vllm_client
from trl.generation.vllm_client import VLLMClient


@pytest.fixture
def client(monkeypatch):
    client = object.__new__(VLLMClient)
    client.base_url = "http://test-server"
    client._updating_weights = False
    client.communicator = None
    client.events = []
    client.paused = False
    client.failure = None

    def post(url, **kwargs):
        endpoint = url.removeprefix(client.base_url)
        client.events.append(endpoint)
        if endpoint == client.failure:
            raise RuntimeError("update failed")
        if endpoint == "/pause":
            client.paused = True
        elif endpoint == "/resume":
            client.paused = False
        return {}

    client._post = post
    client.reset_prefix_cache = lambda: client._post(client.base_url + "/reset_prefix_cache")
    monkeypatch.setattr(vllm_client, "_HAS_WEIGHT_UPDATE_LIFECYCLE", True)
    return client


@pytest.mark.parametrize("failure", ["/start_weight_update", "body", "/finish_weight_update", "/reset_prefix_cache"])
def test_failed_publication_leaves_server_paused(client, failure):
    client.failure = failure
    with pytest.raises(RuntimeError, match="update failed"):
        with client.weight_update():
            assert client.paused
            if failure == "body":
                raise RuntimeError("update failed")
    assert client.paused
    assert "/resume" not in client.events
    assert not client._updating_weights
    if failure in ("/start_weight_update", "body"):
        assert "/finish_weight_update" not in client.events

    client.failure = None
    with client.weight_update():
        assert client.paused
    assert not client.paused
    assert client.events[-3:] == ["/finish_weight_update", "/reset_prefix_cache", "/resume"]


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("failure", ["producer", "receiver", None])
def test_interrupted_weight_stream_closes_before_waiting_for_receiver(client, monkeypatch, stateful, failure):
    closed = threading.Event()
    observed = []

    def parameters():
        try:
            yield "weight", torch.ones(2)
            yield "bias", torch.zeros(2)
        finally:
            closed.set()

    params = parameters()

    def send(iterator, **kwargs):
        next(iterator)
        if failure == "producer":
            raise RuntimeError("producer failed")
        list(iterator)

    post = client._post

    def receive(url, **kwargs):
        if url.endswith("/update_weights"):
            observed.append(closed.wait(timeout=2))
            if failure == "receiver":
                raise RuntimeError("receiver failed")
        return post(url, **kwargs)

    client._post = receive
    monkeypatch.setattr(vllm_client, "_HAS_STATEFUL_TRAINER_ENGINE", stateful)
    monkeypatch.setattr(vllm_client, "packed_nccl_broadcast_producer", send, raising=False)
    monkeypatch.setattr(
        vllm_client, "NCCLWeightTransferEngine", SimpleNamespace(trainer_send_weights=send), raising=False
    )
    monkeypatch.setattr(vllm_client, "NCCLTrainerSendWeightsArgs", lambda **kwargs: kwargs, raising=False)
    metadata = [("weight", "float32", [2]), ("bias", "float32", [2])]
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            client.update_named_params(metadata, params)
        assert client.paused
        assert "/resume" not in client.events
    else:
        client.update_named_params(metadata, params)
        assert not client.paused
    assert observed == [True]
    assert params.gi_frame is None


def test_caught_transfer_error_cannot_publish_the_group(client, monkeypatch):
    def send(iterator, **kwargs):
        next(iterator)
        raise RuntimeError("producer failed")

    monkeypatch.setattr(vllm_client, "_HAS_STATEFUL_TRAINER_ENGINE", True)
    monkeypatch.setattr(vllm_client, "packed_nccl_broadcast_producer", send, raising=False)
    with pytest.raises(RuntimeError, match="server remains paused"):
        with client.weight_update():
            with pytest.raises(RuntimeError, match="producer failed"):
                client.update_named_param("weight", torch.ones(2))
    assert client.paused
    assert "/finish_weight_update" not in client.events
    assert "/resume" not in client.events


def test_unsuccessful_cache_reset_cannot_publish_the_group(client, monkeypatch):
    monkeypatch.setattr(vllm_client, "_HAS_RESET_PREFIX_CACHE_SUCCESS", True)
    post = client._post

    def reject_cache_reset(url, **kwargs):
        response = post(url, **kwargs)
        if url.endswith("/reset_prefix_cache"):
            return {"success": False}
        return response

    client._post = reject_cache_reset
    client.reset_prefix_cache = lambda: VLLMClient.reset_prefix_cache(client)
    with pytest.raises(RuntimeError, match="failed to reset the prefix cache"):
        with client.weight_update():
            pass
    assert client.paused
    assert "/resume" not in client.events
