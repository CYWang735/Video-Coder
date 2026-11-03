from __future__ import annotations

import base64
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.tasks.video_coder import similarity as sim_mod
from uni_agent.tasks.video_coder.similarity import (
    SimilarityRewardConfig,
    combine_reward,
    exe_weight,
    parse_score,
    score_rollout,
)
from uni_agent.tasks.video_coder.task import VideoCoderTask, VideoCoderTaskConfig


def _sim_cfg(tmp_path, **overrides) -> SimilarityRewardConfig:
    raw = {
        "ref2va": {"base_url": "http://h3:30010", "shared_dir": str(tmp_path / "shared"), "poll_interval": 0},
        "judge": {"base_url": "http://vlm:8000/v1", "model": "judge-vlm", "rubric": "overall"},
        **overrides,
    }
    return SimilarityRewardConfig(**raw)


class FakeServices:
    """MockTransport playing the MiniMax-H3 /v1/videos API and an OpenAI-compatible VLM."""

    def __init__(self, score_reply='{"reason": "close", "score": 0.7}', fail_job=False):
        self.score_reply, self.fail_job = score_reply, fail_job
        self.submitted, self.judge_body, self.polls = None, None, 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/videos":
            self.submitted = json.loads(request.content)
            return httpx.Response(200, json={"id": "job-1"})
        if request.method == "GET" and path == "/v1/videos/job-1":
            self.polls += 1
            if self.fail_job:
                return httpx.Response(200, json={"status": "failed", "error": {"msg": "oom"}})
            return httpx.Response(200, json={"status": "completed" if self.polls >= 2 else "in_progress"})
        if request.method == "GET" and path == "/v1/videos/job-1/content":
            return httpx.Response(200, content=b"GENERATED-VIDEO")
        if request.method == "POST" and path == "/v1/chat/completions":
            self.judge_body = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": self.score_reply}}]})
        return httpx.Response(404)


async def _no_sleep(_):
    return None


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"score": 0.85}', 0.85),
        ('{"score": .6}', 0.6),
        ('{"score": 1}', 1.0),
        ('{"score": 1.4}', 1.0),
        ('{"score": "0.3"}', 0.3),
    ],
)
def test_parse_score_unit_scale(reply, expected):
    assert parse_score(reply, 1.0) == pytest.approx(expected)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"score": 7, "reason": "x"}', 0.7),
        ('```json\n{"score": "10"}\n```', 1.0),
        ('{"score": 12}', 1.0),
        ('{"score": -3}', 0.0),
        ('Sure. {"reason": "ok", "score": 4.5}', 0.45),
    ],
)
def test_parse_score(reply, expected):
    assert parse_score(reply, 10.0) == pytest.approx(expected)


@pytest.mark.cpu
def test_parse_score_rejects_reply_without_score():
    with pytest.raises(ValueError):
        parse_score("the videos look similar", 10.0)


@pytest.mark.cpu
def test_combine_reward(tmp_path):
    cfg = _sim_cfg(tmp_path)
    assert combine_reward(cfg, False, 0.9) == 0.0
    assert combine_reward(cfg, True, 0.0) == 1.0
    assert combine_reward(cfg, True, 0.8) == pytest.approx(1.4)
    assert combine_reward(cfg, True, 0.8, global_step=100) == pytest.approx(1.4)  # auto without a trainer total


@pytest.mark.cpu
def test_combine_reward_anneals_execution_term(tmp_path):
    cfg = _sim_cfg(tmp_path).model_copy(update={"exe_anneal_steps": 100})
    assert exe_weight(cfg, None) == 1.0  # validation: no step
    assert exe_weight(cfg, 0) == 1.0
    assert combine_reward(cfg, True, 0.8, global_step=50) == pytest.approx(math.exp(-0.5) + 0.4)
    assert combine_reward(cfg, True, 0.8, global_step=100) == pytest.approx(math.exp(-1) + 0.4)
    assert combine_reward(cfg, False, 0.8, global_step=50) == 0.0


@pytest.mark.cpu
def test_exe_anneal_steps_auto_uses_total_training_steps(tmp_path):
    cfg = _sim_cfg(tmp_path)
    assert cfg.exe_anneal_steps == "auto"  # on by default
    assert exe_weight(cfg, 50, 100) == pytest.approx(math.exp(-0.5))
    assert exe_weight(cfg, 50, None) == 1.0  # no trainer total: not annealed
    assert exe_weight(cfg.model_copy(update={"exe_anneal_steps": None}), 50, 100) == 1.0
    with pytest.raises(ValueError):
        SimilarityRewardConfig(**{**cfg.model_dump(), "exe_anneal_steps": 0})


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_score_rollout_runs_ref2va_then_judge(tmp_path):
    cfg = _sim_cfg(tmp_path)
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT-VIDEO")
    services = FakeServices()

    out = await score_rollout(
        cfg, b"WHITE-VIDEO", str(gt), "a ball rolls", tag="s1", transport=httpx.MockTransport(services), sleep=_no_sleep
    )

    assert out["similarity"] == pytest.approx(0.7)
    body = services.submitted
    assert body["task"] == "ref2va" and body["prompt"] == "a ball rolls" and body["seed"] == 2029
    [cond] = body["conditions"]
    assert cond["type"] == "video" and cond["role"] == "reference" and cond["uri"].startswith("file://")
    assert cond["uri"].endswith("/white.mp4") and str(tmp_path / "shared") in cond["uri"]
    videos = [p["video_url"]["url"] for p in services.judge_body["messages"][0]["content"] if p["type"] == "video_url"]
    assert [base64.b64decode(v.split(",", 1)[1]) for v in videos] == [b"GT-VIDEO", b"GENERATED-VIDEO"]
    assert services.judge_body["model"] == "judge-vlm"
    assert list((tmp_path / "shared").iterdir()) == []  # working files removed (keep_files=False)


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_score_rollout_raises_when_ref2va_job_fails(tmp_path):
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    with pytest.raises(RuntimeError, match="failed"):
        await score_rollout(
            _sim_cfg(tmp_path),
            b"W",
            str(gt),
            "p",
            transport=httpx.MockTransport(FakeServices(fail_job=True)),
            sleep=_no_sleep,
        )


# ---- task.run with a fake sandbox/agent -------------------------------------------------------------


class FakeSandbox:
    def __init__(self):
        self.reads = []

    async def read_file(self, path):
        self.reads.append(path)
        return b"RENDER"


class FakeAgent:
    def __init__(self, accepted):
        self.accepted = accepted

    async def run(self, *, sandbox, messages, workdir=None):
        out = {
            "exit_status": "Submitted" if self.accepted else "LimitsExceeded",
            "accepted": self.accepted,
            "final_video": "/tmp/video_coder_final/final_run/output.mp4" if self.accepted else None,
        }
        return AgentResult(output=out, transcript=list(messages), info={}, finished=self.accepted)


def _task(tmp_path, monkeypatch, *, accepted, similarity=True, gt_exists=True, on_error="raise", scorer=None):
    gt = tmp_path / "gt.mp4"
    if gt_exists:
        gt.write_bytes(b"GT")
    raw = {
        "sandbox": {"provider": "apptainer"},
        "prompt": [{"role": "user", "content": "a ball rolls"}],
        "metadata": {"id": "s1", "gt_video": str(gt)},
    }
    if similarity:
        raw["similarity"] = {**_sim_cfg(tmp_path).model_dump(), "on_error": on_error}
    task = VideoCoderTask(VideoCoderTaskConfig(**raw))
    sandbox = FakeSandbox()

    @asynccontextmanager
    async def build_sandbox():
        yield sandbox

    monkeypatch.setattr(task, "build_sandbox", build_sandbox)
    monkeypatch.setattr(task, "build_agent", lambda: FakeAgent(accepted))
    calls, overrides_seen, descriptions = [], [], []
    task.overrides_seen, task.descriptions = overrides_seen, descriptions

    async def fake_score(cfg, render, gt_video, prompt, *, description="", tag="", request_overrides=None):
        calls.append((render, gt_video, prompt, tag))
        overrides_seen.append(request_overrides)
        descriptions.append(description)
        if scorer is not None:
            return await scorer()
        return {"similarity": 0.8, "judge_reply": '{"score": 8}'}

    monkeypatch.setattr("uni_agent.tasks.video_coder.task.score_rollout", fake_score)
    return task, sandbox, calls, gt


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_accepted_rollout_gets_weighted_similarity(tmp_path, monkeypatch):
    task, sandbox, calls, gt = _task(tmp_path, monkeypatch, accepted=True)
    result = await task.run()
    assert result.reward == pytest.approx(1.4)  # 1.0 * 1 + 0.5 * 0.8
    assert result.accuracy == 1.0
    assert sandbox.reads == ["/tmp/video_coder_final/final_run/output.mp4"]
    [(render, gt_video, prompt, tag)] = calls
    assert (render, gt_video, tag) == (b"RENDER", str(gt), "s1")
    assert "Scene Description:\na ball rolls\n" in prompt and prompt.endswith("Reference Video:\n<Video 1>")
    assert "{{" not in prompt
    assert result.extra_info["similarity"]["similarity"] == 0.8


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_anneals_execution_reward_by_global_step(tmp_path, monkeypatch):
    task, _, _, _ = _task(tmp_path, monkeypatch, accepted=True)
    # set by run_task from tools_kwargs["_global_steps"] / ["_total_training_steps"] for training rollouts
    task.config.global_steps, task.config.total_training_steps = 5, 10
    result = await task.run()
    assert result.reward == pytest.approx(math.exp(-0.5) + 0.4)
    assert result.extra_info["exe_weight"] == pytest.approx(math.exp(-0.5))


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_unaccepted_rollout_is_not_scored(tmp_path, monkeypatch):
    task, sandbox, calls, _ = _task(tmp_path, monkeypatch, accepted=False)
    result = await task.run()
    assert result.reward == 0.0 and calls == [] and sandbox.reads == []


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_without_similarity_keeps_acceptance_reward(tmp_path, monkeypatch):
    task, sandbox, calls, _ = _task(tmp_path, monkeypatch, accepted=True, similarity=False)
    result = await task.run()
    assert result.reward == 1.0 and calls == [] and sandbox.reads == []


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_scoring_error_policies(tmp_path, monkeypatch):
    async def boom():
        raise httpx.ConnectError("h3 down")

    task, *_ = _task(tmp_path, monkeypatch, accepted=True, on_error="zero", scorer=boom)
    result = await task.run()
    assert result.reward == 1.0 and "ConnectError" in result.extra_info["similarity"]["error"]

    (tmp_path / "b").mkdir()
    task, *_ = _task(tmp_path / "b", monkeypatch, accepted=True, on_error="raise", scorer=boom)
    with pytest.raises(httpx.ConnectError):
        await task.run()


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_missing_gt_fails_before_rollout(tmp_path, monkeypatch):
    task, sandbox, calls, _ = _task(tmp_path, monkeypatch, accepted=True, gt_exists=False)
    with pytest.raises(FileNotFoundError):
        await task.run()
    assert sandbox.reads == [] and calls == []


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_uses_per_sample_gen_prompt_when_present(tmp_path, monkeypatch):
    task, _, calls, _ = _task(tmp_path, monkeypatch, accepted=True)
    task.config.metadata["gen_prompt"] = "custom GT prompt"
    await task.run()
    assert calls[0][2] == "custom GT prompt"


# ---- multiple H3 servers, judge options ----------------------------------------------------------


class MultiServer(FakeServices):
    """h3-a refuses connections, h3-b and h3-c serve jobs."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.hosts = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/videos" and request.method == "POST":
            self.hosts.append(request.url.host)
            if request.url.host == "h3-a":
                raise httpx.ConnectError("refused", request=request)
        return super().__call__(request)


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_ref2va_fails_over_to_next_server(tmp_path, monkeypatch):
    cfg = _sim_cfg(tmp_path)
    cfg.ref2va.base_urls = ["http://h3-a:30010", "http://h3-b:30010"]
    monkeypatch.setattr(sim_mod.random, "random", lambda: 0.0)  # deterministic order: h3-a first
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    services = MultiServer()
    out = await score_rollout(cfg, b"W", str(gt), "p", transport=httpx.MockTransport(services), sleep=_no_sleep)
    assert services.hosts == ["h3-a", "h3-b"] and out["ref2va_server"] == "http://h3-b:30010"


@pytest.mark.cpu
def test_server_order_prefers_least_loaded(monkeypatch):
    monkeypatch.setattr(sim_mod, "_H3_INFLIGHT", sim_mod.collections.Counter({"http://a:1": 2, "http://b:1": 0}))
    assert sim_mod._server_order(["http://a:1", "http://b:1/"]) == ["http://b:1", "http://a:1"]


@pytest.mark.cpu
def test_single_base_url_is_accepted(tmp_path):
    cfg = SimilarityRewardConfig(
        ref2va={"base_url": "http://h3:1", "shared_dir": str(tmp_path)}, judge={"base_url": "http://j/v1", "model": "m"}
    )
    assert cfg.ref2va.base_urls == ["http://h3:1"]


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_judge_extra_body_and_inline_size_limit(tmp_path):
    cfg = _sim_cfg(tmp_path)
    cfg.judge.extra_body = {"modalities": ["text"]}
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    services = FakeServices()
    await score_rollout(cfg, b"W", str(gt), "p", transport=httpx.MockTransport(services), sleep=_no_sleep)
    assert services.judge_body["modalities"] == ["text"]

    cfg.judge.max_inline_mb = 1e-6  # anything is too big
    with pytest.raises(ValueError, match="max_inline_mb"):
        await score_rollout(cfg, b"W", str(gt), "p", transport=httpx.MockTransport(FakeServices()), sleep=_no_sleep)


# ---- load spreading, transient errors, streaming ------------------------------------------------


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_concurrent_jobs_spread_over_servers(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(sim_mod, "_H3_INFLIGHT", sim_mod.collections.Counter())
    monkeypatch.setattr(sim_mod.random, "random", lambda: 0.0)  # ties resolved by list order
    cfg = _sim_cfg(tmp_path).ref2va
    cfg.base_urls = ["http://h3-a:1", "http://h3-b:1"]
    hosts, polls = [], {}

    async def handler(request):
        await asyncio.sleep(0)  # let the other submission run while this one is in flight
        if request.method == "POST":
            hosts.append(request.url.host)
            return httpx.Response(200, json={"id": f"job-{request.url.host}"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"V")
        polls[request.url.host] = polls.get(request.url.host, 0) + 1
        return httpx.Response(200, json={"status": "completed" if polls[request.url.host] > 2 else "queued"})

    ref = tmp_path / "w.mp4"
    ref.write_bytes(b"W")
    await asyncio.gather(
        *(
            sim_mod.generate_ref2va(
                cfg, ref, "p", tmp_path / f"o{i}.mp4", transport=httpx.MockTransport(handler), sleep=_no_sleep
            )
            for i in range(2)
        )
    )
    assert sorted(hosts) == ["h3-a", "h3-b"]
    assert sum(sim_mod._H3_INFLIGHT.values()) == 0


class Flaky(FakeServices):
    """Status requests fail `fail_polls` times (connection error) before the normal flow."""

    def __init__(self, fail_polls, **kw):
        super().__init__(**kw)
        self.fail_polls = fail_polls

    def __call__(self, request):
        if request.method == "GET" and request.url.path == "/v1/videos/job-1" and self.fail_polls > 0:
            self.fail_polls -= 1
            raise httpx.ConnectError("blip", request=request)
        return super().__call__(request)


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_poll_rides_out_transient_errors_but_not_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_mod, "_H3_INFLIGHT", sim_mod.collections.Counter())
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    cfg = _sim_cfg(tmp_path)  # max_poll_errors=5
    out = await score_rollout(cfg, b"W", str(gt), "p", transport=httpx.MockTransport(Flaky(3)), sleep=_no_sleep)
    assert out["similarity"] == pytest.approx(0.7)

    with pytest.raises(RuntimeError, match="failed 6 times"):
        await score_rollout(cfg, b"W", str(gt), "p", transport=httpx.MockTransport(Flaky(100)), sleep=_no_sleep)
    assert sum(sim_mod._H3_INFLIGHT.values()) == 0  # slot released on failure too


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_submit_read_timeout_does_not_fail_over(tmp_path, monkeypatch):
    """After a read timeout the first server may already run the job: do not start a duplicate."""
    monkeypatch.setattr(sim_mod, "_H3_INFLIGHT", sim_mod.collections.Counter())
    monkeypatch.setattr(sim_mod.random, "random", lambda: 0.0)
    cfg = _sim_cfg(tmp_path).ref2va
    cfg.base_urls = ["http://h3-a:1", "http://h3-b:1"]
    posts = []

    def handler(request):
        posts.append(request.url.host)
        raise httpx.ReadTimeout("slow", request=request)

    ref = tmp_path / "w.mp4"
    ref.write_bytes(b"W")
    with pytest.raises(httpx.ReadTimeout):
        await sim_mod.generate_ref2va(
            cfg, ref, "p", tmp_path / "o.mp4", transport=httpx.MockTransport(handler), sleep=_no_sleep
        )
    assert posts == ["h3-a"] and sum(sim_mod._H3_INFLIGHT.values()) == 0


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_judge_streaming_reply_is_joined(tmp_path):
    cfg = _sim_cfg(tmp_path).judge
    cfg.stream = True
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        chunks = ['{"reason": "close", ', '"score": ', "0.8}"]
        sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
        return httpx.Response(
            200, content=(sse + "data: [DONE]\n\n").encode(), headers={"Content-Type": "text/event-stream"}
        )

    ref, cand = tmp_path / "r.mp4", tmp_path / "c.mp4"
    ref.write_bytes(b"R")
    cand.write_bytes(b"C")
    out = await sim_mod.judge_similarity(cfg, ref, cand, transport=httpx.MockTransport(handler))
    assert seen["body"]["stream"] is True and out["similarity"] == pytest.approx(0.8)


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_per_sample_request_overrides(tmp_path):
    """metadata.gen_request replaces request keys (the GT's own duration/seed); None (parquet fill) is ignored."""
    cfg = _sim_cfg(tmp_path)
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    services = FakeServices()
    overrides = {
        "seconds": 7.0,
        "target": {"short_edge": 480, "aspect_ratio": "auto", "duration_seconds": 7.0},
        "seed": None,
    }
    await score_rollout(
        cfg, b"W", str(gt), "p", transport=httpx.MockTransport(services), sleep=_no_sleep, request_overrides=overrides
    )
    body = services.submitted
    assert body["seconds"] == 7.0 and body["target"]["duration_seconds"] == 7.0
    assert body["seed"] == cfg.ref2va.request["seed"]  # None did not erase the default


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_passes_gen_request_to_scoring(tmp_path, monkeypatch):
    task, *_ = _task(tmp_path, monkeypatch, accepted=True)
    task.config.metadata["gen_request"] = {"seconds": 7.0}
    await task.run()
    assert task.overrides_seen == [{"seconds": 7.0}]


# ---- judge.rubric structured: description + sub-scores, weighted here, faithfulness gate -----------

STRUCTURED_REPLY = (
    '```json\n{"rationale": "Constraints: four people around a table. The table is there.", '
    '"faithfulness": 0.75, "layout": 0.5, "motion": 0.25, "camera": 1, "motion_na": false}\n```'
)
STRUCTURED_SIM = 0.25 * 0.75 + 0.25 * 0.5 + 0.3 * 0.25 + 0.2 * 1.0


def _structured_cfg(tmp_path, **overrides) -> SimilarityRewardConfig:
    return _sim_cfg(
        tmp_path, judge={"base_url": "http://vlm:8000/v1", "model": "judge-vlm", "rubric": "structured"}, **overrides
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reply,scores,motion_na",
    [
        (STRUCTURED_REPLY, {"faithfulness": 0.75, "layout": 0.5, "motion": 0.25, "camera": 1.0}, False),
        (
            'Result: {"rationale": "x", "faithfulness": "0.5", "layout": 0, "motion": 0.5, "camera": 1.5, '
            '"motion_na": "TRUE"} done',
            {"faithfulness": 0.5, "layout": 0.0, "motion": 0.5, "camera": 1.0},
            True,
        ),
        # invalid JSON (unescaped quotes in the rationale): per-key fallback
        (
            '{"rationale": "a "red" cube", "faithfulness": 0.25, "layout": 0.25, "motion": 0, "camera": 0.5}',
            {"faithfulness": 0.25, "layout": 0.25, "motion": 0.0, "camera": 0.5},
            False,
        ),
    ],
)
def test_parse_structured_judgement(reply, scores, motion_na):
    assert sim_mod.parse_structured_judgement(reply) == {"scores": scores, "motion_na": motion_na}


@pytest.mark.cpu
def test_parse_structured_judgement_rejects_missing_sub_score():
    with pytest.raises(ValueError, match="layout"):
        sim_mod.parse_structured_judgement('{"faithfulness": 1, "motion": 1, "camera": 1}')


@pytest.mark.cpu
def test_structured_similarity_weights_and_motion_na():
    weights = sim_mod.DEFAULT_STRUCTURED_WEIGHTS
    scores = {"faithfulness": 0.75, "layout": 0.5, "motion": 0.25, "camera": 1.0}
    assert sim_mod.structured_similarity(scores, False, weights) == pytest.approx(STRUCTURED_SIM)
    without_motion = (0.25 * 0.75 + 0.25 * 0.5 + 0.2 * 1.0) / 0.7
    assert sim_mod.structured_similarity(scores, True, weights) == pytest.approx(without_motion)
    assert sim_mod.structured_similarity(dict.fromkeys(scores, 1.0), True, weights) == pytest.approx(1.0)


@pytest.mark.cpu
def test_faithfulness_gate(tmp_path):
    cfg = _structured_cfg(tmp_path)
    assert combine_reward(cfg, True, 0.6, faithfulness=0.0) == 0.0  # accepted, but not the described scene
    assert combine_reward(cfg, True, 0.6, faithfulness=0.25) == pytest.approx(1.3)
    assert combine_reward(cfg, True, 0.6) == pytest.approx(1.3)  # rubric overall: no faithfulness, no gate
    assert combine_reward(cfg, False, 0.6, faithfulness=1.0) == 0.0
    cfg.faithfulness_gate = None
    assert combine_reward(cfg, True, 0.6, faithfulness=0.0) == pytest.approx(1.3)


@pytest.mark.cpu
def test_judge_weights_are_checked():
    with pytest.raises(ValueError, match="weights"):
        SimilarityRewardConfig.model_validate(
            {
                "ref2va": {"base_url": "http://h3:1", "shared_dir": "/tmp/x"},
                "judge": {"base_url": "http://j/v1", "model": "m", "weights": {"faithfulness": 1.0}},
            }
        )


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_structured_judge_gets_description_and_both_h3_videos(tmp_path):
    cfg = _structured_cfg(tmp_path, keep_files=True)
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT-VIDEO")
    services = FakeServices(score_reply=STRUCTURED_REPLY)

    out = await score_rollout(
        cfg,
        b"WHITE-VIDEO",
        str(gt),
        "ref2va prompt",
        description="  four friends at a table \n",
        tag="s1",
        transport=httpx.MockTransport(services),
        sleep=_no_sleep,
    )

    assert services.submitted["prompt"] == "ref2va prompt"  # H3 still renders the white model
    content = services.judge_body["messages"][0]["content"]
    videos = [base64.b64decode(p["video_url"]["url"].split(",", 1)[1]) for p in content if p["type"] == "video_url"]
    assert videos == [b"GT-VIDEO", b"GENERATED-VIDEO"]  # Video 1 = GT video, Video 2 = H3 video of the render
    prompt = content[-1]["text"]
    assert prompt.startswith("You are a strict, professional evaluator")
    assert prompt.endswith("Scene description:\nfour friends at a table") and "{{" not in prompt
    assert out["scores"] == {"faithfulness": 0.75, "layout": 0.5, "motion": 0.25, "camera": 1.0}
    assert out["motion_na"] is False and out["similarity"] == pytest.approx(STRUCTURED_SIM)
    kept = json.loads((Path(out["work_dir"]) / "judge.json").read_text())
    assert kept["scores"]["camera"] == 1.0 and kept["gt_video"] == str(gt)


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_overall_judge_prompt_is_unchanged(tmp_path):
    gt = tmp_path / "gt.mp4"
    gt.write_bytes(b"GT")
    services = FakeServices()
    await score_rollout(
        _sim_cfg(tmp_path),
        b"W",
        str(gt),
        "p",
        description="d",
        transport=httpx.MockTransport(services),
        sleep=_no_sleep,
    )
    assert services.judge_body["messages"][0]["content"][-1]["text"] == sim_mod.DEFAULT_JUDGE_PROMPT


@pytest.mark.cpu
@pytest.mark.asyncio
async def test_task_structured_judge_and_gate(tmp_path, monkeypatch):
    async def judged(faithfulness):
        scores = {"faithfulness": faithfulness, "layout": 0.5, "motion": 0.5, "camera": 0.5}
        return {"similarity": 0.6, "scores": scores, "motion_na": False, "judge_reply": "{}"}

    (tmp_path / "a").mkdir()
    task, _, calls, _ = _task(tmp_path / "a", monkeypatch, accepted=True, scorer=lambda: judged(0.75))
    result = await task.run()
    assert result.reward == pytest.approx(1.3) and "gated" not in result.extra_info["similarity"]
    assert task.descriptions == ["a ball rolls"]  # the scene description goes to the judge
    assert "Scene Description:\na ball rolls" in calls[0][2]  # and the Ref2VA prompt is still the GT recipe

    (tmp_path / "b").mkdir()
    task, *_ = _task(tmp_path / "b", monkeypatch, accepted=True, scorer=lambda: judged(0.0))
    result = await task.run()
    assert result.reward == 0.0 and result.accuracy == 1.0  # accepted by the harness, gated by the judge
    assert result.extra_info["similarity"]["gated"] is True
