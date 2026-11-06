"""CPU 단위 테스트: 설정(스키마 · 검증) · 샘플러(resume · val) · 체크포인트(정책 · 회전) · 데이터셋(지시문 · ref 결정성).

q3tts_koft/tests/test_common.py 를 옮기고 VoxCPM2 쪽 항목(지시문 · 무감정 ref · 16kHz · 길이 거르기)을 더했다.
데이터셋 테스트는 가짜 tokenizer · 임시 wav 를 써서 모델 · GPU 없이 돈다.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from voxko.common import ckpt
from voxko.common.config import ConfigError, deep_merge, expand_env, load_config, parse_set_args
from voxko.common.sampler import BucketBatchSampler, EvalBatchSampler
from voxko.data.dataset import PATCH_SAMPLES, VoxDataset, build_pool, load_instruct_map, n_patches

HW1, EXP = "rtxpro6000_1gpu", "configs/experiments/sft/A7N100_ref_FT_00_smoke.yaml"
EXP_LORA = "configs/experiments/lora/A7N100_ref_LoRA_00_smoke.yaml"


# ── 설정 ──
def test_merge_env_and_numbers(monkeypatch):
    monkeypatch.setenv("VOX_T", "/env")
    assert expand_env("${VOX_T:-/d}/x") == "/env/x"
    assert expand_env("${VOX_NONE:-/d}/x") == "/d/x"
    assert deep_merge({"a": {"b": 1, "l": [1]}}, {"a": {"l": [2]}}) == {"a": {"b": 1, "l": [2]}}
    assert parse_set_args(["optim.lr=2e-5", "data.max_tokens=null"]) == {"optim": {"lr": 2e-5}, "data": {"max_tokens": None}}


def test_hardware_required():
    with pytest.raises(ConfigError):
        load_config("", EXP, check_files=False)


def test_smoke_config_derives_paths():
    cfg = load_config(HW1, EXP, check_files=False)
    d = cfg["_derived"]
    assert d["run_name"] == "A7N100_ref_FT_00_smoke" and d["run_dir"].endswith("/A7N100_ref_FT_00_smoke")
    assert d["train_jsonl"].endswith("A7N_smoke100/train.jsonl")
    assert d["emotion_map"].endswith("configs/emotion/a7n_vox_v1.yaml")
    assert cfg["hardware"]["expected_num_processes"] == 1 and cfg["lora"]["enable"] is False
    assert load_config(HW1, EXP_LORA, check_files=False)["lora"]["enable"] is True


@pytest.mark.parametrize("bad,msg", [
    (["train.eval_evry=10"], "eval_every"),          # 오타 → 비슷한 키 힌트
    (["optim.scheduler=cosine"], "scheduler"),
    (["precision.mixed_precision=fp16"], "bf16"),
    (["train.max_steps=0"], "max_steps"),
    (["train.log_every=-1"], "log_every"),
    (["data.ref.ratio=1.5"], "ref.ratio"),
    (["data.ref.pool=emotion"], "ref.pool"),
    (["data.ref.min_dur=12"], "min_dur"),
    (["data.emotion.instruct_ratio=-0.1"], "instruct_ratio"),
    (["data.emotion.map_file=configs/emotion/nope.yaml"], "map_file"),
    (["data.max_seq_len=9000"], "max_seq_len"),
    (["export.dtype=fp16"], "export.dtype"),
    (["lora.enable=true", "lora.enable_lm=false", "lora.enable_dit=false"], "enable_lm"),
    (["data.ref.pool_jsonl=null"], "pool_jsonl"),
    (["data.val_unseen_pool_jsonl=null"], "val_unseen"),
    (["train.lambdas.stop=-1"], "lambdas.stop"),
    (["data.icl.ratio=3"], "data.icl"),               # 모르는 키 → 에러
])
def test_config_errors(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(HW1, EXP, sets=bad, check_files=False)


def test_template_and_name_mismatch(tmp_path):
    t = tmp_path / "_TEMPLATE.yaml"
    t.write_text("logging:\n  run_name: x\n")
    with pytest.raises(ConfigError, match="템플릿"):
        load_config(HW1, str(t), check_files=False)
    m = tmp_path / "S99.yaml"
    m.write_text("logging:\n  run_name: other\n")
    with pytest.raises(ConfigError, match="파일명"):
        load_config(HW1, str(m), check_files=False)


def test_empty_section_does_not_erase_defaults(tmp_path):
    e = tmp_path / "S98.yaml"
    e.write_text("logging:\ntrain:\n  max_steps: 7\ndata:\n  ref:\n    ratio: 0\n")
    cfg = load_config(HW1, str(e), check_files=False)
    assert cfg["logging"]["tensorboard"] is True and cfg["train"]["max_steps"] == 7
    assert cfg["data"]["ref"]["pool"] == "neutral" and cfg["data"]["ref"]["min_dur"] == 2.0


def test_resume_config_defaults_and_validation(tmp_path):
    d = tmp_path / "S96.yaml"
    d.write_text("train:\n  max_steps: 7\ndata:\n  ref:\n    ratio: 0\n")
    cfg = load_config(HW1, str(d), check_files=False)
    assert cfg["train"]["resume"] == {"enable": True, "every": 500, "keep": 1}
    e = tmp_path / "S97.yaml"
    e.write_text("train:\n  resume:\n    keep: 0\ndata:\n  ref:\n    ratio: 0\n")
    with pytest.raises(Exception, match="resume.keep"):
        load_config(HW1, str(e), check_files=False)


# ── 샘플러 ──
LENS = [((i * 37) % 200) + 10 for i in range(203)]


@pytest.mark.parametrize("n_rep", [1, 2, 4])
def test_train_sampler_equal_disjoint_and_window_multiple(n_rep):
    shards = [list(BucketBatchSampler(LENS, batch_size=4, num_replicas=n_rep, rank=r, seed=1, multiple_of=3))
              for r in range(n_rep)]
    assert len({len(s) for s in shards}) == 1 and len(shards[0]) % 3 == 0     # 랭크 동일 · 창 배수
    flat = [i for s in shards for b in s for i in b]
    assert len(flat) == len(set(flat))                                         # 중복 없음
    s = BucketBatchSampler(LENS, max_tokens=400, seed=1)
    assert all(len(b) * max(LENS[i] for i in b) <= 400 or len(b) == 1 for b in s)


def test_train_sampler_resume_offset_is_exact_continuation():
    full = list(BucketBatchSampler(LENS, batch_size=4, seed=7))
    s = BucketBatchSampler(LENS, batch_size=4, seed=7)
    s.set_epoch(0, start=10)
    assert list(s) == full[10:]                                               # 정확히 그 다음 배치부터
    s.set_epoch(1)
    assert list(s) != full                                                    # epoch 마다 다른 순서


def test_eval_sampler_same_sample_for_any_gpu_count():
    def picked(n):
        return sorted(i for r in range(n) for b in EvalBatchSampler(LENS, batch_size=4, num_replicas=n, rank=r,
                                                                    seed=3, max_batches=20) for i in b)
    assert picked(1) == picked(2) == picked(4)


def test_epoch_tag():
    sp = BucketBatchSampler([10] * 8, batch_size=4, tag_epoch=True)
    sp.set_epoch(2)
    assert all(isinstance(x, tuple) and x[1] == 2 for b in sp for x in b)


# ── 체크포인트 ──
def _mk(tmp, steps, complete=()):
    for s in steps:
        (tmp / f"step_{s}").mkdir()
    for s in complete:
        ckpt.mark_complete(tmp / f"step_{s}", step=s)


def _mk_state(tmp, step, resume=True):
    d = tmp / f"step_{step}"
    (d / "export_ft").mkdir(parents=True)
    (d / "export_ft" / "model.safetensors").write_text("w")
    if resume:
        for f in ckpt._REQUIRED:
            (d / f).write_text("x")
    ckpt.mark_complete(d, step=step, resume=resume)
    return d


def test_rotate_semantics(tmp_path):
    _mk(tmp_path, (100, 200, 300, 400), complete=(400,))
    ckpt.rotate(tmp_path, 2, protect=tmp_path / "step_100")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_100", "step_300", "step_400"]
    ckpt.rotate(tmp_path, 0, protect=tmp_path / "step_400")
    assert [p.name for p in tmp_path.iterdir()] == ["step_400"]


def test_run_policy(tmp_path):
    P = ckpt.decide_run_policy
    assert P(tmp_path, 100, resume=False, force_overwrite=False, smoke=False)["mode"] == "fresh"
    assert P(tmp_path, 100, resume=True, force_overwrite=False, smoke=False)["mode"] == "fresh"
    _mk(tmp_path, (50, 80), complete=(50,))
    with pytest.raises(SystemExit):
        P(tmp_path, 100, resume=False, force_overwrite=False, smoke=False)
    r = P(tmp_path, 100, resume=True, force_overwrite=False, smoke=False)
    assert r["mode"] == "resume" and r["resume_from"].name == "step_50"
    assert P(tmp_path, 100, resume=True, force_overwrite=False, smoke=True)["mode"] == "smoke"
    with pytest.raises(SystemExit):
        P(tmp_path, 50, resume=True, force_overwrite=False, smoke=False)


def test_resume_files_keep_and_prune(tmp_path):
    for s, r in ((250, False), (500, True), (750, False), (1000, True)):
        _mk_state(tmp_path, s, r)
    assert ckpt.latest_resumable(tmp_path).name == "step_1000"
    gone = ckpt.prune_resume(tmp_path, keep=1)
    assert [p.name for p in gone] == ["step_500"]
    assert (tmp_path / "step_500" / "export_ft" / "model.safetensors").exists()      # export 는 그대로
    assert not (tmp_path / "step_500" / "optimizer.bin").exists()


def test_policy_resumes_from_last_state_and_marks_stale(tmp_path):
    for s, r in ((500, True), (750, False)):
        _mk_state(tmp_path, s, r)
    r = ckpt.decide_run_policy(tmp_path, 3000, resume=True, force_overwrite=False, smoke=False)
    assert r["mode"] == "resume" and r["resume_from"].name == "step_500"
    assert [p.name for p in r["stale"]] == ["step_750"]


# ── 데이터셋 (지시문 · ref · 길이 · collate) ──
def _tok(s):                       # 가짜 tokenizer: 글자 하나 = token 하나
    return [ord(ch) % 5000 + 10 for ch in s]


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)

    def wav(name, sec):
        p = tmp_path / f"{name}.wav"
        sf.write(p, (rng.standard_normal(int(sec * 16000)) * 0.05).astype("float32"), 16000)
        return str(p)

    emos = ["Neutrality", "Happy-A", "Sad-B", "Angry-B"]
    rows = [{"utt": f"s{s}_{i}", "speaker": f"s{s}", "emotion": emos[i % 4], "text": f"문장 {i}  입니다.",
             "audio": wav(f"s{s}_{i}", 1.0 + 0.3 * i), "duration": 1.0 + 0.3 * i} for s in (1, 2) for i in range(8)]
    pool = [{"utt": r["utt"], "speaker": r["speaker"], "audio": r["audio"], "duration": r["duration"]}
            for r in rows if r["emotion"] == "Neutrality" and r["speaker"] == "s1"]          # s2 는 풀 없음
    imap = load_instruct_map("configs/emotion/a7n_vox_v1.yaml")
    return rows, pool, imap


def _ds(toy, **kw):
    rows, pool, imap = toy
    a = dict(instruct_ratio=1.0, ref_ratio=1.0, seed=1986, max_seq_len=1024)
    a.update(kw)
    return VoxDataset(rows, _tok, imap, a["instruct_ratio"], a["ref_ratio"], build_pool(pool, 0.5, 10.0), a["seed"], a["max_seq_len"])


def test_dataset_instruct_and_ref_rules(toy):
    ds = _ds(toy)
    for i, r in enumerate(ds.rows):
        p = ds.plan(i, 0)
        assert p["text"].startswith("(") and p["text"].endswith(r["text"]) and "  " not in p["text"]   # 지시문 + 공백 정리
        if r["speaker"] == "s2":
            assert p["ref"] is None                                                   # 풀 없는 화자 → ref 없이
        else:
            assert p["ref"] is not None and p["ref"]["utt"] != r["utt"]               # 자기 자신은 ref 로 안 씀
    assert ds.n_no_pool == 8
    ds0 = _ds(toy, instruct_ratio=0.0, ref_ratio=0.0)
    assert all(not ds0.plan(i, 0)["text"].startswith("(") and ds0.plan(i, 0)["ref"] is None for i in range(len(ds0)))


def test_dataset_plan_is_deterministic_and_epoch_dependent(toy):
    a, b = _ds(toy, ref_ratio=0.5, instruct_ratio=0.5), _ds(toy, ref_ratio=0.5, instruct_ratio=0.5)
    pa = [(a.plan(i, 3)["text"], (a.plan(i, 3)["ref"] or {}).get("utt")) for i in range(len(a))]
    assert pa == [(b.plan(i, 3)["text"], (b.plan(i, 3)["ref"] or {}).get("utt")) for i in range(len(b))]
    many = [_ds(toy, ref_ratio=0.5, instruct_ratio=0.5).plan(0, e)["text"].startswith("(") for e in range(40)]
    assert 0 < sum(many) < 40                                                         # epoch 마다 바뀜(비율대로)


def test_dataset_length_filter_and_unknown_emotion(toy):
    rows, pool, imap = toy
    ds = _ds(toy, max_seq_len=60)
    assert ds.n_dropped > 0 and all(L <= 60 for L in ds.lengths)
    with pytest.raises(ValueError, match="지시문 표에 없는 감정"):
        VoxDataset([{**rows[0], "emotion": "Surprise"}], _tok, imap, 1.0, 0.0, {}, 1, 1024)


def test_collate_counts_match_packer_rule(toy):
    ds = _ds(toy, ref_ratio=0.5)
    batch = [ds[(i, 0)] for i in range(6)]
    out = VoxDataset.collate(batch)
    assert out["text_tokens"].dtype.is_floating_point is False
    assert int(out["counts"]) == sum(n_patches(len(b["audio"])) for b in batch)
    assert n_patches(PATCH_SAMPLES) == 1 and n_patches(PATCH_SAMPLES + 1) == 2
    if "ref_audio_tokens" in out:
        no_ref = [k for k, b in enumerate(batch) if b["ref"] is None]
        assert all((out["ref_audio_tokens"][k] == -100).all() for k in no_ref)       # ref 없는 줄은 전부 PAD


def test_sample_rate_guard(toy, tmp_path):
    rows, pool, imap = toy
    p = tmp_path / "x24.wav"
    sf.write(p, np.zeros(24000, "float32"), 24000)
    ds = VoxDataset([{**rows[0], "audio": str(p)}], _tok, imap, 1.0, 0.0, {}, 1, 1024)
    with pytest.raises(ValueError, match="16kHz"):
        ds[0]


def test_prepared_smoke_data_if_present():
    """voxko.data.prepare 산출물(있으면): 16kHz · 처음 보는 화자 누출 없음 · ref 풀은 무감정만."""
    import os
    root = os.path.expanduser("~/workspaces/preprocessed/vox_koft/A7N_smoke100")
    if not os.path.isdir(root):
        pytest.skip("A7N_smoke100 없음")
    tr = [json.loads(l) for l in open(f"{root}/train.jsonl")]
    va = [json.loads(l) for l in open(f"{root}/val.jsonl")]
    unseen = set(json.load(open(os.path.expanduser(
        "~/workspaces/preprocessed/q3tts_koft/q3_extra/val_unseen8/dataset_info.json")))["speakers"])
    assert not ({r["speaker"] for r in tr} & unseen)
    assert not ({r["utt"] for r in tr} & {r["utt"] for r in va})
    assert all(sf.info(r["audio"]).samplerate == 16000 for r in tr[:20])
    pool = [json.loads(l) for l in open(f"{root}/ref_pool.jsonl")]
    neutral = {r["utt"] for r in tr if r["emotion"] == "Neutrality"}
    assert all(p["utt"] in neutral or p["source"] == "ref_neutral466" for p in pool)


# ── 웹 조사판(2026-11-05 밤): 지시문 문구 여러 개 · Neutrality 만 지시문 빼기 ──
def test_v2_map_lists_and_paren_guard(tmp_path):
    m = load_instruct_map("configs/emotion/a7n_vox_v2.yaml")
    v1 = load_instruct_map("configs/emotion/a7n_vox_v1.yaml")
    assert set(m) == set(v1) and all(len(v) == 4 for v in m.values())
    assert all(m[k][0] == v1[k][0] for k in m)                                   # ① = v1 문구
    bad = tmp_path / "bad.yaml"
    bad.write_text("instruct:\n  Neutrality: ['calm (very) tone']\n")
    with pytest.raises(ValueError, match="괄호"):
        load_instruct_map(str(bad))


def _ds_v2(toy, ratio=0.9, drop_only=("Neutrality",), ref_ratio=0.5, rows=None):
    rows0, pool, _ = toy
    m = load_instruct_map("configs/emotion/a7n_vox_v2.yaml")
    return VoxDataset(rows or rows0, _tok, m, ratio, ref_ratio, build_pool(pool, 0.5, 10.0), 1986, 1024,
                      drop_only=list(drop_only) if drop_only else None)


def test_drop_only_neutral_keeps_emotion_and_hits_ratio(toy):
    rows0, _, _ = toy
    rows = [{**r, "utt": f"{r['utt']}_{k}"} for k in range(40) for r in rows0]       # 640문장 · Neutrality 1/4
    ds = _ds_v2(toy, rows=rows)
    assert abs(ds.p_keep_drop - (1 - 0.1 * 4)) < 1e-9                               # 0.1 × 640 / 160 = 0.4 를 뺌
    plans = [ds.plan(i, e) for e in range(3) for i in range(len(ds))]
    emo = [p for e in range(3) for i, p in zip(range(len(ds)), plans[e * len(ds):(e + 1) * len(ds)]) if ds.rows[i]["emotion"] != "Neutrality"]
    assert all(p["use_instr"] for p in emo)                                          # 감정 문장은 항상 지시문
    frac = sum(p["use_instr"] for p in plans) / len(plans)
    assert 0.86 < frac < 0.94, frac                                                  # 전체 약 0.9
    phrases = {p["phrase"] for p in plans if p["use_instr"]}
    assert phrases == {0, 1, 2, 3}                                                   # 4문구 모두 쓰임
    again = _ds_v2(toy, rows=rows)
    assert [again.plan(i, 2)["text"] for i in range(50)] == [ds.plan(i, 2)["text"] for i in range(50)]   # 결정적
    item = ds[(0, 1)]
    p = ds.plan(0, 1)
    assert list(item["text_ids"]) == _tok(p["text"])                                 # 토큰 = 고른 문구의 "(지시문)문장"


def test_drop_only_infeasible_and_config_guard(toy):
    with pytest.raises(ValueError, match="불가능"):
        _ds_v2(toy, ratio=0.5)                                                       # 50% 를 Neutrality(25%) 에서 뺄 수 없음
    rows0, pool, _ = toy
    ev = VoxDataset(rows0, _tok, load_instruct_map("configs/emotion/a7n_vox_v2.yaml"), 0.5, 0.5, build_pool(pool, 0.5, 10.0),
                    1986, 1024, name="val", drop_only=["Neutrality"], strict=False)    # 평가 세트: 가능한 만큼 + 경고
    assert ev.p_keep_drop == 0.0 and ev.drop_warning and "평가 세트" in ev.instr_summary()
    with pytest.raises(ConfigError, match="drop_only"):
        load_config(HW1, EXP, sets=["data.emotion.drop_only=Neutrality"], check_files=False)


def test_iv2_configs_differ_only_in_data_rules():
    import copy
    from voxko.common.config import public_view
    for a, b in (("sft/A7N_ref_FT_01", "sft/A7N_ref_FT_02_iv2"), ("lora/A7N_ref_LoRA_01_r32", "lora/A7N_ref_LoRA_02_r32_iv2")):
        ca = public_view(load_config("rtxpro6000_2gpu", f"configs/experiments/{a}.yaml", check_files=False))
        cb = public_view(load_config("rtxpro6000_2gpu", f"configs/experiments/{b}.yaml", check_files=False))
        from voxko.common.config import flatten
        fa, fb = flatten(ca), flatten(cb)
        diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        assert diff == ["data.emotion.drop_only", "data.emotion.instruct_ratio", "data.emotion.map_file",
                        "data.ref.ratio", "logging.run_name"], diff
