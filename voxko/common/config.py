"""설정 로더 — 세 축 병합 + 스키마 검사 + 검증.

병합 순서 (뒤가 우선):
    configs/config.yaml → configs/hardware/<gpu>.yaml → configs/experiments/<계열>/<ID>.yaml
    → --overlay (여러 개, 순서대로) → --set key=value / --max_steps

· configs/config.yaml 이 **스키마**다. 다른 파일·--set 에 config.yaml 에 없는 키가 있으면 에러(비슷한 키 힌트).
  → `train.eval_evry` 같은 오타가 조용히 무시되던 문제 방지(cv3_koft 이식). 새 키는 그 키를 읽는 코드와 함께
    config.yaml 에 추가한다.
· 각 파일을 읽은 직후 ${VAR} / ${VAR:-기본값} 환경변수를 풀고(~ 도 확장) 병합 → 환경변수가 항상 최우선.
· YAML 1.1 은 `1e-5` 를 문자열로 읽는다 → 숫자로 읽는 로더를 쓴다.
· dict 는 재귀 병합, list·스칼라는 통째로 교체.
· 병합 후 `validate` 로 검사하고, 파생값(run_name, run_dir, tb_dir, 모델·데이터 절대경로)을 cfg["_derived"],
  실행 정보(입력 파일·--set)를 cfg["_run"] 에 넣는다. 밑줄 키는 코드가 만든 값이다.

>>> deep_merge({"a": {"b": 1, "c": [1]}}, {"a": {"c": [2]}})
{'a': {'b': 1, 'c': [2]}}
>>> yaml_value("1e-5")
1e-05
"""
from __future__ import annotations

import copy
import difflib
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from .paths import CONFIGS_DIR, DEFAULT_CONFIG, REPO_ROOT, resolve

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
SCHEDULERS = ("constant_warmup", "cosine_warmup")


class ConfigError(ValueError):
    """설정이 잘못됐을 때. 메시지에 고치는 방법을 함께 적는다."""


class _Loader(yaml.SafeLoader):
    """소수점 없는 지수 표기(1e-5)도 float 로 읽는 YAML 로더."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\.[0-9_]+(?:[eE][-+][0-9]+)?
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def yaml_value(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader) or {}


def expand_env(value: Any) -> Any:
    """str/dict/list 에 ${VAR}·${VAR:-default} 와 선두 ~ 를 재귀 적용한다.

    >>> os.environ["VOX_DOCTEST"] = "/x"
    >>> expand_env({"p": "${VOX_DOCTEST:-/y}/a", "q": ["${VOX_NOPE:-/d}"]})
    {'p': '/x/a', 'q': ['/d']}
    """
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, str):
        prev = None
        while prev != value:              # 중첩 ${A:-${B}} 까지 풀릴 때까지
            prev = value
            value = _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
        if value.startswith("~"):
            value = os.path.expanduser(value)
        return value
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if v is None and isinstance(out.get(k), dict):
            continue                      # 내용 없는 섹션(`logging:` 만 있는 줄)이 기본 섹션 전체를 지우지 않게
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_set_args(items: Iterable[str]) -> dict:
    """['optim.lr=2e-5', 'data.batch_size=4'] → 중첩 dict (값은 YAML 로 해석: 숫자·bool·list·null)."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"--set 은 key.sub=value 형식이어야 합니다: {it!r}")
        k, v = it.split("=", 1)
        cur = out
        parts = k.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml_value(v)
    return out


def check_keys(schema: dict, layer: dict, src: str, prefix: str = "") -> None:
    """layer 의 모든 키가 schema(config.yaml)에 있는지. 없으면 비슷한 키를 힌트로 ConfigError."""
    for k, v in (layer or {}).items():
        key = f"{prefix}{k}"
        if k not in schema:
            near = difflib.get_close_matches(str(k), [str(s) for s in schema], n=3, cutoff=0.6)
            hint = f" (혹시: {', '.join(prefix + n for n in near)})" if near else ""
            raise ConfigError(f"알 수 없는 설정 키 '{key}' — {src}{hint}\n"
                              f"  configs/config.yaml 에 없는 키는 쓸 수 없습니다(오타 방지). 새 키가 필요하면\n"
                              f"  그 키를 읽는 코드와 함께 configs/config.yaml 에 추가하세요.")
        if isinstance(v, dict) and isinstance(schema[k], dict):
            check_keys(schema[k], v, src, prefix=f"{key}.")


def resolve_named(value: str, subdir: str, what: str) -> Path:
    """'rtxpro6000_4gpu' 같은 이름 또는 경로 → configs/<subdir>/<이름>.yaml."""
    for cand in (resolve(value), CONFIGS_DIR / subdir / f"{value}.yaml"):
        if cand.is_file():
            return cand.resolve()
    avail = sorted(p.stem for p in (CONFIGS_DIR / subdir).glob("*.yaml") if not p.stem.startswith("_"))
    raise ConfigError(f"{what} 파일을 찾을 수 없습니다: {value}\n  사용 가능: {', '.join(avail) or '(없음)'}")


def load_config(hardware: str, experiment: Optional[str] = None, config: str | os.PathLike = DEFAULT_CONFIG,
                overlays: Iterable[str] = (), sets: Iterable[str] = (), overrides: Optional[dict] = None,
                check_files: bool = True) -> dict:
    """설정을 병합·검증하고 파생값을 계산해 돌려준다.

    hardware 는 필수(조용한 기본값 금지 — 1GPU 박스에 4GPU 설정이 먹는 사고 방지).
    sets: ["optim.lr=2e-5", ...] (스키마 검사 대상). overrides: 코드가 만든 덮어쓰기 dict(예: --max_steps).
    check_files: train/val JSONL·초기 모델 존재 여부까지 검사(테스트에서 False 로 끌 수 있음).
    """
    if not hardware:
        raise ConfigError("--hardware 는 필수입니다 (예: rtxpro6000_1gpu)\n  사용 가능: "
                          + ", ".join(sorted(p.stem for p in (CONFIGS_DIR / "hardware").glob("*.yaml"))))
    schema_path = resolve(config)
    schema = load_yaml(schema_path)
    layers = [("hardware", resolve_named(hardware, "hardware", "하드웨어"))]
    exp_path = None
    if experiment:
        exp_path = resolve_named(experiment, "experiments", "실험")
        if exp_path.stem.startswith("_"):
            raise ConfigError(f"'_' 로 시작하는 파일은 템플릿입니다: {exp_path.name} — 복사해 이름을 붙여 쓰세요.")
        layers.append(("experiment", exp_path))
    layers += [(f"overlay{i}", resolve(o)) for i, o in enumerate(overlays or ())]

    cfg = expand_env(copy.deepcopy(schema))
    for _, path in layers:
        if not Path(path).is_file():
            raise ConfigError(f"설정 파일이 없습니다: {path}")
        layer = load_yaml(path)
        check_keys(schema, layer, _rel(path))
        cfg = deep_merge(cfg, expand_env(layer))
    set_layer = parse_set_args(sets)
    check_keys(schema, set_layer, "--set")
    cfg = deep_merge(cfg, expand_env(set_layer))
    if overrides:
        check_keys(schema, overrides, "CLI")
        cfg = deep_merge(cfg, expand_env(overrides))

    cfg["_run"] = {"sources": {"config": str(schema_path), **{k: str(p) for k, p in layers}},
                   "sets": list(sets or []), "experiment": str(exp_path) if exp_path else None}
    cfg["_sources"] = [str(schema_path)] + [str(p) for _, p in layers]
    cfg["_derived"] = derive(cfg, exp_path)
    validate(cfg, check_files=check_files)
    return cfg


def _rel(path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def derive(cfg: dict, exp_path: Optional[Path]) -> dict:
    """run_name·경로 파생값. run_name = 실험 파일명(stem) — logging.run_name 이 다르면 에러."""
    paths, log, model, data = cfg.get("paths", {}), cfg.get("logging", {}), cfg.get("model", {}), cfg.get("data", {})
    stem = exp_path.stem if exp_path else None
    if stem and log.get("run_name") and log["run_name"] != stem:
        raise ConfigError(f"logging.run_name({log['run_name']}) 이 실험 파일명({stem}) 과 다릅니다.\n"
                          f"  규약: 파일명 == run_name == 체크포인트·TB·exp_reports 폴더명. 둘을 같게 하세요.")
    run_name = stem or log.get("run_name")
    if not run_name:
        raise ConfigError("run_name 을 정할 수 없습니다 — --experiment 를 주세요.")
    ckpt_root, data_root = Path(paths.get("ckpt_root", ".")), Path(paths.get("data_root", "."))
    save_root = Path(paths.get("save_root", REPO_ROOT / "outputs"))
    tb_root = Path(paths.get("tb_root", save_root / "runs"))

    def under(root: Path, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        q = Path(os.path.expanduser(p))
        return str(q if q.is_absolute() else root / q)

    em = (data.get("emotion") or {}).get("map_file")
    return {"run_name": run_name, "run_dir": str(save_root / run_name), "tb_dir": str(tb_root / run_name),
            "tb_root": str(tb_root), "save_root": str(save_root),
            "init_model_dir": under(ckpt_root, model.get("init_model")),
            "train_jsonl": under(data_root, data.get("train_jsonl")),
            "val_jsonl": under(data_root, data.get("val_jsonl")),
            "val_unseen_jsonl": under(data_root, data.get("val_unseen_jsonl")),
            "val_unseen_pool_jsonl": under(data_root, data.get("val_unseen_pool_jsonl")),
            "ref_pool_jsonl": under(data_root, (data.get("ref") or {}).get("pool_jsonl")),
            "emotion_map": str(resolve(em)) if em else None,
            "extra_sources": [{**e, "train_jsonl": under(data_root, e.get("train_jsonl"))}
                              for e in (data.get("extra_sources") or []) if isinstance(e, dict)]}


# 목록형 설정의 항목 스키마(check_keys 는 list 안을 보지 않으므로 여기서 검사) — 값 = 기본값(None = 필수)
EXTRA_SOURCE_KEYS = {"name": None, "train_jsonl": None, "per_step": None}
_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _check_item_keys(item, allowed: dict, where: str, errs: list) -> bool:
    if not isinstance(item, dict):
        errs.append(f"{where} 는 dict 여야 합니다 (현재: {item!r})")
        return False
    for k in item:
        if k not in allowed:
            near = difflib.get_close_matches(str(k), list(allowed), n=3, cutoff=0.6)
            errs.append(f"{where}: 알 수 없는 키 '{k}'" + (f" (혹시: {', '.join(near)})" if near else "")
                        + f" — 가능한 키: {', '.join(allowed)}")
    return True


def with_defaults(item: dict, allowed: dict) -> dict:
    """항목 dict 에 기본값을 채운 사본(코드가 읽는 값)."""
    return {**{k: v for k, v in allowed.items() if v is not None}, **item}


def _ratio(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0


def _validate_lists(cfg: dict, errs: list, check_files: bool) -> None:
    """data.extra_sources 항목 검사."""
    dc, d = cfg.get("data", {}), cfg["_derived"]
    es = dc.get("extra_sources") or []
    if not isinstance(es, list):
        errs.append("data.extra_sources 는 목록이어야 합니다 (끄려면 [])")
        return
    if es and dc.get("max_tokens"):
        errs.append("data.extra_sources 는 data.max_tokens(동적 배치)와 함께 쓸 수 없습니다 — per_step 을 문장 수로 맞추려면 batch_size 고정이 필요")
    bs, world = dc.get("batch_size"), cfg.get("hardware", {}).get("expected_num_processes")
    enames = []
    for i, e in enumerate(es):
        w = f"data.extra_sources[{i}]"
        if not _check_item_keys(e, EXTRA_SOURCE_KEYS, w, errs):
            continue
        n = e.get("name")
        if not (isinstance(n, str) and _NAME_RE.match(n)) or n in enames:
            errs.append(f"{w}.name={n!r} — 영문·숫자·_ 로, 겹치지 않게")
        enames.append(n)
        if not e.get("train_jsonl"):
            errs.append(f"{w}.train_jsonl 이 비어 있습니다")
        ps = e.get("per_step")
        if not isinstance(ps, int) or ps <= 0:
            errs.append(f"{w}.per_step 은 양의 정수여야 합니다 (현재: {ps!r})")
        elif isinstance(bs, int) and isinstance(world, int) and ps % (bs * world):
            errs.append(f"{w}.per_step={ps} 는 batch_size × GPU 수 = {bs} × {world} = {bs * world} 의 배수여야 합니다")
    if not (check_files and cfg["_run"].get("experiment")):
        return
    for i, e in enumerate(d.get("extra_sources", [])):
        f = e.get("train_jsonl")
        if f and not Path(f).exists():
            errs.append(f"data.extra_sources[{i}]({e.get('name')}).train_jsonl 이 없습니다: {f}")


def validate(cfg: dict, check_files: bool = True) -> None:
    errs = []
    d, tr, oc, dc = cfg["_derived"], cfg.get("train", {}), cfg.get("optim", {}), cfg.get("data", {})
    for k in ("ckpt_root", "data_root", "save_root", "tb_root"):
        if not cfg.get("paths", {}).get(k):
            errs.append(f"paths.{k} 가 비어 있습니다")
    ms = tr.get("max_steps")
    if not isinstance(ms, int) or ms <= 0:
        errs.append(f"train.max_steps 는 양의 정수여야 합니다 (현재: {ms!r})")
    if not isinstance(tr.get("grad_accum"), int) or tr["grad_accum"] < 1:
        errs.append(f"train.grad_accum 은 1 이상의 정수여야 합니다 (현재: {tr.get('grad_accum')!r})")
    for k in ("log_every", "eval_every", "save_every"):
        if not isinstance(tr.get(k), int) or tr[k] < 0:
            errs.append(f"train.{k} 는 0 이상의 정수여야 합니다(0 = 끔) (현재: {tr.get(k)!r})")
    rc = tr.get("resume") or {}
    if not isinstance(rc.get("enable"), bool):
        errs.append(f"train.resume.enable 은 true | false 여야 합니다 (현재: {rc.get('enable')!r})")
    elif rc["enable"]:
        ev, kp = rc.get("every"), rc.get("keep")
        if not isinstance(ev, int) or ev < 0:
            errs.append(f"train.resume.every 는 0 이상의 정수여야 합니다 (현재: {ev!r})")
        if not isinstance(kp, int) or kp < 1:
            errs.append(f"train.resume.keep 은 1 이상의 정수여야 합니다 (현재: {kp!r})")
    lam = tr.get("lambdas") or {}
    for k in ("diff", "stop"):
        if not isinstance(lam.get(k), (int, float)) or lam[k] < 0:
            errs.append(f"train.lambdas.{k} 는 0 이상의 수여야 합니다 (현재: {lam.get(k)!r})")
    if oc.get("scheduler") not in SCHEDULERS:
        errs.append(f"optim.scheduler={oc.get('scheduler')!r} — 가능한 값: {', '.join(SCHEDULERS)}")
    if cfg.get("precision", {}).get("mixed_precision") != "bf16":
        errs.append("precision.mixed_precision 은 bf16 만 허용합니다 (fp16 은 NaN 위험)")
    ec, rf = dc.get("emotion") or {}, dc.get("ref") or {}
    if not d.get("emotion_map") or not Path(d["emotion_map"]).is_file():
        errs.append(f"data.emotion.map_file 이 없습니다: {ec.get('map_file')!r}")
    do = ec.get("drop_only")
    if do is not None and not (isinstance(do, list) and do and all(isinstance(x, str) for x in do)):
        errs.append(f"data.emotion.drop_only 는 null 또는 감정 이름 목록이어야 합니다 (현재: {do!r})")
    if not _ratio(ec.get("instruct_ratio")):
        errs.append(f"data.emotion.instruct_ratio 는 0~1 이어야 합니다 (현재: {ec.get('instruct_ratio')!r})")
    if not _ratio(rf.get("ratio")):
        errs.append(f"data.ref.ratio 는 0~1 이어야 합니다 (현재: {rf.get('ratio')!r})")
    if rf.get("pool") not in ("neutral", "any"):
        errs.append(f"data.ref.pool={rf.get('pool')!r} — 가능한 값: neutral | any")
    try:
        if not (0 < float(rf.get("min_dur")) <= float(rf.get("max_dur"))):
            raise ValueError
    except (TypeError, ValueError):
        errs.append("data.ref: 0 < min_dur ≤ max_dur 이어야 합니다")
    if bool(dc.get("val_unseen_jsonl")) != bool(dc.get("val_unseen_pool_jsonl")):
        errs.append("data.val_unseen_jsonl 과 data.val_unseen_pool_jsonl 은 함께 지정해야 합니다(처음 보는 화자의 ref 풀)")
    if float(rf.get("ratio") or 0) > 0 and rf.get("pool") == "neutral" and not rf.get("pool_jsonl"):
        errs.append("data.ref.ratio > 0 · pool=neutral 이면 data.ref.pool_jsonl(무감정 ref 풀)이 필요합니다")
    if not isinstance(dc.get("max_seq_len"), int) or not (64 <= dc["max_seq_len"] <= 8192):
        errs.append(f"data.max_seq_len 은 64~8192 정수여야 합니다(공식 packer 가 8192 에서 자름) (현재: {dc.get('max_seq_len')!r})")
    lc = cfg.get("lora") or {}
    if lc.get("enable"):
        if not isinstance(lc.get("r"), int) or lc["r"] < 1:
            errs.append(f"lora.r 은 1 이상의 정수여야 합니다 (현재: {lc.get('r')!r})")
        if not (lc.get("enable_lm") or lc.get("enable_dit") or lc.get("enable_proj")):
            errs.append("lora.enable_lm · enable_dit · enable_proj 중 하나는 true 여야 합니다")
    if cfg.get("export", {}).get("dtype") not in ("bfloat16", "float32"):
        errs.append(f"export.dtype 은 bfloat16 | float32 (현재: {cfg.get('export', {}).get('dtype')!r})")
    if not cfg.get("hardware", {}).get("expected_num_processes"):
        errs.append("hardware.expected_num_processes 가 비어 있습니다 (configs/hardware/*.yaml)")
    if check_files and cfg["_run"].get("experiment"):
        for k in ("val_unseen_jsonl", "val_unseen_pool_jsonl", "ref_pool_jsonl"):      # 선택 항목: 지정했으면 있어야 함
            if d.get(k) and not Path(d[k]).exists():
                errs.append(f"{k} 가 없습니다: {d[k]}")
        for k in ("train_jsonl", "val_jsonl", "init_model_dir"):
            if not d.get(k):
                errs.append(f"{k} 가 지정되지 않았습니다 (실험 yaml 의 data.* / model.init_model)")
            elif not Path(d[k]).exists():
                hint = " — python -m voxko.data.prepare_a7n 으로 만드세요" if "jsonl" in k else \
                       " — tools/download_ckpts.py 로 받으세요"
                errs.append(f"{k} 가 없습니다: {d[k]}{hint}")
    _validate_lists(cfg, errs, check_files)
    if errs:
        raise ConfigError("설정 오류:\n  - " + "\n  - ".join(errs))


def public_view(cfg: dict) -> dict:
    """스냅샷용: 밑줄(파생) 키를 뺀, 파일로 다시 쓸 수 있는 병합 결과."""
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


def flatten(d: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} → {'a.b': 1} (TensorBoard config 표·배너용). '_' 로 시작하는 키는 건너뛴다."""
    out = {}
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
