#!/usr/bin/env python
"""가상 화자 일관성 — "같은 모델 · 같은 목표 화자로 만든 중립 + 감정 6칸이 **서로** 같은 목소리인가" (2026-11-06, 사용자 아이디어).

왜: q3tts 학습(D안 데이터)에 필요한 조건은 "ref(중립)와 정답(감정)이 같은 목소리"이지 "진짜 REF 와 같음"이 아니다.
    그래서 REF 와 조금 달라도 세트 안에서 한 사람처럼 들리면 "가상 화자"로 쓸 수 있다.
    위험 = 중립은 REF 쪽, 감정은 다른 쪽으로 갈라지는 경우 → (d) 갈라짐 지표.

재료: d_eval 합성음(wav/<tag>/ · 목표 16명 × 중립 2 + 감정 6칸 × 2) + 시범(wav_trial/<tag>/ · 중립 20) · 잣대 WavLM-large · ERes2Net(q3ko.eval.sv).
지표(모델마다 · 잣대마다 · 같은 성별 목표 8명 안에서 겨룸 → 우연 = 1/8 = 12.5%)
  (a) 중립→감정 세트 1위: 감정 출력 x 가 "자기 화자의 생성 중립 평균"에 다른 목표 화자들의 생성 중립 평균보다 가까운 비율 + 여유(margin)
  (b) 감정끼리(공정한 잣대 · q3tts tools/diag/fair_timbre_probe.py 방식): 묶음 g(중립 · 기쁨 · 슬픔 · 분노, A/B 합침)의 출력 x 를
      자기 화자의 같은 묶음 **다른** 출력 k 개 평균 vs 다른 화자의 같은 묶음 k 개 평균(무작위 3회 평균)에 견줌 → 1위 비율
  (c) 천장 = 진짜 녹음 화자 16명(A7N 여8 · 남8 · 중립 ≥3 · 묶음마다 ≥4)에 같은 구조(REF 1 · 중립 2 · 묶음마다 4)로 같은 측정
  (d) 갈라짐 = sim(생성 중립 평균, REF) − sim(생성 감정 평균, REF) — 진짜 녹음의 같은 값(감정 자체로 생기는 차이)과 비교
      + 세트↔REF 거리 = sim(세트 평균, REF) — 낮으면 "REF 와 다른 가상 화자"

    conda: env q3tts (임베딩) · 나머지 단계는 아무 env(numpy · pandas)
    CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python tools/vspk.py emb [--tags …]   # 합성음 · 진짜 녹음 임베딩(캐시)
    python tools/vspk.py report                                                             # 표 → d_eval/vspk/
    ~/miniforge3/envs/q3tts/bin/python tools/vspk.py refcheck                               # REF 음향 점검
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
O = D / "vspk"
VD = H / "preprocessed/vox_koft/A7N"
SEED = 1986
DRAWS = 3
KEYS = ("w", "e")                                         # WavLM-large · ERes2Net
NAME = {"w": "WavLM", "e": "ERes2Net"}


def grp(e: str) -> str:
    return "Neutrality" if e == "Neutrality" else e.split("-")[0]


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# ───────────────────────── 진짜 녹음 천장 화자 고르기(결정적)
def real_plan():
    rows = rd(VD / "train.jsonl") + rd(VD / "val_unseen8/val.jsonl")
    by = defaultdict(lambda: defaultdict(list))
    gen = {}
    for r in rows:
        by[r["speaker"]][grp(r["emotion"])].append(r["audio"])
        gen[r["speaker"]] = r["gender"]
    rng = random.Random(f"{SEED}|vspk_real")
    ok = sorted(s for s in by if len(by[s]["Neutrality"]) >= 3 and all(len(by[s][g]) >= 4 for g in ("Happy", "Sad", "Angry")))
    pick = []
    for g in ("female", "male"):
        c = [s for s in ok if gen[s] == g]
        pick += rng.sample(c, 8)
    out = {}
    for s in pick:
        n = rng.sample(sorted(by[s]["Neutrality"]), 3)
        items = [dict(id=f"{s}_N{k}", grp="Neutrality", audio=a) for k, a in enumerate(n[1:])]
        for g in ("Happy", "Sad", "Angry"):
            items += [dict(id=f"{s}_{g}{k}", grp=g, audio=a) for k, a in enumerate(rng.sample(sorted(by[s][g]), 4))]
        out[s] = dict(gender=gen[s], ref=n[0], items=items)
    return out


# ───────────────────────── emb (env q3tts)
def emb(a):
    import librosa
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    O.mkdir(parents=True, exist_ok=True)

    def vecs(paths):
        W = [librosa.load(str(p), sr=16000)[0] for p in paths]
        return {k: np.stack([F[k](w, 16000) for w in W]) for k in KEYS}
    P = json.load(open(D / "plan.json"))
    # REF 16개
    if not (O / "emb_ref.npz").exists():
        ids = list(P["spk"])
        v = vecs([P["spk"][s]["ref"] for s in ids])
        np.savez(O / "emb_ref.npz", ids=np.array(ids), **v)
    # 진짜 녹음 천장
    if not (O / "emb_real.npz").exists():
        R = real_plan()
        ids, meta, paths = [], [], []
        for s, v in R.items():
            for x in [dict(id=f"{s}_REF", grp="REF", audio=v["ref"])] + v["items"]:
                ids.append(x["id"]); paths.append(x["audio"]); meta.append(dict(spk=s, gender=v["gender"], grp=x["grp"]))
        v = vecs(paths)
        np.savez(O / "emb_real.npz", ids=np.array(ids), meta=np.array(json.dumps(meta)), **v)
        print("진짜 녹음 천장", len(ids), flush=True)
    for sub, plan, pre in (("wav", D / "plan.json", "emb_"), ("wav_trial", D / "plan_trial.json", "emb_trial_")):
        if not plan.exists() or not (D / sub).exists():
            continue
        PP = json.load(open(plan))
        for t in sorted(p.name for p in (D / sub).iterdir() if p.is_dir()):
            if a.tags and t not in a.tags:
                continue
            f = O / f"{pre}{t}.npz"
            ids = [x["id"] for x in PP["items"] if (D / sub / t / f"{x['id']}.wav").exists()]
            if f.exists() and set(np.load(f)["ids"]) >= set(ids):
                continue
            if len(ids) < len(PP["items"]):
                print(f"⚠️ {sub}/{t}: wav {len(ids)}/{len(PP['items'])} — 덜 만들어짐(건너뜀)", flush=True)
                continue
            v = vecs([D / sub / t / f"{i}.wav" for i in ids])
            np.savez(f, ids=np.array(ids), **v)
            print(f"{sub}/{t} {len(ids)}", flush=True)


# ───────────────────────── 측정
def measure(sets: dict, gender: dict, ref: dict, k: str, label: str, n_key="Neutrality") -> dict:
    """sets[s][grp] = (n, D) 벡터 · gender[s] · ref[s] = (D,) → 지표 dict + 화자별 표."""
    rng = np.random.default_rng(SEED + zlib.crc32(f"{label}|{k}".encode()))
    spk = list(sets)
    ncent = {s: unit(sets[s][n_key].mean(0)) for s in spk}
    a_hit, a_mar, b_hit, b_mar = [], [], [], []
    per = {}
    for s in spk:
        rivals = [j for j in spk if j != s and gender[j] == gender[s]]
        emo = np.concatenate([sets[s][g] for g in sets[s] if g != n_key])
        sa = []
        for x in emo:                                                  # (a)
            own = float(x @ ncent[s])
            oth = max(float(x @ ncent[j]) for j in rivals)
            a_hit.append(own > oth); a_mar.append(own - oth); sa.append(own > oth)
        sb = []
        for g, X in sets[s].items():                                   # (b) 감정끼리 · leave-one-out
            if len(X) < 2:
                continue
            kk = len(X) - 1
            for i in range(len(X)):
                own = float(X[i] @ unit(np.delete(X, i, 0).mean(0)))
                best = -1.0
                for j in rivals:
                    Y = sets[j].get(g)
                    if Y is None or len(Y) < kk:
                        continue
                    sims = [float(X[i] @ unit(Y[rng.choice(len(Y), kk, replace=False)].mean(0))) for _ in range(DRAWS)]
                    best = max(best, float(np.mean(sims)))
                b_hit.append(own > best); b_mar.append(own - best); sb.append(own > best)
        allv = np.concatenate(list(sets[s].values()))
        r = ref[s]
        per[s] = dict(a=100 * np.mean(sa), b=100 * np.mean(sb), set_ref=float(unit(allv.mean(0)) @ r),
                      n_ref=float(ncent[s] @ r), e_ref=float(unit(emo.mean(0)) @ r),
                      split=float(ncent[s] @ r) - float(unit(emo.mean(0)) @ r), n_e=float(ncent[s] @ unit(emo.mean(0))))
    m = dict(a_top1=100 * np.mean(a_hit), a_margin=float(np.mean(a_mar)), b_top1=100 * np.mean(b_hit), b_margin=float(np.mean(b_mar)),
             set_ref=float(np.mean([v["set_ref"] for v in per.values()])), split=float(np.mean([v["split"] for v in per.values()])),
             n_e=float(np.mean([v["n_e"] for v in per.values()])))
    return dict(m=m, per=per)


def load_sets(tag: str, trial: bool):
    P = json.load(open(D / "plan.json"))
    z = np.load(O / f"emb_{tag}.npz")
    idx = {i: n for n, i in enumerate(z["ids"])}
    zr = np.load(O / "emb_ref.npz")
    ref = {k: {s: unit(zr[k][n]) for n, s in enumerate(zr["ids"])} for k in KEYS}
    sets = {k: defaultdict(lambda: defaultdict(list)) for k in KEYS}
    for x in P["items"]:
        for k in KEYS:
            sets[k][x["tgt"]][grp(x["emo"])].append(z[k][idx[x["id"]]])
    if trial:
        T = json.load(open(D / "plan_trial.json"))
        zt = np.load(O / f"emb_trial_{tag}.npz")
        it = {i: n for n, i in enumerate(zt["ids"])}
        for x in T["items"]:
            for k in KEYS:
                sets[k][x["tgt"]]["Neutrality"].append(zt[k][it[x["id"]]])
    gender = {s: v["gender"] for s, v in P["spk"].items()}
    out = {k: {s: {g: np.stack([unit(v) for v in L]) for g, L in sets[k][s].items()} for s in sets[k]} for k in KEYS}
    return out, gender, ref


def report(a):
    import pandas as pd
    P = json.load(open(D / "plan.json"))
    rows, per_rows = [], []
    # 천장
    zr = np.load(O / "emb_real.npz")
    meta = json.loads(str(zr["meta"]))
    for k in KEYS:
        sets, ref, gender = defaultdict(lambda: defaultdict(list)), {}, {}
        for n, (i, mt) in enumerate(zip(zr["ids"], meta)):
            v = unit(zr[k][n])
            gender[mt["spk"]] = mt["gender"]
            if mt["grp"] == "REF":
                ref[mt["spk"]] = v
            else:
                sets[mt["spk"]][mt["grp"]].append(v)
        S = {s: {g: np.stack(L) for g, L in sets[s].items()} for s in sets}
        r = measure(S, gender, ref, k, "real")
        rows.append(dict(model="진짜 녹음(천장)", kind="real", embed=NAME[k], **r["m"]))
    tags = sorted(f.stem[4:] for f in O.glob("emb_*.npz") if not f.stem.startswith(("emb_trial_", "emb_ref", "emb_real")))
    for t in tags:
        variants = [(t, False)] + ([(t + " +시범중립20", True)] if (O / f"emb_trial_{t}.npz").exists() else [])
        for lab, tr in variants:
            sets, gender, ref = load_sets(t, tr)
            for k in KEYS:
                r = measure(sets[k], gender, ref[k], k, lab)
                rows.append(dict(model=lab, kind="trial" if tr else "main", embed=NAME[k], **r["m"]))
                for s, v in r["per"].items():
                    per_rows.append(dict(model=lab, embed=NAME[k], spk=s, gender=gender[s], src=P["spk"][s]["src"], **v))
    df, pf = pd.DataFrame(rows), pd.DataFrame(per_rows)
    O.mkdir(exist_ok=True)
    df.to_csv(O / "vspk_all.csv", index=False)
    pf.to_csv(O / "vspk_per_speaker.csv", index=False)
    # 두 잣대 평균 = 공정한 순위
    g = df.groupby(["model", "kind"], sort=False)[["a_top1", "b_top1", "a_margin", "b_margin", "set_ref", "split", "n_e"]].mean().reset_index()
    g["consist"] = (g["a_top1"] + g["b_top1"]) / 2
    g = g.sort_values("consist", ascending=False)
    g.to_csv(O / "vspk_rank.csv", index=False)
    pd.set_option("display.width", 220)
    print(g.round(3).to_string(index=False))
    json.dump(dict(rank=g.to_dict("records"), all=df.to_dict("records")), open(O / "vspk_summary.json", "w"), ensure_ascii=False, indent=1, default=float)


# ───────────────────────── REF 음향 점검
def acoustics(path, f0=False) -> dict:
    """한 파일의 음향 특징 — 16kHz 로 맞춰 비교(대역 비교가 공정하도록) + 원래 샘플레이트."""
    import librosa
    import soundfile as sf
    sr0 = sf.info(str(path)).samplerate
    y, sr = librosa.load(str(path), sr=16000)
    hop = 160
    n = len(y) // hop
    db = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    act = db > db.max() - 40
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    spec = S[:, np.where(act[:: max(1, 256 // hop)][: S.shape[1]])[0]].sum(1) if act.any() else S.sum(1)
    ldb = 10 * np.log10(spec / spec.max() + 1e-12)
    out = dict(sr0=sr0, dur=round(len(y) / sr, 2), speech_db=round(float(np.median(db[act])), 1),
               noise_db=round(float(np.percentile(db, 5)), 1), snr_db=round(float(np.median(db[act]) - np.percentile(db, 5)), 1),
               peak=round(float(np.abs(y).max()), 3), lead_sil=round(float(np.argmax(act)) * 0.01, 2),
               tail_sil=round(float(len(act) - 1 - np.where(act)[0][-1]) * 0.01, 2) if act.any() else 0.0,
               bw_cut_hz=int(f[np.where(ldb > -50)[0].max()]),                  # 장기 스펙트럼이 최대 −50dB 안에 드는 가장 높은 주파수
               hi4k=round(float(spec[f >= 4000].sum() / spec.sum()), 4),
               centroid=int(np.sum(f * spec) / spec.sum()),
               flat=round(float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(S)))), 4))
    if f0:
        z = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=1024)[0]
        out["f0_med"] = round(float(np.nanmedian(z)), 1)
    return out


def refcheck(a):
    import pandas as pd
    P = json.load(open(D / "plan.json"))
    real = json.load(open(D / "sv_real.json"))
    up = {r["speaker"]: r for r in rd(VD / "val_unseen8/ref_pool.jsonl")}
    rows = []
    for s, v in P["spk"].items():
        src = v["ref"]
        if s in up:                                                    # unseen8 ref = 16k 변환본 → q3tts 쪽 원본 파일
            q = H / "preprocessed/q3tts_koft/q3_extra/val_unseen8/wavs" / f"{up[s]['utt']}.wav"
            src = str(q) if q.exists() else src
        r = acoustics(src, a.f0)
        own = [acoustics(x, a.f0) for x in v["real"]]                  # 같은 화자 진짜 녹음(목소리 잣대의 기준) 평균
        o = {k: float(np.mean([x[k] for x in own])) for k in own[0] if k != "sr0"}
        rows.append(dict(spk=s, src=v["src"], gender=v["gender"], file=Path(src).name, **r,
                         **{f"own_{k}": round(o[k], 2) for k in ("speech_db", "noise_db", "snr_db", "bw_cut_hz", "hi4k", "centroid", "flat", "dur") + (("f0_med",) if a.f0 else ())},
                         own_sr0=sorted({x["sr0"] for x in own}), n_own=len(own),
                         ref_vs_own_real_W=round(real[s]["t_sim"], 3), ref_vs_own_real_E=round(real[s]["e_t_sim"], 3)))
    df = pd.DataFrame(rows)
    O.mkdir(exist_ok=True)
    df.to_csv(O / "ref_acoustics.csv", index=False)
    pd.set_option("display.width", 300)
    cols = ["spk", "sr0", "own_sr0", "dur", "speech_db", "own_speech_db", "noise_db", "own_noise_db", "snr_db", "own_snr_db",
            "bw_cut_hz", "own_bw_cut_hz", "centroid", "own_centroid", "flat", "own_flat"] + (["f0_med", "own_f0_med"] if a.f0 else []) + ["ref_vs_own_real_W"]
    print(df[cols].to_string(index=False))


def page(a):
    """청음 페이지 ② 가상 화자 세트 묶음 → d_eval/page_vset.html (d_eval.py page 가 끼워 넣음)."""
    import html
    import pandas as pd
    E = html.escape
    P = json.load(open(D / "plan.json"))
    T = json.load(open(D / "plan_trial.json")) if (D / "plan_trial.json").exists() else None
    models = [t for t in (a.tags or []) if (D / "wav_trial" / t).exists()]
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    KO = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}
    rk = pd.read_csv(O / "vspk_rank.csv") if (O / "vspk_rank.csv").exists() else None
    pf = pd.read_csv(O / "vspk_per_speaker.csv") if (O / "vspk_per_speaker.csv").exists() else None
    h = ["<h2>② 가상 화자 세트 — 같은 모델 · 같은 목표 화자로 만든 중립과 감정이 <u>서로</u> 같은 목소리인가</h2>",
         "<ul><li>q3tts 학습 데이터(D안)에 필요한 것은 <b>세트 안에서 한 사람</b>(중립 ref 와 감정 정답이 같은 목소리)입니다. REF 와 똑같을 필요는 없습니다.</li>",
         f"<li>모델: {', '.join(E(m) for m in models)} — 가상 화자 일관성(자동 잣대) 상위. 줄마다: REF(참고) · 생성 중립 4개(시범 생성 · 학습 텍스트에 없는 문장) · 생성 감정 6칸(①의 첫 문장).</li>",
         "<li>판정 두 가지: <b>세트</b> = 이 줄의 생성물(중립 + 감정)이 모두 한 사람 같은가(같음 · 애매 · 다름) / <b>REF</b> = 그 사람이 REF 와 같은 사람인가(같음 · 비슷 · 다름 — 참고).</li>",
         "<li>판정 규칙(제안): 세트 '같음'이 대부분이면 REF 가 '다름'이어도 가상 화자로 쓸 수 있음. 세트에서 <b>중립만 다른 사람</b>이면 위험(갈라짐).</li></ul>"]
    if T is None or not models:
        h.append("<p>(시범 생성 전)</p>")
    for m in models:
        h.append(f"<details><summary><b>{E(m)}</b> — 목표 16명</summary><table><tr><th>화자</th><th>REF</th><th>생성 중립 ×4</th>"
                 + "".join(f"<th>{KO[e]}</th>" for e in ["Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]) + "<th>판정</th></tr>")
        for s, v in P["spk"].items():
            num = ""
            if pf is not None:
                r = pf[(pf.model == m + " +시범중립20") & (pf.spk == s)]
                if len(r):
                    num = "<div class='n num'>" + " · ".join(f"{e}: 세트1위 {x.a:.0f}% · 감정끼리 {x.b:.0f}% · 세트↔REF {x.set_ref:.2f} · 갈라짐 {x.split:+.3f}" for e, x in zip(r.embed, r.itertuples())) + "</div>"
            neu = "".join(f"<div>{A(f'wav_trial/{m}/{s}_T{k:02d}.wav')}</div>" for k in (0, 1, 8, 9))
            emo = "".join(f"<td>{A(f'wav/{m}/{s}_{e}_0.wav')}</td>" for e in ["Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"])
            h.append(f"<tr><td>{E(s)}<br><span class=n>{'여' if v['gender'] == 'female' else '남'} · {v['src']}</span>{num}</td><td>{A(f'ref/{s}_ref.wav')}</td><td>{neu}</td>{emo}"
                     f"<td>세트{btn(f'{m}|vset_{s}|set', ['같음', '애매', '다름'])}REF{btn(f'{m}|vset_{s}|ref', ['같음', '비슷', '다름'])}</td></tr>")
        h.append("</table></details>")
    if rk is not None:
        h.append("<details><summary><b>📊 가상 화자 일관성 순위 — 참고용(귀로 판정한 뒤)</b></summary><table style='width:auto'><tr><th>모델</th><th>세트 1위<br>(중립→감정)</th><th>감정끼리<br>1위</th><th>일관성<br>(평균)</th><th>세트↔REF</th><th>갈라짐</th></tr>")
        for r in rk.itertuples():
            h.append(f"<tr><td>{E(r.model)}</td><td>{r.a_top1:.0f}%</td><td>{r.b_top1:.0f}%</td><td><b>{r.consist:.0f}%</b></td><td>{r.set_ref:.3f}</td><td>{r.split:+.3f}</td></tr>")
        h.append("</table><p class=n>두 잣대(WavLM-large · ERes2Net) 평균 · 같은 성별 목표 8명 안에서 겨룸(우연 = 12.5%). 세트 1위 = 감정 출력이 자기 화자의 생성 중립 평균에 가장 가까운 비율. "
                 "감정끼리 = 같은 묶음(기쁨 · 슬픔 · 분노 · 중립)의 다른 출력 평균에 가장 가까운 비율(q3tts 공정한 잣대 방식). 갈라짐 = sim(중립, REF) − sim(감정, REF) — 진짜 녹음(천장)의 값과 비교.</p></details>")
    (D / "page_vset.html").write_text("\n".join(h), encoding="utf-8")
    print("✅ page_vset.html", models)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["emb", "report", "refcheck", "page"])
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--f0", action="store_true", help="refcheck: F0 중앙값도(느림)")
    a = ap.parse_args()
    {"emb": emb, "report": report, "refcheck": refcheck, "page": page}[a.step](a)


if __name__ == "__main__":
    main()
