#!/usr/bin/env python
"""D 재측정 — 🆕 감정 ref 판(2026-11-07): 목표 화자의 **감정 녹음**을 reference 로 주고 7칸을 만든다.

묻는 것: ① 출력이 지시문 감정을 따르는가 ② ref 의 감정이 새어 나오는가 ③ 목소리는 목표 화자인가
  — "다른 칸 ref" 학습(A7NX · A7NX3 · A7NX2R)이 노리는 것이 바로 이것(목소리는 ref, 감정은 지시문).

대상: 처음 보는 화자 중 감정 녹음이 있는 **unseen8 8명**(542 8명은 무감정 녹음뿐이라 제외)
  ref = 화자마다 2개 — 기쁨 녹음 1 · 분노 녹음 1(val_unseen8 의 A/B 중 있는 것 · 2~10초 · utt 순 첫 번째 · 결정적)
  문장 = D 재측정과 같은 14문장(7칸 × 2) → 화자당 2 × 14 = 28 · 모델당 224개
  지시문 · seed 1986 · cfg 2.0 · timesteps 10 · 준비 합성 뒤 = D 재측정과 같음
목소리 잣대: d_eval 과 같음(같은 성별 150명 중 목표 1위 · 두 잣대) — 단 목표 화자 대표에서 **ref 로 쓴 녹음은 뺌**.
새는 정도(leak): ref 묶음과 **다른** 칸에서 e2v 가 ref 감정을 고른 비율 — 같은 모델 · 같은 문장 · 무감정 ref(d_eval) 의 같은 비율과 짝 비교
  + arousal 차이(감정 ref − 무감정 ref, 같은 문장 짝).

  python tools/d_eval_eref.py prep
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/d_eval_eref.py vox --tag LoRA01_s1500
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python     tools/d_eval_eref.py sv        # + 임베딩
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python tools/d_eval_eref.py ser
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/d_eval_eref.py quality
  python tools/d_eval_eref.py report                                                       # → d_eval_eref/report.json · tables.md
  python tools/d_eval_eref.py page                                                         # → d_eval_eref/index.html (작은 청음 페이지 · 8줄 × 모델 4)
결과 폴더: ~/workspaces/tmp_listen/vox_koft/d_eval_eref/ · 숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOX_D_EVAL_OUT", str(Path.home() / "workspaces/tmp_listen/vox_koft/d_eval"))
from voxko.eval import d_eval as DE  # noqa: E402

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
E = H / "tmp_listen/vox_koft/d_eval_eref"
CK = H / "outputs/vox_koft/ckpt"
BASE = H / "ourckpts/voxcpm/VoxCPM2"
RUN = {"LoRA01": "A7N_ref_LoRA_01_r32", "LoRA02iv2": "A7N_ref_LoRA_02_r32_iv2", "FT01": "A7N_ref_FT_01", "FT02iv2": "A7N_ref_FT_02_iv2",
       "A7NX": "A7NX_xref_LoRA_01_r32", "A7NX3": "A7NX3_xref_LoRA_01_r32",
       "X2R01": "A7NX2R_xref_LoRA_01_r32", "X2R02lr5e5": "A7NX2R_xref_LoRA_02_r32_lr5e5",
       "X2Rs01lr5e5": "A7NX2Rs_xref_LoRA_01_r32_lr5e5",
       "X2Rs6lr5e5": "A7NX2Rs6_xref_LoRA_01_r32_lr5e5",
       "NUX2Rs6lr5e5": "A7NUX2Rs6_xref_LoRA_01_r32_lr5e5", "NUX2slr5e5": "A7NUX2s_xref_LoRA_01_r32_lr5e5"}
REF_GROUPS = ["Happy", "Angry"]


def model_path(tag: str) -> Path:
    if tag == "base":
        return BASE
    short, s = tag.rsplit("_s", 1)
    d = CK / RUN[short] / f"step_{s}"
    return d / ("export_lora" if (d / "export_lora").is_dir() else "export_ft")


def use():
    DE.OUT, DE.PLANF, DE.WAVD, DE.PFX = E, E / "plan.json", E / "wav", ""
    return json.load(open(E / "plan.json")) if (E / "plan.json").exists() else None


def prep(a):
    P = json.load(open(D / "plan.json"))
    rows = DE.rd(DE.VD / "val_unseen8/val.jsonl")
    spk = {s: dict(v) for s, v in P["spk"].items()}
    refs = {}
    for s, v in P["spk"].items():
        if v["src"] != "unseen8":
            continue
        for g in REF_GROUPS:
            c = sorted([r for r in rows if r["speaker"] == s and DE.base(r["emotion"]) == g and 2.0 <= float(r["duration"]) <= 10.0],
                       key=lambda r: r["utt"])
            if c:
                refs[f"{s}|{g}"] = dict(spk=s, grp=g, emotion=c[0]["emotion"], audio=c[0]["audio"], utt=c[0]["utt"], duration=c[0]["duration"])
        used = {Path(r["audio"]).stem for k, r in refs.items() if r["spk"] == s}
        spk[s]["real"] = [x for x in v["real"] if Path(x).stem not in used]           # 목표 대표에서 ref 로 쓴 녹음 빼기
    items = []
    for k, r in refs.items():
        for x in P["items"]:
            if x["tgt"] == r["spk"]:
                items.append(dict(id=f"{r['spk']}_R{r['grp']}_{x['id'][len(r['spk']) + 1:]}", tgt=r["spk"], emo=x["emo"], text=x["text"],
                                  base_id=x["id"], ref=r["audio"], ref_grp=r["grp"], ref_emotion=r["emotion"], ref_utt=r["utt"]))
    E.mkdir(parents=True, exist_ok=True)
    json.dump(dict(spk=spk, distract=P["distract"], sents=P["sents"], refs=refs, items=items, seed=DE.SEED,
                   created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), open(E / "plan.json", "w"), ensure_ascii=False, indent=1)
    desc = ", ".join(k + "=" + v["emotion"] for k, v in refs.items())                 # (q3tts_ser env 는 Python 3.11 — f-string 안 같은 따옴표 금지)
    nreal = " · ".join(s + " " + str(len(spk[s]["real"])) for s in spk if spk[s]["src"] == "unseen8")
    print(f"ref {len(refs)}개({desc}) · 문항 {len(items)}/모델 · 목표 대표 수 {nreal}")


def vox(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    P = use()
    W = DE.WAVD / a.tag
    W.mkdir(parents=True, exist_ok=True)
    eng = Engine(str(model_path(a.tag)), emotion_map=str(DE.EMAP))
    t0, tot, n = time.time(), 0.0, 0
    for x in P["items"]:
        p = W / f"{x['id']}.wav"
        if p.exists():
            continue
        w = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=x["ref"], seed=DE.SEED, max_len=600)
        sf.write(p, w, eng.sr)
        tot += len(w) / eng.sr
        n += 1
    x = P["items"][-1]                                                     # 재현: 마지막 문항 다시
    w2 = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=x["ref"], seed=DE.SEED, max_len=600)
    w1 = sf.read(W / f"{x['id']}.wav")[0]
    meta = dict(tag=a.tag, model=str(model_path(a.tag)), n=len(P["items"]), new=n, max_len=600,
                repro=bool(len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3),
                rtf=(time.time() - t0) / tot if tot else None, time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(meta, open(E / f"vox_{a.tag}.json", "w"), indent=1)
    print(json.dumps(meta, ensure_ascii=False))


def sv(a):
    import librosa
    P = use()
    DE.sv(argparse.Namespace(tags=a.tags, force=False))
    from q3ko.eval.sv import load_embedder
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    for t in (a.tags or DE.tags()):
        f = E / f"emb_{t}.npz"
        ids = [x["id"] for x in P["items"]]
        if f.exists() and list(np.load(f)["ids"]) == ids:
            continue
        W = [librosa.load(str(DE.WAVD / t / f"{i}.wav"), sr=16000)[0] for i in ids]
        np.savez(f, ids=np.array(ids), **{k: np.stack([F[k](w, 16000) for w in W]) for k in F})
        print("emb", t, len(ids), flush=True)


def ser(a):
    use()
    DE.ser(argparse.Namespace(tags=a.tags, force=False))


def quality(a):
    use()
    DE.quality(argparse.Namespace(tags=a.tags, force=False, device="cuda"))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def report(a):
    P = use()
    DE.OUT = D
    thr = DE._sq_thr()
    out, md = {}, ["# 감정 ref 판 D 재측정 — 표(참고용)", "",
                   "ref = unseen8 8명 × (기쁨 녹음 1 · 분노 녹음 1) · 문장 = D 재측정 14문장 · 짝 = 같은 모델 · 같은 문장의 무감정 ref 판(d_eval)", ""]
    tags = sorted([p.name for p in (E / "wav").iterdir() if p.is_dir() and (E / f"ser_{p.name}.json").exists() and (E / f"sv_{p.name}.json").exists()],
                  key=lambda t: (t != "base", t))
    rows_md = []
    for t in tags:
        sv_, se, qa = (json.load(open(E / f"{n}_{t}.json")) for n in ("sv", "ser", "quality"))
        nse = json.load(open(D / f"ser_{t}.json")) if (D / f"ser_{t}.json").exists() else {}
        nsv = json.load(open(D / f"sv_{t}.json")) if (D / f"sv_{t}.json").exists() else {}
        # 자기 생성 중립(무감정 ref 판 d_eval 중립 2개) 대비 유사도
        own = {}
        zf, zn = E / f"emb_{t}.npz", D / "vspk" / f"emb_{t}.npz"
        if zf.exists() and zn.exists():
            z, zz = np.load(zf), np.load(zn)
            nc = {}
            for k in ("w", "e"):
                by = defaultdict(list)
                for i, v in zip(zz["ids"], zz[k]):
                    if "_Neutrality_" in i:
                        by[str(i).split("_Neutrality_")[0]].append(unit(v))
                nc[k] = {s: unit(np.mean(v, 0)) for s, v in by.items()}
            idx = {x["id"]: x for x in P["items"]}
            for j, i in enumerate(z["ids"]):
                s = idx[str(i)]["tgt"]
                own[str(i)] = float(np.mean([unit(z[k][j]) @ nc[k][s] for k in ("w", "e")]))
        res = {}
        for x in P["items"]:
            i, b = x["id"], x["base_id"]
            g = DE.base(x["emo"])
            q, r = se[i], sv_[i]
            res[i] = dict(emo=x["emo"], grp=g, ref_grp=x["ref_grp"], hit=q["top"] == DE.E2V[g], leak=q["top"] == DE.E2V[x["ref_grp"]],
                          aro=q["arousal"], cer=q["cer"], top1=(r["top1"] + r["e_top1"]) / 2, odd=bool(DE.odd(qa[i], thr)), own=own.get(i, np.nan),
                          n_hit=(nse[b]["top"] == DE.E2V[g]) if b in nse else None, n_leak=(nse[b]["top"] == DE.E2V[x["ref_grp"]]) if b in nse else None,
                          n_aro=nse[b]["arousal"] if b in nse else None, n_top1=((nsv[b]["top1"] + nsv[b]["e_top1"]) / 2) if b in nsv else None)
        V = list(res.values())
        other = [v for v in V if v["grp"] != v["ref_grp"]]                  # ref 묶음과 다른 칸
        pct = lambda xs: 100 * float(np.mean(xs)) if xs else float("nan")  # noqa: E731
        m = dict(n=len(V), hit=pct([v["hit"] for v in V]), n_hit=pct([v["n_hit"] for v in V if v["n_hit"] is not None]),
                 leak=pct([v["leak"] for v in other]), n_leak=pct([v["n_leak"] for v in other if v["n_leak"] is not None]),
                 d_aro=float(np.mean([v["aro"] - v["n_aro"] for v in V if v["n_aro"] is not None])),
                 d_aro_calm=float(np.mean([v["aro"] - v["n_aro"] for v in V if v["n_aro"] is not None and v["grp"] in ("Neutrality", "Sad")])),
                 top1=pct([v["top1"] for v in V]), n_top1=pct([v["n_top1"] for v in V if v["n_top1"] is not None]),
                 own=float(np.nanmean([v["own"] for v in V])) if own else float("nan"),
                 cer=pct([v["cer"] for v in V]), odd=pct([v["odd"] for v in V]),
                 neutral_ok=pct([v["hit"] for v in V if v["grp"] == "Neutrality"]))
        m["cells"] = {e: dict(hit=pct([v["hit"] for v in V if v["emo"] == e]), aro=float(np.mean([v["aro"] for v in V if v["emo"] == e])),
                              leak=pct([v["leak"] for v in V if v["emo"] == e and v["grp"] != v["ref_grp"]]),
                              own=float(np.nanmean([v["own"] for v in V if v["emo"] == e])) if own else float("nan"))
                      for e in DE.EMO7}
        m["repro"] = json.load(open(E / f"vox_{t}.json")).get("repro")
        out[t] = m
        rows_md.append(f"| {t} | {m['hit']:.0f}% ({m['n_hit']:.0f}) | {m['leak']:.0f}% ({m['n_leak']:.0f}) | {m['d_aro']:+.3f} · 차분 {m['d_aro_calm']:+.3f} | "
                       f"{m['top1']:.0f}% ({m['n_top1']:.0f}) | {m['own']:.3f} | {m['neutral_ok']:.0f}% | {m['cer']:.1f}% | {m['odd']:.0f}% |")
    md += ["| 모델 | 지시문 감정 e2v (무감정 ref) | ref 감정 새어 나옴 (무감정 ref) | arousal 차 · 중립·슬픔 칸 | 목표1위 (무감정 ref) | 자기 중립 유사도 | 중립 칸 중립 | CER | 이상 |",
           "|---|---|---|---|---|---|---|---|---|"] + rows_md
    md += ["", "- 새어 나옴 = ref 와 다른 묶음 칸(예: 분노 ref → 기쁨 · 슬픔 · 중립 칸)에서 e2v 가 ref 감정(분노)을 고른 비율. 괄호 = 같은 문장을 무감정 ref 로 만든 값.",
           "- arousal 차 = 감정 ref 판 − 무감정 ref 판(같은 문장 짝). 기쁨 · 분노 ref 는 arousal 이 높아서 새면 + 로 나옴. '차분' = 중립 · 슬픔 칸만.",
           "- 자기 중립 유사도 = 같은 모델의 무감정 ref 판 생성 중립 2개 평균과의 유사도(목소리가 같은 사람으로 남는가)."]
    (E / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(out, open(E / "report.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(md))


PAGE_MODELS = ["LoRA01_s1500", "A7NX_s1500", "X2R01_s4500", "X2R02lr5e5_s4500", "X2Rs01lr5e5_s4500", "X2Rs6lr5e5_s4500",
               "NUX2Rs6lr5e5_s4500", "NUX2slr5e5_s4500"]  # 11/07 18시 X2Rs 두 판 · 21시 A7NU 두 판 추가
PAGE_SPK = ["aihub_0028", "aihub_0012"]                          # 여 1 · 남 1 (CFG 시험 페이지와 같은 화자)
PAGE_ROWS = [("Happy", "Neutrality"), ("Happy", "Sad-B"), ("Angry", "Neutrality"), ("Angry", "Happy-B")]   # (ref 묶음, 지시문 칸)


def page(a):
    """작은 청음 페이지(2026-11-07): 화자 2 × (기쁨 ref → 중립 · 슬픔B / 분노 ref → 중립 · 기쁨B) = 8줄 × 모델 4 — "ref 감정이 새어 나오는가"."""
    import html
    import shutil
    import yaml
    P = use()
    Es = html.escape
    instr = (yaml.safe_load(open(DE.EMAP, encoding="utf-8")) or {}).get("instruct", {})
    first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731
    btn = lambda i, opts: "<div class=b data-id='%s'>%s</div>" % (Es(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))  # noqa: E731
    KO = {"Neutrality": "중립", "Sad-B": "슬픔B", "Happy-B": "기쁨B", "Happy": "기쁨", "Angry": "분노"}
    (E / "ref").mkdir(exist_ok=True)
    h = ["""<!doctype html><meta charset=utf-8><title>vox_koft 감정 ref 판 — ref 감정이 새는가</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:20px auto;line-height:1.45}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}th{background:#f3f3f3}.n{color:#666;font-size:.8em}audio{width:170px}button{margin:1px;font-size:.8em}button.on{background:#1565c0;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}</style>
<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick="dl()">CSV 내려받기</button> · <button onclick="if(confirm('판정을 모두 지울까요?')){localStorage.removeItem(K);location.reload()}">판정 지우기</button></div>
<h1>🎧 감정 ref 판 — reference 의 감정이 출력에 새어 나오는가</h1>
<h2>청음 가이드</h2><ul>
<li>처음 보는 화자(unseen8)의 <b>감정 녹음</b>(기쁨 · 분노)을 reference 로 주고, 지시문은 <b>다른 칸</b>(중립 · 슬픔B · 기쁨B)을 줍니다.</li>
<li>바라는 것: <b>목소리는 ref 화자, 감정은 지시문</b>. 예: 분노 ref + 중립 지시문 → 차분한 중립이어야 하고 화난 말투가 묻어나면 "샘".</li>
<li>열: LoRA01_s1500(무감정 ref 로만 학습) · A7NX_s1500 · X2R01_s4500(A7NX2R lr 1e-4) · X2R02_s4500(A7NX2R lr 5e-5) — 뒤 셋은 "다른 칸 ref" 로 학습.</li>
<li>버튼(소리마다): <b>지시문 따름</b> · 애매 · <b>ref 감정 샘</b>. 이 브라우저에 저장되고 CSV 로 받습니다. seed 1986 · cfg 2.0.</li>
<li>자동 잣대(224문항 · e2v 새는 비율 등)는 참고용 — 맨 아래 접힌 곳. <b>귀가 정본</b>입니다.</li></ul>
<table><tr><th>화자</th><th>ref(감정 녹음)</th><th>지시문 칸 · 문장</th>""" + "".join(f"<th>{m}</th>" for m in PAGE_MODELS) + "</tr>"]
    for s_ in PAGE_SPK:
        g = "여" if P["spk"][s_]["gender"] == "female" else "남"
        for rg, emo in PAGE_ROWS:
            x = next(x for x in P["items"] if x["tgt"] == s_ and x["ref_grp"] == rg and x["emo"] == emo)
            rp = E / "ref" / Path(x["ref"]).name
            if not rp.exists():
                shutil.copy(x["ref"], rp)
            cells = "".join(f"<td>{A('wav/' + m + '/' + x['id'] + '.wav')}{btn(m + '|' + x['id'], ['지시문 따름', '애매', 'ref 감정 샘'])}</td>" for m in PAGE_MODELS)
            h.append(f"<tr><td>{Es(s_)} ({g})</td><td>{KO[rg]} · {Es(x['ref_emotion'])}<br>{A('ref/' + rp.name)}</td>"
                     f"<td><b>{KO[emo]}</b><br>{Es(x['text'])}<div class=n>({Es(first(instr.get(emo, '')))})</div></td>{cells}</tr>")
    h.append("</table>")
    if (E / "tables.md").exists():
        h.append("<details><summary><b>📊 자동 잣대 표 — 참고용 (귀로 판정한 뒤 열기)</b></summary><pre style='white-space:pre-wrap;font-size:.8em'>"
                 + Es((E / "tables.md").read_text(encoding="utf-8")) + "</pre></details>")
    h.append("""<script>const K='vox_koft_eref';let S=JSON.parse(localStorage.getItem(K)||'{}');
function draw(){document.querySelectorAll('.b').forEach(d=>{d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v))});document.getElementById('cnt').textContent=Object.keys(S).length}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;S[id]=S[id]===b.dataset.v?undefined:b.dataset.v;if(!S[id])delete S[id];localStorage.setItem(K,JSON.stringify(S));draw()});
function dl(){const r=['model,id,value'].concat(Object.entries(S).map(([k,v])=>k.split('|').join(',')+','+v));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+r.join('\\n')],{type:'text/csv'}));a.download='vox_koft_eref_judgments.csv';a.click()}
draw()</script>""")
    (E / "index.html").write_text("\n".join(h), encoding="utf-8")
    print("✅ page →", E / "index.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "vox", "sv", "ser", "quality", "report", "page"])
    ap.add_argument("--tag")
    ap.add_argument("--tags", nargs="*")
    a = ap.parse_args()
    {"prep": prep, "vox": vox, "sv": sv, "ser": ser, "quality": quality, "report": report, "page": page}[a.step](a)


if __name__ == "__main__":
    main()
