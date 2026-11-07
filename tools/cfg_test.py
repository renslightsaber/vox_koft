#!/usr/bin/env python
"""CFG 시험 — 다시 학습하지 않고 추론 cfg 만 올려 감정을 되살릴 수 있는가 (2026-11-07 · 지시문 q3tts 세션 11:45 §4).

대상: A7NX2R_xref_LoRA_02_r32_lr5e5 의 s2000 · s4500 × cfg 2.0(기준) · 2.5 · 3.0
입력: D 재측정(voxko/eval/d_eval.py)과 똑같음 — 목표 16명 × 7칸 × 문장 2 = 224 · "(v1 지시문)문장" + 무감정 reference · seed 1986 · timesteps 10 · 준비 합성 뒤
      cfg 2.0 은 D 재측정 결과(d_eval/wav/X2R02lr5e5_s*)를 그대로 씀(같은 입력 · 같은 seed → 같은 소리) · 새로 만드는 것은 cfg 2.5 · 3.0 뿐

🔎 CFG 가 지시문에도 걸리는가(코드 근거): VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py solve_euler
   — 조건 쪽 = mu(LM 은닉 = 문장 + 지시문 + reference 를 모두 본 결과) · 무조건 쪽 = mu 0(이전 소리 조각 cond 는 같음)
   → cfg 는 "지시문만"이 아니라 **LM 조건 전체(문장 · 지시문 · reference 목소리)를 함께** 세게 함. 지시문에도 걸리지만 간접적.

단계(채점은 d_eval 함수를 그대로 불러 같은 잣대 · 결과는 ~/workspaces/tmp_listen/vox_koft/cfg_test/)
  python tools/cfg_test.py prep
  CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/voxcpm/bin/python    tools/cfg_test.py vox --tag X2R02_s2000_cfg25   (… _cfg30 · s4500)
  CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts/bin/python     tools/cfg_test.py sv
  CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts_ser/bin/python tools/cfg_test.py ser
  CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/voxcpm/bin/python    tools/cfg_test.py quality
  ~/miniforge3/envs/voxcpm/bin/python tools/cfg_test.py report   # → cfg_test/tables.md · report.json
  ~/miniforge3/envs/voxcpm/bin/python tools/cfg_test.py page     # → cfg_test/index.html (작은 청음 페이지)
숫자는 모두 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOX_D_EVAL_OUT", str(Path.home() / "workspaces/tmp_listen/vox_koft/d_eval"))
from voxko.eval import d_eval as DE  # noqa: E402

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
C = H / "tmp_listen/vox_koft/cfg_test"
CK = H / "outputs/vox_koft/ckpt"
RUN = "A7NX2R_xref_LoRA_02_r32_lr5e5"
STEPS = [2000, 4500]
CFGS = {"cfg20": 2.0, "cfg25": 2.5, "cfg30": 3.0}
CELLS6 = ["Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
PAGE_CELLS = ["Happy-B", "Angry-A", "Angry-B", "Sad-B"]
PAGE_SPK = ["aihub_0028", "aihub_0012"]          # unseen8 여 1 · 남 1 (감정 녹음이 있는 화자 — 귀 기준 비교용)
KO = {"Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B",
      "Neutrality": "중립"}


def tag(step: int, c: str) -> str:
    return f"X2R02_s{step}_{c}"


def src_of(step: int, c: str) -> tuple[Path, str]:
    """(결과 폴더, d_eval 이름표) — cfg 2.0 은 D 재측정 것."""
    return (D, f"X2R02lr5e5_s{step}") if c == "cfg20" else (C, tag(step, c))


def use_c():
    DE.OUT, DE.PLANF, DE.WAVD, DE.PFX = C, C / "plan.json", C / "wav", ""


def prep(a):
    C.mkdir(parents=True, exist_ok=True)
    shutil.copy(D / "plan.json", C / "plan.json")
    shutil.copy(D / "quality_base.json", C / "quality_base.json")      # 깨짐 문턱(원본 SQUIM 하위 5%)을 D 재측정과 같게
    json.dump(dict(run=RUN, steps=STEPS, cfgs=CFGS, created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
              open(C / "conds.json", "w"), indent=1)
    print("plan 224 · base 문턱 복사 →", C)


def vox(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    use_c()
    st, c = a.tag.split("_")[1][1:], a.tag.split("_")[2]
    cfg = CFGS[c]
    P = json.load(open(C / "plan.json"))
    W = C / "wav" / a.tag
    W.mkdir(parents=True, exist_ok=True)
    eng = Engine(str(CK / RUN / f"step_{st}/export_lora"), emotion_map=str(DE.EMAP))   # 준비 합성 포함
    t0, tot, chk = time.time(), 0.0, None
    for x in P["items"]:
        p = W / f"{x['id']}.wav"
        if p.exists():
            continue
        w = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=P["spk"][x["tgt"]]["ref"], seed=DE.SEED,
                      cfg_value=cfg, max_len=600)
        sf.write(p, w, eng.sr)
        tot += len(w) / eng.sr
        chk = x
    rep = None
    if chk:
        w2 = eng.synth(eng.compose(chk["text"], emotion=chk["emo"], normalize=False), ref=P["spk"][chk["tgt"]]["ref"], seed=DE.SEED,
                       cfg_value=cfg, max_len=600)
        w1 = sf.read(W / f"{chk['id']}.wav")[0]
        rep = len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3
    meta = dict(tag=a.tag, run=RUN, step=int(st), cfg=cfg, repro=rep, n=len(P["items"]),
                rtf=(time.time() - t0) / tot if tot else None, time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(meta, open(C / f"vox_{a.tag}.json", "w"), indent=1)
    print(json.dumps(meta, ensure_ascii=False))


def score(a, fn):
    use_c()
    a.tags = a.tags or DE.tags()
    fn(a)


def summaries():
    DE.OUT = D
    thr = DE._sq_thr()
    out = {}
    for st in STEPS:
        for c in CFGS:
            o, t = src_of(st, c)
            DE.OUT = o
            P = json.load(open(o / "plan.json"))
            if not (o / f"sv_{t}.json").exists() or not (o / f"ser_{t}.json").exists():
                continue
            s = DE.summarize(P, t, thr)
            k = s["칸별"]
            out[f"s{st}|{c}"] = dict(
                step=st, cfg=CFGS[c], e2v6=s["전체"]["emo_hit6"], e2v7=s["전체"]["emo_hit"],
                arousal6=float(np.mean([k[e]["arousal"] for e in CELLS6])),
                fair6=float(np.mean([(k[e]["top1"] + k[e].get("e_top1", k[e]["top1"])) / 2 for e in CELLS6])),
                t_sim=s["전체"]["t_sim"], cer=s["전체"]["cer"], odd=s["전체"].get("odd", float("nan")),
                score=s["전체"]["score"], cells={e: dict(hit=k[e]["emo_hit"], aro=k[e]["arousal"],
                                                         top1=(k[e]["top1"] + k[e].get("e_top1", k[e]["top1"])) / 2) for e in k})
    return out


def report(a=None):
    S = summaries()
    md = ["## CFG 시험 — A7NX2R_xref_LoRA_02_r32_lr5e5 (D 재측정 입력 224 · seed 1986 · 숫자는 참고용)", "",
          "| step · cfg | e2v 6칸 | arousal 6칸 | 목표 1위(두 잣대 · 6칸) | 목표 유사도 | CER | 이상 | 참고 점수 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in S.items():
        md.append(f"| {k.replace('|', ' · ')} | {v['e2v6']:.0f}% | {v['arousal6']:.3f} | {v['fair6']:.0f}% | {v['t_sim']:.3f} | {v['cer']:.1f}% | {v['odd']:.0f}% | {v['score']:.0f} |")
    md += ["", "### 칸별 e2v 맞힘 · arousal", "", "| step · cfg | " + " | ".join(KO[e] for e in CELLS6) + " |", "|---|" + "---|" * 6]
    for k, v in S.items():
        md.append(f"| {k.replace('|', ' · ')} | " + " | ".join(f"{v['cells'][e]['hit']:.0f}% · {v['cells'][e]['aro']:.2f}" for e in CELLS6) + " |")
    md += ["", "- 한 칸 32개 · 6칸 192개 → e2v ±3.6%p 는 잡음 수준(이항 표준오차)."]
    (C / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(S, open(C / "report.json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(md))
    return S


def page(a=None):
    import yaml
    E = html.escape
    P = json.load(open(C / "plan.json"))
    instr = (yaml.safe_load(open(DE.EMAP, encoding="utf-8")) or {}).get("instruct", {})
    first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731
    S = json.load(open(C / "report.json")) if (C / "report.json").exists() else {}

    def wav(st, c, i):
        return f"../d_eval/wav/X2R02lr5e5_s{st}/{i}.wav" if c == "cfg20" else f"wav/{tag(st, c)}/{i}.wav"

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    h = [f"""<!doctype html><meta charset=utf-8><title>vox_koft CFG 시험</title>
<style>body{{font-family:sans-serif;max-width:1700px;margin:20px auto;line-height:1.45}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}}th{{background:#f3f3f3}}.n{{color:#666;font-size:.8em}}audio{{width:170px}}button{{margin:1px;font-size:.8em}}button.on{{background:#1565c0;color:#fff}}#bar{{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}}.num{{display:none}}body.shownum .num{{display:block}}</style>
<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick="dl()">CSV 내려받기</button> · <label><input type=checkbox onchange="document.body.classList.toggle('shownum',this.checked)"> 숫자 보기(먼저 귀로!)</label> · <button onclick="if(confirm('판정을 모두 지울까요?')){{localStorage.removeItem(K);location.reload()}}">판정 지우기</button></div>
<h1>🎧 CFG 시험 — 다시 학습하지 않고 cfg 만 올려 감정이 살아나는가</h1>
<h2>청음 가이드</h2><ul>
<li>모델: <b>{RUN}</b>(A7NX2R · lr 5e-5) — 목소리는 좋지만 감정이 약했던 판. step <b>2000 · 4500</b>.</li>
<li>열: 같은 문장 · 같은 reference · seed 1986 으로 <b>cfg 2.0(기본 = D 재측정 · 노트북과 같은 소리) · 2.5 · 3.0</b>.</li>
<li>cfg 는 지시문만이 아니라 <b>문장 · 지시문 · reference 목소리를 한꺼번에</b> 세게 합니다(VoxCPM 코드). 그래서 감정이 세지는지와 함께 <b>소리가 거칠어지거나 목소리가 바뀌지 않는지</b>도 들어 주세요.</li>
<li>줄: 감정이 약했던 칸 <b>기쁨B · 분노A · 분노B · 슬픔B</b> × 처음 보는 화자 2명(여 aihub_0028 · 남 aihub_0012) · 문장 1개.</li>
<li>버튼(소리마다): <b>감정</b> 약함 · OK · 과함 / <b>목소리 · 음질</b> 좋음 · 거칢. 이 브라우저에 저장되고 CSV 로 받습니다.</li>
<li>숫자(자동 잣대 · 224문항 전체)는 <b>참고용</b> — 맨 아래 접힌 곳.</li></ul>"""]
    h.append("<table><tr><th>칸</th><th>화자 · 문장</th><th>목표 REF</th>"
             + "".join(f"<th>s{st} · cfg {CFGS[c]}</th>" for st in STEPS for c in CFGS) + "</tr>")
    for e in PAGE_CELLS:
        for s_ in PAGE_SPK:
            x = next(x for x in P["items"] if x["tgt"] == s_ and x["emo"] == e)
            i = x["id"]
            g = "여" if P["spk"][s_]["gender"] == "female" else "남"
            cells = "".join(f"<td>{A(wav(st, c, x['id']))}{btn(f'{st}|{c}|{i}|e', ['약함', 'OK', '과함'])}"
                            f"{btn(f'{st}|{c}|{i}|v', ['좋음', '거칢'])}</td>" for st in STEPS for c in CFGS)
            h.append(f"<tr><td>{KO[e]}</td><td>{E(s_)} ({g})<br>{E(x['text'])}<div class=n>({E(first(instr.get(e, '')))})</div></td>"
                     f"<td>{A(f'../d_eval/ref/{s_}_ref.wav')}</td>{cells}</tr>")
    h.append("</table>")
    if (C / "tables.md").exists():
        h.append("<details><summary><b>📊 자동 잣대 표 — 참고용 (귀로 판정한 뒤 열기)</b></summary><pre style='white-space:pre-wrap;font-size:.8em'>"
                 + E((C / "tables.md").read_text(encoding="utf-8")) + "</pre></details>")
    h.append("""<script>const K='vox_koft_cfg_test';let S=JSON.parse(localStorage.getItem(K)||'{}');
function draw(){document.querySelectorAll('.b').forEach(d=>{d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v))});document.getElementById('cnt').textContent=Object.keys(S).length}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;S[id]=S[id]===b.dataset.v?undefined:b.dataset.v;if(!S[id])delete S[id];localStorage.setItem(K,JSON.stringify(S));draw()});
function dl(){const r=['step,cfg,id,axis,value'].concat(Object.entries(S).map(([k,v])=>k.split('|').join(',')+','+v));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+r.join('\\n')],{type:'text/csv'}));a.download='vox_koft_cfg_test_judgments.csv';a.click()}
draw()</script>""")
    (C / "index.html").write_text("\n".join(h), encoding="utf-8")
    print("✅ page →", C / "index.html", "· 표본 수", len(S))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "vox", "sv", "ser", "quality", "report", "page"])
    ap.add_argument("--tag")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.step in ("sv", "ser", "quality"):
        score(a, getattr(DE, a.step))
    else:
        globals()[a.step](a)


if __name__ == "__main__":
    main()
