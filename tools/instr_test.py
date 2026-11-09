#!/usr/bin/env python
"""(A) 분노B 지시문 시험 — 음이 너무 높아지는 것이 지시문 문구 탓인가 · `!` 탓인가 · 데이터 탓인가 (2026-11-08 · 지시문 q3tts 세션 14:40 §2)

재학습 없이 **추론 때 문구만** 바꿔 본다(학습 표 a7n_vox_v1.yaml 은 그대로).
  모델: A7NUX2sN step_3000 export_lora 하나 · 화자: listen_pages.SPK6(6명) · 분노B 문장 4개(d_eval 2 + 새 2)
  C0 지금(v1 분노B) · C1 부드럽게 · C2 v1 분노A 문구 · C3 = C0 문구 + `!` → `.`(`!` 문장 1개만) → 6 × 4 × 3 + 6 = 78개
  조건은 d_eval vox() 와 같음: Engine · REF = d_eval plan spk ref · seed 1986 · cfg 2.0 · timesteps 10 · max_len 600 · 준비 합성
  재현: C0 의 d_eval 문장 2개(× 6명)는 d_eval wav 와, 새 문장 2개(× 6명)는 listen_set6 wav 와 같아야 함(길이 같음 · max diff < 1e-3)
측정: F0 중앙값(librosa pyin · 16kHz · 화자 범위 [REF F0/2, REF F0 +20반음] · 유성 프레임만 · REF 는 60~800Hz) — C0 대비 · REF 대비 반음 · 목표 유사도 · e2v · CER(d_eval sv · ser 코드 그대로)
거르기: C1 · C2 · C3 중 하나가 C0 보다 F0 1.5반음 이상 낮고 · e2v top 이 angry · 목표 유사도(WavLM) ≥ C0 → 통과(화자, 문장)만 최대 8줄 페이지에

  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python tools/instr_test.py synth
  ~/miniforge3/envs/voxcpm/bin/python tools/instr_test.py f0
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts/bin/python tools/instr_test.py measure --what sv
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts_ser/bin/python tools/instr_test.py measure --what ser
  ~/miniforge3/envs/voxcpm/bin/python tools/instr_test.py page      → instr_test/index.html · results.csv
숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listen_pages as LP  # noqa: E402

OUT = LP.TL / "instr_test"
TAG = "NUX2sNlr5e5_s3000"
COND = {"C0": ("지금(v1 분노B)", "Furious and shouting, strong anger, loud and harsh"),
        "C1": ("부드럽게", "Strong anger, firm and forceful, speaking without shouting"),
        "C2": ("분노A 문구", "Coldly irritated, restrained anger, firm and clipped"),
        "C3": ("`!` 빼기(C0 문구)", "Furious and shouting, strong anger, loud and harsh")}
SENT_KEYS = ["Angry-B_0", "Angry-B_1", "Angry-B_new1", "Angry-B_new2"]       # s0 · s1 = d_eval · s2 · s3 = 새 문장
PASS_ST, MAX_ROWS = 1.5, 8
MIN_VOICED = 0.3                              # 초 — 이보다 짧으면 F0 비교에서 뺌


def sentences():
    P = LP.jl(LP.D / "plan.json")
    it = {x["id"]: x for x in P["items"]}
    new = {k: t for _, k, t in LP.ROWS if t}
    s0 = LP.SPK6[0]
    return [it[f"{s0}_{k}"]["text"] if k in ("Angry-B_0", "Angry-B_1") else new[k] for k in SENT_KEYS]


def plan():
    P = LP.jl(LP.D / "plan.json")
    S = sentences()
    items = []
    for s in LP.SPK6:
        for c in ("C0", "C1", "C2", "C3"):
            for k, t in enumerate(S):
                if c == "C3":
                    if "!" not in t:
                        continue
                    t = t.replace("!", ".")
                items.append(dict(id=f"{s}_{c}_s{k}", tgt=s, emo="Angry-B", cond=c, sk=k, text=t, instr=COND[c][1]))
    return dict(spk=P["spk"], distract=P["distract"], items=items, sents=S, cond=COND, tag=TAG, seed=LP.SEED)


def cmd_synth(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    PL = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(PL, open(OUT / "plan_instr.json", "w"), ensure_ascii=False, indent=1)
    W = OUT / "wav" / TAG
    W.mkdir(parents=True, exist_ok=True)
    model = LP.jl(LP.D / f"vox_{TAG}.json")["model"]
    eng = Engine(model, emotion_map=str(LP.EMAP))                                # 준비 합성 포함
    items = [x for x in PL["items"] if not a.smoke or x["tgt"] == LP.SPK6[0]]
    for x in items:
        p = W / f"{x['id']}.wav"
        if not p.exists():
            txt = eng.compose(x["text"], instruct=x["instr"], normalize=False)   # 문장은 이미 ko_tts_norm(바뀌는 것 없음)
            sf.write(p, eng.synth(txt, ref=PL["spk"][x["tgt"]]["ref"], seed=LP.SEED, max_len=LP.MAXLEN), eng.sr)
    # 재현: C0 = d_eval(s0 · s1) · listen_set6(s2 · s3) 와 같아야
    rep = []
    for x in items:
        if x["cond"] != "C0":
            continue
        k = SENT_KEYS[x["sk"]]
        ref = (LP.D / "wav" / TAG / f"{x['tgt']}_{k}.wav") if x["sk"] < 2 else (LP.SET / "wav" / TAG / f"{x['tgt']}_{k}.wav")
        if not ref.exists():
            rep.append(dict(id=x["id"], against=str(ref), ok=None))
            continue
        w0, w1 = sf.read(ref)[0], sf.read(W / f"{x['id']}.wav")[0]
        rep.append(dict(id=x["id"], against=str(ref.relative_to(LP.TL)), ok=LP.same(w0, w1), len=[len(w0), len(w1)],
                        maxdiff=float(np.abs(w0 - w1).max()) if len(w0) == len(w1) else None))
    meta = dict(tag=TAG, model=model, n=len(items), repro=rep, repro_ok=all(r["ok"] for r in rep if r["ok"] is not None),
                text_example=eng.compose(items[0]["text"], instruct=items[0]["instr"], normalize=False), smoke=bool(a.smoke),
                time_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)))
    json.dump(meta, open(OUT / "synth_meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"합성 {len(items)} · 재현 {sum(bool(r['ok']) for r in rep)}/{sum(r['ok'] is not None for r in rep)}", "✅" if meta["repro_ok"] else "❌")


def f0_med(path, fmin=60.0, fmax=800.0):
    import librosa
    y, sr = librosa.load(str(path), sr=16000)
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=160)
    v = f0[vf & np.isfinite(f0)]
    return (float(np.median(v)) if len(v) else float("nan")), float(len(v) * 0.01)


F0_UP = 20                                    # 화자 범위 위쪽 = REF F0 + 20반음(아래 = REF / 2)


def cmd_f0(a):
    """F0 중앙값. 60~800Hz 로 넓게 재면 남성 고함에서 배음(600~800Hz)을 잡는 일이 있어서(aihub_0115 · 0156 C0),
    화자마다 범위를 [REF F0 / 2, REF F0 × 2^(20/12)] 로 좁혀 잰다. 넓은 범위 값(f0_wide)도 함께 남긴다."""
    PL = LP.jl(OUT / "plan_instr.json")
    res = {}
    for s in LP.SPK6:
        res[f"REF|{s}"] = dict(zip(("f0", "voiced_s"), f0_med(PL["spk"][s]["ref"])))
    for x in PL["items"]:
        p = OUT / "wav" / TAG / f"{x['id']}.wav"
        if p.exists():
            fr = res[f"REF|{x['tgt']}"]["f0"]
            lo, hi = fr / 2, fr * 2 ** (F0_UP / 12)
            f, vs = f0_med(p, lo, hi)
            fw, vw = f0_med(p)
            res[x["id"]] = dict(f0=f, voiced_s=vs, fmin=lo, fmax=hi, f0_wide=fw, voiced_s_wide=vw)
    json.dump(res, open(OUT / "f0.json", "w"), indent=1)
    print("F0", len(res), "개 · REF:", {k.split("|")[1]: round(v["f0"]) for k, v in res.items() if k.startswith("REF|")})


def cmd_measure(a):
    LP.de_measure(OUT, OUT / "plan_instr.json", a.what, [TAG])


def st(a, b):
    return 12 * np.log2(a / b) if a > 0 and b > 0 else float("nan")


def table():
    """줄마다 지표(results.csv 와 같은 모양)."""
    PL = LP.jl(OUT / "plan_instr.json")
    F, SV, SR = LP.jl(OUT / "f0.json"), LP.jl(OUT / f"sv_{TAG}.json"), LP.jl(OUT / f"ser_{TAG}.json")
    rows = []
    for x in PL["items"]:
        i = x["id"]
        if i not in SV or i not in SR or i not in F:
            continue
        c0 = f"{x['tgt']}_C0_s{x['sk']}"
        f, fc, fr = F[i]["f0"], F[c0]["f0"], F[f"REF|{x['tgt']}"]["f0"]
        if F[i]["voiced_s"] < MIN_VOICED:                                      # 유성 구간이 너무 짧으면 F0 를 믿지 않음
            f = float("nan")
        if F[c0]["voiced_s"] < MIN_VOICED:
            fc = float("nan")
        rows.append(dict(id=i, spk=x["tgt"], group=PL["spk"][x["tgt"]]["src"], gender=PL["spk"][x["tgt"]]["gender"], cond=x["cond"], sk=x["sk"],
                         text=x["text"], instr=x["instr"], f0=f, st_vs_c0=st(f, fc), st_vs_ref=st(f, fr), t_sim=SV[i]["t_sim"],
                         top1=bool(SV[i]["top1"]), e_t_sim=SV[i]["e_t_sim"], e_top1=bool(SV[i]["e_top1"]), e2v_top=SR[i]["top"],
                         angry=SR[i]["angry"], arousal=SR[i]["arousal"], cer=SR[i]["cer"], asr=SR[i]["asr"]))
    return PL, rows


def cmd_page(a):
    E = html.escape
    PL, rows = table()
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    R = {r["id"]: r for r in rows}
    # 조건별 평균
    summ = {}
    for c in COND:
        rs = [r for r in rows if r["cond"] == c]
        summ[c] = dict(n=len(rs), n_f0=int(np.sum(np.isfinite([r['st_vs_c0'] for r in rs]))), st_c0=np.nanmean([r["st_vs_c0"] for r in rs]), st_ref=np.nanmean([r["st_vs_ref"] for r in rs]),
                       t_sim=np.mean([r["t_sim"] for r in rs]), top1=100 * np.mean([(r["top1"] + r["e_top1"]) / 2 for r in rs]),
                       angry=100 * np.mean([r["e2v_top"] == "angry" for r in rs]), p_angry=np.mean([r["angry"] for r in rs]),
                       cer=100 * np.mean([r["cer"] for r in rs]))
        if c == "C3":                                                          # C3 는 `!` 문장만 — 같은 문장의 C0 와 맞춰 비교
            rs0 = [R[f"{r['spk']}_C0_s{r['sk']}"] for r in rs]
            summ[c]["c0_same"] = dict(t_sim=np.mean([r["t_sim"] for r in rs0]), angry=100 * np.mean([r["e2v_top"] == "angry" for r in rs0]))
    # 거르기
    passed = []
    for s in LP.SPK6:
        for k in range(len(PL["sents"])):
            c0 = R.get(f"{s}_C0_s{k}")
            if not c0:
                continue
            ok = [R[f"{s}_{c}_s{k}"] for c in ("C1", "C2", "C3") if f"{s}_{c}_s{k}" in R]
            ok = [r for r in ok if np.isfinite(r["st_vs_c0"]) and r["st_vs_c0"] <= -PASS_ST and r["e2v_top"] == "angry" and r["t_sim"] >= c0["t_sim"]]
            if ok:
                best = max(ok, key=lambda r: r["t_sim"])                       # 가장 나은 대안 = 통과한 것 중 목표 유사도 최고
                passed.append((c0, best, [r["cond"] for r in ok]))
    passed.sort(key=lambda p: p[1]["st_vs_c0"])
    shown = passed[:MAX_ROWS]
    rel = ".."
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731
    num = lambda r: (f"<div class=n>F0 {r['f0']:.0f}Hz · C0 대비 {r['st_vs_c0']:+.1f}반음 · REF 대비 {r['st_vs_ref']:+.1f}반음<br>"  # noqa: E731
                     f"목표 {r['t_sim']:.3f}{' ✓1위' if r['top1'] else ''} · e2v {r['e2v_top']}(angry {r['angry']:.2f}) · CER {100 * r['cer']:.0f}%</div>")

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    meta = LP.jl(OUT / "synth_meta.json")
    rp = [r for r in meta["repro"] if r["ok"] is not None]
    h = [f"<!doctype html><meta charset=utf-8><title>A · 분노B 지시문 시험</title><style>{LP.CSS}</style>",
         "<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick='dl()'>CSV 내려받기</button> · <a href='results.csv'>숫자 results.csv</a></div>",
         "<h1>🎧 A · 분노B 지시문 시험 — 음이 너무 높은 것은 문구 탓인가?</h1>",
         f"<p>모델 <b>A7NUX2sN step 3000</b>(LoRA) 하나 · 화자 6명 · 분노B 문장 4개 · <b>재학습 없이 추론 때 문구만</b> 바꿈 · seed {LP.SEED} · cfg 2.0. "
         f"재현 확인 {sum(r['ok'] for r in rp)}/{len(rp)} 같은 소리(C0 = D 재측정 · 공통 세트 음원과 비교).</p>",
         "<h2>조건별 평균 (참고용)</h2><table><tr><th>조건</th><th>문구</th><th>개수</th><th>F0 C0 대비</th><th>F0 REF 대비</th><th>목표 유사도</th><th>목표 1위<br>(두 잣대 평균)</th>"
         "<th>e2v angry 1위</th><th>angry 확률</th><th>CER</th></tr>"]
    for c, m in summ.items():
        extra = f"<div class=n>같은 문장의 C0: 유사도 {m['c0_same']['t_sim']:.3f} · angry {m['c0_same']['angry']:.0f}%</div>" if "c0_same" in m else ""
        h.append(f"<tr><td><b>{c}</b> {E(COND[c][0])}</td><td class=n>{E(COND[c][1])}{' + ! → .' if c == 'C3' else ''}</td><td>{m['n']}</td>"
                 f"<td>{m['st_c0']:+.2f}반음<div class=n>F0 잰 {m['n_f0']}개</div></td><td>{m['st_ref']:+.2f}반음</td><td>{m['t_sim']:.3f}{extra}</td><td>{m['top1']:.0f}%</td>"
                 f"<td>{m['angry']:.0f}%</td><td>{m['p_angry']:.2f}</td><td>{m['cer']:.1f}%</td></tr>")
    h.append("</table><p class=n>F0 = librosa pyin(16kHz) 유성 프레임 중앙값(유성 구간이 0.3초보다 짧은 출력과 그 비교는 뺌) · 범위는 화자마다 [REF F0 / 2, REF F0 + 20반음](60~800Hz 로 넓게 재면 남성 고함에서 배음을 잡아서) · REF 는 60~800Hz · 반음 = 12·log2(F0 / 기준). 목표 유사도 = WavLM(목표 화자 진짜 녹음 평균과 코사인) · "
             "1위 = 같은 성별 후보(목표 16명 + 542 처음 보는 화자 134명) 중 1위 비율 · e2v = emotion2vec · CER = whisper. D 재측정과 같은 코드입니다.</p>")
    h.append(f"<h2>들을 줄 — 통과 {len(passed)}개 중 {len(shown)}개</h2>"
             f"<p class=n>통과 = C1 · C2 · C3 중 하나가 C0 보다 F0 가 {PASS_ST}반음 이상 낮고 · e2v 1위가 여전히 angry · 목표 유사도가 C0 이상. "
             "대안 = 통과한 것 중 목표 유사도가 가장 높은 조건. F0 가 많이 내려간 순.</p>")
    if not passed:
        h.append("<p style='font-size:1.6em;color:#b71c1c'><b>문구로는 음 높이가 안 내려감 → 데이터 쪽 문제</b></p><p>통과한 줄이 없어서 들을 것은 없습니다. 아래 '전체 78개'는 참고용입니다.</p>")
    else:
        h.append("<table><tr><th>화자</th><th>문장 · REF</th><th>C0 지금</th><th>대안</th><th>판정 · 메모</th></tr>")
        for c0, b, oks in shown:
            v = PL["spk"][c0["spk"]]
            pid = f"{c0['spk']}_s{c0['sk']}"
            refp, w0, wb = f"{rel}/d_eval/ref/{c0['spk']}_ref.wav", f"wav/{TAG}/{c0['id']}.wav", f"wav/{TAG}/{b['id']}.wav"
            bid = f"{pid}|{b['cond']}"
            h.append(f"<tr><td><b>{E(c0['spk'])}</b><div class=n>{'여' if v['gender'] == 'female' else '남'} · {v['src']}</div></td>"
                     f"<td>{E(c0['text'])}{A(refp)}<div class=n>↑ REF</div></td>"
                     f"<td>{A(w0)}{num(c0)}</td>"
                     f"<td><b>{b['cond']}</b> {E(COND[b['cond']][0])}<div class=n>({E(b['text'] if b['cond'] == 'C3' else b['instr'])})</div>"
                     f"{A(wb)}{num(b)}<div class=n>통과 조건: {', '.join(oks)}</div></td>"
                     f"<td>{btn(bid, ['대안이 낫다', '같다', '못하다'])}<textarea rows=3 data-id='{pid}|memo'></textarea></td></tr>")
        h.append("</table>")
    # 참고: 전체
    h.append("<details><summary><b>(참고) 전체 78개 — 조건별로 다 듣고 싶을 때</b></summary><table><tr><th>화자</th><th>문장</th>"
             + "".join(f"<th>{c} {E(COND[c][0])}</th>" for c in COND) + "</tr>")
    for s in LP.SPK6:
        for k, t in enumerate(PL["sents"]):
            cells = "".join((f"<td>{A(f'wav/{TAG}/{s}_{c}_s{k}.wav')}{num(R[f'{s}_{c}_s{k}'])}</td>" if f"{s}_{c}_s{k}" in R else "<td class=n>-</td>") for c in COND)
            h.append(f"<tr><td>{E(s)}</td><td>{E(t)}</td>{cells}</tr>")
    h.append("</table></details>")
    rmeta = {f"{c0['spk']}_s{c0['sk']}": dict(spk=c0["spk"], text=c0["text"], alt=b["cond"]) for c0, b, _ in shown}
    h.append("<script>const PAGE=" + json.dumps(dict(key="instr_judg", rows=rmeta), ensure_ascii=False) + ";</script>" + r"""<script>
const K=PAGE.key;let S=JSON.parse(localStorage.getItem(K)||'{}');
function save(){localStorage.setItem(K,JSON.stringify(S));draw()}
function draw(){document.querySelectorAll('.b').forEach(d=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v)));
 document.querySelectorAll('textarea[data-id]').forEach(t=>{if(document.activeElement!==t)t.value=S[t.dataset.id]||''});
 document.getElementById('cnt').textContent=Object.keys(S).filter(k=>!k.endsWith('|memo')).length}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;if(S[id]===b.dataset.v)delete S[id];else S[id]=b.dataset.v;save()});
document.querySelectorAll('textarea[data-id]').forEach(t=>t.oninput=()=>{if(t.value.trim())S[t.dataset.id]=t.value;else delete S[t.dataset.id];save()});
function q(s){s=(s==null?'':String(s));return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function dl(){const L=['페이지,화자,문장,대안,판정,메모'];for(const [pid,r] of Object.entries(PAGE.rows)){const j=S[pid+'|'+r.alt],m=S[pid+'|memo'];
 if(j||m)L.push(['A 지시문 시험',r.spk,r.text,r.alt,j||'',m||''].map(q).join(','))}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+L.join('\n')],{type:'text/csv'}));a.download='vox_koft_instr_test.csv';a.click()}
draw();</script>""")
    (OUT / "index.html").write_text("\n".join(h), encoding="utf-8")
    json.dump(dict(summary=summ, passed=len(passed), shown=[(c0["id"], b["id"]) for c0, b, _ in shown]), open(OUT / "summary.json", "w"),
              ensure_ascii=False, indent=1, default=float)
    print("✅", OUT / "index.html", f"· 통과 {len(passed)} · 올림 {len(shown)}")
    for c, m in summ.items():
        print(f"  {c}: n {m['n']} · F0 C0 대비 {m['st_c0']:+.2f} · REF 대비 {m['st_ref']:+.2f} · 유사도 {m['t_sim']:.3f} · 1위 {m['top1']:.0f}% · "
              f"angry {m['angry']:.0f}% ({m['p_angry']:.2f}) · CER {m['cer']:.1f}%" + (f" | 같은 문장 C0: {m['c0_same']}" if "c0_same" in m else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["synth", "f0", "measure", "page"])
    ap.add_argument("--what", choices=["sv", "ser"])
    ap.add_argument("--smoke", action="store_true", help="synth: 화자 1명만")
    a = ap.parse_args()
    {"synth": cmd_synth, "f0": cmd_f0, "measure": cmd_measure, "page": cmd_page}[a.cmd](a)


if __name__ == "__main__":
    main()
