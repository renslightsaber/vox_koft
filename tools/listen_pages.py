#!/usr/bin/env python
"""작은 청음 페이지 — (B) 4모델 비교 · (C) run 별 (2026-11-08 · 지시문 q3tts 세션 14:40)

D 재측정 페이지(16명 × 14문장 × 많은 열)는 듣기에 너무 커서, **화자 6명 × 9줄(54줄)** 공통 세트로 작게 만든다.
  · 36줄 = d_eval 음원 재사용(상대 경로로 가리킴 · 복사 안 함) · 18줄/모델 = 새 문장 3개(기쁨A 1 · 분노B 2) × 6명 새 합성
  · 새 합성은 d_eval vox() 와 같은 조건(Engine · REF = plan.json spk ref · seed 1986 · cfg 2.0 · timesteps 10 · max_len 600 · 준비 합성)
  · 재현 확인: 모델마다 d_eval 기존 문항 1개를 같은 엔진으로 다시 합성 → d_eval wav 와 같아야(길이 같음 · max diff < 1e-3) 새 합성을 함
  · 지표는 d_eval sv() · ser() 코드를 그대로 불러 같은 잣대로(OUT · PLANF · WAVD 만 바꿈)
  · 줄마다 목소리(같음 · 애매 · 다름) · 감정(OK · 약함 · 틀림) 버튼 · 메모 · 모델 총평 · CSV 내려받기 · localStorage(B: cmp4_ · C: run_<run>_)

단계(각 env 의 python):
  1) python tools/listen_pages.py set                                                      # set.json · plan_new.json
  2) CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python tools/listen_pages.py synth --tags <tag ...> [--smoke]
  3) CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts/bin/python tools/listen_pages.py measure --what sv
     CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts_ser/bin/python tools/listen_pages.py measure --what ser
  4) python tools/listen_pages.py page
→ http://127.0.0.1:8765/vox_koft/listen_cmp/index.html · http://127.0.0.1:8765/vox_koft/listen_runs/index.html
숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxko.eval import d_eval as DE  # noqa: E402

H = Path.home() / "workspaces"
TL = H / "tmp_listen/vox_koft"
D = TL / "d_eval"
SET = TL / "listen_set6"
CMP, RUNS = TL / "listen_cmp", TL / "listen_runs"
EMAP = ROOT / "configs/emotion/a7n_vox_v1.yaml"
EMAP_V3 = ROOT / "configs/emotion/a7n_vox_v3.yaml"


def emap_of(tag: str) -> Path:
    """이름표 → 지시문 표. iv3 run(NUX2sNiv3lr5e5_*)은 학습 · D 재측정과 같은 v3 · 나머지는 v1(2026-11-09)."""
    return EMAP_V3 if tag.startswith("NUX2sNiv3lr5e5_") else EMAP
SEED, MAXLEN = 1986, 600

# 1-1 화자 6명(여 3 · 남 3 · 사용자 결정 11/08) — 음질 · 이상 지적 화자(0012 · 0109 · 0113 · 0110 · 0028)는 뺌
SPK6 = ["a542_2214_G2A2E7_JCE", "a542_2108_G2A5E7_KKH", "a542_1837_G1A2E7_KHG", "aihub_0152", "aihub_0115", "aihub_0156"]
# 1-2 줄 9개: (칸, 줄 키, 새 문장 | None = d_eval 문항 재사용)
ROWS = [("Neutrality", "Neutrality_0", None),
        ("Happy-A", "Happy-A_new", "바다가 너무 예뻐서 기분이 좋았어."),      # 출처 A7N val 466_0023_G1A3E1S0C0_KSH_001822
        ("Happy-B", "Happy-B_1", None), ("Sad-A", "Sad-A_1", None), ("Sad-B", "Sad-B_1", None),
        ("Angry-A", "Angry-A_0", None), ("Angry-B", "Angry-B_0", None),
        ("Angry-B", "Angry-B_new1", "너 자꾸 나 빡치게 할래?"),               # 사용자 문장
        ("Angry-B", "Angry-B_new2", "몇 번이나 말해야 알아듣겠어?")]          # 사용자 문장
CELLS = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
B_TAGS = ["NUX2sNlr5e5_s3000", "X2Rs6lr5e5_s4500", "A7NX3_s4500", "X2Rs01lr5e5_s4000"]
B_WHY = {"NUX2sNlr5e5_s3000": "통합 1위 · 감정 고른 판(542 지름길 없음)", "X2Rs6lr5e5_s4500": "통합 2위 · 목소리 상위",
         "A7NX3_s4500": "감정(e2v) 최고", "X2Rs01lr5e5_s4000": "목소리 유사도 최고"}
# run(tag 앞부분) → (ckpt 폴더, 한 줄 설명)
RUN = {"NUX2sNlr5e5": ("A7NUX2sN_xref_LoRA_01_r32_lr5e5", "A7NUX2sN — 542 0 · 무감정 10.69% 로 채움 · 다른 칸 ref · 새 5목소리 · LoRA lr 5e-5"),
       "NUX2sNiv3lr5e5": ("A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3", "A7NUX2sN iv3 — A7NUX2sN 과 같고 지시문 v3(분노B 문구에서 shouting · loud 뺌 · 합성도 v3 문구) · 11/09"),
       "NUX2sNnlr5e5": ("A7NUX2sNn_xref_LoRA_01_r32_lr5e5", "A7NUX2sNn — A7NUX2sN 과 같고 추가 무감정 행 ref 만 같은 화자 무감정 녹음 · 11/09"),
       "X2Rs6lr5e5": ("A7NX2Rs6_xref_LoRA_01_r32_lr5e5", "A7NX2Rs6 — 다른 칸 ref · 542 3.0%(화자당 6문장) · 무감정 ref 1개 · LoRA lr 5e-5"),
       "NUX2slr5e5": ("A7NUX2s_xref_LoRA_01_r32_lr5e5", "A7NUX2s — A7NUX2Rs6 에서 542 0(무감정 8.1%) · LoRA lr 5e-5"),
       "A7NX3": ("A7NX3_xref_LoRA_01_r32", "A7NX3 — 정답마다 같은 사람 다른 칸 ref 3개 · LoRA lr 1e-4"),
       "X2Rs01lr5e5": ("A7NX2Rs_xref_LoRA_01_r32_lr5e5", "A7NX2Rs — 다른 칸 ref · 542 5.8% · 무감정 ref 1개 · LoRA lr 5e-5"),
       "NUX2Rs6lr5e5": ("A7NUX2Rs6_xref_LoRA_01_r32_lr5e5", "A7NUX2Rs6 — A7NX2Rs6 + 새 5목소리(542 2.9%) · LoRA lr 5e-5"),
       "LoRA01": ("A7N_ref_LoRA_01_r32", "LoRA_01 — A7N · 같은 화자 무감정 ref · LoRA r32 lr 1e-4"),
       "X2R02lr5e5": ("A7NX2R_xref_LoRA_02_r32_lr5e5", "A7NX2R — 다른 칸 ref 2개 + 542 무감정 23% · LoRA lr 5e-5"),
       "FT01": ("A7N_ref_FT_01", "FT_01 — A7N · 같은 화자 무감정 ref · 전체 FT"),
       "X2R01": ("A7NX2R_xref_LoRA_01_r32", "A7NX2R — 다른 칸 ref 2개 + 542 무감정 23% · LoRA lr 1e-4"),
       "A7NX": ("A7NX_xref_LoRA_01_r32", "A7NX — 정답마다 같은 사람 다른 칸 ref 1개 · LoRA lr 1e-4"),
       "LoRA02iv2": ("A7N_ref_LoRA_02_r32_iv2", "LoRA_02 iv2 — 지시문 v2(감정마다 4문구) · ref 0.5 · LoRA r32"),
       "FT02iv2": ("A7N_ref_FT_02_iv2", "FT_02 iv2 — 지시문 v2(감정마다 4문구) · ref 0.5 · 전체 FT")}
REPRO_ROW = "Angry-B_0"                       # 재현 확인에 쓰는 d_eval 문항(화자는 모델마다 돌려 씀)


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def rows_of(P):
    """화자 6 × 줄 9 = 54줄 — 줄 id · 화자 · 칸 · 문장 · 음원 출처."""
    it = {x["id"]: x for x in P["items"]}
    out = []
    for s in SPK6:
        for emo, key, new in ROWS:
            rid = f"{s}_{key}"
            out.append(dict(rid=rid, spk=s, emo=emo, key=key, text=new if new else it[rid]["text"], new=bool(new)))
    return out


# ───────────────────────────── 1) set
def top_steps(S, run):
    """run 의 d_eval 통합 점수 상위 — 기본 3개 · 4위가 1위와 2점 안이면 4개."""
    v = sorted(((m["전체"]["score"], t) for t, m in S.items() if t.rsplit("_s", 1)[0] == run and t.rsplit("_s", 1)[-1].isdigit()), reverse=True)
    n = 4 if len(v) >= 4 and v[0][0] - v[3][0] <= 2.0 else 3
    return [(t, sc) for sc, t in v[:n]], [(t, sc) for sc, t in v[:4]]


def cmd_set(a):
    from voxko.engine.synth import ko_normalize
    P, S = jl(D / "plan.json"), jl(D / "summary.json")["models"]
    for s in SPK6:
        assert s in P["spk"], s
    C, top4 = {}, {}
    for run in RUN:
        c_, t_ = top_steps(S, run)
        if c_:                                   # 아직 D 재측정 점수가 없는 run 은 건너뜀(다음 set 때 들어감)
            C[run], top4[run] = c_, t_
        else:
            print(f"⏳ {run}: summary 에 점수 없음 — 건너뜀")
    order = sorted(C, key=lambda r: -C[r][0][1])
    tags = sorted({t for r in C for t, _ in C[r]} | set(B_TAGS))
    model = {t: jl(D / f"vox_{t}.json")["model"] for t in tags}
    for t, p in model.items():
        assert Path(p).is_dir(), p
    norm = {new: ko_normalize(new) for _, _, new in ROWS if new}
    rows = rows_of(P)
    SET.mkdir(parents=True, exist_ok=True)
    json.dump(dict(spk=SPK6, rows=rows, norm=norm, B=B_TAGS, C={r: C[r] for r in order}, top4={r: top4[r] for r in order}, model=model,
                   created=time.strftime("%Y-%m-%d %H:%M KST", time.gmtime(time.time() + 9 * 3600))),
              open(SET / "set.json", "w"), ensure_ascii=False, indent=1)
    # 새 문장 계획 — spk · distract 는 d_eval 그대로(같은 후보로 목소리 순위), items 는 새 줄만
    items = [dict(id=r["rid"], tgt=r["spk"], emo=r["emo"], text=norm[r["text"]]) for r in rows if r["new"]]
    json.dump(dict(spk=P["spk"], distract=P["distract"], items=items, seed=SEED), open(SET / "plan_new.json", "w"), ensure_ascii=False, indent=1)
    print("정규화:", json.dumps(norm, ensure_ascii=False))
    print(f"줄 {len(rows)}(새 {len(items)}) · 모델 {len(tags)}개(B {len(B_TAGS)} · C {sum(len(v) for v in C.values())})")
    for r in order:
        print(f"  {r:<14}", " · ".join(f"{t.rsplit('_', 1)[1]} {sc:.1f}" for t, sc in C[r]), " | 4위:", f"{top4[r][3][1]:.1f}" if len(top4[r]) > 3 else "-")


# ───────────────────────────── 2) synth (env voxcpm)
def same(w1, w2):
    return len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3


def cmd_synth(a):
    import soundfile as sf
    import torch
    from voxko.engine.synth import Engine
    St, P, PN = jl(SET / "set.json"), jl(D / "plan.json"), jl(SET / "plan_new.json")
    items = [x for x in PN["items"] if not a.smoke or x["tgt"] == SPK6[0]]
    for t in a.tags:
        W = SET / "wav" / t
        W.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        eng = Engine(St["model"][t], emotion_map=str(emap_of(t)))                  # 준비 합성 포함(iv3 는 v3 문구)
        # 재현: d_eval 문항 1개(화자는 이름표 순서로 돌려 씀) — 같지 않으면 이 모델은 새 합성 안 함
        k = sorted(St["model"]).index(t) % len(SPK6)
        x = next(y for y in P["items"] if y["id"] == f"{SPK6[k]}_{REPRO_ROW}")
        w = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=P["spk"][x["tgt"]]["ref"], seed=SEED, max_len=MAXLEN)
        w0 = sf.read(D / "wav" / t / f"{x['id']}.wav")[0]
        ok = same(w, w0)
        meta = dict(tag=t, model=St["model"][t], repro=ok, repro_id=x["id"], repro_len=[len(w0), len(w)],
                    repro_maxdiff=float(np.abs(w - w0).max()) if len(w) == len(w0) else None)
        if ok:
            n = 0
            for y in items:
                p = W / f"{y['id']}.wav"
                if p.exists():
                    continue
                txt = eng.compose(y["text"], emotion=y["emo"], normalize=False)     # 텍스트는 set 에서 미리 ko_tts_norm
                sf.write(p, eng.synth(txt, ref=P["spk"][y["tgt"]]["ref"], seed=SEED, max_len=MAXLEN), eng.sr)
                n += 1
            meta.update(new=n, text_example=eng.compose(items[0]["text"], emotion=items[0]["emo"], normalize=False))
        meta.update(sec=round(time.time() - t0, 1), smoke=bool(a.smoke), time_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)))
        if not a.smoke:
            json.dump(meta, open(SET / f"synth_{t}.json", "w"), ensure_ascii=False, indent=1)
        print(("✅" if ok else "❌ 재현 실패 — 이 모델은 멈춤"), json.dumps(meta, ensure_ascii=False), flush=True)
        del eng
        torch.cuda.empty_cache()


# ───────────────────────────── 3) measure (sv: env q3tts · ser: env q3tts_ser) — d_eval 코드 그대로
def de_measure(out: Path, plan: Path, what: str, tags):
    """d_eval sv() · ser() 를 그대로 — 계획은 모든 이름표에 wav 가 있는 문항만(smoke · 덜 만든 모델 대비)."""
    P = jl(plan)
    P["items"] = [x for x in P["items"] if all((out / "wav" / t / f"{x['id']}.wav").exists() for t in tags)]
    tmp = out / f"_plan_measure_{what}.json"
    json.dump(P, open(tmp, "w"), ensure_ascii=False)
    print(f"{what}: 이름표 {len(tags)} · 문항 {len(P['items'])}", flush=True)
    DE.OUT, DE.PLANF, DE.WAVD, DE.PFX = out, tmp, out / "wav", ""
    ns = argparse.Namespace(tags=tags, force=False)
    {"sv": DE.sv, "ser": DE.ser}[what](ns)


def cmd_measure(a):
    tags = a.tags or sorted(p.name for p in (SET / "wav").iterdir() if p.is_dir() and (SET / f"synth_{p.name}.json").exists())
    de_measure(SET, SET / "plan_new.json", a.what, tags)


# ───────────────────────────── 4) page
CSS = """body{font-family:sans-serif;max-width:1900px;margin:16px auto;line-height:1.45;padding:0 10px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}th{background:#f3f3f3}.n{color:#666;font-size:.8em}audio{width:190px}
button{margin:1px;font-size:.8em}button.on{background:#1565c0;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}
body.hidenum .num{display:none}textarea{width:96%;font-size:.85em}tr.hide{display:none}.tag{font-weight:bold}"""

JS = r"""<script>
const K=PAGE.key;let S=JSON.parse(localStorage.getItem(K)||'{}');
function save(){localStorage.setItem(K,JSON.stringify(S));cnt()}
function cnt(){document.getElementById('cnt').textContent=Object.keys(S).filter(k=>!k.endsWith('|memo')&&!k.startsWith('총평|')).length}
function draw(){document.querySelectorAll('.b').forEach(d=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v)));
 document.querySelectorAll('textarea[data-id]').forEach(t=>t.value=S[t.dataset.id]||'');cnt()}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;if(S[id]===b.dataset.v)delete S[id];else S[id]=b.dataset.v;save();draw()});
document.querySelectorAll('textarea[data-id]').forEach(t=>t.oninput=()=>{if(t.value.trim())S[t.dataset.id]=t.value;else delete S[t.dataset.id];save()});
function filt(){const c=document.getElementById('fc').value,g=document.getElementById('fg').value;let n=0;
 document.querySelectorAll('tr.r').forEach(r=>{const ok=(c==='all'||r.dataset.emo===c)&&(g==='all'||r.dataset.grp===g);r.classList.toggle('hide',!ok);n+=ok});
 document.getElementById('shown').textContent=n}
function q(s){s=(s==null?'':String(s));return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function dl(){const L=[['페이지','화자','칸','문장','모델','목소리','감정','메모'].join(',')];
 for(const m of PAGE.models){if(S['총평|'+m])L.push([PAGE.name,'','','',m,'','',S['총평|'+m]].map(q).join(','))}
 for(const [rid,r] of Object.entries(PAGE.rows)){
  for(const m of PAGE.models){const v=S[m+'|'+rid+'|v'],e=S[m+'|'+rid+'|e'];if(v||e)L.push([PAGE.name,r.spk,r.cell,r.text,m,v||'',e||'',''].map(q).join(','))}
  if(S[rid+'|memo'])L.push([PAGE.name,r.spk,r.cell,r.text,'','','',S[rid+'|memo']].map(q).join(','))}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+L.join('\n')],{type:'text/csv'}));a.download=PAGE.csv;a.click()}
document.getElementById('fc').onchange=filt;document.getElementById('fg').onchange=filt;filt();draw();
</script>"""


def metrics_for(tag):
    """줄 id → (sv, ser) — 재사용 줄은 d_eval, 새 줄은 listen_set6."""
    sv, sr = jl(D / f"sv_{tag}.json"), jl(D / f"ser_{tag}.json")
    if (SET / f"sv_{tag}.json").exists():
        sv.update(jl(SET / f"sv_{tag}.json"))
    if (SET / f"ser_{tag}.json").exists():
        sr.update(jl(SET / f"ser_{tag}.json"))
    return sv, sr


def head_stats(S, P, t):
    """열 머리: 통합 점수 · 목표 유사도 · 1위 비율(두 잣대 평균) · e2v(전체 · 542 A · unseen8 A) — 모두 d_eval 224문항."""
    m = S[t]["전체"]
    q = jl(D / f"ser_{t}.json")
    hitA = lambda g: 100 * np.mean([q[x["id"]]["top"] == DE.E2V[DE.base(x["emo"])] for x in P["items"]  # noqa: E731
                                    if P["spk"][x["tgt"]]["src"] == g and x["emo"].endswith("-A")])
    return dict(score=m["score"], t_sim=m["t_sim"], top1=m.get("fair_top1", m["top1"]), e2v=m["emo_hit"], a542=hitA("542"), aun=hitA("unseen8"))


def render(name, key, models, labels, out: Path, intro: str, csvname: str):
    """models = 이름표 목록(열 순서) · labels[t] = 열 머리 html."""
    E = html.escape
    St, P, S = jl(SET / "set.json"), jl(D / "plan.json"), jl(D / "summary.json")["models"]
    rel = os.path.relpath(TL, out.parent)
    M = {t: metrics_for(t) for t in models}
    import yaml
    instr = (yaml.safe_load(open(emap_of(models[0]), encoding="utf-8")) or {}).get("instruct", {})
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    h = [f"<!doctype html><meta charset=utf-8><title>{E(name)}</title><style>{CSS}</style>",
         "<div id=bar>판정 <span id=cnt>0</span>개 · 보이는 줄 <span id=shown>0</span> · 칸 <select id=fc>"
         + "".join(f"<option value='{c}'{' selected' if c == 'Neutrality' else ''}>{DE.KO[c]}</option>" for c in CELLS)
         + "<option value=all>전체</option></select> · 묶음 <select id=fg><option value=all>전체</option><option value=542>542</option><option value=unseen8>unseen8</option></select>"
         " · <button onclick='dl()'>CSV 내려받기</button> · <label><input type=checkbox onchange=\"document.body.classList.toggle('hidenum',this.checked)\"> 숫자 숨기기(귀 먼저)</label>"
         " · <button onclick=\"if(confirm('이 페이지의 판정 · 메모를 모두 지울까요?')){localStorage.removeItem(PAGE.key);location.reload()}\">판정 지우기</button></div>",
         intro]
    h.append("<table><tr><th style='width:150px'>화자 · 칸</th><th style='width:210px'>문장 · REF</th>"
             + "".join(f"<th>{labels[t]}<div class=n>이 모델 총평</div><textarea rows=2 data-id='총평|{E(t)}'></textarea></th>" for t in models)
             + "<th style='width:180px'>메모</th></tr>")
    rows, rmeta = St["rows"], {}
    for r in rows:
        v = P["spk"][r["spk"]]
        rid = r["rid"]
        refp = f"{rel}/d_eval/ref/{r['spk']}_ref.wav"
        rmeta[rid] = dict(spk=r["spk"], cell=DE.KO[r["emo"]], text=r["text"])
        cells = []
        for t in models:
            src = (f"{rel}/listen_set6/wav/{t}/{rid}.wav" if r["new"] else f"{rel}/d_eval/wav/{t}/{rid}.wav")
            sv, sr = M[t][0].get(rid), M[t][1].get(rid)
            if r["new"] and not (SET / "wav" / t / f"{rid}.wav").exists():
                cells.append("<td><span class=n>없음(아직 합성 안 함)</span></td>")
                continue
            num = ""
            if sv and sr:
                rk = "✓1위" if sv["top1"] else f"{sv['rank']}위"
                num = f"<div class='n num'>목표 {sv['t_sim']:.2f} · {rk} · e2v {sr['top']} · CER {100 * sr['cer']:.0f}%</div>"
            cells.append(f"<td>{A(src)}{num}{btn(f'{t}|{rid}|v', ['같음', '애매', '다름'])}{btn(f'{t}|{rid}|e', ['OK', '약함', '틀림'])}</td>")
        h.append(f"<tr class=r data-emo='{r['emo']}' data-grp='{v['src']}'><td><b>{E(r['spk'])}</b><div class=n>{'여' if v['gender'] == 'female' else '남'} · {v['src']}</div>"
                 f"{DE.KO[r['emo']]}{' 🆕' if r['new'] else ''}</td>"
                 f"<td>{E(r['text'])}<div class=n>({E(instr.get(r['emo'], ''))})</div>{A(refp)}<div class=n>↑ REF(목표 화자 무감정)</div></td>"
                 + "".join(cells) + f"<td><textarea rows=3 data-id='{E(rid)}|memo'></textarea></td></tr>")
    h.append("</table>")
    h.append(f"<p class=n>입력은 모든 열이 같습니다: \"(v1 지시문)문장\" + 목표 화자 무감정 REF · seed {SEED} · cfg 2.0 · timesteps 10. "
             "🆕 = 이 세트에서 새로 합성한 문장(나머지는 D 재측정 음원 그대로). 숫자: 목표 = 목표 화자 진짜 녹음 평균과의 WavLM 유사도 · n위 = 같은 성별 후보 중 순위 · "
             "e2v = emotion2vec 1위 감정 · CER = whisper. 숫자는 참고용 — 판정은 귀로.</p>")
    page = dict(key=key, name=name, models=models, rows=rmeta, csv=csvname)
    h.append("<script>const PAGE=" + json.dumps(page, ensure_ascii=False) + ";</script>" + JS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(h), encoding="utf-8")
    return out


def label(S, P, t, prefix=""):
    s = head_stats(S, P, t)
    return (f"<span class=tag>{prefix}{html.escape(t)}</span><div class=n>통합 {s['score']:.1f} · 목표 유사도 {s['t_sim']:.3f} · 1위 {s['top1']:.0f}%<br>"
            f"e2v 전체 {s['e2v']:.0f}% · 542 A {s['a542']:.0f}% · unseen8 A {s['aun']:.0f}%</div>")


def ready(t):
    m = SET / f"synth_{t}.json"
    return m.exists() and jl(m)["repro"] and (SET / f"sv_{t}.json").exists() and (SET / f"ser_{t}.json").exists()


def cmd_page(a):
    St, P, S = jl(SET / "set.json"), jl(D / "plan.json"), jl(D / "summary.json")["models"]
    E = html.escape
    guide = ("<ul><li>화자 6명(542 여2 · 남1 · unseen8 여1 · 남2) × 9줄 = 54줄. 기본은 <b>칸 하나씩</b>(위 '칸' 고르기) 봅니다.</li>"
             "<li>열마다 두 가지를 누릅니다: <b>목소리</b>(REF 와 같은 사람인가: 같음 · 애매 · 다름) · <b>감정</b>(칸의 감정이 분명한가: OK · 약함 · 틀림).</li>"
             "<li>줄마다 메모 칸, 열 머리마다 '이 모델 총평' 칸이 있습니다. 판정과 메모는 이 브라우저에 저장되고 <b>CSV 내려받기</b>로 받습니다.</li>"
             "<li>열 머리 숫자는 D 재측정 224문항 전체 값입니다(참고용).</li></ul>")
    only = set(a.runs or [])                     # --runs: 이 run 페이지만 새로(B · 다른 run 페이지는 그대로) · 목차는 늘 새로
    # B
    miss = [t for t in St["B"] if not ready(t)]
    lb = {t: label(S, P, t) + f"<div class=n>{E(B_WHY[t])}</div>" for t in St["B"]}
    if not only:
      render("B · 4모델 비교", "cmp4_judg", St["B"], lb, CMP / "index.html",
           f"<h1>🎧 vox_koft B · 4모델 비교 청음</h1><p class=n>열 = 통합 점수 순서 · 만든 때 {time.strftime('%m/%d %H:%M', time.gmtime(time.time() + 9 * 3600))} KST · "
           f"<a href='../listen_runs/index.html'>run 별 페이지(C)</a></p>{guide}" + (f"<p>⚠️ 아직 준비 안 된 열: {miss}</p>" if miss else ""),
           "vox_koft_cmp4.csv")
    # C
    idx = [f"<!doctype html><meta charset=utf-8><title>C · run 별 청음</title><style>{CSS}</style><h1>🎧 vox_koft C · run 별 청음 (11/06 ~ 09 학습 {len(St['C'])}판)</h1>",
           "<p>👉 <a href='../listen_cmp/index.html'><b>B · 4모델 비교 페이지</b></a></p>",
           "<p class=n>run 마다 한 페이지 · 열 = 그 run 의 D 재측정 통합 점수 상위 step(기본 3개 · 4위가 1위와 2점 안이면 4개). 1위 점수 순. 세트는 B 와 같습니다(6명 × 9줄).</p>",
           "<table><tr><th>순</th><th>run</th><th>설명</th><th>열(점수)</th><th>페이지</th></tr>"]
    for i, (run, cols) in enumerate(St["C"].items(), 1):
        tags = [t for t, _ in cols]
        miss = [t for t in tags if not ready(t)]
        lb = {t: label(S, P, t, f"{k}위 s{t.rsplit('_s', 1)[1]} · {sc:.1f}<br>") for k, (t, sc) in enumerate(cols, 1)}
        f = RUNS / f"{run}.html"
        if not only or run in only:
          f = render(f"C · {run}", f"run_{run}_judg", tags, lb, RUNS / f"{run}.html",
                   f"<h1>🎧 {E(run)} — {E(RUN[run][1])}</h1><p class=n>ckpt {E(RUN[run][0])} · 열 = 통합 점수 상위 {len(tags)}개 · "
                   f"<a href='index.html'>목차</a> · <a href='../listen_cmp/index.html'>B 비교</a></p>{guide}" + (f"<p>⚠️ 아직 준비 안 된 열: {miss}</p>" if miss else ""),
                   f"vox_koft_run_{run}.csv")
        colstr = " · ".join(f"s{t.rsplit('_s', 1)[1]} {sc:.1f}" for t, sc in cols)
        idx.append(f"<tr><td>{i}</td><td><b>{E(run)}</b></td><td>{E(RUN[run][1])}</td><td>{colstr}</td>"
                   f"<td><a href='{f.name}'>열기</a>{' ⚠️' if miss else ''}</td></tr>")
    idx.append("</table>")
    (RUNS / "index.html").write_text("\n".join(idx), encoding="utf-8")
    print("✅", CMP / "index.html", "·", RUNS / "index.html", f"· run 페이지 {len(St['C'])}개")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["set", "synth", "measure", "page"])
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="synth: 화자 1명만(메타 저장 안 함)")
    ap.add_argument("--what", choices=["sv", "ser"])
    ap.add_argument("--runs", nargs="*", help="page: 이 run 페이지만 새로 만듦(B 페이지 · 다른 run 페이지는 그대로 · 목차는 늘 새로)")
    a = ap.parse_args()
    {"set": cmd_set, "synth": cmd_synth, "measure": cmd_measure, "page": cmd_page}[a.cmd](a)


if __name__ == "__main__":
    main()
