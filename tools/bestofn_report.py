"""tools/bestofn.py 의 report · page 단계(분석 · 청음 페이지) — 직접 실행하지 말고 `python tools/bestofn.py report|page`.

자기 중립 유사도(own_n) = 출력 ↔ "같은 모델 · 같은 목표 화자의 d_eval 생성 중립 2개 평균"의 코사인(WavLM · ERes2Net 평균)
  — tools/vspk.py 와 같은 임베딩 · 같은 중립(d_eval wav/<model>/<s>_Neutrality_{0,1}). ② 의 prompt 조건도 같은 중립에 견줌.
① 고르기: 문장마다 seed 5개 중 [CER ≤ 10% · 이상 없음 · (e2v 가 칸의 감정 묶음 or arousal 이 진짜 녹음 같은 칸 10~90% 범위)]
          을 만족하는 것 가운데 own_n 최대 1개(없으면 "후보 없음")
③ 문턱: 진짜 녹음 천장(vspk emb_real · 16명)의 같은 구조 값 = 감정 녹음 ↔ 그 사람 중립 2개 평균 — 묶음(기쁨 · 슬픔 · 분노)별
        하위 10% · 20% 값보다 낮으면 버림
"""
from __future__ import annotations

import html
import json
from collections import defaultdict

import numpy as np

import bestofn as BO

DE, B, D = BO.DE, BO.B, BO.D
KLAB = {"cfg15": "cfg 1.5", "cfg20": "cfg 2.0(기본 · d_eval 과 같은 소리)", "cfg25": "cfg 2.5",
        "prompt": "prompt 모드 ⚠️ 지시문을 소리 내어 읽음", "prompt_front": "prompt 모드 · 지시문 맨 앞"}
KO = {"Angry-A": "분노A", "Happy-B": "기쁨B", "Angry-B": "분노B", "Sad-B": "슬픔B"}
E2V = DE.E2V


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ncent(model: str) -> dict:
    """{key: {spk: 중립 평균 단위벡터}} — d_eval 생성 중립 2개."""
    z = np.load(D / "vspk" / f"emb_{model}.npz")
    ids = list(z["ids"])
    out = {}
    for k in ("w", "e"):
        by = defaultdict(list)
        for i, v in zip(ids, z[k]):
            if "_Neutrality_" in i:
                by[i.split("_Neutrality_")[0]].append(unit(v))
        out[k] = {s: unit(np.mean(vs, 0)) for s, vs in by.items()}
    return out


def real_thr() -> dict:
    """진짜 녹음: 묶음별 own_n 분포 → {grp: dict(p10, p20, mean, vals)}."""
    z = np.load(D / "vspk/emb_real.npz")
    meta = json.loads(str(z["meta"]))
    ids = list(z["ids"])
    sims = defaultdict(list)
    spk = sorted({m["spk"] for m in meta})
    for s in spk:
        idx = [j for j, m in enumerate(meta) if m["spk"] == s]
        nidx = [j for j in idx if meta[j]["grp"] == "Neutrality"]
        for j in idx:
            g = meta[j]["grp"]
            if g in ("Happy", "Sad", "Angry"):
                sims[g].append(float(np.mean([unit(z[k][j]) @ unit(np.mean([unit(z[k][n]) for n in nidx], 0)) for k in ("w", "e")])))
    assert len(ids) == len(meta)
    return {g: dict(p10=float(np.percentile(v, 10)), p20=float(np.percentile(v, 20)), mean=float(np.mean(v)), n=len(v)) for g, v in sims.items()}


def aro_range() -> dict:
    r = json.load(open(B / "real_arousal.json"))
    by = defaultdict(list)
    for v in r.values():
        by[v["emo"]].append(v["arousal"])
    return {e: (float(np.percentile(v, 10)), float(np.percentile(v, 90)), len(v)) for e, v in by.items()}


def load_items(setname: str, tag: str, model: str, thr_sq, ar):
    """한 이름표의 문항별 지표 dict."""
    pf, wd, px = BO.SETS[setname]
    P = json.load(open(B / pf))
    z = np.load(B / f"{px}emb_{tag}.npz")
    zi = {i: j for j, i in enumerate(z["ids"])}
    sv, sr, qa = (json.load(open(B / f"{px}{n}_{tag}.json")) for n in ("sv", "ser", "quality"))
    nc = ncent(model)
    out = {}
    for x in P["items"]:
        i, s, e = x["id"], x["tgt"], x["emo"]
        j = zi[i]
        own = float(np.mean([unit(z[k][j]) @ nc[k][s] for k in ("w", "e")]))
        r, q, a = sv[i], sr[i], qa[i]
        hit = q["top"] == E2V[DE.base(e)]
        lo, hi, _ = ar[e]
        o = DE.odd(a, thr_sq)
        emo_ok = hit or (lo <= q["arousal"] <= hi)
        out[i] = dict(id=i, base_id=x["base_id"], tgt=s, emo=e, seed=x["seed"], text=x["text"], own=own,
                      ref=(r["ref_sim"] + r["e_ref_sim"]) / 2, top1=(r["top1"] + r["e_top1"]) / 2, w_top1=r["top1"], e_top1=r["e_top1"],
                      hit=hit, arousal=q["arousal"], top=q["top"], cer=q["cer"], odd=o, emo_ok=emo_ok,
                      ok=(q["cer"] <= 0.10 and not o and emo_ok), dur=a["dur"])
    return out


def pick(items: dict, T: dict):
    """문장(base_id)마다 ① 고르기 + ③ 문턱."""
    by = defaultdict(list)
    for v in items.values():
        by[v["base_id"]].append(v)
    res = {}
    for b, vs in by.items():
        vs = sorted(vs, key=lambda v: v["seed"])
        s86 = [v for v in vs if v["seed"] == 1986][0]
        c = [v for v in vs if v["ok"]]
        ch = max(c, key=lambda v: v["own"]) if c else None
        g = DE.base(vs[0]["emo"])
        lvl = None if ch is None else ("p20" if ch["own"] >= T[g]["p20"] else "p10" if ch["own"] >= T[g]["p10"] else "drop")
        res[b] = dict(s86=s86, all=vs, chosen=ch, n_ok=len(c), lvl=lvl, grp=g,
                      s86_lvl="p20" if s86["own"] >= T[g]["p20"] else "p10" if s86["own"] >= T[g]["p10"] else "drop")
    return res


RULES = {  # ① 고르기 조건 비교(고른 뒤 = 조건을 지킨 것 중 자기 중립 유사도 최대)
    "V1 e2v 또는 arousal 범위(페이지)": lambda v, vs, ar: v["ok"],
    "V2 e2v 맞힘 + arousal 범위": lambda v, vs, ar: v["cer"] <= 0.10 and not v["odd"] and v["hit"] and ar[v["emo"]][0] <= v["arousal"] <= ar[v["emo"]][1],
    "V3 V1 + arousal ≥ 다섯 개 중앙값": lambda v, vs, ar: v["ok"] and v["arousal"] >= float(np.median([u["arousal"] for u in vs])),
}


def rule_table(items, T, ar):
    by = defaultdict(list)
    for v in items.values():
        by[v["base_id"]].append(v)
    out = {}
    for name, f in RULES.items():
        rows = defaultdict(list)
        for vs in by.values():
            c = [v for v in vs if f(v, vs, ar)]
            ch = max(c, key=lambda v: v["own"]) if c else None
            s86 = [v for v in vs if v["seed"] == 1986][0]
            for k in (s86["emo"], "전체"):
                rows[k].append((s86, ch))
        out[name] = {e: dict(own=float(np.mean([c["own"] for _, c in rs if c])), hit86=100 * float(np.mean([x["hit"] for x, _ in rs])),
                             hit=100 * float(np.mean([c["hit"] for _, c in rs if c])), aro86=float(np.mean([x["arousal"] for x, _ in rs])),
                             aro=float(np.mean([c["arousal"] for _, c in rs if c])),
                             keep20=100 * sum(1 for _, c in rs if c and c["own"] >= T[DE.base(c["emo"])]["p20"]) / len(rs)) for e, rs in rows.items()}
    return out


def agg(vs):
    vs = [v for v in vs if v]
    if not vs:
        return dict(n=0)
    m = lambda k: float(np.mean([v[k] for v in vs]))  # noqa: E731
    return dict(n=len(vs), own=m("own"), ref=m("ref"), top1=100 * m("top1"), hit=100 * m("hit"), arousal=m("arousal"),
                cer=100 * m("cer"), odd=100 * float(np.mean([bool(v["odd"]) for v in vs])), ok=100 * m("ok"))


def report(a=None):
    DE.OUT = D
    thr_sq = DE._sq_thr()
    T, ar = real_thr(), aro_range()
    out = dict(real_thr=T, aro_range=ar, squim_thr=thr_sq, bon={}, knob={})
    md = ["# best-of-N · 문턱 · 노브 — 표(참고용)", ""]
    md += ["진짜 녹음 문턱(감정 녹음 ↔ 그 사람 중립 2개 평균 · 두 잣대 평균): " +
           " · ".join(f"{g} 평균 {v['mean']:.3f} · 하위10% {v['p10']:.3f} · 하위20% {v['p20']:.3f}" for g, v in T.items()), "",
           "arousal 범위(A7N val 진짜 녹음 10~90%): " + " · ".join(f"{KO[e]} {lo:.2f}~{hi:.2f}(n{n})" for e, (lo, hi, n) in ar.items()), ""]
    for model in BO.MODELS:
        if not (B / f"bon_emb_{model}.npz").exists():
            continue
        it = load_items("bon", model, model, thr_sq, ar)
        R = pick(it, T)
        rows = {}
        for e in BO.CELLS + ["전체"]:
            rs = [r for r in R.values() if e == "전체" or r["s86"]["emo"] == e]
            n = len(rs)
            s86 = agg([r["s86"] for r in rs])
            all5 = agg([v for r in rs for v in r["all"]])
            ch = agg([r["chosen"] for r in rs])
            k10 = agg([r["chosen"] for r in rs if r["lvl"] in ("p10", "p20")])
            k20 = agg([r["chosen"] for r in rs if r["lvl"] == "p20"])
            rows[e] = dict(n=n, s86=s86, all5=all5, chosen=ch, keep10=k10, keep20=k20,
                           cover=100 * sum(r["chosen"] is not None for r in rs) / n,
                           s86_keep10=100 * sum(r["s86_lvl"] in ("p10", "p20") and r["s86"]["ok"] for r in rs) / n,
                           s86_keep20=100 * sum(r["s86_lvl"] == "p20" and r["s86"]["ok"] for r in rs) / n,
                           keep10_pct=100 * k10["n"] / n, keep20_pct=100 * k20["n"] / n,
                           trap=bool(ch["n"] and (ch["hit"] < s86["hit"] - 5 or ch["arousal"] < s86["arousal"] - 0.02)))
        vm = json.load(open(B / f"bon_vox_{model}.json"))
        out["bon"][model] = dict(rows=rows, repro=vm.get("repro_vs_d_eval"), rtf=vm.get("rtf"),
                                 picks={b: dict(chosen=(r["chosen"] or {}).get("id"), seed=(r["chosen"] or {}).get("seed"), lvl=r["lvl"],
                                                n_ok=r["n_ok"]) for b, r in R.items()})
        md += [f"## ① · ③ — {model} (seed 1986 재현 {vm.get('repro_vs_d_eval')})", "",
               "| 칸 | 자기중립 1986 → ① | e2v 1986 → ① | arousal 1986 → ① | 목표1위 1986 → ① | ① 후보 있음 | ①+③ 남음 10% · 20% | 1986+③ 남음 10% · 20% | 함정 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for e, r in rows.items():
            s, c = r["s86"], r["chosen"]
            md.append(f"| {KO.get(e, e)} | {s['own']:.3f} → {c.get('own', float('nan')):.3f} | {s['hit']:.0f} → {c.get('hit', float('nan')):.0f}% | "
                      f"{s['arousal']:.2f} → {c.get('arousal', float('nan')):.2f} | {s['top1']:.0f} → {c.get('top1', float('nan')):.0f}% | {r['cover']:.0f}% | "
                      f"{r['keep10_pct']:.0f} · {r['keep20_pct']:.0f}% | {r['s86_keep10']:.0f} · {r['s86_keep20']:.0f}% | {'⚠️' if r['trap'] else '-'} |")
        rt = rule_table(it, T, ar)
        out["bon"][model]["rules"] = rt
        md += ["", "고르기 조건 비교(전체 · 칸별 '자기중립 / e2v 1986→고른 것 / arousal 1986→고른 것 / ③20% 남음')", ""]
        for name, d in rt.items():
            md.append(f"- {name}: " + " · ".join(f"{KO.get(e, e)} {v['own']:.3f} / {v['hit86']:.0f}→{v['hit']:.0f}% / {v['aro86']:.2f}→{v['aro']:.2f} / {v['keep20']:.0f}%"
                                                for e, v in d.items()))
        md += ["", f"(5개 평균: 자기중립 {rows['전체']['all5']['own']:.3f} · e2v {rows['전체']['all5']['hit']:.0f}% · arousal {rows['전체']['all5']['arousal']:.2f} · "
                   f"조건 통과 {rows['전체']['all5']['ok']:.0f}% — 1986 하나도 같은 분포의 한 표본)", ""]
    # ② 노브
    kn = {}
    for c in BO.KNOB:
        if (B / f"knob_emb_{c}.npz").exists() and (B / f"knob_quality_{c}.json").exists():
            kn[c] = load_items("knob", c, BO.KNOB_MODEL, thr_sq, ar)
    if kn:
        md += ["## ② 노브 — LoRA01_s1500 · 목표 4명 × 4칸 × 2문장 · seed 1986", "",
               "| 조건 | 칸 | 자기중립 | REF | 목표1위 | e2v | arousal | CER | 이상 |", "|---|---|---|---|---|---|---|---|---|"]
        for c, it in kn.items():
            out["knob"][c] = {}
            vm = json.load(open(B / f"knob_vox_{c}.json"))
            out["knob"][c]["repro"] = vm.get("repro_vs_d_eval")
            out["knob"][c]["rtf"] = vm.get("rtf")
            for e in BO.CELLS + ["전체"]:
                m = agg([v for v in it.values() if e == "전체" or v["emo"] == e])
                out["knob"][c][e] = m
                md.append(f"| {c} | {KO.get(e, e)} | {m['own']:.3f} | {m['ref']:.3f} | {m['top1']:.0f}% | {m['hit']:.0f}% | {m['arousal']:.2f} | {m['cer']:.1f}% | {m['odd']:.0f}% |")
        md.append("")
    (B / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(out, open(B / "report.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(md))
    return out


# ───────────────────────────── page
def page(a=None):
    import yaml
    out = report()
    DE.OUT = D
    thr_sq = DE._sq_thr()
    T, ar = real_thr(), aro_range()
    E = html.escape
    P = json.load(open(B / "plan_bon.json"))
    instr = (yaml.safe_load(open(DE.EMAP, encoding="utf-8")) or {}).get("instruct", {})
    first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))

    def judge(i):
        return btn(i + "|v", ["같은 사람", "애매", "다른 사람"]) + btn(i + "|e", ["감정 OK", "약함"])

    def num(v):
        if not v:
            return ""
        return (f"<div class='n num'>자기중립 {v['own']:.3f} · REF {v['ref']:.3f} · 목표1위 W{'✓' if v['w_top1'] else '✗'} E{'✓' if v['e_top1'] else '✗'}"
                f"<br>e2v {v['top']} · arousal {v['arousal']:.2f} · CER {100 * v['cer']:.0f}%" + (f" · ⚠️{'/'.join(v['odd'])}" if v["odd"] else "") + "</div>")
    badge = {"p20": "✅ ③ 통과", "p10": "🟡 ③ 10% 문턱만 통과(20% 문턱이면 탈락)", "drop": "❌ ③ 탈락 — 자기 중립과 너무 멂(진짜 녹음 하위 10% 아래)"}
    h = [f"""<!doctype html><meta charset=utf-8><title>vox_koft best-of-N · 문턱 · 노브</title>
<style>body{{font-family:sans-serif;max-width:1800px;margin:20px auto;line-height:1.45}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}}th{{background:#f3f3f3}}.n{{color:#666;font-size:.8em}}audio{{width:170px}}button{{margin:1px;font-size:.8em}}button.on{{background:#1565c0;color:#fff}}#bar{{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}}.num{{display:none}}body.shownum .num{{display:block}}details{{margin:6px 0}}td.pick{{background:#f4fbf4}}</style>
<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick="dl()">CSV 내려받기</button> · <label><input type=checkbox onchange="document.body.classList.toggle('shownum',this.checked)"> 숫자 보기(먼저 귀로!)</label> · <button onclick="if(confirm('판정을 모두 지울까요?')){{localStorage.removeItem(K);location.reload()}}">판정 지우기</button></div>
<h1>🎧 센 감정 칸 목소리 흔들림 — ① 여러 번 만들어 고르기 + ③ 문턱 · ② 추론 노브</h1>
<h2>청음 가이드</h2><ul>
<li>질문은 하나입니다: <b>감정 출력이 "그 화자의 생성 중립"과 같은 사람인가?</b> (D안 데이터는 생성 중립 = ref, 감정 = 정답으로 쓰기 때문)</li>
<li>입력은 D 재측정과 같습니다: <b>"(v1 영어 지시문)문장"</b> + 목표 화자 무감정 reference · cfg 2.0 · 준비 합성 뒤. 칸은 흔들림이 컸던 <b>분노A · 기쁨B · 분노B · 슬픔B</b>.</li>
<li>① 절 열: <b>목표 REF</b>(진짜 녹음) · <b>생성 중립</b>(같은 모델 · 같은 화자 · d_eval 중립 문장) · <b>seed 1986</b>(지금 노트북 · d_eval 과 같은 소리) · <b>① 고른 것</b>(seed 1986~1990 다섯 개 중 자동으로 고른 1개).
 <ul><li>고르기 = 발음(CER ≤ 10%) · 깨짐 없음 · 감정 있음(e2v 또는 arousal) 을 지킨 것 중 <b>생성 중립에 가장 가까운 것</b>. 고른 것이 seed 1986 이면 두 칸이 같은 소리입니다.</li>
 <li>③ 표시: ✅ 통과 · 🟡 느슨한 문턱(10%)만 통과 · ❌ 탈락. 문턱 = 진짜 녹음에서 "같은 사람의 감정 ↔ 중립" 거리의 하위 10% · 20%.</li></ul></li>
<li>판정 버튼(seed 1986 · ① 고른 것): <b>목소리</b> 같은 사람 · 애매 · 다른 사람 / <b>감정</b> OK · 약함. 이 브라우저에 저장되고 CSV 로 받습니다.</li>
<li>② 절: 같은 4칸을 <b>cfg 1.5 · 2.0 · 2.5</b> 와 <b>prompt 모드</b>(reference 녹음을 prompt 로도 줌)로. "목소리 ↑ 감정 ↓" 맞바꿈을 들어 봅니다.</li>
<li>숫자(자동 잣대)는 <b>참고용</b> — 맨 위 한눈에 표 · "숫자 보기" · 맨 아래 접힌 곳. 귀가 정본입니다.</li></ul>"""]
    # 한눈에
    h.append("<h2>📋 한눈에 (참고용)</h2><table><tr><th>모델</th><th>칸</th><th>자기중립 유사도<br>seed 1986 → ① 고른 것</th><th>e2v 감정<br>1986 → ①</th>"
             "<th>arousal<br>1986 → ①</th><th>①+③ 남는 비율<br>문턱 10% · 20%</th><th>(비교) 1986 하나 + ③</th></tr>")
    for model, o in out["bon"].items():
        for e in BO.CELLS + ["전체"]:
            r = o["rows"][e]
            s, c = r["s86"], r["chosen"]
            h.append(f"<tr><td>{model}</td><td>{KO.get(e, '<b>4칸 전체</b>')}</td><td>{s['own']:.3f} → <b>{c['own']:.3f}</b></td>"
                     f"<td>{s['hit']:.0f}% → {c['hit']:.0f}%{' ⚠️' if r['trap'] else ''}</td><td>{s['arousal']:.2f} → {c['arousal']:.2f}</td>"
                     f"<td><b>{r['keep10_pct']:.0f}% · {r['keep20_pct']:.0f}%</b></td><td>{r['s86_keep10']:.0f}% · {r['s86_keep20']:.0f}%</td></tr>")
    h.append("</table><p class=n>진짜 녹음 참고(같은 잣대): 감정 ↔ 그 사람 중립 평균 = " + " · ".join(f"{g} {v['mean']:.3f}" for g, v in T.items())
             + " · ⚠️ = 고른 것의 감정이 seed 1986 보다 약해짐(e2v −5%p 또는 arousal −0.02 이상) — '감정 약한 것만 골라지는 함정' 표시.</p>")
    # ① · ③
    for model, o in out["bon"].items():
        it = load_items("bon", model, model, thr_sq, ar)
        h.append(f"<h2>① · ③ — {model}</h2>")
        for s_, v in P["spk"].items():
            emo_of = {x["base_id"]: x["emo"] for x in P["items"] if x["tgt"] == s_}
            bids = sorted(emo_of, key=lambda b: (BO.CELLS.index(emo_of[b]), b))
            nd = sum(o["picks"][b]["lvl"] in ("drop", None) for b in bids)
            h.append(f"<details><summary><b>{E(s_)}</b> ({'여' if v['gender'] == 'female' else '남'} · {v['src']}) — {len(bids)}문장 · ③ 탈락/후보 없음 {nd}</summary>")
            h.append("<table><tr><th>칸</th><th>문장</th><th>목표 REF</th><th>생성 중립</th><th>seed 1986</th><th>① 고른 것</th></tr>")
            for b in bids:
                pk = o["picks"][b]
                s86 = it[f"{b}_s1986"]
                ch = it.get(pk["chosen"]) if pk["chosen"] else None
                e = s86["emo"]
                ref = A(f"../d_eval/ref/{s_}_ref.wav")
                neu = A(f"../d_eval/wav/{model}/{s_}_Neutrality_0.wav")
                c86 = f"{A(f'wav_bon/{model}/{b}_s1986.wav')}{num(s86)}{judge(f'{model}|{b}|s1986')}"
                if ch:
                    note = f"<div><b>seed {ch['seed']}</b>{' (= seed 1986)' if ch['seed'] == 1986 else ''} · {badge[pk['lvl']]}</div>"
                    cch = f"<td class=pick>{note}{A(f'wav_bon/{model}/{ch["id"]}.wav')}{num(ch)}{judge(f'{model}|{b}|pick')}</td>"
                else:
                    cch = "<td>❌ 후보 없음 — 다섯 개 모두 발음 · 깨짐 · 감정 조건 중 하나를 못 지킴</td>"
                h.append(f"<tr><td>{KO[e]}</td><td>{E(s86['text'])}<div class=n>({E(first(instr.get(e, '')))})</div></td>"
                         f"<td>{ref}</td><td>{neu}</td><td>{c86}</td>{cch}</tr>")
            h.append("</table></details>")
    # ②
    if out["knob"]:
        Pk = json.load(open(B / "plan_knob.json"))
        kn = {c: load_items("knob", c, BO.KNOB_MODEL, thr_sq, ar) for c in out["knob"]}
        h.append(f"<h2>② 추론 노브 — {BO.KNOB_MODEL} · 목표 4명 · seed 1986</h2>"
                 "<p class=n>cfg = 지시문(조건)을 얼마나 세게 따를지(기본 2.0). prompt = reference 녹음을 이어 말하기 prompt 로도 줌(전사문 포함) — 목소리를 더 세게 묶으려는 시험. "
                 "VoxCPM2 는 prompt 모드에서 글을 'prompt 전사문 + 문장' 으로 이어 붙여서, 요청대로 '(지시문)문장' 을 주면 지시문이 가운데 끼어 <b>영어 지시문을 소리 내어 읽습니다</b>(prompt 열). "
                 "그래서 지시문을 맨 앞('(지시문)prompt 전사문' + 문장)에 둔 변형(prompt_front 열)을 더했습니다.</p>")
        for s_ in BO.KNOB_SPK:
            v = Pk["spk"][s_]
            xs = sorted([x for x in Pk["items"] if x["tgt"] == s_], key=lambda x: (BO.CELLS.index(x["emo"]), x["id"]))
            h.append(f"<details><summary><b>{E(s_)}</b> ({'여' if v['gender'] == 'female' else '남'} · {v['src']})</summary>")
            h.append("<table><tr><th>칸</th><th>문장</th><th>목표 REF</th><th>생성 중립</th>" + "".join(f"<th>{KLAB.get(c, c)}</th>" for c in kn) + "</tr>")
            for x in xs:
                i = x["id"]
                cells = "".join(f"<td>{A(f'wav_knob/{c}/{i}.wav')}{num(kn[c][i])}{judge(f'knob_{c}|{i}|-')}</td>" for c in kn)
                h.append(f"<tr><td>{KO[x['emo']]}</td><td>{E(x['text'])}</td><td>{A(f'../d_eval/ref/{s_}_ref.wav')}</td>"
                         f"<td>{A(f'../d_eval/wav/{BO.KNOB_MODEL}/{s_}_Neutrality_0.wav')}</td>{cells}</tr>")
            h.append("</table></details>")
    # 맨 아래 숫자
    md = (B / "tables.md").read_text(encoding="utf-8")
    h.append("<details><summary><b>📊 자동 잣대 전체 표 — 참고용 (귀로 판정한 뒤 열기)</b></summary><pre style='white-space:pre-wrap;font-size:.8em'>"
             + E(md) + "</pre></details>")
    h.append("""<script>const K='vox_koft_bestofn';let S=JSON.parse(localStorage.getItem(K)||'{}');
function draw(){document.querySelectorAll('.b').forEach(d=>{d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v))});document.getElementById('cnt').textContent=Object.keys(S).length}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;S[id]=S[id]===b.dataset.v?undefined:b.dataset.v;if(!S[id])delete S[id];localStorage.setItem(K,JSON.stringify(S));draw()});
function dl(){const r=['model,id,column,axis,value'].concat(Object.entries(S).map(([k,v])=>k.split('|').join(',')+','+v));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+r.join('\\n')],{type:'text/csv'}));a.download='vox_koft_bestofn_judgments.csv';a.click()}
draw()</script>""")
    (B / "index.html").write_text("\n".join(h), encoding="utf-8")
    print("✅ page →", B / "index.html")
