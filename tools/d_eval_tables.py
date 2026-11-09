#!/usr/bin/env python
"""D 재측정 표 정리 — (a) run 별 best 5 · (b) 통합 참고 점수 상위 5 ckpt + 성공 기준(plan §1-2) ✅/❌ (2026-11-06).

    python tools/d_eval_tables.py            # → d_eval/tables.md · tables.json · page_top.html(페이지 맨 위 접힌 요약)
숫자는 모두 참고용 — 판정은 청음.
성공 기준(docs/plan/20261105_success_criteria_and_next_runs.md §1-2 의 숫자 부분 · 원본 대비)
  · 목표 1위 −10%p 이내(WavLM · ERes2Net 각각) · 목표 유사도 −0.03 이내(두 잣대 각각)
  · CER 이 원본보다 크게 나빠지지 않음 → 여기서는 "원본 + 1%p 이하"로 둠(가정)
  · e2v 감정 맞힘 +15%p 이상(감정 6칸 기준 — 원래 기준이 6칸이라서 · 7칸 값도 함께 표시)
  · arousal A<B 순서(기쁨 · 슬픔 · 분노 3쌍 중 몇 쌍)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
ROOT = Path(__file__).resolve().parents[1]
RUNS = [("FT_01", "FT01", "sft/A7N_ref_FT_01"), ("LoRA_01_r32", "LoRA01", "lora/A7N_ref_LoRA_01_r32"),
        ("FT_02_iv2", "FT02iv2", "sft/A7N_ref_FT_02_iv2"), ("LoRA_02_r32_iv2", "LoRA02iv2", "lora/A7N_ref_LoRA_02_r32_iv2"),
        ("A7NX_xref_LoRA_01", "A7NX", "lora/A7NX_xref_LoRA_01_r32"), ("A7NX3_xref_LoRA_01", "A7NX3", "lora/A7NX3_xref_LoRA_01_r32"),  # 2026-11-07
        ("A7NX2R_xref_LoRA_01", "X2R01", "lora/A7NX2R_xref_LoRA_01_r32"), ("A7NX2R_xref_LoRA_02_lr5e5", "X2R02lr5e5", "lora/A7NX2R_xref_LoRA_02_r32_lr5e5"),
        ("A7NX2Rs_xref_LoRA_01_lr5e5", "X2Rs01lr5e5", "lora/A7NX2Rs_xref_LoRA_01_r32_lr5e5"),
        ("A7NX2Rs6_xref_LoRA_01_lr5e5", "X2Rs6lr5e5", "lora/A7NX2Rs6_xref_LoRA_01_r32_lr5e5"),
        ("A7NUX2Rs6_xref_LoRA_01_lr5e5", "NUX2Rs6lr5e5", "lora/A7NUX2Rs6_xref_LoRA_01_r32_lr5e5"),  # 11/07 18:40
        ("A7NUX2s_xref_LoRA_01_lr5e5", "NUX2slr5e5", "lora/A7NUX2s_xref_LoRA_01_r32_lr5e5"),
        ("A7NUX2sN_xref_LoRA_01_lr5e5", "NUX2sNlr5e5", "lora/A7NUX2sN_xref_LoRA_01_r32_lr5e5"),
        ("A7NUX2sN_xref_LoRA_02_lr5e5_iv3", "NUX2sNiv3lr5e5", "lora/A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3"),   # 11/09 · v3 문구로 잼
        ("A7NUX2sNn_xref_LoRA_01_lr5e5", "NUX2sNnlr5e5", "lora/A7NUX2sNn_xref_LoRA_01_r32_lr5e5")]
CELLS = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KO = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}
OLD = D / "_backup_20261106_0955_before_neutral/summary.json"


def main():
    S = json.load(open(D / "summary.json"))["models"]
    old = json.load(open(OLD))["models"] if OLD.exists() else {}
    base = S["base"]["전체"]
    md, out = [], {"best5": {}, "top5": []}
    # (a) best 5
    md += ["### (a) run 별 best 5 (참고 점수 = e2v 감정(7칸) + 목표 1위(두 잣대 평균) − CER − 이상 비율)", "",
           "| 노트북 | best 5 |", "|---|---|"]
    for name, short, rel in RUNS:
        ev = {int(t.split("_s")[1]): m["전체"]["score"] for t, m in S.items() if t.startswith(short + "_s")}
        ev_old = {int(t.split("_s")[1]): m["전체"]["score"] for t, m in old.items() if t.startswith(short + "_s")}
        order = sorted(ev, key=lambda s: -ev[s])
        order_old = sorted(ev_old, key=lambda s: -ev_old[s])
        if not ev:                                                  # 아직 평가 전 run(예: 학습 중)은 건너뜀
            continue
        an = ROOT / "exp_reports" / rel / "run_backup/analysis.json"
        tb = json.load(open(an))["best"]["val_unseen/loss"]["top5"] if an.exists() else []
        pick = [(s, True) for s in order[:5]] + [(s, False) for s in tb if s not in order][: max(0, 5 - len(order))]
        cell = " · ".join(f"{s}" if ok else f"{s}*" for s, ok in pick)
        md.append(f"| {name} | {cell} |")
        out["best5"][name] = dict(pick=[s for s, _ in pick], evaluated=order, scores={s: round(ev[s], 1) for s in order},
                                  order_6cells=order_old, changed=order[:len(order_old)] != order_old, tb_top5=tb)
    md += ["", "`*` = 평가하지 않은 step(TB 처음 보는 화자 val loss 상위로 채움). run 마다 평가한 step 은 4~5개뿐이라 5칸 중 1칸 정도가 `*` 입니다.", ""]
    ch = [n for n, v in out["best5"].items() if v["changed"]]
    md.append("중립 칸을 넣은 뒤 평가 step 의 순위가 바뀐 run: " + (", ".join(
        f"{n}(전 {' · '.join(map(str, out['best5'][n]['order_6cells']))} → 지금 {' · '.join(map(str, out['best5'][n]['evaluated']))})" for n in ch) if ch else "없음"))
    md.append("")
    # (b) 통합 상위 5
    cand = sorted([t for t in S if t != "base"], key=lambda t: -S[t]["전체"]["score"])[:5]
    md += ["### (b) 통합 참고 점수 상위 5 ckpt", ""]
    md += ["**목소리 · 발음 · 이상**", "", "| ckpt | 점수 | 1위 W | 1위 E | 유사도 W | 유사도 E | CER | 이상 |", "|---|---|---|---|---|---|---|---|"]
    rows = ["base"] + cand
    for t in rows:
        m = S[t]["전체"]
        md.append(f"| {t} | {m['score']:.0f} | {m['top1']:.0f}% | {m['e_top1']:.0f}% | {m['t_sim']:.3f} | {m['e_t_sim']:.3f} | {m['cer']:.1f}% | {m['odd']:.0f}% |")
    md += ["", "**감정 — e2v 맞힘(칸별)**", "", "| ckpt | 6칸 | 7칸 | " + " | ".join(KO[c] for c in CELLS) + " |", "|---|---|---|" + "---|" * len(CELLS)]
    for t in rows:
        m, k = S[t]["전체"], S[t]["칸별"]
        md.append(f"| {t} | {m['emo_hit6']:.0f}% | {m['emo_hit']:.0f}% | " + " | ".join(f"{k[c]['emo_hit']:.0f}" for c in CELLS) + " |")
    md += ["", "**감정 — arousal(칸별)**", "", "| ckpt | " + " | ".join(KO[c] for c in CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for t in rows:
        k = S[t]["칸별"]
        md.append(f"| {t} | " + " | ".join(f"{k[c]['arousal']:.2f}" for c in CELLS) + " |")
    # 성공 기준
    md += ["", "**성공 기준(§1-2 숫자 부분 · 원본 대비) — 참고용**", "",
           "| ckpt | 1위 −10%p | 유사도 −0.03 | CER | e2v +15%p | A<B |", "|---|---|---|---|---|---|"]
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    for t in cand:
        m, k = S[t]["전체"], S[t]["칸별"]
        c1 = m["top1"] >= base["top1"] - 10 and m["e_top1"] >= base["e_top1"] - 10
        c2 = m["t_sim"] >= base["t_sim"] - 0.03 and m["e_t_sim"] >= base["e_t_sim"] - 0.03
        c3 = m["cer"] <= base["cer"] + 1.0
        d6 = m["emo_hit6"] - S["base"]["전체"]["emo_hit6"]
        c4 = d6 >= 15
        ab = sum(k[f"{e}-A"]["arousal"] < k[f"{e}-B"]["arousal"] for e in ("Happy", "Sad", "Angry"))
        md.append(f"| {t} | {ok(c1)} | {ok(c2)} ({m['t_sim'] - base['t_sim']:+.3f}) | {ok(c3)} | {ok(c4)} ({d6:+.0f}%p) | {ab}/3 |")
        out["top5"].append(dict(tag=t, score=m["score"], ok_voice_top1=c1, ok_voice_sim=c2, ok_cer=c3, ok_e2v=c4, e2v_delta6=d6, ab_pairs=ab,
                                **{k_: m[k_] for k_ in ("top1", "e_top1", "t_sim", "e_t_sim", "cer", "odd", "emo_hit", "emo_hit6")},
                                cells={c: dict(e2v=k[c]["emo_hit"], arousal=k[c]["arousal"]) for c in CELLS}))
    bk = S["base"]["칸별"]
    md += ["", f"원본 기준선: 목표 1위 W {base['top1']:.0f}% · E {base['e_top1']:.0f}% · 유사도 W {base['t_sim']:.3f} · E {base['e_t_sim']:.3f} · "
           f"CER {base['cer']:.1f}% · e2v 6칸 {S['base']['전체']['emo_hit6']:.0f}% · 7칸 {base['emo_hit']:.0f}% · "
           f"A<B {sum(bk[f'{e}-A']['arousal'] < bk[f'{e}-B']['arousal'] for e in ('Happy', 'Sad', 'Angry'))}/3",
           "CER 기준 \"크게 나빠지지 않음\"은 원본 + 1%p 이하로 두었습니다(가정). 감정 기준은 원래 6칸으로 정했으므로 6칸 값으로 판정합니다."]
    vr = D / "vspk/vspk_rank.csv"                                              # (c) 가상 화자 일관성(tools/vspk.py report)
    if vr.exists():
        import csv
        rk = list(csv.DictReader(open(vr)))
        md += ["", "### (c) 가상 화자 일관성 순위(두 잣대 평균 · 우연 12.5%)", "",
               "| 모델 | 세트 1위 | 감정끼리 | 평균 | 세트↔REF |", "|---|---|---|---|---|"]
        for r in rk[:8] + [r for r in rk[8:] if r["model"] == "base"]:
            md.append(f"| {r['model']} | {float(r['a_top1']):.0f}% | {float(r['b_top1']):.0f}% | {float(r['consist']):.0f}% | {float(r['set_ref']):.3f} |")
        md.append("세트 1위 = 감정 출력이 자기 화자의 생성 중립 평균에 가장 가까운 비율 · 감정끼리 = 같은 묶음 다른 출력에 가장 가까운 비율(공정한 잣대) · +시범중립20 = 생성 중립 22개로 잰 값.")
    (D / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(out, open(D / "tables.json", "w"), ensure_ascii=False, indent=1, default=float)
    # 페이지 맨 위(접힘) — md 표를 간단히 HTML 로
    hh = ["<details><summary><b>📋 한눈에 — best 5 · 통합 상위 5 · 성공 기준 (숫자는 참고용 · 귀로 판정한 뒤 여세요)</b></summary>"]
    tbl = []
    for line in md:
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not tbl else "td"
            tbl.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if tbl:
                hh.append("<table style='width:auto'>" + "".join(tbl) + "</table>")
                tbl = []
            if line.startswith("### "):
                hh.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("**"):
                hh.append(f"<p><b>{html.escape(line.strip('*'))}</b></p>")
            elif line.strip():
                hh.append(f"<p class=n>{html.escape(line).replace('`', '')}</p>")
    if tbl:
        hh.append("<table style='width:auto'>" + "".join(tbl) + "</table>")
    hh.append("</details>")
    (D / "page_top.html").write_text("\n".join(hh), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
