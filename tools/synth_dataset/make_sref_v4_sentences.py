"""SREF v4 새 문장 만들기(2026-11-07 · q3tts_koft 세션 · Claude 작성 · 템플릿 조합 · seed 1986)
왜(근거 = q3tts docs/plan/20261106_SREF_v4_notes.md)
  · 이사님 메일 "2번 예제 '~라고' 오류" → 영어 뒤에 붙는 **인용 · 조사로 시작하는 조각** 보강(ko_quote)
  · 사용자 청음: 숫자 · 날짜가 든 영어가 "애매"로 많이 나옴 → **영어 숫자 · 날짜 · 시각**(en_only · en_type=number)
  · v3 에서 떨어진 영어만 조각 444 · 코드스위치/영어 의문문 일부 200 → 후보 풀(여러 seed · SREF2 태그)로 다시 시도(retry)
출력: ~/workspaces/preprocessed/q3tts_koft/SREF/sentences/v4/{ko_quote,en_num,retry}.jsonl
~/miniforge3/envs/q3tts/bin/python make_sref_v4_sentences.py
"""
import json, random
from pathlib import Path
H = Path.home() / "workspaces"; S = H / "preprocessed/q3tts_koft/SREF"; OUT = S / "sentences/v4"
SEED = 1986
USER_DROP = {"v3eo_0247", "v3eo_0580", "v3eo_0206", "v3eo_0029"}     # 사용자 청음 '빼야 함'(11/06) — 다시 시도하지 않음
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ── 1) 인용 · 조사로 시작하는 조각(서빙이 영어 덩어리를 떼어 내면 남는 앞머리) ──
HEAD = ["라고", "이라고", "하고", "는", "은", "를", "을", "이에요.", "예요.", "라는", "이라는", "가", "이", "도", "처럼"]
TAIL = {
    "라고": ["말해 봐요.", "대답해 볼까요?", "인사해요.", "물어보면 돼요.", "쓰여 있어요.", "따라 해 보세요.", "부르기도 해요.", "적어 두세요."],
    "이라고": ["말해 봐요.", "읽으면 돼요.", "대답했어요.", "불러요.", "해요.", "적혀 있네요.", "말하는 거예요."],
    "하고": ["대답하면 돼요.", "말하면서 손을 흔들어요.", "인사한 다음 자리에 앉아요.", "따라 하세요."],
    "는": ["영어로 뭐라고 할까요?", "이렇게 읽어요.", "아주 쉬운 표현이에요.", "인사할 때 써요.", "다 같이 읽어 볼까요?"],
    "은": ["무슨 뜻일까요?", "이렇게 발음해요.", "자주 쓰는 말이에요.", "꼭 기억해 두세요."],
    "를": ["다시 한번 읽어 볼까요?", "크게 따라 해 보세요.", "천천히 말해 봐요.", "공책에 써 보세요.", "기억해 두세요."],
    "을": ["세 번 따라 해 봐요.", "큰 소리로 읽어요.", "짝꿍에게 말해 보세요.", "잘 들어 보세요."],
    "이에요.": ["따라 해 보세요.", "잘 들어 보세요.", "다 같이 말해 봐요.", "정말 잘했어요!", "한 번 더 해 볼까요?"],
    "예요.": ["잘 들어 보세요.", "따라 읽어 봐요.", "어렵지 않죠?", "같이 해 봐요.", "기억나요?"],
    "라는": ["말을 배워 볼까요?", "단어를 써 봐요.", "표현은 인사예요.", "문장을 읽어 봐요."],
    "이라는": ["단어가 나왔어요.", "말은 무슨 뜻일까요?", "표현을 배워요."],
    "가": ["무슨 뜻인지 알아요?", "어디에 나왔나요?", "정답이에요."],
    "이": ["바로 오늘의 단어예요.", "들리나요?", "정답이에요!"],
    "도": ["같이 배워 볼까요?", "따라 해 봐요.", "알아 두면 좋아요."],
    "처럼": ["말해 보세요.", "또박또박 읽어 봐요.", "자신 있게 말해요."],
}
EXTRA = ["", "", "다 같이 해 봐요.", "잘했어요!", "정말 멋져요.", "천천히 해도 괜찮아요.", "한 번 더 해 볼까요?", "선생님이 먼저 해 볼게요.",
         "손을 들고 말해 보세요.", "틀려도 괜찮아요.", "목소리가 아주 좋네요.", "이제 짝꿍 차례예요."]
EN_SHORT = ["Hi", "Hello", "Yes", "No", "OK", "Thank you", "Sorry", "Good job", "Bye", "Wow", "Apple", "Cat", "Dog", "Book", "Pen",
            "Red", "Blue", "Happy", "Ready", "Sit down", "Stand up", "Look", "Listen", "Excuse me", "Good morning", "Good night", "See you",
            "Nice to meet you", "How are you?", "What's this?", "I'm fine", "Me too", "Let's go", "Well done", "Great"]
EN_TAIL = {"라고": ["말해 봐요.", "대답해요.", "인사해요.", "따라 해 볼까요?"], "는": ["무슨 뜻일까요?", "이렇게 써요."],
           "를": ["세 번 말해 봐요.", "따라 해 보세요."], "예요.": ["같이 읽어 봐요."], "이에요.": ["같이 읽어 봐요."]}


def josa(word, head):
    """영어 뒤 조사(받침 대신 읽기 기준 — 마지막 글자가 자음 소리면 이라고 · 이에요 · 은 · 을)."""
    vowel_end = word.rstrip("?!. ").lower()[-1:] in "aeiouy" or word.rstrip("?!. ").lower().endswith(("e", "o"))
    pairs = {"라고": ("라고", "이라고"), "는": ("는", "은"), "를": ("를", "을"), "예요.": ("예요.", "이에요."), "이에요.": ("예요.", "이에요.")}
    a, b = pairs[head]
    return a if vowel_end else b


def ko_quote(rng, seen):
    out = []
    combos = [(h, t, x) for h in HEAD for t in TAIL[h] for x in EXTRA]
    rng.shuffle(combos)
    for h, t, x in combos:
        s = f"{h} {t} {x}".strip()
        if s in seen:
            continue
        seen.add(s); out.append(dict(text=s, en_type="head_particle", en=[]))
        if len(out) >= 400:
            break
    combos = [(w, h, t) for w in EN_SHORT for h in EN_TAIL for t in EN_TAIL[h]]
    rng.shuffle(combos); n = 0
    for w, h, t in combos:
        s = f"\"{w}\"{josa(w, h)} {t}"
        if s in seen:
            continue
        seen.add(s); out.append(dict(text=s, en_type="en_particle", en=[w])); n += 1
        if n >= 200:
            break
    return [dict(id=f"v4kq_{i:04d}", kind="ko_quote", **r) for i, r in enumerate(out)]


# ── 2) 영어 숫자 · 날짜 · 시각(영어만 조각) ──
NOUN = ["apples", "pencils", "books", "cats", "dogs", "students", "chairs", "cookies", "balloons", "birds", "stickers", "crayons"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
       "fifteenth", "twentieth", "twenty-first", "twenty-fifth", "thirtieth"]
PERSON = ["My sister", "My brother", "My cousin", "My friend", "Our teacher", "The baby", "My grandma", "Tom", "Emma", "Jake"]


def en_num(rng, seen):
    T = []
    for n in range(2, 21):
        for nn in NOUN:
            T += [f"I have {n} {nn}.", f"There are {n} {nn} on the desk.", f"Can you count {n} {nn}?", f"We need {n} more {nn}."]
    for p in PERSON:
        for a in range(3, 13):
            T += [f"{p} is {a} years old.", f"Is {p.lower() if p.startswith(('My', 'Our', 'The')) else p} {a} years old?"]
    for m in MONTH:
        for o in ORD:
            T += [f"Today is {m} {o}.", f"My birthday is on {m} {o}.", f"Is your birthday on {m} {o}?"]
    for h in range(1, 13):
        T += [f"It's {h} o'clock.", f"School starts at {h} o'clock.", f"Let's meet at {h} o'clock.", f"Is it {h} o'clock now?"]
    for n in (5, 10, 15, 20, 30, 50, 100):
        T += [f"It costs {n} dollars.", f"Open your book to page {n}.", f"Count from one to {n}.", f"How much is {n} plus {n}?"]
    T = sorted(set(T)); rng.shuffle(T); out = []
    for t in T:
        if t in seen:
            continue
        seen.add(t); out.append(t)
        if len(out) >= 300:
            break
    return [dict(id=f"v4en_{i:04d}", text=t, kind="en_only", en_type="number", en=[t]) for i, t in enumerate(out)]


# ── 3) v3 에서 떨어진 것 다시 시도 ──
def retry(rng):
    sent = {s["id"]: s for s in jl(S / "v3/new/sentences.jsonl")}
    resc = json.load(open(S / "v3/new/en_rescore.json"))["items"]
    en_fail = sorted(i for i, r in resc.items() if not r["ok"] and i not in USER_DROP)
    rej = [r for r in jl(S / "v3/new/rejected.jsonl") if r["kind"] in ("codeswitch", "en_q_embed")]
    cs = sorted(r["utt"] for r in rej); rng.shuffle(cs); cs = sorted(cs[:200])
    out = []
    for i in en_fail + cs:
        s = sent[i]
        out.append(dict(id=f"v4r_{i}", text=s["raw"], kind=s["kind"], en_type=s.get("en_type"), en=s.get("en", []), retry_of=i))
    return out


def main():
    rng = random.Random(SEED)
    seen = set()
    for p in (S / "sentences/v3").glob("*.jsonl"):
        seen |= {r["text"] for r in jl(p)}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ko_quote", ko_quote(rng, seen)), ("en_num", en_num(rng, seen)), ("retry", retry(rng))):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), rows[0]["text"], "|", rows[-1]["text"])


if __name__ == "__main__":
    main()
