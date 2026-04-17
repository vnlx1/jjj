from __future__ import annotations

import asyncio
import os
import random
import re
import statistics
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="BTEC Stealth Humanizer API", version="2.0")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

Mode = Literal["subtle", "balanced", "aggressive", "academic", "stealth"]

# Free, no-key LLM endpoint. See https://pollinations.ai — no auth required.
POLLINATIONS_URL = os.getenv(
    "POLLINATIONS_URL", "https://text.pollinations.ai/openai"
)
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai")
POLLINATIONS_TIMEOUT = float(os.getenv("POLLINATIONS_TIMEOUT", "90"))


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    mode: Mode = "balanced"
    intensity: int = Field(65, ge=0, le=100)
    passes: int | None = None
    target_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Optional: if set, the server re-runs humanization up to 2 extra "
            "times if the heuristic AI-likeness score exceeds this threshold."
        ),
    )


class PassTrace(BaseModel):
    stage: str
    chars: int
    ai_score: float
    burstiness: float


class HumanizeResponse(BaseModel):
    humanized: str
    original_chars: int
    humanized_chars: int
    passes: int
    model: str
    ai_score: float = Field(
        ...,
        description="Heuristic AI-likeness score (0 = human, 100 = very AI-like).",
    )
    burstiness: float = Field(
        ..., description="Sentence-length standard deviation, higher is more human."
    )
    trace: list[PassTrace] = []


# ---------------------------------------------------------------------------
# Three-pass humanization pipeline. Each pass narrows in on one axis that AI
# detectors (GPTZero, Originality.ai, ZeroGPT, Turnitin, Copyleaks) use to
# flag machine text. Chaining focused prompts empirically beats a single
# monolithic prompt because each pass gets the model's full attention on one
# objective.
# ---------------------------------------------------------------------------

_COMMON_HEADER = (
    "You are rewriting text so it reads like natural human writing. "
    "Output ONLY the rewritten text. "
    "No preamble. No quotes around the output. No explanation. No labels. "
    "No markdown unless the original used markdown. "
    "Preserve all facts, numbers, names, and the overall meaning. "
    "Keep the length within plus-or-minus 25% of the original.\n"
    "\n"
    "HARD BANS (never do these, they are AI tells):\n"
    "- No similes of the shape 'like X after Y' or 'like X-ing Y'. "
    "Examples to avoid: 'like spotting a rainbow after a storm', "
    "'like picking the right spaghetti', 'like chewing bubblegum', "
    "'like a well-oiled machine'. If you want a comparison, make it "
    "specific and domain-relevant, not folksy.\n"
    "- No 'game-changer', 'real gains', 'smart move', 'bottom line', "
    "'at the end of the day', 'in this fast-paced world'.\n"
    "- No em-dashes. No triple-parallel lists unless the source had one.\n"
    "- No hollow intensifiers: truly, vitally, significantly, undoubtedly, "
    "profoundly, meticulously.\n"
    "- No hedges stacked at sentence starts (e.g. 'Honestly, I think that, to be fair, ...').\n"
    "\n"
)

_PASS1_FLOW = (
    "Pass 1 of 3: NATURAL FLOW.\n"
    "Rewrite so it sounds like a thoughtful person typing a draft, not a "
    "finished essay. Apply:\n"
    "- Contractions where natural: it's, don't, they're, can't, there's, won't, I'm.\n"
    "- Replace textbook transitions (Furthermore, Moreover, Additionally, "
    "Consequently, Therefore, In conclusion) with ordinary ones (Also, Plus, "
    "So, Still, Anyway, That said, One more thing).\n"
    "- Swap AI tell-words for ordinary ones: delve -> dig into, utilize -> "
    "use, comprehensive -> thorough, crucial -> key, leverage -> use, "
    "robust -> solid, realm -> area, tapestry -> mix, plethora -> a lot of, "
    "pivotal -> major, ascertain -> figure out, facilitate -> help, "
    "endeavor -> effort, paradigm -> model, elucidate -> explain.\n"
    "- Prefer concrete, specific nouns and verbs over abstract ones.\n"
    "- Do not add new metaphors. Do not add similes. If the source had none, "
    "output should have none.\n"
)

_PASS2_BURSTINESS = (
    "Pass 2 of 3: BURSTINESS + PERPLEXITY.\n"
    "Keep the meaning identical. Rewrite so sentence lengths vary sharply. "
    "This is the single most important pass. Follow ALL of these:\n"
    "- At least ONE sentence must be 3 to 6 words long. Short, blunt.\n"
    "- At least ONE sentence must be 22+ words long with a mid-sentence clause.\n"
    "- No two consecutive sentences may have lengths within 3 words of each other.\n"
    "- Break parallel structure. Three parallel clauses -> reshape two so the "
    "grammatical forms differ.\n"
    "- Inject a few unexpected but accurate word choices. Pick the correct "
    "but less obvious word once or twice (perplexity).\n"
    "- Optional: up to ONE personal hedge per paragraph (not per sentence) "
    "from: honestly, to be fair, arguably, more or less, which is fine. "
    "Do not stack hedges.\n"
    "- You MAY start up to two sentences with And, But, So, or Still.\n"
    "\n"
    "Example of the burstiness pattern you want (different topic, shown "
    "only for rhythm):\n"
    "'The plan works. On paper it looks solid, and in the first dry run the "
    "numbers held up better than anyone had predicted. But nothing ever "
    "survives first contact with a real user. So we iterated.'\n"
    "Notice: 3 words, then 27 words, then 10 words, then 2 words. Copy that "
    "kind of uneven rhythm.\n"
)

_PASS3_CLEANUP = (
    "Pass 3 of 3: STRIP AI TELLS.\n"
    "Keep the sentence-length rhythm from the previous pass. Scrub any "
    "remaining machine-sounding patterns.\n"
    "- Delete any of: 'In today's rapidly evolving', 'It is important to "
    "note', 'As we navigate', 'In conclusion', 'In summary', 'delve into', "
    "'a testament to', 'the landscape of', 'in the realm of', 'plays a "
    "crucial role', 'stands as a', 'bottom line', 'at the end of the day', "
    "'game-changer', 'real gains'.\n"
    "- Remove any simile of the shape 'like X after Y' or 'like X-ing Y'. "
    "Replace with a plain statement.\n"
    "- No three-item parallel lists unless the original had one.\n"
    "- Delete hollow intensifiers: truly, vitally, significantly, "
    "undoubtedly, profoundly, meticulously.\n"
    "- Prefer active voice. Drop passive unless it is naturally better.\n"
    "- Fix any em-dashes or en-dashes. Use commas or periods.\n"
    "- Do NOT add hedges or colloquialisms that were not already in the "
    "text. This is a clean-up pass, not a voice-injection pass.\n"
    "- Output must still read fluently. Do not introduce grammar errors.\n"
)

_MODE_FLAVOR: dict[Mode, str] = {
    "subtle": (
        "Tone: subtle. Keep the original's register, just sand off AI edges. "
        "Light contractions, minor rewording, no slang."
    ),
    "balanced": (
        "Tone: balanced. Conversational but still competent. Clear "
        "burstiness, no slang, no forced personality."
    ),
    "aggressive": (
        "Tone: aggressive. Direct and blunt. Short punchy sentences next to "
        "rambly ones. Plain vocabulary, no jargon."
    ),
    "academic": (
        "Tone: academic human. Grad student draft, not textbook. Keep domain "
        "terms, add at most one hedge per paragraph (arguably, roughly, "
        "which is to say), break rhythm with a short sentence."
    ),
    "stealth": (
        "Tone: maximum stealth. Push burstiness hard. Break every parallel "
        "structure. Pick less-obvious but accurate word choices. Feel like "
        "a focused first draft by a competent human who edits as they go. "
        "Do not reach for metaphors or similes."
    ),
}


def _intensity_hint(intensity: int) -> str:
    if intensity < 25:
        rewrite = "Barely touch the wording."
    elif intensity < 50:
        rewrite = "Light rewrite, keep most phrasing."
    elif intensity < 75:
        rewrite = "Moderate rewrite, reword most sentences."
    else:
        rewrite = "Heavy rewrite, rework sentence structure and vocabulary."
    return f"Intensity: {intensity}/100. {rewrite}"


def build_pass_prompt(text: str, stage: str, mode: Mode, intensity: int) -> str:
    stage_block = {
        "flow": _PASS1_FLOW,
        "burstiness": _PASS2_BURSTINESS,
        "cleanup": _PASS3_CLEANUP,
    }[stage]
    return (
        _COMMON_HEADER
        + stage_block
        + "\n"
        + _MODE_FLAVOR[mode]
        + "\n"
        + _intensity_hint(intensity)
        + "\n\nText to rewrite:\n---\n"
        + text
        + "\n---\n"
    )


# ---------------------------------------------------------------------------
# Heuristic AI-likeness scorer. Fast local approximation of what detectors
# measure — not a replacement, but a useful feedback signal for the retry
# loop. Combines burstiness (sentence-length stddev), AI-tell word rate,
# AI-phrase hits, and sentence-start repetition. Score is 0..100 (higher is
# more AI-like).
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")

_AI_TELL_WORDS = {
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "delve", "utilize", "utilizes", "utilized", "comprehensive", "crucial",
    "leverage", "leverages", "leveraged", "robust", "realm", "tapestry",
    "plethora", "ascertain", "facilitate", "navigate", "endeavor", "paradigm",
    "elucidate", "pivotal", "underscore", "underscores", "emphasize",
    "emphasizes", "exemplifies", "encapsulate", "encapsulates",
    "multifaceted", "meticulous", "meticulously", "profound", "profoundly",
    "intricate", "intricacies", "nuanced", "paramount", "quintessential",
    "ubiquitous", "disseminate", "juxtaposition", "dichotomy",
}

_AI_TELL_PHRASES = [
    re.compile(p, re.I)
    for p in (
        r"\bin today'?s rapidly evolving\b",
        r"\bit is important to note\b",
        r"\bas we navigate\b",
        r"\bin conclusion\b",
        r"\bin summary\b",
        r"\ba testament to\b",
        r"\bthe landscape of\b",
        r"\bin the realm of\b",
        r"\bplays a (?:crucial|pivotal|significant) role\b",
        r"\bstands as a\b",
    )
]


def _sentence_lengths(text: str) -> list[int]:
    sents = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return [len(_WORD_RE.findall(s)) for s in sents]


def burstiness(text: str) -> float:
    lens = _sentence_lengths(text)
    if len(lens) < 2:
        return 0.0
    return round(statistics.pstdev(lens), 2)


def ai_score(text: str) -> float:
    """0..100 heuristic. Higher = more AI-like."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    tell_rate = sum(1 for w in words if w in _AI_TELL_WORDS) / max(len(words), 1)
    phrase_hits = sum(1 for p in _AI_TELL_PHRASES if p.search(text))

    lens = _sentence_lengths(text)
    if len(lens) >= 2:
        mean = statistics.mean(lens)
        stddev = statistics.pstdev(lens)
        cv = stddev / mean if mean else 0.0
    else:
        cv = 0.0

    starts = [s.strip().split()[0].lower() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    start_diversity = len(set(starts)) / max(len(starts), 1) if starts else 1.0

    score = 0.0
    score += min(tell_rate * 1500, 40)              # up to 40 pts from AI tells
    score += min(phrase_hits * 10, 20)              # up to 20 pts from AI phrases
    score += max(0, (0.55 - cv) * 80)               # low burstiness -> up to ~44 pts
    score += max(0, (1 - start_diversity) * 20)     # repetitive starts -> up to 20 pts

    if len(lens) < 3:
        score *= 0.7

    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# Deterministic post-processing to catch tells the LLM left behind.
# ---------------------------------------------------------------------------

_AI_CLICHE_SUBS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bfurthermore\b", re.I), ["also", "plus", "on top of that", "and"]),
    (re.compile(r"\bmoreover\b", re.I), ["also", "plus", "besides", "and"]),
    (re.compile(r"\badditionally\b", re.I), ["also", "plus", "on top of that"]),
    (re.compile(r"\bdelve\s+into\b", re.I), ["dig into", "get into", "look at"]),
    (re.compile(r"\butilize\b", re.I), ["use", "work with", "rely on"]),
    (re.compile(r"\bcomprehensive\b", re.I), ["thorough", "complete", "full"]),
    (re.compile(r"\bcrucial\b", re.I), ["key", "really important", "vital"]),
    (re.compile(r"\bpivotal\b", re.I), ["key", "major", "central"]),
    (re.compile(r"\bleverage\b", re.I), ["use", "tap into", "lean on"]),
    (
        re.compile(r"\bin\s+today'?s\s+rapidly\s+evolving\b", re.I),
        ["these days", "right now", "lately"],
    ),
    (
        re.compile(r"\bit\s+is\s+important\s+to\s+note\s+that\b", re.I),
        ["worth noting,", "one thing:", "note that"],
    ),
    (re.compile(r"\bin\s+conclusion\b", re.I), ["so", "all told,", "so really,"]),
    (re.compile(r"\bin\s+summary\b", re.I), ["so basically,", "to sum up,", "put simply,"]),
    (re.compile(r"\bas\s+we\s+navigate\b", re.I), ["as we work through", "going through"]),
    (re.compile(r"\brealm\b", re.I), ["area", "world", "field"]),
    (re.compile(r"\btapestry\b", re.I), ["mix", "blend", "web"]),
    (re.compile(r"\bplethora\b", re.I), ["a lot of", "loads of", "plenty of"]),
    (re.compile(r"\bfacilitate\b", re.I), ["help", "make easier", "enable"]),
    (re.compile(r"\bascertain\b", re.I), ["figure out", "find out", "work out"]),
    (re.compile(r"\brobust\b", re.I), ["solid", "strong", "sturdy"]),
    (re.compile(r"\bmeticulous(?:ly)?\b", re.I), ["careful", "close", "tight"]),
    (re.compile(r"\bprofound(?:ly)?\b", re.I), ["deep", "serious", "real"]),
    (re.compile(r"\bmultifaceted\b", re.I), ["layered", "wide-ranging"]),
    (re.compile(r"\bunderscore(?:s|d)?\b", re.I), ["show", "highlight", "make clear"]),
    (re.compile(r"\bgame[-\s]?changer\b", re.I), ["big shift", "turning point"]),
    (re.compile(r"\bat\s+the\s+end\s+of\s+the\s+day\b", re.I), ["in the end", "ultimately"]),
    (re.compile(r"\bbottom\s+line\b", re.I), ["point is", "net-net,"]),
    (re.compile(r"\bin\s+this\s+(?:fast[-\s]?paced|modern)\s+world\b", re.I), ["these days", "right now"]),
]

# Kill the hokey "like X after Y" simile template outright — both GPT and the
# humanizer model reach for it when asked for "personality" and it is a
# strong AI tell.
_SIMILE_KILLER = re.compile(
    r",?\s*(?:almost\s+|just\s+)?(?:like|as|as\s+if)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"\w+(?:ing\s+|\s+)"
    r"[^.,;!?]*?\b(?:after|through|before)\s+[^.,;!?]*",
    re.I,
)
_FOLKSY_SIMILE = re.compile(
    r",?\s*(?:almost\s+|just\s+)?(?:like|as\s+if)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(?:well-oiled machine|rainbow|spaghetti|bubblegum|pancake|clockwork|"
    r"ninja|rockstar|wizard|magic|champ|hero|boss|breeze)[^.,;!?]*",
    re.I,
)

_EM_DASH = re.compile(r"\s*[\u2014\u2013]\s*")
_QUOTED_WRAP = re.compile(r'^"(.+)"$', re.S)
_LEADING_HERE = re.compile(
    r"^\s*(here(?:'s| is) (?:the|your|a) (?:rewritten|rewrite|humanized|revised)[^\n:]*:?\s*)",
    re.I,
)
_SENT_FINAL_PUNCT = re.compile(r"([.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter that preserves the trailing punctuation on each sentence."""
    parts: list[str] = []
    buf = text.strip()
    if not buf:
        return parts
    # Split on ".!?" followed by whitespace; keep the delimiter attached.
    pieces = re.split(r"(?<=[.!?])\s+", buf)
    return [p.strip() for p in pieces if p.strip()]


_SPLIT_CONJUNCTIONS = (", and ", ", but ", ", or ", ", so ", ", yet ", " and ", " but ", " or ", " so ", " yet ")


def _find_split_point(sentence: str, min_word: int = 7) -> int | None:
    """Return a char index inside `sentence` to split at (after word ``min_word``)."""
    words = sentence.split(" ")
    if len(words) < min_word + 3:
        return None
    # Prefer commas first.
    char_pos = 0
    for i, w in enumerate(words):
        char_pos += len(w) + 1  # +1 for the trailing space
        if i + 1 >= min_word and w.endswith(","):
            return char_pos  # split right after the comma
    # Fall back to conjunctions.
    lowered = sentence.lower()
    for conj in _SPLIT_CONJUNCTIONS:
        idx = lowered.find(conj, _char_index_after_word(sentence, min_word))
        if idx != -1:
            # Split at the start of the conjunction.
            return idx + 1 if conj.startswith(", ") else idx + 1
    return None


def _char_index_after_word(sentence: str, n: int) -> int:
    words = sentence.split(" ")
    if len(words) < n:
        return len(sentence)
    return sum(len(w) + 1 for w in words[:n])


def _enforce_burstiness(text: str) -> str:
    """Mechanically split sentences until stddev >= 4 or no safe split remains.

    Detectors flag uniform sentence lengths as AI-written. This post-processing
    pass guarantees at least one short sentence exists and breaks overly
    uniform rhythm. It splits the longest sentence at a comma (preferred) or
    a coordinating conjunction that lands past word 7.
    """
    sents = _split_sentences(text)
    if len(sents) < 3:
        return text

    for _ in range(3):  # up to 3 splits
        word_counts = [len(_WORD_RE.findall(s)) for s in sents]
        if statistics.pstdev(word_counts) >= 4 and min(word_counts) <= 7:
            break
        # Pick the longest sentence to split.
        idx_long = max(range(len(sents)), key=lambda i: word_counts[i])
        if word_counts[idx_long] < 10:
            break
        long_s = sents[idx_long]
        split_at = _find_split_point(long_s, min_word=6)
        if split_at is None:
            break
        first = long_s[:split_at].rstrip(", ").rstrip()
        rest = long_s[split_at:].lstrip()
        # Strip leading coordinating conjunction if present.
        for conj in ("and ", "but ", "or ", "so ", "yet "):
            if rest.lower().startswith(conj):
                rest = rest[len(conj):]
                break
        if not first.endswith((".", "!", "?")):
            first += "."
        if rest and rest[0].islower():
            rest = rest[0].upper() + rest[1:]
        if not rest.endswith((".", "!", "?")):
            rest += "."
        if len(_WORD_RE.findall(first)) < 3 or len(_WORD_RE.findall(rest)) < 3:
            break  # splitting produced fragments
        sents[idx_long] = first
        sents.insert(idx_long + 1, rest)

    return " ".join(sents)


def post_process(text: str, intensity: int) -> str:
    out = text.strip()
    out = _LEADING_HERE.sub("", out).strip()
    m = _QUOTED_WRAP.match(out)
    if m:
        out = m.group(1).strip()
    out = _EM_DASH.sub(", ", out)

    # Kill cringe similes first — they are high-signal AI tells.
    out = _SIMILE_KILLER.sub("", out)
    out = _FOLKSY_SIMILE.sub("", out)

    p = 0.35 + (intensity / 200.0)  # 0.35 .. 0.85
    rng = random.Random(len(out))
    for pattern, replacements in _AI_CLICHE_SUBS:

        def _sub(match: re.Match[str]) -> str:
            if rng.random() > p:
                return match.group(0)
            return rng.choice(replacements)

        out = pattern.sub(_sub, out)

    # Mechanical burstiness enforcement on the final text.
    out = _enforce_burstiness(out)

    # Tidy whitespace and dangling punctuation from simile removal.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r",\s*([.!?])", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    return out.strip()


# ---------------------------------------------------------------------------
# LLM call with defensive response parsing. Pollinations returns an
# OpenAI-shaped response but, for reasoning models like gpt-oss-20b,
# `content` can be empty or the text can land in `reasoning_content`. We
# also retry once on empty output.
# ---------------------------------------------------------------------------


def _extract_content(data: dict) -> str:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(msg, dict):
        return ""
    # Only trust `content` — `reasoning_content` is the model's internal
    # thinking and is NOT the final answer for reasoning models.
    v = msg.get("content")
    if isinstance(v, str) and v.strip():
        return v.strip()
    delta = msg.get("delta") or {}
    if isinstance(delta, dict):
        v = delta.get("content")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# Attempt configs for call_llm retries. Pollinations (especially
# gpt-oss-20b) sometimes returns an empty `content`. Rotating temperature,
# reasoning_effort, and model between attempts gets us out of bad seeds
# without needing a paid provider.
_LLM_ATTEMPTS: list[dict] = [
    {"model": POLLINATIONS_MODEL, "temperature": 0.9, "reasoning_effort": "low"},
    {"model": POLLINATIONS_MODEL, "temperature": 1.1, "reasoning_effort": "low"},
    {"model": POLLINATIONS_MODEL, "temperature": 0.7, "reasoning_effort": "medium"},
    {"model": "openai-fast", "temperature": 0.95, "reasoning_effort": "low"},
    {"model": "mistral", "temperature": 0.9},
]


async def call_llm(prompt: str) -> str:
    system = (
        "You are a silent text rewriter. You output only the rewritten "
        "text. You never explain. You never add quotation marks, preamble, "
        "or meta-commentary."
    )
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=POLLINATIONS_TIMEOUT) as client:
        for i, cfg in enumerate(_LLM_ATTEMPTS):
            payload: dict = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": cfg.get("temperature", 0.9),
                "top_p": 0.95,
            }
            if "reasoning_effort" in cfg:
                payload["reasoning_effort"] = cfg["reasoning_effort"]
            try:
                resp = await client.post(POLLINATIONS_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                last_err = e
                if i < len(_LLM_ATTEMPTS) - 1:
                    await asyncio.sleep(0.4)
                    continue
                break

            content = _extract_content(data)
            if content:
                return content
            # Empty response on this attempt; rotate config.
            await asyncio.sleep(0.4)

    if last_err is not None:
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {last_err}"
        )
    raise HTTPException(
        status_code=502,
        detail="LLM returned an empty response after multiple attempts.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/humanize", response_model=HumanizeResponse)
async def humanize(req: HumanizeRequest) -> HumanizeResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    if req.passes is not None:
        pass_count = max(1, min(3, req.passes))
    elif req.mode == "stealth":
        pass_count = 3
    elif req.mode == "aggressive":
        pass_count = 3
    else:
        pass_count = 2

    stages = ["flow", "burstiness", "cleanup"][:pass_count]

    trace: list[PassTrace] = [
        PassTrace(
            stage="input",
            chars=len(text),
            ai_score=ai_score(text),
            burstiness=burstiness(text),
        )
    ]

    current = text
    for stage in stages:
        prompt = build_pass_prompt(current, stage, req.mode, req.intensity)
        try:
            current = await call_llm(prompt)
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502, detail=f"LLM provider error: {e}"
            ) from None
        trace.append(
            PassTrace(
                stage=stage,
                chars=len(current),
                ai_score=ai_score(current),
                burstiness=burstiness(current),
            )
        )
        await asyncio.sleep(0.15)

    humanized = post_process(current, req.intensity)

    retries = 0
    if req.target_score is not None:
        while ai_score(humanized) > req.target_score and retries < 2:
            retries += 1
            prompt = build_pass_prompt(
                humanized, "burstiness", "stealth", max(req.intensity, 80)
            )
            try:
                redo = await call_llm(prompt)
            except httpx.HTTPError:
                break
            humanized = post_process(redo, max(req.intensity, 80))
            trace.append(
                PassTrace(
                    stage=f"retry-{retries}",
                    chars=len(humanized),
                    ai_score=ai_score(humanized),
                    burstiness=burstiness(humanized),
                )
            )

    return HumanizeResponse(
        humanized=humanized,
        original_chars=len(text),
        humanized_chars=len(humanized),
        passes=pass_count + retries,
        model=POLLINATIONS_MODEL,
        ai_score=ai_score(humanized),
        burstiness=burstiness(humanized),
        trace=trace,
    )


class DetectScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)


@app.post("/api/detect-score")
async def detect_score(req: DetectScoreRequest) -> dict[str, float]:
    return {
        "ai_score": ai_score(req.text),
        "burstiness": burstiness(req.text),
    }
