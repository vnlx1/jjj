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
    "You are rewriting text to read like natural human writing. "
    "Output ONLY the rewritten text. "
    "No preamble. No quotes around the output. No explanation. "
    "No markdown unless the original used markdown. "
    "Preserve all facts, numbers, names, and the overall meaning. "
    "Keep the length within ~20% of the original.\n\n"
)

_PASS1_FLOW = (
    "Pass 1 of 3: NATURAL FLOW.\n"
    "Rewrite the text so it sounds like a thoughtful person typing a draft, "
    "not a finished essay. Apply:\n"
    "- Contractions where natural: it's, don't, they're, can't, there's, won't, I'm.\n"
    "- Replace textbook transitions (Furthermore, Moreover, Additionally, "
    "Consequently, Therefore, In conclusion) with casual ones (Also, Plus, "
    "So, Still, Anyway, Look, Thing is, Bottom line).\n"
    "- Replace AI tell-words with ordinary alternatives: delve -> dig into, "
    "utilize -> use, comprehensive -> thorough, crucial -> key, leverage -> use, "
    "robust -> solid, realm -> area, tapestry -> mix, plethora -> a ton of, "
    "pivotal -> major, ascertain -> figure out, facilitate -> help, "
    "navigate -> get through, endeavor -> effort, paradigm -> model, "
    "elucidate -> explain.\n"
    "- Prefer concrete, specific words over abstract ones.\n"
    "- Do not use em-dashes. Use commas or periods.\n"
)

_PASS2_BURSTINESS = (
    "Pass 2 of 3: BURSTINESS + PERPLEXITY.\n"
    "Keep the meaning identical. Rewrite so sentence lengths vary a lot.\n"
    "- Mix short 3-6 word sentences next to longer 25+ word sentences.\n"
    "- Never leave three sentences in a row of similar length. Break the "
    "rhythm on purpose.\n"
    "- Break parallel structure. If a list has three parallel clauses, "
    "reshape two of them so the grammatical forms differ.\n"
    "- Inject a few unexpected but accurate word choices (perplexity). Pick "
    "the word that's right but not the obvious one.\n"
    "- Add one personal aside or hedge per paragraph: 'honestly', 'to be "
    "fair', 'I guess', 'more or less', 'which, fine', 'weirdly'.\n"
    "- Start one or two sentences with And, But, So, or Still.\n"
)

_PASS3_CLEANUP = (
    "Pass 3 of 3: STRIP AI TELLS.\n"
    "Keep the meaning and structure. Scrub any remaining machine-sounding "
    "patterns.\n"
    "- Remove any of: 'In today's rapidly evolving', 'It is important to "
    "note', 'As we navigate', 'In conclusion', 'In summary', 'delve into', "
    "'a testament to', 'the landscape of', 'in the realm of'.\n"
    "- No tri-colons (three-item parallel lists) unless the original had "
    "one. Prefer two items or four.\n"
    "- Remove hollow intensifiers: 'truly', 'vitally', 'significantly', "
    "'undoubtedly' unless they add real meaning.\n"
    "- Prefer active voice. Drop passive unless it's naturally better.\n"
    "- Fix any em-dashes or en-dashes. Use commas or periods.\n"
    "- Output must still read fluently. Do not introduce grammar errors.\n"
)

_MODE_FLAVOR: dict[Mode, str] = {
    "subtle": (
        "Tone: subtle. Keep the original's register, just sand off AI edges. "
        "Light contractions and minor rewording."
    ),
    "balanced": (
        "Tone: balanced. Conversational but still competent. Visible "
        "burstiness but not slangy."
    ),
    "aggressive": (
        "Tone: aggressive. Strong personal voice. Short punchy sentences "
        "alongside rambly ones. Casual vocabulary."
    ),
    "academic": (
        "Tone: academic human. Think grad student draft, not a textbook. "
        "Keep domain words, add hedges (arguably, roughly, which is to say), "
        "break rhythm."
    ),
    "stealth": (
        "Tone: maximum stealth. Push burstiness hard. One unexpected word "
        "per sentence. Two personal asides per paragraph. Break every "
        "parallel structure. Add a colloquialism. Feel like a first draft "
        "by a thoughtful but tired human."
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
    (re.compile(r"\bpivotal\b", re.I), ["key", "major", "huge"]),
    (re.compile(r"\bleverage\b", re.I), ["use", "tap into", "lean on"]),
    (
        re.compile(r"\bin\s+today'?s\s+rapidly\s+evolving\b", re.I),
        ["these days", "right now", "lately"],
    ),
    (
        re.compile(r"\bit\s+is\s+important\s+to\s+note\s+that\b", re.I),
        ["worth noting,", "fair point:", "one thing:"],
    ),
    (re.compile(r"\bin\s+conclusion\b", re.I), ["so", "bottom line,", "all told,"]),
    (re.compile(r"\bin\s+summary\b", re.I), ["so basically,", "to sum up,", "net-net,"]),
    (re.compile(r"\bas\s+we\s+navigate\b", re.I), ["as we work through", "going through"]),
    (re.compile(r"\brealm\b", re.I), ["area", "world", "field"]),
    (re.compile(r"\btapestry\b", re.I), ["mix", "blend", "web"]),
    (re.compile(r"\bplethora\b", re.I), ["a ton of", "loads of", "plenty of"]),
    (re.compile(r"\bfacilitate\b", re.I), ["help", "make easier", "enable"]),
    (re.compile(r"\bascertain\b", re.I), ["figure out", "find out", "work out"]),
    (re.compile(r"\brobust\b", re.I), ["solid", "strong", "sturdy"]),
]

_EM_DASH = re.compile(r"\s*[\u2014\u2013]\s*")
_QUOTED_WRAP = re.compile(r'^"(.+)"$', re.S)
_LEADING_HERE = re.compile(
    r"^\s*(here(?:'s| is) (?:the|your|a) (?:rewritten|rewrite|humanized|revised)[^\n:]*:?\s*)",
    re.I,
)


def post_process(text: str, intensity: int) -> str:
    out = text.strip()
    out = _LEADING_HERE.sub("", out).strip()
    m = _QUOTED_WRAP.match(out)
    if m:
        out = m.group(1).strip()
    out = _EM_DASH.sub(", ", out)

    p = 0.25 + (intensity / 200.0)  # 0.25 .. 0.75
    rng = random.Random(len(out))
    for pattern, replacements in _AI_CLICHE_SUBS:

        def _sub(match: re.Match[str]) -> str:
            if rng.random() > p:
                return match.group(0)
            return rng.choice(replacements)

        out = pattern.sub(_sub, out)

    out = re.sub(r"  +", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


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


async def call_llm(prompt: str, *, attempt: int = 0) -> str:
    payload = {
        "model": POLLINATIONS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a silent text rewriter. You output only the "
                    "rewritten text. You never explain. You never add "
                    "quotation marks, preamble, or meta-commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        # Reasoning models on Pollinations honor this hint; it dramatically
        # shortens latency and keeps the final answer in `content`.
        "reasoning_effort": "low",
    }
    async with httpx.AsyncClient(timeout=POLLINATIONS_TIMEOUT) as client:
        resp = await client.post(POLLINATIONS_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = _extract_content(data)
    if content:
        return content

    if attempt < 1:
        await asyncio.sleep(0.3)
        return await call_llm(prompt, attempt=attempt + 1)

    raise HTTPException(
        status_code=502,
        detail="LLM returned an empty response. Try again in a moment.",
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
