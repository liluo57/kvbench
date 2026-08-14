"""RULER-style synthetic data generator for KVBench (NIAH / VT / CWE).

Generates the JSONL records the RULER tasks in ``tasks/Niah.py`` consume. The
prompt templates and record layout follow the official RULER benchmark (Hsieh
et al., COLM 2024) so results stay comparable with the reference numbers:

    {index, input, outputs, length, length_w_model_temp, answer_prefix,
     token_position_answer, needle}

``input`` + ``answer_prefix`` is the complete Mistral-formatted prompt. The
needle sentence (``needle``) is a KVBench extension used by ``NIAHShuffleTask``
to split the prompt into the A/B/C segments.

Usage::

    python GenerateRuler.py --task niah --max_seq_length 8192 --num_samples 10
    python GenerateRuler.py --task vt   --max_seq_length 8192 --num_samples 10
    python GenerateRuler.py --task cwe  --max_seq_length 8192 --num_samples 10

Records are written to ``data/ruler/<task>_len<length>.jsonl``. The essay
haystack corpus (Paul Graham essays) is downloaded on first use into
``data/ruler/corpus/paul_graham.json``; the word list comes from
``/usr/share/dict/words`` (fallback: an embedded list). Token counts use the
model's own HuggingFace tokenizer (config ``ModelPath`` unless ``--tokenizer``).
"""

import argparse
import html
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------
# Templates (from RULER ``scripts/data/synthetic/constants.py``, ``meta-chat``
# wrapper ``[INST] ... [/INST]``, single-needle / single-value NIAH).
# --------------------------------------------------------------------------

_NiahTemplate = (
    "[INST] A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards.\n"
    "{context}\n"
    "What is the special magic number for {query} mentioned in the provided text? [/INST]"
)
_NiahAnswerPrefix = " The special magic number for {query} mentioned in the provided text is"
_NiahNeedle = "One of the special magic numbers for {key} is: {value}."
_NiahPrefixMarker = " The speci"

_VtTemplate = (
    "[INST] Memorize and track the chain(s) of variable assignment hidden in "
    "the following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above. [/INST]"
)
_VtAnswerPrefix = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)
_VtPrefixMarker = " Answer: "

_CweTemplate = (
    "[INST] Below is a numbered list of words. In these words, some appear "
    "more often than others. Memorize the ones that appear most often.\n"
    "{context}\nQuestion: What are the 10 most common words in the above list? [/INST]"
)
_CweAnswerPrefix = " Answer: The top 10 words that appear most often in the list are:"
_CwePrefixMarker = " Answer: "

_NoiseSentences = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

#: tokens_to_generate budget per task (RULER constants.py).
_TaskBudgets = {"niah": 128, "vt": 30, "cwe": 120}

_FallbackWords = (
    "apple bear castle desert elephant forest garden hammer island jungle "
    "kettle lemon mountain night ocean paper queen river silver tiger umbrella "
    "valley window yellow zebra anchor blossom candle diamond eagle falcon "
    "glacier horizon island jacket kernel lantern meadow nickel olive pepper "
    "quartz ribbon saddle thunder village willow anchor branch candle meadow "
    "pepper quartz saddle thunder village willow young zephyr bridge cloud "
    "drum ember flame galaxy harbor ivory jade kite lantern marble nebula "
    "opening palace quiet river sunrise tulip unity velvet wave xylem yearn "
    "bloom crater desert ember frost granite hollow island jewel kelp lark "
    "mist nectar orbit pebble quartz ravine stone trail undercurrent violet "
    "whisper yonder zenith amber birch cascade dawn echo flame glow harbor "
    "icicle jungle knoll lagoon moss nova oasis pearl quill ridge summit "
    "thicket urn valley willow marshl"
).split()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _randomNumber(numDigits: int = 7) -> str:
    """A random integer with exactly ``numDigits`` digits (as a string)."""
    lower = 10 ** (numDigits - 1)
    upper = 10 ** numDigits - 1
    return str(random.randint(lower, upper))


def _splitSentences(text: str) -> List[str]:
    """Rough sentence splitter (stand-in for ``nltk.sent_tokenize``)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _numTokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:  # transformers >= 5 dropped add_special_tokens from encode
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _splitAnswerPrefix(fullPrompt: str, prefixMarker: str) -> Tuple[str, str]:
    """Split a full prompt into (input, answer_prefix) like RULER's generators."""
    idx = fullPrompt.rfind(prefixMarker)
    if idx < 0:
        raise RuntimeError(f"answer prefix marker {prefixMarker!r} not found")
    return fullPrompt[:idx], fullPrompt[idx:]


# --------------------------------------------------------------------------
# Word list / essay corpus
# --------------------------------------------------------------------------


def _loadWords(wordListPath: Optional[Path]) -> List[str]:
    """Lowercase alphabetic words, longest first, deduped."""
    if wordListPath is not None and wordListPath.exists():
        words = [w.strip().lower() for w in wordListPath.read_text().splitlines() if w.strip()]
        words = [w for w in words if re.fullmatch(r"[a-z]{3,12}", w)]
        if words:
            return sorted(set(words))
    print(f"[ruler] {wordListPath} unusable; using embedded word list", flush=True)
    return _FallbackWords


def _fetch(url: str) -> Optional[str]:
    """Download ``url`` via ``curl`` (urllib is unreliable in this env)."""
    try:
        proc = subprocess.run(
            ["curl", "-sL", "--max-time", "25", "-A", "Mozilla/5.0", url],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            print(f"[ruler]   skip {url} (curl {proc.returncode})", flush=True)
            return None
        return proc.stdout.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - a failing essay must not abort
        print(f"[ruler]   skip {url}: {exc}", flush=True)
        return None


def _downloadEssays(urlsFile: Path) -> str:
    """Download the Paul Graham corpus into one concatenated string."""
    urls = [line.strip() for line in urlsFile.read_text().splitlines() if line.strip()]
    chunks: List[str] = []
    for url in urls:
        raw = _fetch(url)
        if not raw:
            continue
        if ".html" in url:
            body = re.search(r"<font[^>]*>(.*?)</font>", raw, re.S)
            text = body.group(1) if body else raw
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text)
        else:
            text = raw
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 2000:  # skip teaser/short pages
            chunks.append(text)
    if not chunks:
        raise RuntimeError("essay download failed: no usable essay text")
    return "\n".join(chunks)


def _extractCorpusFromData(dataDir: Path) -> Optional[str]:
    """Reuse the essay haystack of already-downloaded niah samples as a corpus."""
    essays: List[str] = []
    for path in sorted(dataDir.glob("niah_len*.jsonl")):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    body = rec["input"]
                    # strip the instruction + question, keep the middle essay
                    body = re.split(r"\nWhat is the special magic", body, maxsplit=1)[0]
                    start = body.find(". ")
                    essays.append(body[start + 1:] if start >= 0 else body)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not essays:
        return None
    return " ".join(essays)


def _loadEssayCorpus(corpusDir: Path, dataDir: Path, downloadFirst: bool = False) -> str:
    """The essay haystack text: reuse downloaded samples, else download once."""
    jsonPath = corpusDir / "paul_graham.json"
    if jsonPath.exists():
        return json.loads(jsonPath.read_text())["text"]

    text = ""
    if downloadFirst:
        try:
            print("[ruler] downloading Paul Graham essay corpus ...", flush=True)
            text = _downloadEssays(corpusDir / "PaulGrahamEssaysUrls.txt")
            if len(text) < 20000:
                raise RuntimeError(f"corpus too small ({len(text)} chars)")
        except Exception as exc:  # noqa: BLE001 - fall back to downloaded data
            print(f"[ruler] corpus download unavailable ({exc}); extracting", flush=True)
    if not text:
        print("[ruler] extracting essay corpus from downloaded niah data ...", flush=True)
        text = _extractCorpusFromData(dataDir)
    if not text:
        raise RuntimeError("no essay corpus available (pass --download_corpus)")

    corpusDir.mkdir(parents=True, exist_ok=True)
    jsonPath.write_text(json.dumps({"text": text}))
    print(f"[ruler] corpus ready: {len(text)} chars", flush=True)
    return text


# --------------------------------------------------------------------------
# Binary search: largest haystack size that still fits max_seq_length
# --------------------------------------------------------------------------


def _largestFit(
    buildFull: Callable[[int], Tuple[str, Any]],
    tokenizer,
    maxSeqLength: int,
    tokensToGenerate: int,
    incremental: int,
) -> int:
    sampleFull, _ = buildFull(incremental)
    tokensPerUnit = _numTokens(tokenizer, sampleFull) / incremental
    lower = incremental
    upper = max(int((maxSeqLength / tokensPerUnit) * 3), incremental * 2)
    best = None
    while lower <= upper:
        mid = (lower + upper) // 2
        full, _ = buildFull(mid)
        total = _numTokens(tokenizer, full) + tokensToGenerate
        if total <= maxSeqLength:
            best = mid
            lower = mid + 1
        else:
            upper = mid - 1
    return best if best is not None else incremental


# --------------------------------------------------------------------------
# NIAH (essay haystack, word key, number value)  == RULER ``niah_single_2``
# --------------------------------------------------------------------------


def GenerateNiah(tokenizer, essayText: str, words: List[str], numSamples: int,
                 maxSeqLength: int, tokensToGenerate: int, randomSeed: int,
                 outPath: Path) -> None:
    random.seed(randomSeed)
    haystack = re.sub(r"\s+", " ", essayText).strip().split(" ")

    def buildFull(numHaystack: int) -> Tuple[str, Dict[str, Any]]:
        if numHaystack <= len(haystack):
            text = " ".join(haystack[:numHaystack])
        else:
            repeats = (numHaystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * repeats)[:numHaystack])
        sentences = _splitSentences(text)
        depth = random.randint(0, 100)
        insertPos = int(len(sentences) * depth / 100)
        key = random.choice(words)
        value = _randomNumber(7)
        needle = _NiahNeedle.format(key=key, value=value)
        context = " ".join(sentences[:insertPos] + [needle] + sentences[insertPos:])
        full = (
            _NiahTemplate.format(context=context, query=key)
            + _NiahAnswerPrefix.format(query=key)
        )
        meta = {"needle": needle, "key": key, "value": value, "depth": depth}
        return full, meta

    numHaystack = _largestFit(buildFull, tokenizer, maxSeqLength, tokensToGenerate, incremental=500)
    records: List[Dict[str, Any]] = []
    for idx in range(numSamples):
        full, meta = buildFull(numHaystack)
        inputText, answerPrefix = _splitAnswerPrefix(full, _NiahPrefixMarker)
        length = _numTokens(tokenizer, full) + tokensToGenerate
        answerOffset = inputText.find(meta["value"])
        tokenPos = _numTokens(tokenizer, inputText[:answerOffset])
        records.append({
            "index": answerOffset,
            "input": inputText,
            "outputs": [meta["value"]],
            "length": length,
            "length_w_model_temp": length,
            "answer_prefix": answerPrefix,
            "token_position_answer": tokenPos,
            "needle": meta["needle"],
            "depth_percent": meta["depth"],
        })
    _writeRecords(outPath, records)


# --------------------------------------------------------------------------
# Variable Tracking  == RULER ``vt`` (1 chain, 4 hops)
# --------------------------------------------------------------------------


def _generateChains(numChains: int, numHops: int) -> Tuple[List[List[str]], List[List[str]]]:
    varsAll = [''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) for _ in range((numHops + 1) * numChains)]
    while len(set(varsAll)) < numChains * (numHops + 1):
        varsAll.append(''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)))
    varsRet: List[List[str]] = []
    chainsRet: List[List[str]] = []
    for i in range(0, len(varsAll), numHops + 1):
        thisVars = varsAll[i:i + numHops + 1]
        varsRet.append(thisVars)
        chain = [f"VAR {thisVars[0]} = {random.randint(10000, 99999)}"]
        for j in range(numHops):
            chain.append(f"VAR {thisVars[j + 1]} = VAR {thisVars[j]} ")
        chainsRet.append(chain)
    return varsRet, chainsRet


def GenerateVt(tokenizer, numSamples: int, maxSeqLength: int, tokensToGenerate: int,
               randomSeed: int, outPath: Path) -> None:
    random.seed(randomSeed)
    numChains, numHops = 1, 4
    noise = _NoiseSentences

    def buildFull(numNoises: int) -> Tuple[str, Dict[str, Any]]:
        varsRet, chains = _generateChains(numChains, numHops)
        value = chains[0][0].split("=")[-1].strip()
        sentences = [noise] * numNoises
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences)), len(chain)))
            for offset, j in enumerate(range(len(chain))):
                sentences.insert(positions[offset] + offset, chain[j])
        context = "\n".join(sentences).replace(". \n", ".\n")
        full = (
            _VtTemplate.format(context=context, query=value)
            + _VtAnswerPrefix.format(query=value, num_v=numHops + 1)
        )
        return full, {"vars": varsRet[0]}

    numNoises = _largestFit(buildFull, tokenizer, maxSeqLength, tokensToGenerate, incremental=5)
    records: List[Dict[str, Any]] = []
    for idx in range(numSamples):
        full, meta = buildFull(numNoises)
        inputText, answerPrefix = _splitAnswerPrefix(full, _VtPrefixMarker)
        length = _numTokens(tokenizer, full) + tokensToGenerate
        records.append({
            "index": idx,
            "input": inputText,
            "outputs": meta["vars"],
            "length": length,
            "length_w_model_temp": length,
            "answer_prefix": answerPrefix,
        })
    _writeRecords(outPath, records)


# --------------------------------------------------------------------------
# Common Words Extraction  == RULER ``cwe``
# --------------------------------------------------------------------------


def GenerateCwe(tokenizer, words: List[str], numSamples: int, maxSeqLength: int,
                tokensToGenerate: int, randomSeed: int, outPath: Path) -> None:
    random.seed(randomSeed)
    freqCw, freqUcw, numCw = 30, 3, 10

    def buildFull(numWords: int) -> Tuple[str, Dict[str, Any]]:
        pool = words if numWords <= len(words) else words + words
        wordList = random.sample(pool, min(numWords, len(pool)))
        common = wordList[:numCw]
        uncommon = wordList[numCw:]
        items = list(common) * freqCw + list(uncommon) * freqUcw
        random.shuffle(items)
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(items))
        full = _CweTemplate.format(context=context) + _CweAnswerPrefix
        return full, {"common": common}

    numWords = _largestFit(buildFull, tokenizer, maxSeqLength, tokensToGenerate, incremental=50)
    records: List[Dict[str, Any]] = []
    for idx in range(numSamples):
        full, meta = buildFull(numWords)
        inputText, answerPrefix = _splitAnswerPrefix(full, _CwePrefixMarker)
        length = _numTokens(tokenizer, full) + tokensToGenerate
        records.append({
            "index": idx,
            "input": inputText,
            "outputs": meta["common"],
            "length": length,
            "length_w_model_temp": length,
            "answer_prefix": answerPrefix,
        })
    _writeRecords(outPath, records)


def _writeRecords(outPath: Path, records: List[Dict[str, Any]]) -> None:
    outPath.parent.mkdir(parents=True, exist_ok=True)
    with outPath.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[ruler] wrote {len(records)} samples -> {outPath}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _defaultModelPath() -> str:
    cfgPath = Path(__file__).parent / "config.yaml"
    try:
        cfg = yaml.safe_load(cfgPath.read_text()) or {}
        return str(cfg.get("ModelPath", ""))
    except OSError:
        return ""


def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["niah", "vt", "cwe"], required=True)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--tokens_to_generate", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=Path("data/ruler"))
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer path (default: config ModelPath)")
    parser.add_argument("--words", type=Path, default=Path("/usr/share/dict/words"))
    parser.add_argument("--download_corpus", action="store_true",
                        help="download the Paul Graham essay corpus instead of "
                             "reusing the downloaded niah samples' haystack")
    args = parser.parse_args()

    tokenizerPath = args.tokenizer or _defaultModelPath()
    if not tokenizerPath:
        raise SystemExit("no tokenizer: pass --tokenizer or set ModelPath in config.yaml")
    print(f"[ruler] loading tokenizer {tokenizerPath}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizerPath)

    words = _loadWords(args.words)
    tokensToGenerate = args.tokens_to_generate or _TaskBudgets[args.task]
    outPath = args.data_dir / f"{args.task}_len{args.max_seq_length}.jsonl"

    if args.task == "niah":
        essayText = _loadEssayCorpus(args.data_dir / "corpus", args.data_dir,
                                     downloadFirst=args.download_corpus)
        GenerateNiah(tokenizer, essayText, words, args.num_samples, args.max_seq_length,
                     tokensToGenerate, args.random_seed, outPath)
    elif args.task == "vt":
        GenerateVt(tokenizer, args.num_samples, args.max_seq_length,
                   tokensToGenerate, args.random_seed, outPath)
    elif args.task == "cwe":
        GenerateCwe(tokenizer, words, args.num_samples, args.max_seq_length,
                    tokensToGenerate, args.random_seed, outPath)


if __name__ == "__main__":
    Main()
