"""Shared document/prefix contract for mixed training and task validation.

Separate source documents never share a cache chunk. A lone document is split
at paragraph, dialogue-turn, sentence or word boundaries, in that order. Token
budgets truncate documents rather than moving tokens across their boundaries.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import re


POLICY = {
    "format": "task-documents-prefix-v1",
    "max_tokens": 16384,
    "max_documents": 32,
    "single_document_parts": 4,
    "multi_document_policy": "One source document per chunk; original order, first 32 documents",
    "single_document_policy": "Natural paragraphs/turns/sentences; word-boundary fallback",
    "budget_policy": "Max-min fair token allocation across documents; preserve first prefix",
    "prefix_policy": "Task-specific prefix on first chunk only; RULER retains its embedded instruction",
}


@lru_cache(None)
def task_prefix(task):
    from tasks.Musique import MusiqueTask
    from tasks.TwoWikiMultiHopQA import TwoWikiMultiHopQATask
    from tasks.HotpotQA import HotpotQATask
    from tasks.TriviaQA import TriviaQATask
    from tasks.Samsum import SamsumTask
    from tasks.GovReport import GovReportTask
    from tasks.MultiNews import MultiNewsTask
    classes = {"corag": MusiqueTask, "musique": MusiqueTask,
        "2wikimultihopqa": TwoWikiMultiHopQATask, "hotpotqa": HotpotQATask,
        "triviaqa": TriviaQATask, "samsum": SamsumTask,
        "govreport": GovReportTask, "multinews": MultiNewsTask}
    if task in ("niah_shuffle", "cwe_shuffle", "vt_shuffle"):
        return ""
    return classes[task].prefixPrompt


def natural_parts(text, count=4):
    """Keep text byte-for-byte; choose cuts only at natural text boundaries."""
    if count < 2:
        raise ValueError("At least two parts are needed for residual supervision")
    cuts,kind=[],'unsegmentable'
    for pattern,boundary,label in (
        (r"(?:\r?\n[ \t]*){2,}",'start','paragraph'),
        (r"\r?\n",'end','line/turn'),
        (r"[.!?](?:[\"')\]]+)?(?=\s+\S)",'end','sentence'),
        (r"\s+\S",'start','word'),
    ):
        candidate=[];previous=0
        for match in re.finditer(pattern,text):
            position=getattr(match,boundary)()
            if 0<position<len(text) and text[previous:position].strip():
                candidate.append(position);previous=position
        while candidate and not text[candidate[-1]:].strip():candidate.pop()
        if candidate:
            cuts,kind=candidate,label;break
    n = min(count, len(cuts) + 1)
    selected, first = [], 0
    for part in range(1, n):
        last = len(cuts) - (n - part)
        index = min(range(first, last + 1), key=lambda i: abs(cuts[i] - len(text)*part/n))
        selected.append(cuts[index]); first = index + 1
    boundaries = [0, *selected, len(text)]
    parts = [text[a:b] for a, b in zip(boundaries, boundaries[1:])]
    assert "".join(parts) == text and all(p.strip() for p in parts)
    return parts, kind


def fair_quotas(lengths, budget, first_reserve=0):
    """Preserve every document and the first instruction before fair clipping."""
    minimum = [min(n, max(2, first_reserve) if i == 0 else 2) for i, n in enumerate(lengths)]
    if sum(minimum) > budget:
        raise ValueError("Token budget cannot retain document boundaries and task prefix")
    result = minimum[:]; remaining = min(budget, sum(lengths)) - sum(result)
    active = [i for i, n in enumerate(lengths) if n > result[i]]
    while remaining and active:
        share = max(1, remaining // len(active))
        for i in active:
            amount = min(share, lengths[i] - result[i], remaining)
            result[i] += amount; remaining -= amount
            if not remaining:
                break
        active = [i for i in active if lengths[i] > result[i]]
    return result


def compile_documents(documents, task, tokenizer, *, max_tokens=16384, max_documents=32,
                      single_document_parts=4):
    documents = [str(d) for d in documents if str(d).strip()]
    if not documents:
        raise ValueError("Empty document input")
    original_count = len(documents)
    if task.endswith('_shuffle'):
        text=documents[0].strip()
        if text.startswith('[INST]'):text=text[len('[INST]'):].lstrip()
        if text.endswith('[/INST]'):text=text[:-len('[/INST]')].rstrip()
        documents=[text]
    single_kind = None
    if len(documents) == 1:
        documents, single_kind = natural_parts(documents[0], single_document_parts)
    if len(documents) < 2:
        raise ValueError("Single source has no natural boundary for residual supervision")
    documents = documents[:max_documents]
    prefix = task_prefix(task)
    # Match the old KB document presentation without merging source documents.
    if original_count > 1 and task in ("musique", "2wikimultihopqa", "hotpotqa", "triviaqa"):
        formatted = []
        for d in documents:
            title, sep, body = d.partition("\n")
            formatted.append(title + "\n\n" + body + "\n\n" if sep else d + "\n\n")
    elif original_count > 1:
        formatted = [d + '\n\n' for d in documents]
    else:
        formatted = documents
    texts = [(prefix if i == 0 else "") + d for i, d in enumerate(formatted)]
    encoded = [tokenizer.encode(t, add_special_tokens=False, truncation=True,
        max_length=max_tokens+1) for t in texts]
    if not all(encoded):
        raise ValueError("Tokenizer produced an empty source document")
    prefix_count = len(tokenizer.encode(prefix, add_special_tokens=False))
    quotas = fair_quotas(list(map(len, encoded)), max_tokens, prefix_count+2)
    ending=tokenizer.encode('\n\n',add_special_tokens=False)
    chunks = [ids[:quota-len(ending)]+ending
        if original_count>1 and len(ids)>quota and quota>len(ending) else ids[:quota]
        for ids, quota in zip(encoded, quotas)]
    info = {"chunk_policy": POLICY["format"], "source_document_count": original_count,
        "documents_kept": len(chunks), "dropped_documents": max(0, original_count-max_documents),
        "single_document_boundary": single_kind,
        "prefix_tokens": prefix_count, "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "embedded_ruler_instruction": task.endswith("_shuffle"),
        "truncated_documents": sum(len(ids)>quota for ids, quota in zip(encoded, quotas)),
        "tokens": sum(map(len, chunks))}
    assert len(chunks) >= 2 and all(chunks) and info["tokens"] <= max_tokens
    return chunks, info
