"""Inventory, acquire, decontaminate and compile a deterministic CoRAG pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from .decontam import MinHashLSH, screen, window_hashes
from .eval_inventory import build_inventory, text_hash
from .sources import (
    CORPUS_REVISION, QUERY_REVISION, SOURCE_CONFIGS, acquire, inventory,
    local_tokenizer_spec, write_json,
)


def compile_examples(records, passages, tokenizer, *, seed, train_size, validation_size,
                     max_tokens=256, max_chunk_tokens=128, max_chunks=4):
    """Split connected source documents before selecting/truncating chunk groups."""
    if train_size < 1 or validation_size < 1 or max_chunks < 2 or max_chunk_tokens < 16:
        raise ValueError("Invalid pilot size or chunk limits")
    from tasks.Musique import MusiqueTask

    prefix = MusiqueTask.prefixPrompt
    prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    if max_tokens < prefix_length + 32 or max_chunk_tokens < prefix_length + 16:
        raise ValueError("Token limits must leave document content after the real KB instruction prefix")
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(right)] = find(left)

    owners, spans, near = {}, {}, MinHashLSH()
    for index, row in enumerate(records):
        keys = [("query", text_hash(row["query"]))]
        for identifier in row["passage_ids"]:
            passage = passages[identifier]
            keys.extend([("wikipedia", passage["wikipedia_id"]), ("text", text_hash(passage["contents"]))])
            ids = tokenizer.encode(passage["contents"], add_special_tokens=False)
            for digest in window_hashes(ids):
                # A hash collision can only conservatively merge split components.
                digest = int(digest)
                if digest in spans:
                    union(index, spans[digest])
                else:
                    spans[digest] = index
            for other, _ in near.query(passage["contents"]):
                union(index, int(other.split(":", 1)[0]))
            near.add(f"{index}:{identifier}", passage["contents"])
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    components = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    rank = lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    groups = sorted(components.values(), key=lambda group: rank(sorted(
        records[i]["config"] + ":" + records[i]["query_id"] for i in group)))
    selected = {"train": [], "validation": []}
    for group in groups:
        split = "validation" if len(selected["validation"]) < validation_size else "train"
        limit = validation_size if split == "validation" else train_size
        available = limit - len(selected[split])
        selected[split].extend(sorted(group, key=lambda i: rank(records[i]["config"] + records[i]["query_id"]))[:available])
    if len(selected["train"]) < train_size or len(selected["validation"]) < validation_size:
        raise ValueError(f"Insufficient independent clean source groups: {len(components)} components, "
                         f"train={len(selected['train'])}, validation={len(selected['validation'])}; acquire more records")

    output, manifest = {}, []
    for split, indices in selected.items():
        output[split] = []
        for index in indices:
            row = records[index]
            rng = random.Random(int(rank(row["config"] + ":" + row["query_id"]), 16))
            maximum = min(max_chunks, len(row["passage_ids"]), (max_tokens - prefix_length) // 16)
            count = rng.randint(2, maximum)
            identifiers = row["passage_ids"][:count]
            minimum = [prefix_length + 16] + [16] * (count - 1)
            budget = rng.randint(sum(minimum), min(max_tokens, max_chunk_tokens * count))
            quotas = minimum[:]
            while sum(quotas) < budget:
                available = [i for i in range(count) if quotas[i] < max_chunk_tokens]
                quotas[rng.choice(available)] += 1
            chunks, provenance = [], []
            for position, (identifier, quota) in enumerate(zip(identifiers, quotas)):
                passage = passages[identifier]
                title, contents = passage["title"], passage["contents"]
                body = contents[len(title) + 1:] if contents.startswith(title + "\n") else contents
                formatted = (prefix if position == 0 else "") + title + "\n\n" + body + "\n\n"
                ids = tokenizer.encode(formatted, add_special_tokens=False)
                ending = tokenizer.encode("\n\n", add_special_tokens=False)
                chunk = ids if len(ids) <= quota else ids[:quota - len(ending)] + ending
                if not chunk:
                    raise ValueError("Builder produced an empty chunk")
                chunks.append(chunk)
                provenance.append({"id": identifier, "wikipedia_id": passage["wikipedia_id"],
                                   "source_file": passage["source_file"], "source_row": passage["source_row"],
                                   "content_sha256": text_hash(contents), "retrieval_rank": row["original_passage_ids"].index(identifier),
                                   "tokens": len(chunk), "truncated": len(ids) > quota})
            example_id = hashlib.sha256(json.dumps(chunks, separators=(",", ":")).encode()).hexdigest()
            output[split].append({"chunks": chunks})
            manifest.append({"example_id": example_id, "split": split, "seed": seed,
                             "source": row["source"], "source_revision": row["revision"],
                             "source_config": row["config"], "source_split": row["split"],
                             "query_id": row["query_id"], "query": row["query"], "answers": row["answers"],
                             "source_file": row["source_file"], "source_row": row["source_row"],
                             "corpus_revision": CORPUS_REVISION, "passages": provenance,
                             "chunk_count": count, "token_count": sum(map(len, chunks)),
                             "component": find(index), "deduplication": "screened against local evaluation inventory"})
    return output, manifest


def build(args):
    from transformers import AutoTokenizer

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    coverage = json.loads((Path(args.evaluation_inventory) / "manifest.json").read_text())
    missing = [key for key, value in coverage["coverage"].items()
               if key != "agent_benchflow" and value["status"] != "available"]
    if missing:
        raise ValueError(f"Required static evaluation datasets unavailable: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_spec = local_tokenizer_spec(args.model)
    acquired = acquire(args.cache, records_per_source=args.records_per_source,
                       passages_per_query=args.passages_per_query, workers=args.workers)
    screening_key = hashlib.sha256(json.dumps({
        "query_revision": QUERY_REVISION, "corpus_revision": CORPUS_REVISION,
        "records": acquired["records"], "tokenizer": tokenizer_spec,
        "inventory": coverage,
    }, sort_keys=True).encode()).hexdigest()
    screening_cache = Path(args.cache) / f"screen-{screening_key}.json"
    if screening_cache.is_file():
        audit = json.loads(screening_cache.read_text())
    else:
        audit = screen(acquired, args.evaluation_inventory, tokenizer)
        write_json(screening_cache, audit)
    rejected = set(audit["removed_passage_ids"]) | set(audit["pending_review_passage_ids"])
    rejected_queries = {(item["config"], item["query_id"]) for item in audit["question_matches"]}
    clean, rejected_records = [], []
    for row in acquired["records"]:
        ids, seen = [], set()
        for identifier in row["passage_ids"]:
            passage = acquired["passages"][identifier]
            digest = text_hash(passage["contents"])
            if identifier not in rejected and digest not in seen and passage["contents"].strip():
                ids.append(identifier)
                seen.add(digest)
        if (row["config"], row["query_id"]) in rejected_queries or len(ids) < 2:
            rejected_records.append({"source_config": row["config"], "query_id": row["query_id"],
                                     "reason": "question_provenance" if (row["config"], row["query_id"]) in rejected_queries
                                     else "fewer_than_two_clean_passages"})
        else:
            clean.append({**row, "original_passage_ids": row["passage_ids"], "passage_ids": ids})
    examples, manifest = compile_examples(
        clean, acquired["passages"], tokenizer, seed=args.seed, train_size=args.train_size,
        validation_size=args.validation_size, max_tokens=args.max_tokens,
        max_chunk_tokens=args.max_chunk_tokens, max_chunks=args.max_chunks,
    )
    train_documents = {passage["wikipedia_id"] for row in manifest if row["split"] == "train" for passage in row["passages"]}
    validation_documents = {passage["wikipedia_id"] for row in manifest if row["split"] == "validation" for passage in row["passages"]}
    if train_documents & validation_documents:
        raise AssertionError("Source documents leaked between training and validation")
    destination.mkdir(parents=True, exist_ok=False)
    for split, rows in examples.items():
        with (destination / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    with (destination / "examples.manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps({**row, "tokenizer": tokenizer_spec}, ensure_ascii=False) + "\n")
    by_workload = defaultdict(set)
    for item in audit["removals"]:
        for match in item["evaluation"]:
            by_workload[match["workload"]].add(item["passage_id"])
    removed = set(audit["removed_passage_ids"])
    audit.update({
        "rejected_records": rejected_records, "seed": args.seed,
        "tokenizer": tokenizer_spec, "query_revision": QUERY_REVISION, "corpus_revision": CORPUS_REVISION,
        "coverage": coverage["coverage"], "evaluation_inventory": str(Path(args.evaluation_inventory).resolve()),
        "match_events_by_reason": dict(Counter(item["reason"] for item in audit["removals"])),
        "removed_passages_by_reason": {reason: len({item["passage_id"] for item in audit["removals"] if item["reason"] == reason})
                                       for reason in {item["reason"] for item in audit["removals"]}},
        "removed_passages_by_workload": {k: len(v) for k, v in by_workload.items()},
        "screening_results_by_workload": {name: {
            "coverage": details["status"], "confirmed_passages": len(by_workload.get(name, ())),
            "no_confirmed_match_in_complete_inventory": not bool(by_workload.get(name))
                if details["status"] == "available" else None,
        } for name, details in coverage["coverage"].items()},
        "by_source": {config: {"initial_records": sum(row["config"] == config for row in acquired["records"]),
                               "initial_passages": len({identifier for row in acquired["records"] if row["config"] == config
                                                        for identifier in row["passage_ids"]}),
                               "removed_passages": len({identifier for row in acquired["records"] if row["config"] == config
                                                        for identifier in row["passage_ids"] if identifier in removed}),
                               "discarded_records": sum(row["source_config"] == config for row in rejected_records),
                               "train": sum(row["source_config"] == config and row["split"] == "train" for row in manifest),
                               "validation": sum(row["source_config"] == config and row["split"] == "validation" for row in manifest)}
                      for config in SOURCE_CONFIGS},
        "final_sizes": {split: len(rows) for split, rows in examples.items()},
        "train_validation_source_documents_disjoint": True,
        "length_statistics": {split: {
            "total_tokens": sum(row["token_count"] for row in manifest if row["split"] == split),
            "min_tokens": min(row["token_count"] for row in manifest if row["split"] == split),
            "max_tokens": max(row["token_count"] for row in manifest if row["split"] == split),
            "chunk_counts": dict(Counter(row["chunk_count"] for row in manifest if row["split"] == split)),
        } for split in examples},
        "review": {"confirmed_fuzzy_removals": 0, "retained_after_review": 0,
                   "pending_review": len(audit["pending_review_passage_ids"])},
        "split_policy": "connected components by query/content/Wikipedia IDs, shared token spans and LSH candidates, before chunk construction",
        "chunk_policy": "ordered surviving first-k retrievals; deterministic 2..max_chunks; KB title/body format and Musique instruction; independently tokenized, no chat template",
        "settings": {key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool))},
    })
    write_json(destination / "deduplication.json", audit)
    files = {f"{split}.jsonl": hashlib.sha256((destination / f"{split}.jsonl").read_bytes()).hexdigest() for split in examples}
    write_json(destination / "dataset.json", {"format": "kvbench-qwen38-pilot-v1", "files_sha256": files,
                                             "sizes": audit["final_sizes"], "tokenizer": tokenizer_spec,
                                             "max_tokens": args.max_tokens, "seed": args.seed})
    summary = ["# CacheRepair pilot 数据去重报告", "", f"候选记录 {audit['initial_records']}，独立 passages {audit['initial_passages']}。",
               f"扫描评测内容 {audit['evaluation_content_units_scanned']} 段、{audit['evaluation_tokens_scanned']} tokens。",
               f"确认污染并排除 {len(removed)} 个 passages；待审查近重复 {len(audit['pending_review_passage_ids'])} 个，未纳入 pilot。",
               f"最终训练 {args.train_size} 条，验证 {args.validation_size} 条。", "",
               "所有本地静态数据均扫描；AgentBenchFlow 只覆盖 task.md/SKILL.md，动态工具输出、任务资源及生成对话未完成筛查。",
               "数据是固定 CoRAG revision 的小型近似构造，不是官方 50,000 条训练集的复现。", "",
               "逐项匹配来源、理由、覆盖情况、source split 和 revision 见 deduplication.json / examples.manifest.jsonl。"]
    (destination / "deduplication.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({"dataset": str(destination), "sizes": audit["final_sizes"], "removed_passages": len(removed)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("inventory", help="Hub metadata only, no corpus download")
    source.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluation-inventory")
    evaluation.add_argument("--data-root", default="data")
    evaluation.add_argument("--skills-root")
    evaluation.add_argument("--output", required=True)
    dataset = commands.add_parser("build")
    dataset.add_argument("--model", required=True)
    dataset.add_argument("--evaluation-inventory", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--cache", default=".cache/cacherepair-qwen38-sources")
    for name, default in (("records-per-source", 16), ("passages-per-query", 6), ("workers", 6),
                          ("train-size", 32), ("validation-size", 8), ("max-tokens", 256),
                          ("max-chunk-tokens", 128), ("max-chunks", 4), ("seed", 7)):
        dataset.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.command == "inventory":
        with Path(args.output).open("x") as stream:
            json.dump(inventory(), stream, indent=2)
    elif args.command == "evaluation-inventory":
        build_inventory(args.data_root, args.output, skills_root=args.skills_root)
    else:
        build(args)


if __name__ == "__main__":
    main()
