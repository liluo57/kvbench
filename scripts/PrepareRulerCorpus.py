"""Cache every official RULER essay, then run its unchanged parser offline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import time
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def _transport_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.split("/")
    if parts.hostname == "github.com" and len(path) > 5 and path[3] == "raw":
        return "https://raw.githubusercontent.com/" + "/".join(path[1:3] + path[4:])
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, parts.fragment))


def _hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def _fetch(url: str, cache: Path) -> dict:
    path = cache / "essays" / (_hash(url.encode()) + ".raw")
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        entry = json.loads(metadata_path.read_text())
        if entry["url"] != url or _hash(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"cached RULER essay failed integrity verification: {path}")
        return entry
    transport = _transport_url(url)
    error = None
    for attempt in range(4):
        try:
            # curl provides a second TLS transport for hosts that reject urllib.
            if attempt % 2 == 0:
                with urlopen(Request(transport, headers={"User-Agent": "KVBench/RULER"}), timeout=25) as response:
                    body = response.read()
            else:
                completed = subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--connect-timeout", "10", "--max-time", "40", transport],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=45,
                )
                body = completed.stdout
            if not body.strip():
                raise ValueError("empty essay")
            if urlsplit(transport).hostname == "raw.githubusercontent.com" and body.lstrip().lower().startswith(b"<!doctype html"):
                raise ValueError("GitHub returned an HTML page instead of a raw essay")
            entry = {"url": url, "transport_url": transport, "file": str(path.resolve()),
                     "sha256": _hash(body), "bytes": len(body)}
            _write(path, body)
            _write(metadata_path, (json.dumps(entry, indent=2) + "\n").encode())
            return entry
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = str(exc)
            if attempt < 3:
                time.sleep(attempt + 1)
    raise RuntimeError(f"could not fetch official RULER essay {url}: {error}")


def prepare_sources(urls_path: Path, cache: Path) -> Path:
    urls = [url.strip() for url in urls_path.read_text().splitlines() if url.strip()]
    if not urls or len(set(urls)) != len(urls):
        raise ValueError("official RULER URL list must be non-empty and unique")
    entries, failures = {}, []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, url, cache): url for url in urls}
        for completed_count, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                entries[url] = future.result()
            except Exception as exc:
                failures.append({"url": url, "error": str(exc)})
            if completed_count % 20 == 0 or completed_count == len(urls):
                print(f"[dataset] RULER essay sources checked {completed_count}/{len(urls)}, failures={len(failures)}", flush=True)
    if failures:
        _write(cache / "essay_failures.json", (json.dumps(failures, indent=2) + "\n").encode())
        raise RuntimeError(f"{len(failures)} official RULER essays are unavailable; no partial corpus will be installed; see {cache / 'essay_failures.json'}")
    manifest = cache / "corpus_sources_manifest.json"
    _write(manifest, (json.dumps({"urls_sha256": _hash(urls_path.read_bytes()),
                                 "sources": [entries[url] for url in urls]}, indent=2) + "\n").encode())
    return manifest


def render(script: Path, manifest: Path) -> None:
    entries = json.loads(manifest.read_text())["sources"]
    sources = {}
    for entry in entries:
        body = Path(entry["file"]).read_bytes()
        if _hash(body) != entry["sha256"]:
            raise ValueError(f"RULER essay cache was modified: {entry['file']}")
        sources[entry["url"]] = body

    def cached_urlopen(url, *args, **kwargs):
        if isinstance(url, Request):
            url = url.full_url
        if url not in sources:
            raise ValueError(f"official parser requested an uncached URL: {url}")
        return io.BytesIO(sources[url])

    with patch("urllib.request.urlopen", cached_urlopen):
        runpy.run_path(str(script.resolve()), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    render(args.render, args.manifest)
