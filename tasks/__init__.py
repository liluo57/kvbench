"""Concrete benchmark tasks.

Each task's dataset is resolved by *name* against ``DatasetPath`` from
``config.yaml`` (see ``core.Config``):

- :class:`NIAHTask` / :class:`NIAHShuffleTask` — RULER needle-in-a-haystack
  (:mod:`tasks.Niah`), read from ``<DatasetPath>/ruler/niah_len*.jsonl``.
- :class:`VTTask` / :class:`VTShuffleTask` — RULER variable tracking
  (:mod:`tasks.Vt`), read from ``<DatasetPath>/ruler/vt_len*.jsonl``.
- :class:`CWETask` / :class:`CWEShuffleTask` — RULER common-words extraction
  (:mod:`tasks.Cwe`), read from ``<DatasetPath>/ruler/cwe_len*.jsonl``.

  Each RULER family ships a *shuffle* variant: the prompt's informative units
  (needle segments / chain assignments / numbered-list blocks) are handed to
  reuse methods in the original order as a chunk-isolated context
  (``prepare_input``) while ``run_input`` is their non-identity permutation —
  a method that detects the change recomputes, a naive one serves stale KV.

- :class:`MusiqueTask` / :class:`WikimQATask` / :class:`SamsumTask` — the
  knowledge-base workloads the original CacheBlend repo evaluates on
  (``<DatasetPath>/musique``, ``/wikimqa``, ``/samsum``; each in its own module
  sharing the machinery in :mod:`tasks.bases.KBBase`).
"""

from .Cwe import CWEShuffleTask, CWETask
from .Musique import MusiqueTask
from .Niah import NIAHShuffleTask, NIAHTask
from .Samsum import SamsumTask
from .Vt import VTShuffleTask, VTTask
from .WikimQA import WikimQATask

__all__ = [
    "CWEShuffleTask",
    "CWETask",
    "MusiqueTask",
    "NIAHShuffleTask",
    "NIAHTask",
    "SamsumTask",
    "VTShuffleTask",
    "VTTask",
    "WikimQATask",
]
