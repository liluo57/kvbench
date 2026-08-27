"""Concrete benchmark tasks.

Each task's dataset is resolved by *name* against ``DatasetPath`` from
``config.yaml`` (see ``core.Config``):

- :class:`NIAHTask` / :class:`NIAHShuffleTask` — RULER needle-in-a-haystack
  (:mod:`tasks.Niah`), read from ``<DatasetPath>/ruler/niah_len*.jsonl``.
- :class:`VTTask` / :class:`VTShuffleTask` — RULER variable tracking
  (:mod:`tasks.Vt`), read from ``<DatasetPath>/ruler/vt_len*.jsonl``.
- :class:`CWETask` / :class:`CWEShuffleTask` — RULER common-words extraction
  (:mod:`tasks.Cwe`), read from ``<DatasetPath>/ruler/cwe_len*.jsonl``.

  Each RULER family ships a *shuffle* variant. Its ``Case.input`` is an
  :class:`workload.RAGWorkload.RAGInput`: ``prepare_input`` contains the
  original-order informative units and ``run_input`` contains their
  non-identity permutation. These are RAGInput fields, not Case fields. A
  method that detects the change recomputes; a naive one serves stale KV.

- :class:`MusiqueTask` / :class:`WikimQATask` / :class:`SamsumTask` — the
  knowledge-base workloads the original CacheBlend repo evaluates on
  (``<DatasetPath>/musique``, ``/wikimqa``, ``/samsum``; each in its own module
  sharing the machinery in :mod:`tasks.bases.KBBase`).
- :class:`FreshGapTask` — a synthetic interleaved-reuse check where a short
  fresh span appears between two reusable chunks.
- :class:`KVCommMMLUTask` / :class:`KVCommGSM8KTask` /
  :class:`KVCommHumanEvalTask` / :class:`KVCommCopyTask` — multi-agent
  workloads whose sequential agent outputs can be retained and reused.
"""

from .AgentBenchFlowTask import AgentBenchFlowTask
from .Cwe import CWEShuffleTask, CWETask
from .Musique import MusiqueTask
from .Niah import NIAHShuffleTask, NIAHTask
from .Samsum import SamsumTask
from .Vt import VTShuffleTask, VTTask
from .WikimQA import WikimQATask
from .KVCommTasks import KVCommCopyTask, KVCommGSM8KTask, KVCommHumanEvalTask, KVCommMMLUTask

__all__ = [
    "AgentBenchFlowTask",
    "CWEShuffleTask",
    "CWETask",
    "MusiqueTask",
    "NIAHShuffleTask",
    "NIAHTask",
    "SamsumTask",
    "VTShuffleTask",
    "VTTask",
    "WikimQATask",
    "KVCommMMLUTask",
    "KVCommGSM8KTask",
    "KVCommHumanEvalTask",
    "KVCommCopyTask",
]
