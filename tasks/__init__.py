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
  :class:`workflow.RAGWorkflow.RAGInput`: ``prepare_input`` contains the
  original-order informative units and ``run_input`` contains their
  non-identity permutation. These are RAGInput fields, not Case fields. A
  method that detects the change recomputes; a naive one serves stale KV.

- :class:`MusiqueTask` / :class:`WikimQATask` / :class:`SamsumTask` — the
  knowledge-base workflows the original CacheBlend repo evaluates on
  (``<DatasetPath>/musique``, ``/wikimqa``, ``/samsum``; each in its own module
  sharing the machinery in :mod:`tasks.bases.KBBase`).
- :class:`HotpotQATask` / :class:`TriviaQATask` — local LongBench QA tasks;
  :class:`MultiNewsTask` / :class:`GovReportTask` — local LongBench
  summarization tasks.  Their snapshots are kept under ``data/`` and are not
  resolved from the HYPIC checkout.
- :class:`FreshGapTask` — a synthetic interleaved-reuse check where a short
  fresh span appears between two reusable chunks.
- :class:`KVCommMMLUTask` / :class:`KVCommGSM8KTask` /
  :class:`KVCommHumanEvalTask` / :class:`KVCommCopyTask` — multi-agent
  workflows whose sequential agent outputs can be retained and reused.
"""

from .AgentBenchFlowTask import AgentBenchFlowTask
from .Cwe import CWEShuffleTask, CWETask
from .GovReport import GovReportTask
from .HotpotQA import HotpotQATask
from .Musique import MusiqueTask
from .MultiNews import MultiNewsTask
from .Niah import NIAHShuffleTask, NIAHTask
from .Samsum import SamsumTask
from .TriviaQA import TriviaQATask
from .Vt import VTShuffleTask, VTTask
from .WikimQA import WikimQATask
from .KVCommTasks import KVCommCopyTask, KVCommGSM8KTask, KVCommHumanEvalTask, KVCommMMLUTask

__all__ = [
    "AgentBenchFlowTask",
    "CWEShuffleTask",
    "CWETask",
    "GovReportTask",
    "HotpotQATask",
    "MusiqueTask",
    "MultiNewsTask",
    "NIAHShuffleTask",
    "NIAHTask",
    "SamsumTask",
    "TriviaQATask",
    "VTShuffleTask",
    "VTTask",
    "WikimQATask",
    "KVCommMMLUTask",
    "KVCommGSM8KTask",
    "KVCommHumanEvalTask",
    "KVCommCopyTask",
]
