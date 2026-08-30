"""Drive a SkillsBench rollout end-to-end: HTTP endpoint + sandbox + agent + verifier.

The :class:`~helpers.benchflow.helper.BenchflowHelper` owns four things and
exposes only the queue/sync the Workload needs (mirroring the original split
— endpoint plumbing is hidden from the Workload, and Method is untouched):

- **HTTP server**: an OpenAI-compatible ``POST /v1/chat/completions`` endpoint
  backed by ``http.server.ThreadingHTTPServer``. Each agent request becomes a
  ``(prompt, future)`` pair in a queue; the handler blocks on the future
  until the Workload responds.
- **Sandbox**: the SkillsBench task's environment. The agent and the verifier
  both run *inside* the sandbox (DockerSandbox in production; LocalSandbox
  for dev / smoke tests). The sandbox is what guarantees the agent sees the
  task's installed dependencies and the verifier's expected paths
  (``/root/answer.json``, ``/logs/verifier/reward.txt``).
- **Watchdog**: detects when the agent exits, runs the verifier inside the
  sandbox, extracts the reward, and writes a synthetic ``result.json``.

The Helper's ``sandbox_type`` selects the isolation strategy:

- ``docker`` (default): builds the task's ``environment/Dockerfile``, runs the
  agent + verifier inside containers with the SkillsBench-expected mounts,
  extracts the reward from ``/logs/verifier/reward.txt``.
- ``apptainer``: pulls the ``FROM``-line base image of the task's
  ``environment/Dockerfile`` as a rootless SIF via ``apptainer pull`` and
  executes the agent + verifier with bind mounts.
- ``local``: runs everything on the host. Faster (no image build), no
  isolation — only for dev / smoke tests.
"""

from .Helper import BenchflowHelper
from .Sandbox import (
    ApptainerSandbox,
    DockerSandbox,
    LocalSandbox,
    Sandbox,
)
from .Util import FindFreePort, ResolveHostSitePackages


__all__ = [
    "ApptainerSandbox",
    "BenchflowHelper",
    "DockerSandbox",
    "FindFreePort",
    "LocalSandbox",
    "ResolveHostSitePackages",
    "Sandbox",
]