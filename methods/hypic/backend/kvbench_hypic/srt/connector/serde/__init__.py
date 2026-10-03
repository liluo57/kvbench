# SPDX-License-Identifier: Apache-2.0

# inspired by LMCache
from typing import Optional, Tuple

import torch

from methods.hypic.backend.kvbench_hypic.srt.connector.serde.safe_serde import SafeDeserializer, SafeSerializer
from methods.hypic.backend.kvbench_hypic.srt.connector.serde.serde import Deserializer, Serializer


def create_serde(serde_type: str) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "safe":
        s = SafeSerializer()
        d = SafeDeserializer()
    else:
        raise ValueError(f"Unknown serde type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "SafeSerializer",
    "SafeDeserializer",
    "create_serde",
]
