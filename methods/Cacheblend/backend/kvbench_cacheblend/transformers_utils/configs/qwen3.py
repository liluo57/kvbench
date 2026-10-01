"""Dense Qwen3 configuration for the bundled Transformers 4.44 runtime."""

from transformers import LlamaConfig


class Qwen3Config(LlamaConfig):
    model_type = "qwen3"

    def __init__(self, head_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
