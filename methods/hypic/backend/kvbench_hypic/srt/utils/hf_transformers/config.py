# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config loading utilities."""

from pathlib import Path
from typing import Optional

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from methods.hypic.backend.kvbench_hypic.srt.configs.model_config_parser_registry import (
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from methods.hypic.backend.kvbench_hypic.srt.connector import create_remote_connector
from methods.hypic.backend.kvbench_hypic.srt.utils import is_remote_url, lru_cache_frozenset

from .common import (
    _CONFIG_REGISTRY,
    AutoConfig,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
    resolve_runai_obj_uri,
)
from .mistral_utils import is_mistral_model, load_mistral_config


def _set_architectures(config, arch_name):
    config.update({"architectures": [arch_name]})


def _apply_deepseek_ocr_overrides(config, model):
    _override_v_head_dim_if_zero(config)
    _set_architectures(config, "DeepseekOCRForCausalLM")
    config._name_or_path = model


@register_model_config_parser("hf")
class HfModelConfigParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code: bool, revision=None, **kwargs):
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code,
                                           revision=revision, **kwargs)
        config._name_or_path = model
        return config


@register_model_config_parser("mistral")
class MistralModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        return load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    model_config_parser: str = "auto",
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        if model_config_parser not in ("auto", "hf"):
            raise ValueError(
                f"model_config_parser={model_config_parser!r} is incompatible "
                "with GGUF inputs; only 'hf' (or 'auto') is supported."
            )
        kwargs["gguf_file"] = model
        model = Path(model).parent
        # Skip auto-resolution for GGUF: the name-based Mistral heuristic
        # would misfire on the rewritten parent dir.
        model_config_parser = "hf"

    model = resolve_runai_obj_uri(model)

    if is_remote_url(model):
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if model_config_parser == "auto":
        # `model` is post-rewrite (gguf parent / runai uri / remote pull).
        model_config_parser = "mistral" if is_mistral_model(model) else "hf"

    parser = get_model_config_parser(model_config_parser)
    config = parser.parse(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )

    if model_override_args:
        config.update(model_override_args)

    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config
