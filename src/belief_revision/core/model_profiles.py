"""Target identities and architecture settings, independent of compute hosting."""

import re
import json
import yaml
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TargetModelProfile:
    """Requested target configuration; live backend compatibility is separate."""

    model_id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    task: str = "text-generation"
    loader: str = "AutoModelForCausalLM"
    decoder_path: str = "model"
    layers_path: str = "model.layers"
    lm_head_path: str = "lm_head"
    final_norm_path: str = "model.norm"
    logits_path: str = "output.logits"
    system_role: str = "native"
    assistant_end_token: str | None = None
    assistant_end_suffix: str = ""
    layer_output_index: int | None = None
    expected_layers: int | None = None
    expected_hidden_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "task", "loader"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not self.loader.startswith("AutoModel") or not self.loader.isidentifier():
            raise ValueError("loader must name a Transformers AutoModel class")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be a nonempty string or null")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.system_role not in ("native", "merge_system_into_user"):
            raise ValueError("system_role must be native or merge_system_into_user")
        for name in ("decoder_path", "layers_path", "lm_head_path", "final_norm_path", "logits_path"):
            _validate_path(getattr(self, name))
        if self.assistant_end_token is not None and (not isinstance(self.assistant_end_token, str) or not self.assistant_end_token):
            raise ValueError("assistant_end_token must be a nonempty string or null")
        if not isinstance(self.assistant_end_suffix, str):
            raise ValueError("assistant_end_suffix must be a string")
        if self.layer_output_index is not None and (type(self.layer_output_index) is not int or self.layer_output_index < 0):
            raise ValueError("layer_output_index must be a nonnegative integer or null")
        for name in ("expected_layers", "expected_hidden_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _validate_path(path: str) -> None:
    if not isinstance(path, str) or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", path) is None:
        raise ValueError(f"invalid model attribute path: {path!r}")


def profile_from_dict(data: Mapping[str, object]) -> TargetModelProfile:
    """Load only recognized profile fields; reject typos instead of ignoring them."""

    if not isinstance(data, Mapping):
        raise ValueError("model profile must be a mapping")
    unknown = set(data) - {field.name for field in fields(TargetModelProfile)}
    if unknown:
        raise ValueError("unknown model profile field(s): " + ", ".join(sorted(map(str, unknown))))
    if "model_id" not in data:
        raise ValueError("model profile requires model_id")
    return TargetModelProfile(**data)


def default_target_profile(model_id: str | None = None) -> TargetModelProfile:
    """Provide compatibility defaults; custom architectures use an explicit YAML."""

    from .defaults import TARGET_MODEL

    model_id = TARGET_MODEL if model_id is None else model_id
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    options: dict[str, object] = {"model_id": model_id}
    if model_id.startswith("meta-llama/"):
        options["assistant_end_token"] = "<|eot_id|>"
    elif model_id.startswith("Qwen/"):
        options.update(assistant_end_token="<|im_end|>", assistant_end_suffix="\n")
    elif model_id.startswith("google/gemma-"):
        options.update(system_role="merge_system_into_user", assistant_end_token="<end_of_turn>", assistant_end_suffix="\n")
        if model_id.startswith("google/gemma-3-"):
            options.update(task="image-text-to-text", loader="AutoModelForImageTextToText", decoder_path="model.language_model", layers_path="model.language_model.layers", final_norm_path="model.language_model.norm")
    elif model_id.startswith("mistralai/Ministral-3-"):
        options.update(task="image-text-to-text", loader="AutoModelForImageTextToText", decoder_path="model.language_model", layers_path="model.language_model.layers", final_norm_path="model.language_model.norm", assistant_end_token="</s>")
    return TargetModelProfile(**options)


def load_target_profile(path: str | Path | None = None, *, model_id: str | None = None) -> TargetModelProfile:
    """Read a flat YAML profile, or resolve a compatible model-ID default."""

    if path is not None:
        if model_id is not None:
            raise ValueError("choose a model profile or a model ID, not both")
        with Path(path).open(encoding="utf-8") as source:
            return profile_from_dict(yaml.safe_load(source))
    return default_target_profile(model_id)


def resolve_attribute(root: object, path: str) -> object:
    """Resolve configured attributes without evaluating Python expressions."""

    _validate_path(path)
    result = root
    for part in path.split("."):
        try:
            result = getattr(result, part)
        except AttributeError as error:
            raise ValueError(f"model does not expose configured path {path!r}") from error
    return result


def model_tokenizer(model: object) -> object:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model must expose tokenizer or processor.tokenizer")
    return tokenizer


def get_text_config(model_or_config: object) -> object:
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def get_model_layers(model: object, profile: TargetModelProfile) -> object:
    layers = resolve_attribute(model, profile.layers_path)
    if profile.expected_layers is not None and len(layers) != profile.expected_layers:
        raise ValueError("configured expected_layers does not match the loaded model")
    if profile.expected_hidden_size is not None and get_text_config(model).hidden_size != profile.expected_hidden_size:
        raise ValueError("configured expected_hidden_size does not match the loaded model")
    return layers


def layer_output(output: object, profile: TargetModelProfile) -> object:
    return output if profile.layer_output_index is None else output[profile.layer_output_index]


def prepare_profile_messages(messages: list[dict[str, str]], profile: TargetModelProfile) -> list[dict[str, str]]:
    """Apply the explicit chat-role policy without mutating canonical messages."""

    if profile.system_role == "native":
        return messages
    system_text = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    prepared = [dict(message) for message in messages if message["role"] != "system"]
    if system_text:
        for message in prepared:
            if message["role"] == "user":
                message["content"] = f"{system_text}\n\n{message['content']}"
                break
        else:
            prepared.insert(0, {"role": "user", "content": system_text})
    return prepared


def load_hf_target(profile: TargetModelProfile, **kwargs: object) -> object:
    """Load a configured local model when explicitly invoked by a research tool."""

    import torch
    import transformers

    loader = getattr(transformers, profile.loader, None)
    if loader is None:
        raise ValueError(f"installed Transformers does not provide {profile.loader}; use a compatible environment")
    options: dict[str, object] = {"torch_dtype": getattr(torch, profile.dtype)}
    if profile.revision is not None:
        options["revision"] = profile.revision
    options.update(kwargs)
    return loader.from_pretrained(profile.model_id, **options)
