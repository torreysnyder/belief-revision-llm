"""Remote language-model inference through NDIF."""

import os
import time
import random
import anyio.streams.tls  # noqa: F401
from .models import (
    ActivationCapture,
    ActivationCaptureRequest,
    ModelGeneration,
)
from dotenv import load_dotenv
from functools import lru_cache
from nnsight import CONFIG, LanguageModel
from nnsight import save as nnsight_save
from openai import OpenAI

load_dotenv()


# TEMPORARY GEMMA SMOKE-TEST SUPPORT: remove this compatibility block after
# Pilot access allows the configured research models to run on NDIF.
GEMMA_MODEL_PREFIX = "google/gemma-"

# TEMPORARY FULL-FLOW SMOKE TEST: cap every model response to reduce NDIF GPU
# memory use while validating mechanics. Remove after the first complete run.
# SMOKE_TEST_MAX_TOKENS = 128
# SMOKE_TEST_MAX_PROMPT_CHARS = 6000


def prepare_messages_for_model(
    model_name: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Adapt chat roles only for models with known template restrictions."""

    if not model_name.startswith(GEMMA_MODEL_PREFIX):
        return messages

    # TEMPORARY GEMMA SMOKE-TEST SUPPORT: Gemma's chat template rejects the
    # system role, so preserve that content inside the first user message.
    system_text = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    compatible_messages = [message.copy() for message in messages if message["role"] != "system"]

    if not system_text:
        return compatible_messages

    for message in compatible_messages:
        if message["role"] == "user":
            message["content"] = f"{system_text}\n\n{message['content']}"
            return compatible_messages

    return [{"role": "user", "content": system_text}, *compatible_messages]


def require_ndif_key() -> None:
    """Raise a clear error when NDIF authentication is not configured."""

    api_key = os.getenv("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError("NDIF_API_KEY is missing. Add it to the local .env file.")

    # nnsight reads the key once at import, before load_dotenv() has run here.
    CONFIG.API.APIKEY = api_key


@lru_cache(maxsize=None)
def get_model(model_name: str) -> LanguageModel:
    """Create and cache one lightweight NDIF model handle per model ID."""

    require_ndif_key()
    return LanguageModel(model_name)


def build_chat_prompt(
    model: LanguageModel,
    model_name: str,
    messages: list[dict[str, str]],
) -> str:
    """Format messages using a native template or a base-model fallback."""

    messages = prepare_messages_for_model(model_name, messages)

    if model.tokenizer.chat_template:
        prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_messages = [f"{message['role'].upper()}:\n{message['content']}" for message in messages]
        prompt = "\n\n".join(formatted_messages) + "\n\nASSISTANT:\n"

    # TEMPORARY FULL-FLOW SMOKE TEST: accumulated dialogue exceeds NDIF's
    # per-job attention-memory limit. Keep the instructions and newest context.
    #    if len(prompt) > SMOKE_TEST_MAX_PROMPT_CHARS:
    #        half_limit = SMOKE_TEST_MAX_PROMPT_CHARS // 2
    #        prompt = (
    #            prompt[:half_limit]
    #            + "\n\n[OLDER CONTEXT OMITTED FOR SMOKE TEST]\n\n"
    #            + prompt[-half_limit:]
    #        )

    return prompt


def generate_remote(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 500,
    capture_requests: tuple[ActivationCaptureRequest, ...] = (),
) -> ModelGeneration:
    """Generate one response using a remotely hosted NDIF model."""

    model = get_model(model_name)
    prompt = build_chat_prompt(model, model_name, messages)

    # TEMPORARY FULL-FLOW SMOKE TEST: this deliberately overrides larger
    # per-call limits without changing the intended experiment settings.
    #    max_tokens = min(max_tokens, SMOKE_TEST_MAX_TOKENS)

    prompt_length = model.tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.shape[-1]

    generation_options = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }

    if temperature > 0:
        generation_options["temperature"] = temperature

    layer_count = len(model.model.layers)
    capture_specs: list[tuple[ActivationCaptureRequest, int]] = []
    for request in capture_requests:
        layers = range(layer_count) if request.layers is None else request.layers
        for layer in layers:
            if layer >= layer_count:
                raise ValueError(f"capture layer {layer} is out of bounds for model with {layer_count} layers")
            capture_specs.append((request, layer))

    saved_activations = []
    with model.generate(
        prompt,
        remote=True,
        **generation_options,
    ) as tracer:
        if capture_specs:
            saved_activations = nnsight_save([])
        for _ in tracer.iter[0]:
            for _, layer in capture_specs:
                saved_activations.append(model.model.layers[layer].output[0, -1, :].float().detach().cpu())
            output_tokens = tracer.result.save()

    generated_tokens = output_tokens[0, prompt_length:]

    text = model.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
    activations = tuple(
        ActivationCapture(
            request_id=request.request_id,
            layer=layer,
            model_name=model_name,
            location=request.location,
            token_position=request.token_position,
            token_index=prompt_length - 1,
            sequence_length=prompt_length,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            tensor=tensor,
        )
        for (request, layer), tensor in zip(capture_specs, saved_activations, strict=True)
    )
    return ModelGeneration(text=text, activations=activations)


NON_RETRYABLE_ERRORS = (
    "not pinned",
    "pilot only",
    "unauthorized",
    "incompatible with the server",
    "gated repo",
    "chat template",
)

API_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/"),
    "google": (
        "GOOGLE_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


def is_retryable_error(error: Exception) -> bool:
    """Return whether an NDIF failure may succeed on another attempt."""

    message = str(error).lower()

    return not any(marker in message for marker in NON_RETRYABLE_ERRORS)


def call_model(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 500,
    retries: int = 5,
    capture_requests: tuple[ActivationCaptureRequest, ...] = (),
    provider: str = "ndif",
) -> ModelGeneration:
    """Route generation to NDIF or a configured chat-compatible API."""

    if provider != "ndif":
        if capture_requests:
            raise ValueError("activation capture is only supported through NDIF")
        return generate_api(provider, model_name, messages, max_tokens, retries)

    for attempt in range(retries):
        try:
            return generate_remote(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                capture_requests=capture_requests,
            )
        except Exception as error:
            if not is_retryable_error(error) or attempt == retries - 1:
                raise

            delay = (2**attempt) + random.random()
            time.sleep(delay)

    raise RuntimeError("NDIF generation failed without returning a response.")


def generate_api(
    provider: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    retries: int,
) -> ModelGeneration:
    """Generate through an official chat-compatible provider API."""

    if provider not in API_PROVIDERS:
        raise ValueError(f"unknown API provider: {provider}")
    key_name, base_url = API_PROVIDERS[provider]
    key = os.getenv(key_name)
    if not key:
        raise RuntimeError(f"{key_name} is missing. Add it to the local .env file.")
    if retries < 1:
        raise ValueError("retries must be at least 1")

    with OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=120.0,
        max_retries=retries - 1,
    ) as client:
        token_parameter = "max_completion_tokens" if provider == "openai" else "max_tokens"
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **{token_parameter: max_tokens},
        )

    if not response.choices:
        raise RuntimeError(f"{provider} returned no completion choices")
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("evaluator response was truncated; increase EVALUATOR_MAX_TOKENS")
    content = choice.message.content
    if not content or not content.strip():
        raise RuntimeError(f"{provider} returned an empty response")
    return ModelGeneration(text=content.strip())
