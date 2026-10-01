"""Remote language-model inference through NDIF."""

from functools import lru_cache
import os
import random
import time

from dotenv import load_dotenv
from nnsight import LanguageModel


load_dotenv()


def require_ndif_key() -> None:
    """Raise a clear error when NDIF authentication is not configured."""

    if not os.getenv("NDIF_API_KEY"):
        raise RuntimeError(
            "NDIF_API_KEY is missing. Add it to the local .env file."
        )


@lru_cache(maxsize=None)
def get_model(model_name: str) -> LanguageModel:
    """Create and cache one lightweight NDIF model handle per model ID."""

    require_ndif_key()
    return LanguageModel(model_name)

def build_chat_prompt(
    model: LanguageModel,
    messages: list[dict[str, str]],
) -> str:
    """Format messages using a native template or a base-model fallback."""

    if model.tokenizer.chat_template:
        return model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted_messages = [
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ]

    return "\n\n".join(formatted_messages) + "\n\nASSISTANT:\n"

def generate_remote(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> str:
    """Generate one response using a remotely hosted NDIF model."""

    model = get_model(model_name)
    prompt = build_chat_prompt(model, messages)

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

    with model.generate(
            prompt,
            remote=True,
            **generation_options,
    ):
        output_tokens = model.generator.output.save()

    generated_tokens = output_tokens[0, prompt_length:]

    return model.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

NON_RETRYABLE_ERRORS = (
    "not pinned",
    "pilot only",
    "unauthorized",
    "incompatible with the server",
    "gated repo",
    "chat template",
)


def is_retryable_error(error: Exception) -> bool:
    """Return whether an NDIF failure may succeed on another attempt."""

    message = str(error).lower()

    return not any(
        marker in message
        for marker in NON_RETRYABLE_ERRORS
    )

def call_model(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 500,
    retries: int = 5,
) -> str:
    """Generate a response with retries for temporary NDIF failures."""

    for attempt in range(retries):
        try:
            return generate_remote(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            if not is_retryable_error(error) or attempt == retries - 1:
                raise

            delay = (2 ** attempt) + random.random()
            time.sleep(delay)

    raise RuntimeError("NDIF generation failed without returning a response.")