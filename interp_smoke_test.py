"""Check that the interpretability environment can load a model and cache activations.

Usage:
    uv run python interp_smoke_test.py [MODEL_NAME]

The default model is gated on Hugging Face: request access on the model page
and set HF_TOKEN in .env before running. Any ungated model (for example
Qwen/Qwen2.5-0.5B-Instruct) works for checking the environment itself.
"""

import sys

import torch
from dotenv import load_dotenv
from transformer_lens.model_bridge import TransformerBridge

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    load_dotenv()
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = TransformerBridge.boot_transformers(model_name, device=device, dtype=dtype)
    print(f"Loaded {model_name}: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # Format as a chat so we probe the assistant, not the base-model continuation.
    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.to_tokens(prompt, prepend_bos=False)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    next_token = model.to_string(logits[0, -1].argmax())
    print(f"Next-token prediction: {next_token!r}")

    mid_layer = model.cfg.n_layers // 2
    resid = cache[f"blocks.{mid_layer}.hook_resid_post"]
    print(f"blocks.{mid_layer}.hook_resid_post shape: {tuple(resid.shape)}  # [batch, seq, d_model]")

    if device == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Environment OK.")


if __name__ == "__main__":
    main()
