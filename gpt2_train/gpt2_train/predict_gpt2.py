"""
GPT-2 Text Generation / Prediction Script

Load a trained model checkpoint (or pretrained weights) and generate text completions.

Usage Examples:
    # 1. Predict from a checkpoint file
    uv run python -m gpt2_train.predict_gpt2 --checkpoint gpt2_train/training_logs/checkpoint_2026-08-18T18:01:54IST/model_05000.pt

    # 2. Custom prompt with 3 samples and temperature 0.8
    uv run python -m gpt2_train.predict_gpt2 --checkpoint /path/to/model_05000.pt --prompt "The secret to life is" -n 3 -t 0.8

    # 3. Greedy deterministic decoding
    uv run python -m gpt2_train.predict_gpt2 --checkpoint /path/to/model.pt --prompt "In a galaxy far far away" --greedy

    # 4. Interactive REPL mode
    uv run python -m gpt2_train.predict_gpt2 --checkpoint /path/to/model.pt --interactive

    # 5. Using standard pretrained weights (without a local checkpoint)
    uv run python -m gpt2_train.predict_gpt2 --from_pretrained gpt2 --prompt "Once upon a time,"
"""

import argparse
import contextlib
import os
from pathlib import Path
import sys
import time
import typing as t

import tiktoken
import torch
import torch.nn.functional as F

# Ensure package root is in sys.path for standalone script execution
current_file = Path(__file__).resolve()
package_root = current_file.parent.parent  # contains the `gpt2_train` directory
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from gpt2_train.train_gpt2 import GPT, GPTConfig
    from gpt2_train.dataset_helpers import get_checkpoint_dir
except ImportError:
    from .train_gpt2 import GPT, GPTConfig
    from .dataset_helpers import get_checkpoint_dir

# Fix for checkpoints serialized where GPTConfig was in __main__
sys.modules["__main__"].GPTConfig = GPTConfig
sys.modules["__main__"].GPT = GPT


def get_device(device_arg: str = "auto") -> tuple[str, str]:
    """Resolve target torch device and device type string."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_type = "mps"
        else:
            device = "cpu"
            device_type = "cpu"
    else:
        device = device_arg
        device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
    return device, device_type


def get_torch_dtype(dtype_arg: str, device_type: str) -> torch.dtype:
    """Resolve torch dtype based on CLI arg and hardware support."""
    if dtype_arg == "auto":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif device_type == "cuda":
            return torch.float16
        return torch.float32
    elif dtype_arg == "bfloat16":
        return torch.bfloat16
    elif dtype_arg == "float16":
        return torch.float16
    elif dtype_arg == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def find_latest_checkpoint(checkpoint_path: Path | str | None = None) -> Path | None:
    """Find checkpoint file by path or search the default checkpoint directory."""
    if checkpoint_path:
        p = Path(checkpoint_path)
        if p.is_file():
            return p
        if p.is_dir():
            ckpts = sorted(p.glob("*.pt"), key=os.path.getmtime)
            if ckpts:
                return ckpts[-1]
        return None

    try:
        ck_dir = get_checkpoint_dir()
        ckpts = sorted(ck_dir.glob("*.pt"), key=os.path.getmtime)
        if ckpts:
            return ckpts[-1]
    except Exception:
        pass
    return None


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    config_overrides: dict[str, t.Any] | None = None,
) -> tuple[GPT, dict[str, t.Any]]:
    """
    Load GPT model and metadata from a saved .pt checkpoint.
    Handles DDP / torch.compile prefixes and custom configs.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Ensure __main__ has GPTConfig mapping for pickle deserialization
    sys.modules["__main__"].GPTConfig = GPTConfig
    sys.modules["__main__"].GPT = GPT

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dictionary
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean wrapper prefixes from state_dict keys
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[len("_orig_mod."):]
        if clean_k.startswith("module."):
            clean_k = clean_k[len("module."):]
        clean_state_dict[clean_k] = v

    # Extract or infer GPTConfig
    config = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        if isinstance(cfg, GPTConfig):
            config = cfg
        elif isinstance(cfg, dict):
            config = GPTConfig(**cfg)
        elif hasattr(cfg, "__dict__"):
            valid_fields = {
                f: getattr(cfg, f)
                for f in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd"]
                if hasattr(cfg, f)
            }
            config = GPTConfig(**valid_fields)

    if config is None:
        # Infer configuration from weights shape
        vocab_size = 50304
        block_size = 1024
        n_layer = 12
        n_head = 12
        n_embd = 768

        if "transformer.wte.weight" in clean_state_dict:
            vocab_size, n_embd = clean_state_dict["transformer.wte.weight"].shape
        if "transformer.wpe.weight" in clean_state_dict:
            block_size, _ = clean_state_dict["transformer.wpe.weight"].shape

        layer_indices = set()
        for k in clean_state_dict:
            if k.startswith("transformer.h."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
        if layer_indices:
            n_layer = max(layer_indices) + 1

        config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

    # Apply manual configuration overrides if specified
    if config_overrides:
        for k, val in config_overrides.items():
            if val is not None and hasattr(config, k):
                setattr(config, k, val)

    model = GPT(config)

    # Load matching weights, filtering out predetermined attention buffers
    model_sd = model.state_dict()
    filtered_sd = {}
    for k, v in clean_state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                filtered_sd[k] = v
            else:
                print(f"[Warning] Shape mismatch for {k}: model {model_sd[k].shape} vs ckpt {v.shape}. Skipped.")
        elif not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias"):
            print(f"[Warning] Unexpected parameter in checkpoint: {k}. Skipped.")

    model.load_state_dict(filtered_sd, strict=False)
    model.to(device)
    model.eval()

    # Extract optional training metadata
    meta = {}
    if isinstance(checkpoint, dict):
        for meta_key in ["step", "val_loss", "acc_norm", "iter", "epoch"]:
            if meta_key in checkpoint:
                meta[meta_key] = checkpoint[meta_key]

    return model, meta


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample next token index from model logits with temperature, top-k, and top-p (nucleus) filtering.
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = -float("Inf")

    # Top-P (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens exceeding cumulative probability
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    if generator is not None:
        return torch.multinomial(probs, num_samples=1, generator=generator)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: GPT,
    prompt: str = "Hello, I'm a language model,",
    num_samples: int = 4,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = 42,
    device: str = "cpu",
    device_type: str = "cpu",
    dtype: torch.dtype = torch.float32,
    stop_on_eos: bool = True,
    stream: bool = False,
) -> list[str]:
    """
    Generate completions for a given prompt using GPT model.

    Args:
        model: Loaded GPT model instance.
        prompt: Initial input prompt string.
        num_samples: Number of completion sequences to generate.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (<= 0.0 for greedy argmax).
        top_k: Top-k filtering threshold (None or <= 0 to disable).
        top_p: Nucleus top-p filtering threshold (None or >= 1.0 to disable).
        repetition_penalty: Penalty multiplier for previously generated tokens (1.0 = none).
        seed: Random seed for reproducibility.
        device: PyTorch device ('cuda', 'cpu', 'mps').
        device_type: Device category for torch.autocast.
        dtype: PyTorch precision dtype for forward passes.
        stop_on_eos: Whether to terminate generation on `<|endoftext|>`.
        stream: Stream tokens to stdout in real-time (effective when num_samples == 1).

    Returns:
        List of generated complete strings (prompt + generated completion).
    """
    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = enc.eot_token  # 50256 for <|endoftext|>

    prompt_tokens = enc.encode(prompt)
    if len(prompt_tokens) == 0:
        prompt_tokens = [eos_token_id]

    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    x = tokens.unsqueeze(0).repeat(num_samples, 1)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    model.eval()

    autocast_context = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type in ["cuda", "cpu"] and dtype in [torch.bfloat16, torch.float16]
        else contextlib.nullcontext()
    )

    if stream and num_samples == 1:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]

            with autocast_context:
                logits, _ = model(x_cond)

            logits = logits[:, -1, :]  # Shape: (num_samples, vocab_size)

            # Apply repetition penalty if enabled
            if repetition_penalty != 1.0:
                for b in range(num_samples):
                    for token_id in set(x[b].tolist()):
                        if logits[b, token_id] < 0:
                            logits[b, token_id] *= repetition_penalty
                        else:
                            logits[b, token_id] /= repetition_penalty

            next_token = sample_next_token(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=rng,
            )

            # If sample already finished on EOS, preserve EOS
            if stop_on_eos:
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_token_id), next_token)
                finished = finished | (next_token.squeeze(1) == eos_token_id)

            x = torch.cat((x, next_token), dim=1)

            if stream and num_samples == 1:
                token_item = next_token[0, 0].item()
                if stop_on_eos and token_item == eos_token_id:
                    break
                sys.stdout.write(enc.decode([token_item]))
                sys.stdout.flush()

            if stop_on_eos and finished.all():
                break

    if stream and num_samples == 1:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Decode completions
    results = []
    for i in range(num_samples):
        seq_tokens = x[i].tolist()
        # Truncate at EOS if stop_on_eos is active
        if stop_on_eos:
            prompt_len = len(prompt_tokens)
            gen_tokens = seq_tokens[prompt_len:]
            if eos_token_id in gen_tokens:
                eos_idx = gen_tokens.index(eos_token_id)
                seq_tokens = seq_tokens[: prompt_len + eos_idx]
        decoded_str = enc.decode(seq_tokens)
        results.append(decoded_str)

    return results


def run_interactive_mode(model: GPT, args: argparse.Namespace, device: str, device_type: str, dtype: torch.dtype):
    """Interactive command-line REPL for text prediction."""
    print("\n" + "=" * 60)
    print("GPT-2 Interactive Inference Mode")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to end.")
    print("=" * 60 + "\n")

    temp = 0.0 if args.greedy else args.temperature

    while True:
        try:
            prompt = input("\nPrompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break

            print("-" * 50)
            t0 = time.time()
            completions = generate(
                model=model,
                prompt=prompt,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                device=device,
                device_type=device_type,
                dtype=dtype,
                stop_on_eos=not args.no_stop_eos,
                stream=args.stream and args.num_samples == 1,
            )
            t1 = time.time()

            if not (args.stream and args.num_samples == 1):
                for idx, completion in enumerate(completions, 1):
                    if args.num_samples > 1:
                        print(f"\n[Sample {idx}/{args.num_samples}]")
                    print(completion)

            elapsed = t1 - t0
            total_tokens = sum(len(tiktoken.get_encoding("gpt2").encode(c)) for c in completions)
            print(f"\n[Time: {elapsed:.2f}s | Speed: {total_tokens / max(elapsed, 1e-5):.1f} tokens/s]")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict / generate sentences using a GPT-2 checkpoint or pretrained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & Checkpoint sources
    model_group = parser.add_argument_group("Model & Checkpoint Options")
    model_group.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to .pt checkpoint file or directory containing checkpoints.",
    )
    model_group.add_argument(
        "--from_pretrained",
        type=str,
        default=None,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Load standard pretrained HuggingFace GPT-2 weights instead of a custom checkpoint.",
    )

    # Prompt & Interaction
    prompt_group = parser.add_argument_group("Prompt Options")
    prompt_group.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Hello, I'm a language model,",
        help="Input prompt text to condition sentence generation.",
    )
    prompt_group.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing prompt(s), one per line.",
    )
    prompt_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive REPL mode where prompts can be typed dynamically.",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Sampling & Generation Parameters")
    gen_group.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=4,
        help="Number of independent sentence completions to generate.",
    )
    gen_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate per completion.",
    )
    gen_group.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Total maximum sequence length (prompt + completion). Overrides max_new_tokens if specified.",
    )
    gen_group.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random, lower = more conservative). Use 0 for greedy decoding.",
    )
    gen_group.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic greedy decoding (equivalent to --temperature 0).",
    )
    gen_group.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=50,
        help="Top-k sampling threshold (keep only top-k most likely tokens; set 0 to disable).",
    )
    gen_group.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (e.g. 0.9; keep tokens with cumulative prob <= p).",
    )
    gen_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty factor (> 1.0 penalizes repeating previous tokens).",
    )
    gen_group.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    gen_group.add_argument(
        "--no_stop_eos",
        action="store_true",
        help="Do not stop generation when encountering the end-of-text token.",
    )
    gen_group.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout in real-time (effective when num_samples=1).",
    )

    # Runtime & Hardware options
    hw_group = parser.add_argument_group("Hardware & Execution Options")
    hw_group.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Device to run inference on ('cuda', 'cuda:0', 'cpu', 'mps', or 'auto').",
    )
    hw_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Data type precision for inference forward pass.",
    )
    hw_group.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile to the model for faster generation throughput.",
    )

    # Model architecture override options
    arch_group = parser.add_argument_group("Model Architecture Overrides (Optional)")
    arch_group.add_argument("--vocab_size", type=int, default=None, help="Override vocabulary size.")
    arch_group.add_argument("--block_size", type=int, default=None, help="Override block size (context length).")
    arch_group.add_argument("--n_layer", type=int, default=None, help="Override number of transformer layers.")
    arch_group.add_argument("--n_head", type=int, default=None, help="Override number of attention heads.")
    arch_group.add_argument("--n_embd", type=int, default=None, help="Override embedding dimension.")

    return parser.parse_args()


def main():
    args = parse_args()

    device, device_type = get_device(args.device)
    dtype = get_torch_dtype(args.dtype, device_type)

    print(f"[*] Target Device: {device} ({device_type}) | Precision: {dtype}")

    # Resolve model checkpoint or pretrained source
    model = None
    meta = {}

    if args.from_pretrained:
        print(f"[*] Loading pretrained weights: {args.from_pretrained}")
        model = GPT.from_pretrained(args.from_pretrained)
        model.to(device)
        model.eval()
    else:
        ckpt_path = find_latest_checkpoint(args.checkpoint)
        if ckpt_path is None:
            if args.checkpoint:
                print(f"[Error] Checkpoint not found at: {args.checkpoint}")
            else:
                print("[Error] No checkpoint specified and none found in default checkpoint directory.")
                print("Please provide a checkpoint with `--checkpoint path/to/model.pt` or use `--from_pretrained gpt2`.")
            sys.exit(1)

        print(f"[*] Loading checkpoint from: {ckpt_path}")
        config_overrides = {
            "vocab_size": args.vocab_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
        }
        model, meta = load_model_from_checkpoint(ckpt_path, device=device, config_overrides=config_overrides)

        if meta:
            meta_info = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in meta.items())
            print(f"[*] Checkpoint metadata: ({meta_info})")

    print(
        f"[*] Model Architecture: {model.config.n_layer} layers, {model.config.n_head} heads, "
        f"{model.config.n_embd} embd, {model.config.vocab_size} vocab size, {model.config.block_size} block size"
    )

    if args.compile:
        print("[*] Compiling model with torch.compile...")
        model = torch.compile(model)

    if args.interactive:
        run_interactive_mode(model, args, device, device_type, dtype)
        return

    # Collect prompts to evaluate
    prompts = []
    if args.prompt_file:
        prompt_file = Path(args.prompt_file)
        if not prompt_file.is_file():
            print(f"[Error] Prompt file not found: {prompt_file}")
            sys.exit(1)
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    prompts.append(line_str)
        print(f"[*] Loaded {len(prompts)} prompts from {prompt_file}")
    else:
        prompts = [args.prompt]

    # Calculate max_new_tokens
    enc = tiktoken.get_encoding("gpt2")
    temp = 0.0 if args.greedy else args.temperature

    top_k = None if args.top_k <= 0 else args.top_k
    top_p = None if (args.top_p is None or args.top_p >= 1.0 or args.top_p <= 0.0) else args.top_p

    print("\n" + "=" * 60)
    print(f"Generation Settings:")
    print(f"  Samples per prompt: {args.num_samples}")
    print(f"  Temperature:        {temp:.2f} ({'Greedy' if temp == 0.0 else 'Sampling'})")
    print(f"  Top-K:              {top_k}")
    print(f"  Top-P:              {top_p}")
    print(f"  Repetition Penalty: {args.repetition_penalty}")
    print(f"  Seed:               {args.seed}")
    print("=" * 60)

    for p_idx, prompt in enumerate(prompts, 1):
        prompt_token_count = len(enc.encode(prompt))
        if args.max_length is not None:
            max_new_tokens = max(1, args.max_length - prompt_token_count)
        else:
            max_new_tokens = args.max_new_tokens

        if len(prompts) > 1:
            print(f"\n>>> [Prompt {p_idx}/{len(prompts)}]: \"{prompt}\"")
        else:
            print(f"\n>>> Prompt: \"{prompt}\"")

        t0 = time.time()
        completions = generate(
            model=model,
            prompt=prompt,
            num_samples=args.num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            device=device,
            device_type=device_type,
            dtype=dtype,
            stop_on_eos=not args.no_stop_eos,
            stream=args.stream and args.num_samples == 1,
        )
        t1 = time.time()

        if not (args.stream and args.num_samples == 1):
            for s_idx, completion in enumerate(completions, 1):
                if args.num_samples > 1:
                    print(f"\n--- Output Sample {s_idx}/{args.num_samples} ---")
                else:
                    print(f"\n--- Generated Completion ---")
                print(completion)

        elapsed = t1 - t0
        total_tokens = sum(len(enc.encode(c)) for c in completions)
        print(f"\n[Generated {total_tokens} total tokens in {elapsed:.2f}s | Throughput: {total_tokens / max(elapsed, 1e-5):.1f} tok/s]")

    print("\n" + "=" * 60)
    print("Generation complete.")


if __name__ == "__main__":
    main()
