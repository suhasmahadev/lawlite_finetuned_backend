# summarizer/utils/ml.py
import os
import threading
import time
import requests

# transformers / torch imports happen lazily (only if local model used)
from .parsers import extract_text_and_pages

# Environment names
HF_TOKEN_ENV = "HUGGINGFACEHUB_API_TOKEN"
HF_LOCAL_DIR_ENV = "HF_LOCAL_MODEL_DIR"  # if set, use local model at this path
HF_MODEL_ID_ENV = "HF_MODEL_ID"          # fallback remote model id

# Remote default (keeps remote fallback working)
DEFAULT_REMOTE_MODEL = os.getenv(HF_MODEL_ID_ENV, "sshleifer/distilbart-cnn-12-6")

# Globals for local model (lazy loaded)
_LOCAL_LOCK = threading.Lock()
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_DEVICE = None
_LOCAL_MAX_CTX = None

###############################
# Local model loader / generator
###############################
def _load_local_model(local_dir: str):
    """
    Load tokenizer + model from a local directory. This is called once (lazy).
    Returns (model, tokenizer, device, max_context).
    May raise RuntimeError with actionable message.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_DEVICE, _LOCAL_MAX_CTX

    with _LOCAL_LOCK:
        if _LOCAL_MODEL is not None:
            return _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_DEVICE, _LOCAL_MAX_CTX

        # lazy import
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(f"Missing transformer/torch packages. Install transformers + torch. Error: {e}")

        model_path = local_dir
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(f"Local model path invalid or not found: '{model_path}'")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Strategy: try bitsandbytes 8-bit -> fp16 device_map auto -> CPU fallback
        load_kwargs = {"trust_remote_code": True}  # allow community models if needed

        model = None
        tokenizer = None

        # prepare tokenizer first
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        except Exception:
            # sometimes tokenizer files are missing fast tokenizer; try without use_fast
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ensure pad token is set
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # attempt loading model in order of preference
        last_exc = None
        try:
            # try 8-bit if bitsandbytes available and cuda is available
            if device == "cuda":
                try:
                    import bitsandbytes as bnb  # noqa: F401
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        load_in_8bit=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    last_exc = e
                    # try fp16 device_map auto
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            trust_remote_code=True
                        )
                    except Exception as e2:
                        last_exc = e2
                        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            else:
                # CPU fallback (very slow)
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load local model from {model_path}. Last error: {e}\nIf model is large, ensure you have torch + CUDA + bitsandbytes installed or try a smaller model.")

        # Put model in eval
        model.eval()

        # Determine model max context length (safe fallback)
        max_ctx = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_ctx", None) or 2048

        # Save globals
        _LOCAL_MODEL = model
        _LOCAL_TOKENIZER = tokenizer
        _LOCAL_DEVICE = device
        _LOCAL_MAX_CTX = int(max_ctx)

        return _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_DEVICE, _LOCAL_MAX_CTX

def _chunk_text_by_tokens(text: str, tokenizer, max_tokens: int, overlap: int = 128):
    """
    Return list of text chunks where each chunk token length <= max_tokens.
    Overlap tokens between chunks helps preserve context between chunks.
    """
    # encode full text to token ids
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    stride = max_tokens - overlap
    for i in range(0, len(token_ids), stride):
        chunk_ids = token_ids[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + max_tokens >= len(token_ids):
            break
    return chunks

def _generate_from_model(model, tokenizer, device, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0):
    """
    Generate text given prompt using the loaded local model. Returns generated string.
    Uses simple deterministic generation (no sampling) for stable summaries.
    """
    try:
        import torch
    except Exception:
        raise RuntimeError("torch not installed. Install torch to use local model.")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # move to device if model placed on cuda (some loaders used device_map)
    # If model is sharded with device_map='auto', model parameters may be on multiple devices
    # so we avoid moving model here. Instead move tensors to model.device if single device.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    except Exception:
        model_device = torch.device("cpu")

    # If model is on CPU but we have CUDA available and model expects cuda, this will be handled by device_map above.
    if model_device.type == "cuda":
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

    # Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            use_cache=True,
        )

    # Many causal models return full sequence (prompt + generated). Strip prompt tokens.
    out_ids = out[0]
    gen_ids = out_ids[len(input_ids[0]):] if out_ids.shape[0] >= len(input_ids[0]) else out_ids
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

def _summarize_with_local_model(text: str, local_dir: str = None):
    """
    Summarize large text using the local model:
      1) chunk text by tokenizer based on model context
      2) summarize each chunk
      3) combine chunk summaries and final-summarize
    """
    model_dir = local_dir or os.getenv(HF_LOCAL_DIR_ENV)
    if not model_dir:
        raise RuntimeError("No local model directory provided. Set HF_LOCAL_MODEL_DIR environment variable to your unzipped model path.")

    model, tokenizer, device, max_ctx = _load_local_model(model_dir)

    # choose conservative chunk size to leave room for generation tokens
    max_input_tokens = min(max_ctx - 256, 2048) if max_ctx and max_ctx > 512 else 1024
    overlap = min(256, max(128, int(max_input_tokens * 0.1)))

    chunks = _chunk_text_by_tokens(text, tokenizer, max_input_tokens, overlap=overlap)
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following text as concise bullet points (one or two short paragraphs), preserving key facts and legal conclusions:\n\n"
            f"{chunk}"
        )
        try:
            s = _generate_from_model(model, tokenizer, device, prompt, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"Local model generation failed on chunk {i}: {e}")
        partial_summaries.append(s or "")

    # If only one chunk, return it
    if len(partial_summaries) == 1:
        return partial_summaries[0].strip()

    # Combine partial summaries and then create a final summary
    combined = "\n\n".join([p for p in partial_summaries if p])
    final_prompt = (
        "You are given multiple partial summaries. Combine them into a single concise bullet-point summary that preserves key facts and legal conclusions.\n\n"
        f"{combined}"
    )
    final_summary = _generate_from_model(model, tokenizer, device, final_prompt, max_new_tokens=300, temperature=0.0)
    return final_summary.strip()

#######################################
# Remote HF inference fallback (keeps your previous behavior)
#######################################
def _call_hf_inference_remote(model_id: str, prompt: str, max_new_tokens: int = 256, max_length: int = 300, timeout: int = 120):
    token = os.getenv(HF_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Missing HF token in env var {HF_TOKEN_ENV}")

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payloads = [
        {"inputs": prompt, "parameters": {"max_length": max_length}, "options": {"wait_for_model": True}},
        {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}, "options": {"wait_for_model": True}},
        {"inputs": prompt, "options": {"wait_for_model": True}},
    ]

    last_error = None
    for payload in payloads:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            txt = e.response.text if e.response is not None else str(e)
            if status == 400:
                last_error = f"400 Bad Request for model {model_id}. HF response: {txt}"
                continue
            if status in (401, 403):
                raise RuntimeError(f"Hugging Face authentication/permission error ({status}): {txt}")
            if status == 404:
                raise RuntimeError(f"Model not found on Hugging Face: '{model_id}'. HF returned 404. Response: {txt}")
            if status in (429, 500, 502, 503):
                raise RuntimeError(f"Hugging Face server error ({status}): {txt}")
            raise RuntimeError(f"Hugging Face HTTP error ({status}): {txt}")
        except requests.RequestException as e:
            last_error = f"Network/request error when calling HF model {model_id}: {str(e)}"
            continue

    raise RuntimeError(f"Hugging Face inference failed for all tried payload formats. Last error: {last_error}")

def _parse_hf_result(result):
    # compatibility parser for remote responses
    if not result:
        return ""
    if isinstance(result, dict) and "choices" in result:
        # router/chat style
        try:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content") or ""
            if "text" in choice:
                return choice["text"]
        except Exception:
            pass
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            for k in ("summary_text", "generated_text", "text", "output_text", "content"):
                if k in first and first[k]:
                    return first[k]
            return str(first)
        else:
            return str(first)
    if isinstance(result, dict):
        for k in ("summary_text", "generated_text", "text", "output_text", "content"):
            if k in result and result[k]:
                return result[k]
        if "error" in result:
            raise RuntimeError(f"Hugging Face error: {result['error']}")
        return str(result)
    return str(result)

#######################################
# Public function kept identical signature
#######################################
def summarize_text(file_bytes: bytes, mimetype: str) -> str:
    """
    Extract the text from file_bytes, then try local model if available,
    otherwise fallback to remote Hugging Face.
    """
    # 1) Extract text
    text, pages = extract_text_and_pages(file_bytes, mimetype)
    if not (text and text.strip()):
        # fallback: try utf-8 decode
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raise RuntimeError("Could not extract text from uploaded file. Implement parsers.extract_text_and_pages.")

    # short-circuit empty
    if not text or not text.strip():
        raise RuntimeError("No textual content found in document.")

    # 2) If local model dir provided, use it
    local_dir = os.getenv(HF_LOCAL_DIR_ENV)
    if local_dir:
        try:
            return _summarize_with_local_model(text, local_dir=local_dir)
        except Exception as e:
            # surface error but allow remote fallback if token present
            last_err = str(e)
            # If no token provided, raise
            if not os.getenv(HF_TOKEN_ENV):
                raise RuntimeError(f"Local summarization failed and no HF token set for remote fallback. Local error: {last_err}")
            # otherwise log and fall through to remote
            # NOTE: in production you may prefer to fail hard
            print(f"[ml] Local summarization failed: {last_err}. Falling back to remote inference.")

    # 3) Remote fallback
    model_id = os.getenv(HF_MODEL_ID_ENV, DEFAULT_REMOTE_MODEL)
    # build a short prompt for seq2seq remote summarizers (they expect raw text)
    prompt = text[:2000]  # truncate to safe size for remote
    result = _call_hf_inference_remote(model_id, prompt, max_new_tokens=300, max_length=300, timeout=240)
    summary = _parse_hf_result(result)
    return summary.strip()
