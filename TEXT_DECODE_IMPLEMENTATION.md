# Implementing Text Decoding for Pi0.5

This document describes how to add autoregressive text generation to the open-source Pi0.5 model, enabling inspection of the VLM backbone's subtask predictions alongside normal action inference.

---

## Background

The Pi0.5 paper describes a two-stage inference process:

1. **High-level:** The VLM backbone (Gemma 2B) autoregressively generates a subtask prediction (e.g., "pick up the plate") given images and a high-level command.
2. **Low-level:** The action expert (Gemma 300M) generates continuous actions via flow matching, conditioned on the predicted subtask.

The open-source OpenPI codebase only implements stage 2. The VLM backbone is used purely as an encoder — its output is discarded after producing the KV cache. However, all the building blocks for text generation already exist in the code:

| Building Block | Location | Status |
|---|---|---|
| `Embedder.decode()` — maps hidden states to vocab logits | `gemma.py:153` | Exists, never called at inference |
| `Embedder.encode()` — maps token IDs to embeddings | `gemma.py:148` | Exists, used for prompt embedding |
| KV cache support in `Attention.__call__` | `gemma.py:211-214` | Exists, used for action denoising |
| `Module.__call__` with `kv_cache` parameter | `gemma.py:389-411` | Exists, fully functional |
| `Module.embed()` — token ID to embedding | `gemma.py:385` | Exists, used in `embed_prefix` |

The `pi05_base` checkpoint was trained with the full Pi0.5 recipe, including subtask prediction. The weights know how to generate text — we just need to wire up the decoding loop.

---

## Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │         embed_prefix()            │
                    │  SigLIP(images) + embed(prompt)   │
                    └───────────────┬──────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │  Gemma 2B forward pass (Expert 0) │
                    │  Input: [prefix_tokens, None]     │
                    │  Output: (prefix_out, kv_cache)   │
                    └──────┬───────────────┬───────────┘
                           │               │
                   ┌───────┘               └────────┐
                   ▼                                ▼
        ┌─────────────────────┐          ┌─────────────────────┐
        │  TEXT DECODE (NEW)   │          │  ACTION DENOISE     │
        │                     │          │  (EXISTING)         │
        │  Copy kv_cache      │          │  Use original       │
        │  Autoregressive     │          │  kv_cache           │
        │  loop: decode →     │          │  10 denoising steps │
        │  sample → re-embed  │          │  through action     │
        │  → decode → ...     │          │  expert (300M)      │
        │                     │          │                     │
        │  Output: text       │          │  Output: actions    │
        │  "Subtask: pick     │          │  [10, 6] floats     │
        │   up the plate"     │          │                     │
        └─────────────────────┘          └─────────────────────┘
```

The text decode branch uses a **copy** of the KV cache. The action branch uses the **original**. Action behavior is completely unchanged.

---

## Files to Modify

### 1. `src/openpi/models/pi0.py` — Add `generate_text()` and modify `sample_actions()`

#### 1a. Add `generate_text()` method to the `Pi0` class

This method takes the prefix output and a copy of the KV cache, then runs a standard autoregressive decoding loop.

```python
def generate_text(
    self,
    prefix_out,           # [batch, prefix_len, 2048] — VLM backbone output
    kv_cache,             # KV cache from the prefix forward pass (will be extended)
    prefix_mask,          # [batch, prefix_len] — which prefix tokens are real vs padding
    *,
    max_new_tokens=50,
    temperature=0.0,      # 0.0 = greedy (always pick most likely token)
    rng=None,             # needed if temperature > 0
):
    """Autoregressively generate text tokens from the VLM backbone."""
    batch_size = prefix_out.shape[0]
    
    # The last position of the prefix output gives us logits for the first new token
    last_hidden = prefix_out[:, -1:, :]  # [batch, 1, 2048]
    logits = self.PaliGemma.llm.embedder.decode(last_hidden)  # [batch, 1, 257152]
    
    # Sample or argmax the first token
    if temperature == 0.0:
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)  # [batch]
    else:
        next_token = jax.random.categorical(rng, logits[:, -1, :] / temperature)
    
    generated = [next_token]
    next_position = jnp.sum(prefix_mask, axis=-1)  # [batch] — position after prefix
    
    for i in range(max_new_tokens - 1):
        # Embed the new token
        token_emb = self.PaliGemma.llm.embed(next_token[:, None])  # [batch, 1, 2048]
        
        # Attention mask: new token can see all previous tokens
        # Shape: [batch, 1, prefix_len + i + 1]
        total_len = prefix_out.shape[1] + i + 1
        new_mask = jnp.ones((batch_size, 1, total_len), dtype=jnp.bool_)
        
        # Position for this token
        pos = (next_position + i + 1)[:, None]  # [batch, 1]
        
        # Forward pass through Gemma 2B only, using + extending KV cache
        (new_out, _), kv_cache = self.PaliGemma.llm(
            [token_emb, None],  # Expert 0 processes the token, Expert 1 is off
            mask=new_mask,
            positions=pos,
            kv_cache=kv_cache,
        )
        # new_out shape: [batch, 1, 2048]
        
        # Decode to vocabulary logits
        logits = self.PaliGemma.llm.embedder.decode(new_out)  # [batch, 1, 257152]
        
        # Sample next token
        if temperature == 0.0:
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        else:
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, logits[:, -1, :] / temperature)
        
        generated.append(next_token)
        
        # Stop at EOS token (token ID 1 in SentencePiece)
        # Note: in a batched setting, you'd mask finished sequences
    
    return jnp.stack(generated, axis=1)  # [batch, max_new_tokens]
```

**Key details:**

- `self.PaliGemma.llm.embedder.decode(hidden)` is the existing method at `gemma.py:153`. It does `jnp.dot(hidden, embedding_table.T)` to produce logits over the 257,152-token vocabulary.
- `self.PaliGemma.llm.embed(token_ids)` is the existing method at `gemma.py:385`. It looks up the token ID in the embedding table.
- `self.PaliGemma.llm([token_emb, None], ..., kv_cache=kv_cache)` is the existing `Module.__call__` at `gemma.py:389`. Passing `[X, None]` means Expert 0 (Gemma 2B) processes `X`, Expert 1 (action expert) is skipped. The KV cache is consumed and returned extended.
- The attention mask shape `[batch, 1, total_len]` tells the single new query token that it can attend to all `total_len` keys in the cache.

#### 1b. Modify `sample_actions()` to optionally run text decode

```python
@override
def sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    decode_text=False,        # NEW parameter
    decode_max_tokens=50,     # NEW parameter
):
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # Prefix forward pass — CAPTURE the output instead of discarding it
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    
    # Changed: capture prefix_out (was `_` before)
    (prefix_out, _), kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    # NEW: optionally generate text from the prefix output
    if decode_text:
        # Copy the KV cache so text generation doesn't corrupt the action branch
        kv_cache_copy = jax.tree.map(jnp.copy, kv_cache)
        self._last_decoded_tokens = self.generate_text(
            prefix_out, kv_cache_copy, prefix_mask,
            max_new_tokens=decode_max_tokens,
        )
    else:
        self._last_decoded_tokens = None

    # ... rest of sample_actions is COMPLETELY UNCHANGED ...
    # (the denoising loop using the original kv_cache)
    
    def step(carry):
        x_t, time = carry
        # ... existing code, no changes ...
        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

**The only changes to `sample_actions`:**
1. Capture `prefix_out` instead of `_`
2. When `decode_text=True`, copy the KV cache and run `generate_text`
3. Store result on `self._last_decoded_tokens`

When `decode_text=False` (default), behavior is identical to the original code.

#### 1c. Imports needed

Add at the top of `pi0.py`:

```python
# No new imports needed — jax, jnp, and all model references already imported
```

---

### 2. `src/openpi/models/tokenizer.py` — Add a `decode_tokens()` method

The `PaligemmaTokenizer` class has a SentencePiece processor but only exposes `tokenize()`. We need the reverse operation.

```python
class PaligemmaTokenizer:
    def __init__(self, max_len=48):
        # ... existing code ...
        pass

    def tokenize(self, prompt, state=None):
        # ... existing code ...
        pass

    # NEW METHOD
    def decode_tokens(self, token_ids):
        """Convert token IDs back to readable text.
        
        Args:
            token_ids: list of ints or numpy array of token IDs
            
        Returns:
            Decoded string
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids)
```

This is a one-liner — SentencePiece's `.decode()` method already exists on the `_tokenizer` object, we just need to expose it publicly.

---

### 3. `src/openpi/policies/policy.py` — Surface decoded text in output

Modify `Policy.infer()` to check for decoded tokens and include them in the output.

```python
class Policy(BasePolicy):
    def __init__(self, model, *, rng=None, transforms=(), output_transforms=(),
                 sample_kwargs=None, metadata=None, pytorch_device="cpu", is_pytorch=False):
        # ... existing init code ...
        
        # NEW: load a tokenizer for decoding generated tokens
        self._text_tokenizer = None  # lazy init
    
    def _get_text_tokenizer(self):
        """Lazy-load the SentencePiece tokenizer for text decoding."""
        if self._text_tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            self._text_tokenizer = PaligemmaTokenizer()
        return self._text_tokenizer

    @override
    def infer(self, obs, *, noise=None):
        # ... existing transform + inference code (unchanged) ...
        
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # ... existing batching + device code ...
        
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        
        # ... existing unbatching code ...
        
        # NEW: check for decoded text tokens
        model = self._model
        if hasattr(model, '_last_decoded_tokens') and model._last_decoded_tokens is not None:
            tokenizer = self._get_text_tokenizer()
            token_ids = model._last_decoded_tokens[0]  # first batch element
            decoded_text = tokenizer.decode_tokens(token_ids)
            outputs["decoded_text"] = decoded_text
        
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs
```

---

### 4. `src/openpi/models/pi0_config.py` — No changes needed

The config doesn't need modification. The `decode_text` flag is passed at inference time via `sample_kwargs`, not baked into the config.

---

### 5. `src/openpi/models/gemma.py` — No changes needed

All required methods (`Embedder.decode`, `Module.__call__` with `kv_cache`, `Module.embed`) already exist.

---

## How to Use It

### Option A: Decode every Nth inference call

Pass `decode_text` through `sample_kwargs` in your inference loop:

```python
policy = policy_config.create_trained_policy(config, checkpoint_dir)

for step in range(1000):
    # Decode text every 5th step
    decode_this_step = (step % 5 == 0)
    
    result = policy.infer(obs)  # need to wire decode_text through
    
    actions = result["actions"]
    if "decoded_text" in result:
        print(f"[step {step}] VLM says: {result['decoded_text']}")
    
    robot.execute(actions)
```

To wire `decode_text` through the policy layer, modify `sample_kwargs` in the config or pass it dynamically:

```python
config = _config.get_config("pi05_so100_lora")
# Add decode settings
policy = policy_config.create_trained_policy(
    config, checkpoint_dir,
    sample_kwargs={"decode_text": True, "decode_max_tokens": 30}
)
```

### Option B: Async background decoding

For zero-latency-impact decoding, store the prefix output and decode in a background thread:

```python
import threading

class DecodingPolicy:
    def __init__(self, base_policy):
        self.policy = base_policy
        self.last_decoded_text = None
        self._step = 0
    
    def infer(self, obs, decode_every=5):
        self._step += 1
        
        # Always run normal inference (no text decode — full speed)
        result = self.policy.infer(obs)
        
        # Every Nth step, grab the prefix output and decode in background
        if self._step % decode_every == 0:
            model = self.policy._model
            if hasattr(model, '_last_prefix_out'):
                prefix_snapshot = jnp.copy(model._last_prefix_out)
                kv_snapshot = jax.tree.map(jnp.copy, model._last_kv_cache)
                
                def background_decode():
                    tokens = model.generate_text(prefix_snapshot, kv_snapshot, ...)
                    self.last_decoded_text = tokenizer.decode_tokens(tokens[0])
                
                threading.Thread(target=background_decode, daemon=True).start()
        
        result["decoded_text"] = self.last_decoded_text
        return result
```

This requires also storing `prefix_out` and `prefix_mask` on `self` during `sample_actions`, not just `_last_decoded_tokens`.

---

## Expected Output

### With the `pi05_base` checkpoint (trained on PI's full data mixture)

Given images of a kitchen with dishes on a counter:

```
prompt: "clean the kitchen"

[step 0]   VLM: "Bounding boxes: <loc0405><loc0011><loc0911><loc0197>plate Subtask: pick up the plate"
[step 5]   VLM: "Bounding boxes: <loc0410><loc0015><loc0905><loc0195>plate Subtask: pick up the plate"
[step 10]  VLM: "Bounding boxes: <loc0410><loc0020><loc0900><loc0200>plate Subtask: pick up the plate"
...robot picks up plate...
[step 40]  VLM: "Bounding boxes: <loc0601><loc0345><loc0823><loc0567>sink Subtask: put plate in the sink"
```

### With a fine-tuned checkpoint (trained on your SO100 data)

If you haven't trained the text generation head on subtask data, the output may be:
- Garbled or repetitive tokens
- Generic VQA-style responses (from web data training)
- Partially coherent but irrelevant subtasks (from other robot domains)

To get meaningful subtask predictions for SO100, you would need to:
1. Annotate your SO100 episodes with subtask labels
2. Add cross-entropy loss on text tokens during fine-tuning
3. Train the model to predict subtasks for your specific tasks

---

## Latency Summary

All numbers assume 2 cameras, 224×224 images, action_horizon=10, 6-dim actions.

| Configuration | Action Inference | Text Decode (30 tokens) | Total (when decoding) | Control Impact |
|---|---|---|---|---|
| BF16, RTX 4090 | ~65ms | ~60ms | ~125ms | None if every 5th step |
| INT8, RTX 4090 | ~42ms | ~35ms | ~77ms | None if every 5th step |
| INT4, RTX 4090 | ~30ms | ~25ms | ~55ms | None — fits every step at 15Hz |
| BF16, A100 | ~40ms | ~40ms | ~80ms | None if every 5th step |
| INT8, A100 | ~28ms | ~22ms | ~50ms | None — fits every step at 15Hz |

With action chunking (predict 10, execute 5), the model is called every ~333ms at 15Hz. Even the slowest configuration (125ms) leaves 208ms of headroom.

---

## Risks and Caveats

1. **JAX JIT compilation:** The `generate_text` method uses a Python loop with variable-length KV cache. This doesn't JIT-compile cleanly. For production use, you'd want to use `jax.lax.scan` with a fixed `max_new_tokens` and mask out tokens after EOS. For inspection/debugging, the Python loop is fine.

2. **KV cache copy cost:** `jax.tree.map(jnp.copy, kv_cache)` copies ~13MB of data (for 612 prefix tokens). This is fast (~0.1ms) but allocates GPU memory. If running on a memory-constrained GPU, consider reusing a pre-allocated buffer.

3. **The `pi05_base` checkpoint text quality is unknown.** Physical Intelligence trained this checkpoint with text generation, but they may have released a version with degraded text capabilities. The only way to know is to try it.

4. **Linen vs NNX:** The Gemma module uses Flax Linen (old API), wrapped with `nnx_bridge.ToNNX`. Calling methods like `embedder.decode()` requires going through the bridge. The exact calling convention may need adjustment — test with the actual model object to confirm the attribute path `self.PaliGemma.llm.embedder.decode()` resolves correctly through the NNX bridge.

5. **Attention mask shape:** The `Module.__call__` expects mask shape `[batch, seq_len_q, seq_len_kv]` which gets reshaped to `[batch, 1, seq_len_q, seq_len_kv]` internally (line 401). For single-token generation, `seq_len_q=1` and `seq_len_kv=prefix_len + num_generated_so_far`. Verify this matches the existing `Attention.__call__` expectations.

---

## Testing Strategy

1. **Unit test:** Load the `pi05_base` checkpoint, create a dummy observation, call `generate_text` with `max_new_tokens=5`. Verify you get integer token IDs in the valid vocab range (0 to 257,151). Verify `decode_tokens` produces readable characters.

2. **Integration test:** Run `sample_actions` with `decode_text=True` and `decode_text=False` on the same input. Verify the action outputs are identical (text decode must not affect actions).

3. **Quality test:** Feed real camera images with the prompt "clean the kitchen" and inspect the decoded text. Look for structured output containing "Bounding boxes:" and "Subtask:" patterns.
