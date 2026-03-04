# Control Loop Improvement Strategies for SO-100 + pi-0.5

## Current State (March 2026)

**Hardware:** Jetson AGX Thor, SO-100 arm (6-DOF, Feetech STS3215), 2x USB cameras  
**Model:** pi-0.5 LoRA, action_horizon=11, action_dim=6, 8 denoising steps  
**Inference latency:** ~355ms (warm, with torch.compile + CUDA 13.0 ptxas)  
**Control rate:** 30 Hz target (33ms per step)  
**Client script:** `examples/so100/run_policy.py`

### Current Control Loop (Synchronous)

```
while running:
    capture cameras            # ~5ms
    send obs, BLOCK for infer  # ~355ms  ← arm is FROZEN here
    for i in range(11):        # ~367ms  (11 steps × 33ms)
        apply action[i] to arm
        sleep to 30Hz
    # Total cycle: ~722ms, arm moves open-loop for last 367ms
```

**Problems:**
1. Arm freezes for 355ms during every inference call
2. Actions are based on observations that are 355-722ms stale by execution time
3. No feedback during the 11-step execution — completely open loop

---

## Strategy 1: Async Overlap (High Priority)

Run inference in a background thread while the arm continues executing the
previous action chunk.

```
Thread 1 (control):          Thread 2 (inference):
  execute chunk A step 0  →    start inference(obs_B)
  execute chunk A step 1       ...computing...
  execute chunk A step 2       ...computing...
  ...                          ...computing...
  execute chunk A step 10      inference done → chunk_B ready
  switch to chunk B step 0  →  start inference(obs_C)
  execute chunk B step 1       ...computing...
  ...
```

**How it works:**
- Capture a fresh observation at the START of each chunk execution
- Fire off inference in a background thread immediately
- Continue executing the current chunk's remaining actions
- When inference returns, swap in the new chunk on the next cycle

**Expected result:**
- Arm never freezes
- Observation-to-execution delay = 1 inference time (~355ms), same as now but
  without the freeze gap
- Effective control rate = 30 Hz continuous

**Implementation:** Rewrite `run_policy.py` main loop with `threading.Thread` or
`concurrent.futures.ThreadPoolExecutor`. The WebSocket client is synchronous so
it blocks its own thread while the control thread keeps running.

**Risk:** The observation used for chunk B is captured ~11 steps before chunk B
starts executing. If the arm drifts significantly during those 11 steps, the
prediction may be inaccurate. In practice this works because the model is trained
on data with similar latency characteristics.

---

## Strategy 2: Receding Horizon (Medium Priority)

Instead of executing all 11 predicted steps, execute only the first K and
re-query with a fresh observation.

```
K=4 example:
  capture obs → infer (355ms) → execute steps 0-3 → capture obs → infer → ...
  Total cycle: 355ms + 4×33ms = 488ms
```

**Tradeoffs by K value:**

| K | Cycle time | Obs staleness (end) | Arm idle? | Notes |
|---|---|---|---|---|
| 1 | 388ms | 388ms | Yes, 355ms freeze | Tightest feedback but wastes 10/11 predictions |
| 4 | 487ms | 487ms | Yes, 355ms freeze | Good balance |
| 8 | 619ms | 619ms | Yes, 355ms freeze | Nearly full chunk |
| 11 | 722ms | 722ms | Yes, 355ms freeze | Current behavior |

**Without async overlap, the arm still freezes during inference regardless of K.**
This strategy is most useful *combined* with async overlap: use K < 11 to get
fresher observations, while the async thread hides the freeze.

**Combined async + receding horizon (K=5):**
- Inference runs in background (~355ms)
- Arm executes 5 steps of current chunk (5 × 33ms = 165ms)
- After 5 steps, inference may not be done yet → options:
  - (a) Wait for it (arm pauses briefly)
  - (b) Continue executing steps 6-11 of the old chunk as fallback
  - (c) Interpolate/slow down to buy time

If inference is faster than K steps (e.g., after TensorRT optimization gets us
to 150ms), we'd have K=5 steps = 165ms > 150ms inference, meaning inference is
always ready before we run out of actions. That's the sweet spot.

---

## Strategy 3: Temporal Ensembling (Lower Priority)

Blend overlapping predictions from consecutive chunks to smooth transitions.

```
Chunk A: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
Chunk B:          [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
                   ↑ overlap zone: blend a2..a10 with b0..b8

Executed: [a0, a1, blend(a2,b0), blend(a3,b1), ...]
```

**How the blending works (exponential weighting):**
```python
w = np.exp(-np.arange(overlap_len) * decay)  # favor newer predictions
blended = w * new_chunk + (1-w) * old_chunk_tail
```

**Pros:** Very smooth motion, reduces jitter at chunk boundaries  
**Cons:** More complex bookkeeping, requires careful timing of when to trigger
new inference, slightly increases effective observation staleness

**When to use:** After async overlap is working well, if you notice jerkiness
at chunk transitions. Not needed if transitions are already smooth.

---

## Strategy 4: Use ActionChunkBroker (Utility)

The openpi client library already has `ActionChunkBroker` at
`packages/openpi-client/src/openpi_client/action_chunk_broker.py`.

It wraps a policy and returns actions one-at-a-time, only calling inference
when the current chunk is exhausted. This simplifies the control loop from
"get 11 actions, iterate" to "call infer every step, it handles the rest."

Useful as a building block but still synchronous — the 355ms freeze happens
every 11th call. Combine with async overlap for best results.

---

## Recommended Implementation Order

1. **Async overlap** — biggest UX improvement (eliminates arm freezing), moderate
   code change to `run_policy.py`
2. **Receding horizon (K=5-8)** — easy to add once async is working, one param
   change in the action execution loop
3. **Temporal ensembling** — only if chunk transitions are jerky, adds complexity
4. **ActionChunkBroker** — use it if rewriting the loop from scratch, optional

---

## Latency Targets and Their Effect on Control Strategy

| Inference latency | Best strategy | Effective control |
|---|---|---|
| **355ms (current)** | Async overlap, K=11 (full chunk just barely covers inference) | 30Hz continuous, 355ms stale |
| **150-200ms** | Async overlap + receding K=6 | 30Hz, 200ms stale, re-query 2x per chunk |
| **100ms** | Async overlap + receding K=3 | 30Hz, 100ms stale, very responsive |
| **<33ms** | No chunking needed — infer every step | 30Hz fully closed-loop |

**Bottom line:** Reducing raw inference latency is strictly better than any
control loop strategy. Every ms saved in inference directly reduces observation
staleness, which directly improves task success rate. The control loop strategies
above are about making the best of whatever latency we have.
