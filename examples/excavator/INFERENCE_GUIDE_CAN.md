# CAN-bus Excavator Inference — Jetson Runbook

This guide is the operator's checklist for running the **8-axis CAN-bus**
excavator policy on the Takeuchi TB20E from the Jetson mounted on the
machine. For the GPU-side (RunPod) deployment, see `RUNPOD_DEPLOY_CAN.md`.

---

## 0. What this does

```
                 +-------------------+        wss://<pod>-8000
                 |   RunPod GPU pod  | <===========================+
                 | pi05_canteleop_   |     8-axis actions          |
                 |     fullft        |                             |
                 +---------+---------+                             |
                           ^                                       |
                           |                                       |
                3 cams + 8-axis state (CAN)                        |
                           |                                       |
                +----------+----------+      TakeuchiClient        |
                |  examples/excavator |--- (UDP -> 2 Arduinos) --->|
                |   run_policy_can.py |                            |
                +---------------------+                            |
                                                                   v
                                                        TB20E excavator
```

Inputs to the model:
- `cab_forward` camera (224x224 RGB)
- `front_left` camera (224x224 RGB)
- `front_right` camera (224x224 RGB)
- 8-axis decoded CAN state (`left_stick_x, left_stick_y, right_stick_x, right_stick_y, left_track, right_track, swing, blade`)
- prompt string

Outputs:
- 8-axis action chunk (typically `[11, 8]`), executed at 50 Hz.
- The first 4 axes are demuxed to `bucket / boom / body / arm` and sent via
  `TakeuchiClient` to the right and left Arduinos.
- The last 4 axes (`left_track / right_track / swing / blade`) are LOGGED
  but NOT actuated — the current bridge does not support them.

---

## 1. Pre-flight checklist

On the Jetson, in this order:

1. **CAN bus up?**

   ```bash
   ip link show can0
   # If state is DOWN:
   sudo ip link set can0 up type can bitrate 500000
   ip link show can0   # confirm UP, bitrate=500000
   ```

2. **CAN traffic flowing?** Run for 1–2 seconds — you should see `~500 Hz`
   of frames:

   ```bash
   timeout 2 candump -t a can0 | wc -l
   ```

3. **Cameras present?** Confirm three USB devices:

   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices
   ```

   The defaults expect: `cab_forward=2`, `front_left=6`, `front_right=0`.
   Pass `--cab-cam`, `--front-left-cam`, `--front-right-cam` if different.

4. **Both Arduinos plugged in?** Stable `/dev/serial/by-id` symlinks are
   required by `TakeuchiClient`:

   ```bash
   ls /dev/serial/by-id/
   # expect two entries like:
   #   usb-Arduino_*1344A474130351A03926-if00
   #   usb-Arduino_*0353638313635121D232-if00
   ```

5. **TB20E decoder repo on disk?**

   ```bash
   ls /home/Actor/Thor-CAN-recording/tb20e_decoder.py
   ```

   If missing, clone it or pass `--decoder-path <dir>`.

6. **RunPod server up + warmed up?** See `RUNPOD_DEPLOY_CAN.md` Sections
   5–6. The first inference will time out on the proxy if you skip warmup.

7. **Cab in operate mode**: key fully on, throttle up, hydraulic lock OFF.
   Without operate mode the cab still broadcasts state but joystick
   channels stay pinned at zero.

---

## 2. First-time dry-run (no actuation)

Validate the full pipeline without touching the hydraulics:

```bash
cd /home/Actor/openpi-actor-labs

sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \
    examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --no-send \
    --prompt "Scoop packing peanuts from large pool and dump into small pool"
```

Expected output:
- `[cam:cab_forward] local /dev/video2 native 640x480` (and similar for the
  other two cams)
- `CAN reader started on can0`
- `WARNING ... Dropped (logged only): left_track, right_track, swing, blade`
- `Server metadata: {...}`
- A scrolling status line:
  ```
  step    25 chunk  1/11 infer  1234ms | lx=+0.21 ly=-0.05 rx=+0.43 ry=-0.12 | lt=+0.00 rt=+0.00 sw=+0.00 bl=+0.01
  ```

If the `state` is always all zeros while `total_frames` is climbing, the
joysticks aren't being deflected by anything — confirm Step 7 of pre-flight.

Press `Q` to quit.

---

## 3. Full run (with actuation)

Same command as the dry-run, drop `--no-send`:

```bash
sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \
    examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --prompt "Scoop packing peanuts from large pool and dump into small pool"
```

Stay close to the kill switch. Use the runtime keys:

| Key   | Action                                              |
|-------|-----------------------------------------------------|
| SPACE | E-STOP — zero all joystick axes immediately         |
| R     | Resume inference after E-STOP                       |
| Q     | Graceful shutdown (sends neutral, releases cameras) |

The watchdog also forces neutral if no new action target arrives within
`--watchdog-timeout` seconds (default `0.5`).

---

## 4. Per-axis tuning

Defaults are pure passthrough. Adjust per axis with these flags (applied in
the order: invert -> gain -> deadzone -> clip -> slew):

```
--gain-{lx|ly|rx|ry} <float>       (default 1.0)
--invert-{lx|ly|rx|ry} <0|1>       (default 0)
--deadzone-{lx|ly|rx|ry} <float>   (default 0.0)
--slew-{lx|ly|rx|ry} <float>       (default -1.0 = unbounded)
```

A safety-conservative setting derived from the CAN capture:

```bash
--gain-lx 0.7  --gain-ly 0.7  --gain-rx 0.7  --gain-ry 0.7 \
--deadzone-lx 0.05 --deadzone-ly 0.05 --deadzone-rx 0.05 --deadzone-ry 0.05 \
--slew-lx 0.5  --slew-ly 0.5  --slew-rx 0.5  --slew-ry 0.5
```

`--slew-* 0.5` at 50 Hz = ~25 axis-units per second, which matches the
fastest human stick deflection observed in the live capture.

If a stick polarity disagrees with the model's output (e.g. the model
predicts "+ry = boom up" but the bridge interprets "+ry = boom down"),
flip with `--invert-ry 1`.

---

## 5. Component reference

| File | Purpose |
|---|---|
| `examples/excavator/run_policy_can.py` | Main entry point. Cameras, CAN reader, policy client, action interpolator, CAN sender, keyboard. |
| `examples/excavator/can_state_reader.py` | Background `socketcan` -> `TB20EDecoder` -> `get_state()` (8 floats). |
| `examples/excavator/action_interpolator.py` | Per-axis `gain/invert/deadzone/slew` and 8 -> 4 axis demux. |
| `src/openpi/policies/excavator_policy.py` | Train-side data transforms (`CANExcavatorInputs/Outputs`). |
| `src/openpi/training/config.py` | Train and serving configs (`pi05_canteleop_fullft`, `pi05_can_teleop`). |
| `RUNPOD_DEPLOY_CAN.md` | GPU-side (RunPod) deploy guide for `verm11/pi05-canteleop-fullft`. |

---

## 6. Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `Failed to open socketcan channel 'can0'` | Interface down | `sudo ip link set can0 up type can bitrate 500000` |
| `CAN state stale (no frames for >0.50s)` | Cab off / cable unplugged / wrong channel | Confirm cab keyed on; try `--can-channel can1` |
| Decoded state all zeros (but bus traffic exists) | Hydraulic lock engaged or cab not in operate mode | Engage operate mode and deflect a stick to confirm |
| `tb20e_decoder.py not found` | Decoder repo missing | Clone Thor-CAN-recording, or `--decoder-path <dir>` |
| `Failed to open ... camera` | Wrong /dev/video index | `v4l2-ctl --list-devices`, pass correct index |
| `TimeoutError: timed out while waiting for handshake` | RunPod JIT not warmed | Run warmup script on the pod (RUNPOD_DEPLOY_CAN.md §6) |
| Excavator twitches | Slew unbounded + sharp model output | Set `--slew-* 0.5` |
| Wrong stick direction | Polarity mismatch | `--invert-{axis} 1` |
| Won't quit cleanly | E-stop active and `Q` ignored | Press `R` then `Q` (E-stop swallows the `Q` only briefly) |
| `serial port not found` from TakeuchiClient | Arduino not plugged in or wrong by-id | `ls /dev/serial/by-id/` |

---

## 7. Stop the run cleanly

Press `Q` (or `Ctrl-C`). The script will:
1. Send a neutral command (zero all axes).
2. Send 5 more neutrals for reliability.
3. Stop the Arduino bridge subprocesses.
4. Close the CAN bus.
5. Release all cameras.
6. Restore the terminal mode.

If the script crashes, the Arduino bridges have a 0.5 s UDP watchdog of
their own and will return to neutral on their own. As a manual safety net:

```bash
pkill -f usb_can_bridge_stick
```
