# Windows setup & run guide (Dum-E)

Your offline runbook for the desktop (RTX 5070). Follow top to bottom.
Motor IDs are already set (they live in the motors); you only need the env,
calibration, and the scripts here.

---

## 0. Prereqs
- Git + Miniconda/Anaconda installed.
- Clone the repo:
  ```bat
  git clone https://github.com/Rudr4Khunt1/dum-e.git
  cd dum-e
  ```

## 1. Python env
```bat
conda create -n lerobot python=3.12 -y
conda activate lerobot
pip install -r requirements.txt

:: cv2.imshow needs the FULL opencv (lerobot pulls the headless build):
pip uninstall -y opencv-python-headless
pip install opencv-python
```

## 2. Verify PyTorch actually drives the 5070 (Blackwell = sm_120)
```bat
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); x=torch.randn(1000,1000,device='cuda'); print(float((x@x).sum()))"
```
- Must print the RTX 5070 name and a number (no "no kernel image" error).
- If it errors, install a newer CUDA build, e.g.:
  ```bat
  pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  ```
- (Not needed for calibration/FOLLOW, but required later for LocateAnything.)

## 3. Find the arm's port
Plug in the follower: **USB + the 12V supply** (never the 5V — that's the leader's).
```bat
lerobot-find-port
```
It lists ports, tells you to unplug USB + press Enter, then prints the port,
e.g. **COM5**. Put that in `config.py` -> `PORT`.

## 4. Calibrate the follower
```bat
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=dum_e_follower
```
- Torque turns OFF so you move the arm by hand.
- It asks you to (1) set a reference pose, then (2) sweep each joint through its
  full range. This doubles as an assembly check — all 6 joints should move freely.
- Keep `--robot.id=dum_e_follower` (the scripts expect that id).
- ! The arm SAGS when power/torque drops — park it low or keep a hand ready.

## 5. Run the scripts — in this order

Edit `config.py` first: set `PORT`. (Camera not needed until step 5b.)

### 5a. control_test.py  — Phase 0 gate (arm obeys code)
```bat
python control_test.py
```
Prints joint angles, nudges shoulder_pan +8 deg and back.
GATE: it moves where told -> Phase 0 done.
SAFETY: clear the workspace, hand near power.

### 5b. camera_test.py  — verify the camera (no arm)
```bat
python camera_test.py
```
A window opens; show your hand -> it draws the hand skeleton + prints the pixel.
- Wrong camera opens? Change `CAMERA_INDEX` (0 -> 1 -> 2) in config.py.
- Use this to check the mount sees the whole workspace + where your hand enters.
- LOCK THE C920 FOCUS in Logitech G HUB (or it may drift and break homography later).

### 5c. follow.py  — Phase 1: it follows your hand
```bat
python follow.py
```
The arm's "head" (base + wrist) tracks your hand. 'q' or Ctrl-C to stop.
GATE: it smoothly follows -> "it's alive".

Tuning (edit `config.py`, no code changes):
- Steers AWAY from your hand -> flip `SIGN_PAN` and/or `SIGN_TILT`.
- Twitchy/oscillating -> lower `GAIN_PAN` / `GAIN_TILT`.
- Sluggish -> raise the gains a little, or raise `SMOOTHING`.
- Jerky/snappy -> lower `SMOOTHING` (more glide = more "alive").
- Too small a range -> raise `PAN_LIMIT_DEG` / `TILT_LIMIT_DEG`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Input voltage error` | Wrong supply — follower needs the **12V**, not the 5V (leader). |
| Port not found / permission | Re-run `lerobot-find-port`; check USB + power; correct `PORT` in config. |
| `cv2.imshow` errors / no window | You have opencv-headless — do the uninstall/install in step 1. |
| Wrong camera opens | Change `CAMERA_INDEX` in config.py (try 1, 2, ...). |
| Camera slow to open on Windows | Normal for the DSHOW backend; give it a few seconds. |
| `mediapipe` install fails on 3.12 | Ping me — we'll pin a working version or use the tasks API. |
| Torch "no kernel image for device" | Newer CUDA build needed — see step 2. |
| Arm won't move but connects | Did you calibrate (step 4) with the same `--robot.id`? |
| Arm flops when I power off | Expected — servos have no torque unpowered. Park it low first. |

When something breaks, copy the full terminal error and send it to me — I'll tell
you the exact tweak. Send photos of the rig too and I'll sanity-check placement.
