# Dum-E 🦾

> A bootstrapped, expressive desktop robot arm — inspired by Iron Man's Dum-E.
> By default it **follows you**; on a **voice command** it picks up a named object and
> hands it to you. The differentiator isn't the picking — it's the **expressive,
> character-driven personality** layered on top.

**Status:** Arm received; **follower motors IDed (1–6) & follower assembled**; C920 in hand.
Next: mount camera + `lerobot-calibrate` on Windows → Phase 1 FOLLOW.
**Owner:** solo build, no funding, POC-first.
**Living doc** — we tweak this as we go.

---

## 1. Vision

Build the cheapest possible arm that feels *alive*. A claw that fetches a phone is a
tool; a claw that **droops when it loses you** and **perks up when you return** is a
character. The personality is the product. The persona is "expressive" — playful and
emotive (puppy-like or human-like, conveyed purely through motion + posture, since there's
no face). The whole point: turn a ₹28k arm into something people *feel something* about.

### Three target behaviors
1. **Follow me** — default mode; the arm's "head" (gripper/wrist) tracks the user.
2. **Voice fetch** — "pick up my phone" → finds it → picks it → hands it over.
3. **Handover** — give/take objects to/from the user's hand.

---

## 2. Core architecture decisions

### Two modes sharing one arm (a state machine)
```
FOLLOW (default) ──[wakeword + command]──► COMMAND (fetch + handover) ──[done]──► FOLLOW
```
- An **always-on wakeword listener** runs in a parallel thread alongside the follow loop.
- Mode transitions are **graceful** (settle to a stable pose, then act).
- After a command, it re-acquires the user and resumes following.

### No VLA, no training, no depth camera (the "Road A" path)
We deliberately **avoid** Vision-Language-Action models / imitation learning / a leader-arm
demo-collection pipeline. Instead:

- **FOLLOW** = classical **visual servoing** (fast tracker → 2 joints). Zero training.
- **COMMAND pick** = **scripted manipulation**: a fixed *recipe* (sequence + gripper
  timing) whose *positions are computed live* from perception. Not hardcoded coordinates —
  hardcoded *logic*, perception-driven *parameters*.
- "No training" still allows **pretrained** models (LocateAnything, Whisper, MediaPipe,
  YOLO) — we just don't fine-tune anything.

**Why:** plays to a software founder's strengths, needs zero demo data, is debuggable, and
both halves are already demonstrated on the SO-101 (see XLeRobot references). The novel part
(personality) is low-risk authored animation.

> Both behaviors have public reference builds on the SO-101:
> - Follow: XLeRobot YOLO object-follow demo (zero training, RGB cam, no depth).
> - Voice fetch + handover: XLeRobot "grab a notebook and give it to human" LLM-agent demo.

### The personality layer (the soul)
Expressiveness is **authored keyframe animation**, not AI. The gripper+wrist is the "head."
Emotion comes from **timing/easing**, not poses — Disney's 12 animation principles applied
to 6 joints (slow-in/slow-out, anticipation, follow-through). Reference: Guy Hoffman's
expressive robotics; Anki Cozmo/Vector.

| Event | Gesture (authored) |
|---|---|
| lost the user (~2s) | slow droop + settle |
| user reappears | perk up + bob |
| heard a command | quick nod |
| fetch success | wiggle / present-with-flourish |
| idle while following | gentle "breathing" sway |

Bonus: an expressive persona makes the cheap arm's clumsiness read as *charm*, not malfunction.

> **Full design → [`soul.md`](./soul.md)** — the personality system (mood + gesture library +
> idle engine + director + mixer), the per-tick blend logic, easing→12-principles mapping,
> puppet-and-record authoring, and a fallback ladder if it gets too taxing.

---

## 3. The pipelines

### FOLLOW loop (~20–30 Hz, low latency)
```
webcam frame → MediaPipe (track the hand) → pixel error vs image center
   → base pan (joint 1, shoulder_pan) + head tilt (joint 4, wrist_flex), proportional
   → smoothing/easing → servo targets   (other 4 joints hold a neutral "perch")
```
Three layers: **track** (eyes) → **aim** (neck) → **personality** (soul). No depth, no training.
Sign of the gain may need flipping on first run if it steers away from the hand.

### COMMAND pipeline (one-shot, ~2–4s "think" then move)
```
voice "pick up my phone"
 → Whisper (STT)
 → intent parse (local Ollama qwen) → {action: PICK, target: "phone"}
 → LocateAnything-3B (open-vocab detect) → 2D box
 → [optional SAM mask → centroid + principal axis = grasp angle]
 → homography H: pixel → table (X,Y) in robot frame;  Z = table height
 → top-down grasp pose → IK → waypoints
 → hover → open → descend → close (watch servo load) → lift
 → move to user's hand (from follow tracker) → pause → open
 → return to FOLLOW
```

### Replacing depth with geometry — the table-plane homography
A 2D camera can't see depth (a pixel = a whole ray). **But if the object is on a known flat
plane (the table), the ray hits exactly one point.** One assumption replaces the depth sensor.
- **Reference point = the gripper TCP** (grasp point between the fingertips) — use it for touching
  the table during calibration, for "table Z", and as where objects end up. Same point everywhere.
- **Pose = gripper pointing straight down (vertical)** for both calibration AND picking, so the
  geometry matches the top-down grasp.
- Calibrate **once** (table-fixed): command the arm so the TCP touches the table at ~8–10 known
  (X,Y) points (stick a colored dot on the TCP to auto-detect it) → `cv2.findHomography` → save `H`.
- **H needs a partner — IK.** H gives *where* (table X,Y); **IK** turns "TCP at (X,Y,Z) gripper-down"
  into joint angles (the *how to reach*). Set up SO-101 kinematics (URDF + a solver) alongside —
  without it, a perfect H is just a number the arm can't act on.
- Limits: only valid for objects on the table; aim at the **bbox bottom (footprint)** to avoid
  parallax on tall objects; **camera must not move** after calibration.
- **Monocular depth** (Depth Anything / YOLO26) was considered and **rejected for v1**: rough/relative,
  ~decimeter metric error — *worse* than the flat-plane homography here. Reserve real depth (Orbbec
  + **Contact-GraspNet**, both pretrained) for varied heights / non-flat scenes.

### Signal path to the arm
Desktop is the brain. Arm connects by **USB serial** → bus-servo adapter board → 6× STS3215
servos (daisy-chained). USB carries data only; a separate PSU powers the motors. **LeRobot**
encodes joint targets into servo register-writes; servos are smart/closed-loop (you command
*target angles*, position control). No Jetson, no Pi.

---

## 4. Hardware (ordered / to buy)

| Item | Price (₹) | Status |
|---|---|---|
| **SO-ARM101 kit** (leader+follower, DIY w/ printed parts) — ThinkRobotics | 27,999 | ✅ received; follower assembled + IDed |
| **Logitech C920 HD Pro** color webcam (main, fixed, side-offset) | ~6,500 | ✅ have |
| USB mic (or existing headset) | 0–1,000 | have? |
| Small camera tripod / flex mount | 500–1,500 | to buy |
| Power strip / surge protector | ~500 | to buy |
| Optional USB speaker (puppy/robot sounds) | ~300 | later |
| **v1 total** | **≈ ₹30–36k** | GPU already owned |

**Kit includes:** both arms, 3D-printed parts, 2 control boards (identical/interchangeable),
**5V 6A + 12V 7.5A PSUs**, USB-C cables, **4 table clamps**. Camera NOT included.

**Kit specifics (Pro edition):** follower = **6× 12V motors** (30 kg·cm, **~200–300 g payload**,
~300 mm reach) — a phone is *at the edge*, start light. Leader = 6× 7.4V motors run at 5V.
⚠️ **Power pairing: 12V → follower, 5V → leader. Never cross them** (5V on a 12V motor → "input
voltage error"). Follower motor IDs: **1**=shoulder_pan(base) · **2**=shoulder_lift ·
**3**=elbow_flex · **4**=wrist_flex · **5**=wrist_roll · **6**=gripper. IDs live in motor EEPROM
(travel with the arm); only calibration is per-host.

### Deliberately deferred
- ⛔ **OV9281 mono cam** — mono breaks color grounding; only useful later as a *wrist* cam.
- ⏸ **Depth camera** — not needed for v1 (planar trick). If added: **Orbbec Gemini 335 (~₹26k)**,
  NOT Intel RealSense (pricier, supply wobbles).
- ⏸ **Leader arm** — unused in the no-VLA build (sits in the box; kept for a possible future VLA).

### Camera placement
**Fixed, side-offset from the arm** (arm on the right short edge), **angled down** at the
workspace, **outside the arm's swept volume** (map it by moving the arm through its range first —
it can rear up high), with the **whole workspace + hand-entry area in frame**. Not on the arm
(breaks the homography); not the laptop cam (wrong position). The arm being *taller* than the
camera is fine — only a *collision that nudges it* matters, since that kills the homography.
So **lock it down hard** (heavy base/clamp, cables routed clear), offset so the arm body doesn't
permanently occlude the workspace. One camera serves both modes: FOLLOW tracks the hand, PICK uses
the homography. **Lock the C920's focus in software** so autofocus can't shift calibration.
*(Wrist cam is a Phase-2+ add — for grasp precision or VLA — not now; the deferred OV9281 is its home.)*

---

## 5. Compute / platform

- **Run on the Windows desktop (RTX 5070 12GB), native.** Existing CUDA + PyTorch reused.
- ❌ **No WSL, no Docker** — both put a VM between code and the USB arm (painful serial
  passthrough). Native talks to the `COM` port directly.
- LeRobot supports Windows natively (needs **PyTorch ≥ 2.8**, **Python 3.12+**).
- ⚠️ **5070 = Blackwell (sm_120)** — verify PyTorch actually runs a GPU op (not just
  `cuda.is_available()`); may need a recent cu128-era build.
- Mac (M4) was the alternative for the no-VLA path (everything runs native on Apple Silicon,
  CUDA not required) — kept as a fallback. Desktop chosen for the 5070 + open VLA option.

### Software stack
`lerobot` · `opencv-python` · `mediapipe` · `faster-whisper` · `ultralytics` (YOLO) ·
LocateAnything-3B (GPU) · Ollama (`qwen`) · a fresh `conda create -n lerobot python=3.12`.

---

## 6. Build order (from unboxing) — with gates 🚦

**A. Unbox & inventory** ✅ — checked kit contents vs listing.

**B. Set motor IDs + assemble FOLLOWER** ✅ — IDs set one-at-a-time via
   `lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>` (one motor on the bus at a
   time, gripper→6 … shoulder_pan→1, **labeled each**). ⚠️ **12V supply + 12V motors only** (the 5V/7.4V
   mix throws "input voltage error"). Assembled base→gripper by ID; horns = M3×6, motors = M2×6.

**C. Windows env** — `conda create -n lerobot python=3.12` → `pip install "lerobot[feetech]"`;
   verify torch sees the 5070 (run a real GPU op, not just `cuda.is_available()` — Blackwell sm_120);
   add Ollama+qwen, mediapipe, faster-whisper. *(Same env already built on the Mac for ID-setting.)*

**D. Phase 0 — bring-up** 🚦 *every joint obeys & reports position.*
   `lerobot-find-port` → `lerobot-calibrate --robot.type=so101_follower --robot.port=<COMx>
   --robot.id=dum_e_follower` (torque off → sweep each joint by hand; doubles as an assembly check)
   → control-test script (move the arm *from code*). ⚠️ Arm **sags on power-off** — park low / catch it.

**E. Phase 1 — FOLLOW** 🚦 *arm smoothly follows your hand ("it's alive").*
   Mount + focus-lock C920 → MediaPipe hand → 2-joint servoing → smoothing.

**F. Phase 2 — table-plane calibration** 🚦 *a pixel maps to the right real spot.*
   Measure Z → touch ~8–10 known points + click in image → `findHomography` → validate.

**G. Phase 3 — scripted pick (keypress)** 🚦 *press key → picks object repeatably.*
   LocateAnything detect → bbox → H → hover/open/descend/close(watch load)/lift → tune.

**H. Phase 4 — handover** 🚦 *places object in your hand.*

**I. Phase 5 — voice + state machine** 🚦 *"pick up my phone" runs hands-free.*
   faster-whisper + wakeword thread → qwen intent → FOLLOW⇄COMMAND state machine.

**J. Phase 6 — personality** 🚦 *feels like a character, not a machine.* → see [`soul.md`](./soul.md)
   Author droop/perk/wiggle/nod/idle-bob → hook to events → tune easing → optional sounds.

**K. Later upgrades** — depth cam + Contact-GraspNet (varied heights); leader + SmolVLA
(reactive handover); wrist cam (embodied eyes).

**Throughline:** D it obeys → E it's alive → F–H it can fetch → I it listens → J it has a soul.
Emotional payoff comes early (E); hard fetch logic builds on the F calibration.

---

## 7. Key risks / what would kill it

- **Grasp reliability** of the scripted/planar approach on odd or cluttered objects → mitigate
  by scoping v1 to rigid, flat-lying, top-down-graspable items (phone, pen, remote).
- **Calibration drift** if the camera/arm base move → keep both rigid; recal is ~20 min.
- **5070 Blackwell / PyTorch** kernel support → verify before building.
- **ThinkRobotics delivery/support** is mixed in reviews → confirm contents on arrival.
- Scripted handover is **non-reactive** (won't chase a moving hand) → acceptable for v1;
  personality layer masks the stiffness; VLA is the later fix.
- **Payload ~200–300 g** — a phone is at the edge. Demo with lighter objects first; keep heavy
  picks **close to the base** (payload drops at full reach); don't hold near-max loads long (servo heat).

---

## 8. Open questions / to decide later
- Train a small custom YOLO on desk objects vs. rely on open-vocab LocateAnything?
- ~~Track hand vs face for FOLLOW~~ → **decided: track the hand** (natural over a desk).
- Local qwen vs Azure OpenAI for intent parsing (have Azure credits).
- When (if ever) to bring in the leader arm + SmolVLA for a smoother handover.

---

## 9. Reference links
- LeRobot install: https://huggingface.co/docs/lerobot/installation
- SO-101 docs: https://huggingface.co/docs/lerobot/so101
- XLeRobot follow demo: https://xlerobot.readthedocs.io/en/latest/software/getting_started/SO101.html
- XLeRobot LLM agent (voice + grab + handover): https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html
- SmolVLA: https://huggingface.co/papers/2506.01844
- NVIDIA LocateAnything-3B: https://huggingface.co/nvidia/LocateAnything-3B
- Arm: https://thinkrobotics.com/products/so-arm101-hugging-face-lerobot
