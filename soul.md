# soul.md — Dum-E's personality system (Stage J)

> How Dum-E *feels alive*. This is the differentiator — the picking is plumbing, this is the product.
> **Status:** design only. If it proves too taxing to build, we simplify (see "Fallback ladder" at the end).

The persona is **expressive** — playful/emotive, conveyed purely through **motion + posture**
(no face). Emotion lives in *timing*, not poses. This is an authored **character-animation
system**, not AI — the same pattern game engines use for living characters.

---

## Core insight: personality is a *layer that modulates*, not a separate mode

Dum-E is always doing something functional (following, reaching, or idling). Personality
doesn't replace that motion — it **rides on top of it**. Every tick, the final joint angles are
a blend:

```
final_pose =  functional_target          # from FOLLOW / COMMAND (or neutral if idle)
            + idle_aliveness(mood, t)     # procedural "breathing", always on
            + active_gesture(mood, t)     # a one-shot emotional beat, when triggered
   → all passed through mood-scaled easing → servos
```

So it emotes *while* it works. That's what separates "alive" from "robot plays a clip, then
goes stiff." It also means the **pick itself becomes expressive for free** — curious head-tilt
on approach, FOCUSED during the grasp, EXCITED + wiggle on success. Fetch *expressively*, not
fetch-then-emote.

---

## The four components

### 1. `Mood` — global emotional state
A small state whose *parameters reshape all other motion* (it never moves the arm directly):

```python
Mood.CONTENT  = { speed: 1.0, amplitude: 1.0, easing: "smooth",    head_bias:  0    }
Mood.EXCITED  = { speed: 1.6, amplitude: 1.4, easing: "overshoot", head_bias: +up   }
Mood.DEJECTED = { speed: 0.5, amplitude: 0.7, easing: "heavy",     head_bias: -down }
Mood.CURIOUS  = { speed: 1.1, amplitude: 1.0, easing: "smooth",    head_bias:  tilt }
Mood.FOCUSED  = { speed: 1.0, amplitude: 0.4, easing: "precise",   head_bias:  0    }
```

One variable change re-colors everything. *Sad = slow + small + heavy + head down.*

### 2. `Gesture` — an authored one-shot animation
A timeline of keyframes, each with its own easing, flagged `additive` or `override`:

```python
Gesture("droop", override=True, keys=[
    (0.0, current_pose),
    (0.6, {wrist_tilt: -40, shoulder: -15, gripper: curl}, ease="easeOut"),
    (1.2, {wrist_tilt: -45, shoulder: -18},                ease="easeInOut"),  # extra sag
])
Gesture("wiggle", additive=True, keys=[...fast base-joint oscillation...])
Gesture("nod",    additive=True, keys=[...quick wrist dip + return...])
```

- **`override`** — takes over the arm. Used when idle (e.g. droop when the user is lost; nothing
  functional to do).
- **`additive`** — adds a small offset on top of functional motion (tail-wag while still
  tracking; nod while listening).

### 3. `IdleEngine` — procedural aliveness (the anti-frozen layer)
Always-on, low-amplitude motion so it never looks dead. Cheapest, highest-impact trick:

```python
offset = sin(t * mood.breath_freq) * mood.amplitude * small   # "breathing" on 1–2 joints
       + occasional micro "look around" when nothing happens
```

Excited → faster/bigger breathing. Dejected → slow/shallow.

### 4. `Director` — the brain that reacts to events
Listens to events from the FOLLOW/COMMAND state machine; sets mood + fires gestures:

| Event | Mood set | Gesture |
|---|---|---|
| lost user (~2s) | DEJECTED | `droop` (override) |
| user reappears | EXCITED → settle to CONTENT | `perk_up` |
| wakeword heard | (keep) | `nod` (additive) |
| pick starts | FOCUSED | — (precise, minimal flourish) |
| pick success | EXCITED | `wiggle` |
| pick fails / can't find | DEJECTED | `shake_head` |
| idle, following you | CONTENT | (breathing only) |

---

## The tick (whole system, one loop, ~30 Hz)

```python
def tick(t):
    base = functional_target()                  # FOLLOW pose / COMMAND waypoint / neutral
    pose = base + idle.offset(mood, t)           # aliveness layer
    if director.active_gesture:
        g = director.active_gesture.sample(t, mood)   # mood scales its speed/size
        pose = g if g.override else pose + g     # takeover vs additive
    pose = ease(pose, prev_pose, mood.easing)    # mood-scaled smoothing
    send_to_servos(pose)
```

---

## Easing is literally the soul (Disney's 12 principles → code)

Poses barely matter; the **curves between them** are everything. Implement ~5 easing functions;
the mood just *picks which one* — that single indirection is where ~80% of the character comes from.

| Principle | Implementation |
|---|---|
| Slow in / slow out | `easeInOut` between keyframes (the #1 "alive" cue) |
| Anticipation | prepend a tiny *reverse* keyframe before a move (lean back before reaching) |
| Follow-through / overshoot | `easeOutBack` on arrival (overshoot + settle, like real mass) |
| Exaggeration | `mood.amplitude` multiplier |
| Timing | `mood.speed` multiplier |
| Secondary action | the additive idle layer running underneath |

Functions to write: `easeInOut`, `easeOutBack`, `easeOutBounce`, `anticipate`, `heavy`.

---

## Authoring gestures: puppet-and-record (the fun part)

Don't type joint angles — **pose the arm and record**:

```
1. Put the follower's servos in compliant / low-torque mode (movable by hand)
2. Pose it into "droop" → press a key → record current joint angles as a keyframe
3. Move to next pose → record → repeat
4. Save the keyframe list as a gesture; tune timing/easing in code by feel
```

⚠️ This is **not** VLA demo-collection — it's just "move the arm, save a few poses," like
keyframing in Blender where the puppet is the real robot. Single arm, no training, no leader.

---

## Full-body vocabulary (use ALL six motors, not just the head)

FOLLOW v1 moves only pan/tilt/roll — reads as "camera gimbal". The other joints
(shoulder_lift, elbow_flex, gripper) are the difference between gimbal and creature.
In payoff order:

1. **Distance lean** ⭐ — hand SIZE in the image (wrist↔middle-base px, One-Euro'd)
   is a free depth proxy. Closer hand → shoulder+elbow lean/extend toward you;
   farther → settle back to perch. The *body* participates in attention.
2. **Counter-phase breathing** ⭐ — idle sine on shoulder_lift + elbow_flex in
   OPPOSITE phase: body rises/falls, head stays level (chest breathing under a
   steady gaze). Head-bob = mechanical; body-breath = alive.
3. **Gripper = mouth** 🐶 — perk: slightly open (panting); droop: closed; "heard
   you": quick open-close flutter (yip); idle: relaxed half-open. Lightest joint,
   zero risk, huge character.
4. **Eyes lead, body follows** — wrist/pan react fast (already do); add a SLOW
   integrator on shoulder/base that re-centers the body so the wrist drifts back
   toward neutral. Spine, not turret.
5. **Mood = whole-body posture** — generalize head_bias → posture_bias per joint:
   EXCITED tall (shoulder up, elbow extended), DEJECTED crouched (dropped+folded),
   CURIOUS leaning in + wrist_roll head-cock. The droop becomes full-body deflation.
6. **Anticipation & startle** — tiny opposite lean before a big pan swing; fast
   incoming hand → slight recoil, then re-engage.

Slots into the existing mixer: posture(mood, distance) + gaze + breathing + gestures
→ clamp_pose → per-joint slew caps → servos.

Amplitudes: standard SO-101 usage swings shoulder/elbow through 60–90° continuously
(teleop/ACT demos), so expressive leans of **20–40°** are normal duty — no need to
be timid. The one real rule: avoid LONG STATIC HOLDS at full horizontal extension
(max moment arm) — that's what trips servo overload, not motion through big angles.

## Optional: sound layer
Tie short sounds (chirps/whirs) to moods/gestures via a cheap USB speaker. High character-per-rupee.

---

## Fallback ladder (if the full system is too taxing)

Build top-down; stop wherever it already feels good enough:

1. **Minimum viable soul:** just the `IdleEngine` (breathing) + mood-scaled easing on existing
   motion. No gestures yet. This alone reads as "alive."
2. **+ A handful of override gestures** played only when idle (`droop`, `perk_up`). Easiest wins.
3. **+ Additive gestures** during functional motion (`nod`, `wiggle`).
4. **+ Full Director** with mood transitions coloring the pick/follow.

Each rung is independently shippable. We can stop at rung 1–2 and it'll still have a soul.

---

## TL;DR
A **mood** variable reshapes all motion; an always-on **idle** layer keeps it breathing; a
**director** fires authored **gestures** on events; everything **mixes + eases** each tick.
Small system, huge character.
