"""
dance.py — Dum-E dances to any song.

    python dance/dance.py song.mp3                     local file
    python dance/dance.py "https://youtu.be/xyz"       yt-dlp grabs the audio first
    python dance/dance.py song.mp3 --no-arm            dry run (no robot, prints)
    python dance/dance.py --set-stage                  capture the dance stance (once)

HOW IT WORKS
  1. URL? -> yt-dlp downloads the audio into dance/songs/ (needs ffmpeg on PATH).
  2. Dance-sheet cache (<song>.dance.json):
       HIT  -> perk-up: "I know this one!" -> joins on the first downbeat
       MISS -> listening head-tilt + gentle sway WHILE librosa genuinely analyzes
               the song (beats, energy, accents, 8-beat phrase tiers) -> cached.
  3. PERFORM: this script plays the audio; the master clock is the audio
     playback position (frames delivered to the device — drift-free). Every
     tick evaluates smooth beat-locked waveforms (dance_moves.py), crossfades
     figures at phrase boundaries, clamps to the dance envelope, applies the
     per-joint slew caps, and sends the pose. Ctrl-C / end of song -> fades out
     and parks via the usual rest ritual.

Same song = same dance (choreography is seeded by the file's hash).
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))     # repo root: config, arm_utils

import config as C                              # noqa: E402
import dance_moves as M                         # noqa: E402
from arm_utils import pose_now, ramp_to, safe_park  # noqa: E402

SONGS_DIR = os.path.join(_HERE, "songs")
STAGE_PATH = os.path.join(_HERE, "stage_pose.json")
PHRASE_BEATS = 8
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")


# ────────────────────────── song acquisition ──────────────────────────

def fetch(source):
    """URL or ytsearch query -> local wav via yt-dlp (cached); local path -> itself."""
    if not source.startswith(("http://", "https://", "ytsearch")):
        if not os.path.exists(source):
            raise SystemExit(f"no such file: {source}")
        return source
    os.makedirs(SONGS_DIR, exist_ok=True)
    key = hashlib.sha1(source.encode()).hexdigest()[:12]
    out = os.path.join(SONGS_DIR, f"{key}.wav")
    if os.path.exists(out):
        return out
    print("fetching audio (yt-dlp)...")
    # prefer the yt-dlp installed next to THIS python (works even when the conda
    # env isn't activated); fall back to whatever is on PATH
    ytdlp = os.path.join(os.path.dirname(sys.executable),
                         "yt-dlp.exe" if os.name == "nt" else "yt-dlp")
    if not os.path.exists(ytdlp):
        ytdlp = "yt-dlp"
    r = subprocess.run(
        [ytdlp, "-x", "--audio-format", "wav", "-o", out, source],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"yt-dlp failed (is ffmpeg installed?):\n{r.stderr[-800:]}")
    return out


# ────────────────────────── analysis (the dance sheet) ──────────────────────────

SHEET_VERSION = 2       # bump when the sheet gains fields -> old caches re-analyze
MOUTH_HOP_S = 0.04      # vocal envelope sample period (25 Hz)


def vocal_envelope(path, y, sr):
    """The lip-sync track: 0..1 loudness envelope of the SINGER, 25 Hz.
    Demucs = true vocal stem (best; fast with CUDA). HPSS fallback = harmonic
    component band-passed to the vocal range (no heavy deps, decent).
    Returns a list of floats, or None when DANCE_VOCALS='off' or extraction fails."""
    import librosa
    mode = getattr(C, "DANCE_VOCALS", "auto")
    if mode == "off":
        return None
    use_demucs = False
    if mode in ("auto", "demucs"):
        try:
            import torch  # noqa: F401
            import demucs.api  # noqa: F401
            use_demucs = (mode == "demucs") or torch.cuda.is_available()
        except Exception:
            use_demucs = False
    try:
        if use_demucs:
            print("separating vocals (demucs)...")
            import torch
            from demucs.api import Separator
            sep = Separator(model="htdemucs")
            _origin, stems = sep.separate_audio_file(path)
            v = stems["vocals"].mean(0).cpu().numpy()
            vsr = sep.samplerate
        else:
            print("estimating vocals (hpss fallback)...")
            import scipy.signal as ss
            harm = librosa.effects.harmonic(y, margin=3.0)
            sos = ss.butter(4, [200, 4000], btype="bandpass", fs=sr, output="sos")
            v = ss.sosfilt(sos, harm)
            vsr = sr
        env = librosa.feature.rms(y=np.asarray(v, dtype=np.float32),
                                  frame_length=2048,
                                  hop_length=int(vsr * MOUTH_HOP_S))[0]
        ref = np.quantile(env, 0.97) + 1e-9
        env = np.clip((env / ref - 0.12) / 0.88, 0.0, 1.0)   # normalize + noise gate
        k = np.ones(3) / 3.0                                  # ~120 ms smoothing
        env = np.convolve(env, k, mode="same")
        return [float(e) for e in env]
    except Exception as e:  # noqa: BLE001 — lip-sync is a garnish, never fatal
        print(f"[warn] vocal extraction failed ({e}) — accent pops only.")
        return None


def analyze(path):
    """librosa -> beats, per-beat energy/accents, 8-beat phrase energy tiers,
    plus the vocal lip-sync envelope."""
    import librosa
    y, sr = librosa.load(path, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < PHRASE_BEATS * 2:
        raise SystemExit("couldn't find enough beats — is this actually music?")

    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    energy = np.interp(beats, rms_t, rms)
    energy = (energy - energy.min()) / (np.ptp(energy) + 1e-9)

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_t = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    beat_onset = np.interp(beats, onset_t, onset)
    accent = (beat_onset > np.quantile(beat_onset, 0.75)).astype(int)

    n_phrases = len(beats) // PHRASE_BEATS
    tiers = []
    for i in range(n_phrases):
        e = energy[i * PHRASE_BEATS:(i + 1) * PHRASE_BEATS].mean()
        tiers.append(0 if e < 0.35 else (1 if e < 0.65 else 2))

    return {
        "v": SHEET_VERSION,
        "tempo": float(np.atleast_1d(tempo)[0]),
        "beats": [float(t) for t in beats],
        "energy": [float(e) for e in energy],
        "accent": [int(a) for a in accent],
        "tiers": tiers,
        "mouth": vocal_envelope(path, y, sr),
        "mouth_hop": MOUTH_HOP_S,
    }


def load_sheet(path, listen_cb=None):
    """Cache hit -> (sheet, True). Miss -> analyze (calling listen_cb while it
    runs, so the arm can do its 'listening' act) -> (sheet, False)."""
    cache = path + ".dance.json"
    if os.path.exists(cache):
        with open(cache) as f:
            sheet = json.load(f)
        if sheet.get("v") == SHEET_VERSION:
            return sheet, True
        print("dance sheet from an older version — re-learning the song.")
    result = {}

    def work():
        result["sheet"] = analyze(path)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while th.is_alive():
        if listen_cb:
            listen_cb()
        time.sleep(1.0 / C.DANCE_HZ)
    th.join()
    if "sheet" not in result:
        raise SystemExit("analysis failed")
    with open(cache, "w") as f:
        json.dump(result["sheet"], f)
    return result["sheet"], False


# ────────────────────────── playback clock ──────────────────────────

class Player:
    """Plays the file; .pos is the authoritative audio time (frames delivered)."""

    def __init__(self, path):
        import soundfile as sf
        self.data, self.sr = sf.read(path, dtype="float32", always_2d=True)
        self._frames = 0
        self._stream = None

    @property
    def pos(self):
        return self._frames / self.sr

    @property
    def done(self):
        return self._frames >= len(self.data)

    def start(self):
        import sounddevice as sd

        def cb(outdata, frames, t, status):
            i = self._frames
            chunk = self.data[i:i + frames]
            outdata[:len(chunk)] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):] = 0
            self._frames = i + frames

        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=self.data.shape[1], callback=cb)
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ────────────────────────── the conductor ──────────────────────────

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Conductor:
    def __init__(self, sheet, stage, seed):
        self.beats = np.array(sheet["beats"])
        self.idx = np.arange(len(self.beats), dtype=float)
        self.accent = sheet["accent"]
        self.tiers = sheet["tiers"] or [1]
        self.stage = stage
        m = sheet.get("mouth")
        self.mouth = np.array(m, dtype=float) if m else None
        self.mouth_hop = float(sheet.get("mouth_hop", 0.04))
        rng = random.Random(seed)
        self.figures = M.pick_figures(rng, self.tiers)
        self.caps = {j: C.MAX_DEG_PER_SEC.get(j, C.DEFAULT_MAX_DEG_PER_SEC) / C.DANCE_HZ
                     for j in JOINTS}
        self.cmd = dict(stage)

    def beat_index(self, t):
        """Continuous beat index at song time t (0 before the first beat)."""
        return float(np.interp(t, self.beats, self.idx))

    def accent_env(self, b):
        """0..1 raised-cosine envelope around accented beats near b."""
        env = 0.0
        for k in (math.floor(b), math.ceil(b)):
            if 0 <= k < len(self.accent) and self.accent[k]:
                env = max(env, M.rc_pulse(b - k, 0.35))
        return env

    def params_at(self, b):
        """Figure params with a 1-beat crossfade at each phrase boundary."""
        ph = int(b // PHRASE_BEATS)
        ph = min(ph, len(self.figures) - 1)
        cur = self.figures[ph]
        into = b - ph * PHRASE_BEATS
        if ph > 0 and into < 1.0:
            return M.lerp_params(self.figures[ph - 1], cur, into), ph
        return cur, ph

    def pose_at(self, t, master=1.0):
        """The smooth dance pose at song time t. master fades the whole dance
        in/out (0..1) without any discontinuity."""
        b = self.beat_index(t - C.DANCE_BEAT_OFFSET_S)
        p, ph = self.params_at(b)
        tier = self.tiers[min(ph, len(self.tiers) - 1)]
        scale = master * C.DANCE_AMPLITUDE * M.TIER_MULT[tier]
        off = M.figure_offsets(b, p, self.accent_env(b), scale)
        off["wrist_flex"] *= C.DANCE_NOD_SIGN
        # lip-sync: the mouth follows the SINGER; accent pops still win in
        # instrumental stretches (max = whichever wants it open more)
        if self.mouth is not None:
            i = (t - C.DANCE_BEAT_OFFSET_S) / self.mouth_hop
            vo = float(np.interp(i, np.arange(len(self.mouth)), self.mouth))
            off["gripper"] = max(off["gripper"],
                                 master * C.DANCE_AMPLITUDE * C.DANCE_MOUTH_MAX * vo)
        off["gripper"] *= C.DANCE_MOUTH_SIGN
        pose = {}
        for j in JOINTS:
            k = j + ".pos"
            lim = C.DANCE_LIMITS.get(j, 15.0)
            target = self.stage[k] + clamp(off.get(j, 0.0), -lim, lim)
            step = clamp(target - self.cmd[k], -self.caps[j], self.caps[j])
            self.cmd[k] += step                      # slew-capped, guaranteed smooth
            pose[k] = self.cmd[k]
        return pose


# ────────────────────────── theatrics ──────────────────────────

def listening_pose(stage, t0):
    """Head cocked, tiny sway — held while analysis genuinely runs. Eases in
    over the first ~0.6 s so it glides into the tilt instead of snapping."""
    t = time.time() - t0
    ease = min(1.0, t / 0.6)
    pose = dict(stage)
    pose["wrist_roll.pos"] += ease * (16.0 + 2.0 * math.sin(2.0 * math.pi * 0.4 * t))
    pose["wrist_flex.pos"] += ease * C.DANCE_NOD_SIGN * -4.0
    pose["shoulder_lift.pos"] += ease * 1.5 * math.sin(2.0 * math.pi * 0.25 * t)
    return pose


def perk(robot, stage):
    """'I know this one!' — quick eager rise + mouth pop, then back to stage."""
    up = dict(stage)
    up["shoulder_lift.pos"] += C.DANCE_NOD_SIGN * -6.0
    up["wrist_flex.pos"] += C.DANCE_NOD_SIGN * -8.0
    up["gripper.pos"] += C.DANCE_MOUTH_SIGN * 12.0
    ramp_to(robot, up, seconds=0.35)
    ramp_to(robot, stage, seconds=0.45)


# ────────────────────────── main ──────────────────────────

def set_stage(robot):
    print("\n== SET STAGE ==  torque OFF. Pose the DANCE stance: upright, alert,")
    print("head up, mid-range on every joint (room to bob/sway both ways). Enter...")
    robot.bus.disable_torque()
    input()
    pose = pose_now(robot)
    robot.bus.enable_torque()
    robot.send_action(pose)
    with open(STAGE_PATH, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"saved {STAGE_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Dum-E dances")
    ap.add_argument("song", nargs="?", help="audio file or YouTube/URL")
    ap.add_argument("--no-arm", action="store_true", help="dry run, no robot")
    ap.add_argument("--set-stage", action="store_true", help="capture the dance stance")
    args = ap.parse_args()

    robot = None
    if not args.no_arm:
        from arm_utils import connect
        robot = connect()

    try:
        if args.set_stage:
            if robot is None:
                raise SystemExit("--set-stage needs the arm")
            set_stage(robot)
            return
        if not args.song:
            raise SystemExit("give me a song (file or URL)")

        path = fetch(args.song)

        if os.path.exists(STAGE_PATH):
            with open(STAGE_PATH) as f:
                stage = json.load(f)
        elif robot is not None:
            print("[warn] no stage pose captured (dance.py --set-stage) — "
                  "using the CURRENT pose as the stage.")
            stage = pose_now(robot)
        else:
            stage = {j + ".pos": 0.0 for j in JOINTS}

        if robot is not None:
            ramp_to(robot, stage, seconds=2.0)

        # ---- recognition moment ----
        t0 = time.time()
        listen = (lambda: robot.send_action(listening_pose(stage, t0))) if robot else None
        sheet, known = load_sheet(path, listen_cb=listen)
        print(f"tempo {sheet['tempo']:.0f} bpm, {len(sheet['beats'])} beats, "
              f"{len(sheet['tiers'])} phrases " + ("[cache hit]" if known else "[learned]"))
        if robot is not None:
            if known:
                print("… I know this one!")
                perk(robot, stage)
            else:
                ramp_to(robot, stage, seconds=0.8)

        # ---- perform ----
        seed = hashlib.sha1(open(path, "rb").read(65536)).hexdigest()
        cond = Conductor(sheet, stage, seed)
        player = Player(path)
        player.start()
        print("dancing — Ctrl-C to stop.")
        period = 1.0 / C.DANCE_HZ
        t_next = time.time()
        try:
            while not player.done:
                pose = cond.pose_at(player.pos)
                if robot is not None:
                    robot.send_action(pose)
                else:
                    b = cond.beat_index(player.pos)
                    print(f"beat {b:7.2f}  pan {pose['shoulder_pan.pos']:+7.2f} "
                          f"flex {pose['wrist_flex.pos']:+7.2f}", end="\r")
                t_next += period
                dt = t_next - time.time()
                if dt > 0:
                    time.sleep(dt)
        finally:
            player.stop()
        # fade out gracefully
        if robot is not None:
            for k in range(int(C.DANCE_HZ * 1.5)):
                fade = 1.0 - (k + 1) / (C.DANCE_HZ * 1.5)
                robot.send_action(cond.pose_at(player.pos, master=fade))
                time.sleep(period)
        print("\nthat's the song.")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if robot is not None:
            safe_park(robot)


if __name__ == "__main__":
    main()
