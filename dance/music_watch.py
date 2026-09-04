"""
music_watch.py — Dum-E dances to whatever YOU play in Chrome (YouTube Music etc.)

    python dance/music_watch.py            watch + dance (Windows only)
    python dance/music_watch.py --no-arm   dry run (prints, no robot)

HOW IT WORKS
  Chrome publishes the current track's title, artist, play state and playback
  POSITION to Windows' media API (SMTC — the volume-flyout media card). We never
  touch the audio itself:
    1. WATCH: poll SMTC until a track from a watched app (config.MUSIC_SOURCE_APPS)
       is playing. Dum-E holds its stage pose with a tiny idle sway.
    2. MATCH: yt-dlp searches "<artist> <title> audio", fetches the same song into
       dance/songs/ and the usual dance-sheet pipeline runs (cache hit = it
       recognizes the song BY NAME -> perk; miss = listening head-tilt while it
       learns). If the fetched track's length differs a lot from what Chrome
       reports, we warn — the beat grid may be a different edit.
    3. SYNC: the dance clock phase-locks to the position Chrome reports,
       extrapolated between polls and nudged gently on each resync (a big jump
       means you seeked -> snap). Pause freezes the clock and the dance eases to
       the stage pose; resume picks it right back up. Track change / autoplay ->
       fade out, back to WATCH, next song.

Needs:  pip install winsdk        (Windows 10/11 only)
Ads on free YouTube Music show up as short "tracks" — anything shorter than
config.MUSIC_MIN_DURATION_S is ignored (Dum-E waits them out, unimpressed).
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))     # repo root: config, arm_utils
sys.path.insert(0, _HERE)                      # dance.py, dance_moves.py

import config as C                              # noqa: E402
from arm_utils import pose_now, ramp_to, safe_park  # noqa: E402
from dance import (Conductor, JOINTS, STAGE_PATH, fetch,  # noqa: E402
                   listening_pose, load_sheet, perk)


# ────────────────────────── SMTC (what is Chrome playing?) ──────────────────────────

@dataclass
class NowPlaying:
    title: str
    artist: str
    playing: bool
    pos: float        # seconds, extrapolated to "right now"
    duration: float   # seconds (0 = unknown)

    @property
    def key(self):
        return (self.title, self.artist)


class Smtc:
    """Thin sync wrapper over Windows' async media-session API."""

    def __init__(self):
        try:
            from winsdk.windows.media import control as wmc
        except ImportError:
            raise SystemExit(
                "winsdk not installed (or not on Windows).\n"
                "    pip install winsdk")
        self._wmc = wmc
        self._loop = asyncio.new_event_loop()
        self._mgr = self._loop.run_until_complete(
            wmc.GlobalSystemMediaTransportControlsSessionManager.request_async())

    def snapshot(self):
        """NowPlaying from the first watched app that has a track, or None.
        A PLAYING session wins over a paused one."""
        try:
            return self._loop.run_until_complete(self._read())
        except OSError:
            return None

    async def _read(self):
        import datetime as dt
        PS = self._wmc.GlobalSystemMediaTransportControlsSessionPlaybackStatus
        best = None
        for s in self._mgr.get_sessions():
            app = (s.source_app_user_model_id or "").lower()
            if not any(a in app for a in C.MUSIC_SOURCE_APPS):
                continue
            try:
                props = await s.try_get_media_properties_async()
                info = s.get_playback_info()
                tl = s.get_timeline_properties()
            except OSError:
                continue
            title = (props.title or "").strip()
            if not title:
                continue
            playing = info.playback_status == PS.PLAYING
            pos = tl.position.total_seconds()
            duration = max(0.0, tl.end_time.total_seconds())
            # extrapolate: SMTC updates are coarse; last_updated_time says how stale
            lu = tl.last_updated_time
            if playing and lu is not None and lu.year > 2000:
                stale = (dt.datetime.now(dt.timezone.utc) - lu).total_seconds()
                pos += max(0.0, stale)
            if duration > 0:
                pos = min(pos, duration)
            np_ = NowPlaying(title, (props.artist or "").strip(), playing, pos, duration)
            if playing:
                return np_
            best = best or np_
        return best


# ────────────────────────── the live clock ──────────────────────────

class LiveClock:
    """Song time driven by SMTC reports. Between polls it free-runs on the
    monotonic clock; each report nudges it (or snaps after a seek). Pausing
    freezes it."""

    def __init__(self, np_):
        self.pos = np_.pos
        self.mono = time.monotonic()
        self.playing = np_.playing

    def t(self):
        if self.playing:
            return self.pos + (time.monotonic() - self.mono)
        return self.pos

    def update(self, np_):
        now = time.monotonic()
        cur = self.t()
        if np_.playing:
            err = np_.pos - cur
            cur = np_.pos if (not self.playing or abs(err) > C.MUSIC_SNAP_S) \
                else cur + 0.2 * err
        self.pos, self.mono, self.playing = cur, now, np_.playing


# ────────────────────────── stage + idle act ──────────────────────────

def load_stage(robot):
    if os.path.exists(STAGE_PATH):
        with open(STAGE_PATH) as f:
            return json.load(f)
    if robot is not None:
        print("[warn] no stage pose (dance/dance.py --set-stage) — using current pose.")
        return pose_now(robot)
    return {j + ".pos": 0.0 for j in JOINTS}


def idle_sway(stage, t0):
    """Barely-there breathing while waiting for you to press play."""
    t = time.time() - t0
    pose = dict(stage)
    s = 0.8 * math.sin(2.0 * math.pi * 0.15 * t)
    pose["shoulder_lift.pos"] += s
    pose["elbow_flex.pos"] -= s
    return pose


def search_query(np_):
    """SMTC metadata -> a yt-dlp search. 'audio' biases toward the track itself
    rather than a music video with a long intro."""
    q = f"{np_.artist} {np_.title} audio" if np_.artist else f"{np_.title} audio"
    return "ytsearch1:" + re.sub(r"\s+", " ", q).strip()


# ────────────────────────── perform (SMTC-clocked) ──────────────────────────

def perform(robot, cond, smtc, np0):
    """Dance until the track changes or disappears. Returns the latest
    NowPlaying (or None). Pause/resume and song-end are handled by easing the
    master envelope — no discontinuities anywhere."""
    clock = LiveClock(np0)
    period = 1.0 / C.DANCE_HZ
    poll_every = max(1, int(C.MUSIC_POLL_S * C.DANCE_HZ))
    fade_step = 1.0 / (1.5 * C.DANCE_HZ)         # 1.5 s fades
    end_t = float(cond.beats[-1]) + 4.0
    master, tick, t_next = 0.0, 0, time.time()
    while True:
        tick += 1
        if tick % poll_every == 0:
            np_ = smtc.snapshot()
            if np_ is None or np_.key != np0.key:
                return np_                        # track changed / player gone
            clock.update(np_)
        t = clock.t()
        want = 1.0 if (clock.playing and t < end_t) else 0.0
        master += max(-fade_step, min(fade_step, want - master))
        pose = cond.pose_at(t, master)
        if robot is not None:
            robot.send_action(pose)
        else:
            print(f"t {t:7.2f}  master {master:4.2f}  "
                  f"pan {pose['shoulder_pan.pos']:+7.2f}", end="\r")
        t_next += period
        dt = t_next - time.time()
        if dt > 0:
            time.sleep(dt)
        else:
            t_next = time.time()


# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dum-E dances to whatever Chrome plays")
    ap.add_argument("--no-arm", action="store_true", help="dry run, no robot")
    args = ap.parse_args()

    smtc = Smtc()

    robot = None
    if not args.no_arm:
        from arm_utils import connect
        robot = connect()

    skip = set()          # (title, artist) we failed to fetch — don't retry forever
    try:
        stage = load_stage(robot)
        if robot is not None:
            ramp_to(robot, stage, seconds=2.0)
        print("watching Chrome — play something on YouTube Music. Ctrl-C to stop.")

        np_ = None
        t0 = time.time()
        while True:
            # ---- WATCH: idle until a real track is playing ----
            while np_ is None or not np_.playing or np_.key in skip or \
                    (0 < np_.duration < C.MUSIC_MIN_DURATION_S):
                if np_ is not None and 0 < np_.duration < C.MUSIC_MIN_DURATION_S \
                        and np_.playing and np_.key not in skip:
                    print(f"\n'{np_.title}' is {np_.duration:.0f}s — probably an ad. "
                          "waiting it out, unimpressed.")
                    skip.add(np_.key)             # each ad only announced once
                deadline = time.time() + C.MUSIC_POLL_S
                while time.time() < deadline:
                    if robot is not None:
                        robot.send_action(idle_sway(stage, t0))
                    time.sleep(1.0 / C.DANCE_HZ)
                np_ = smtc.snapshot()

            # ---- MATCH: fetch + learn/recognize ----
            print(f"\n♪ hearing: {np_.title} — {np_.artist or 'unknown artist'}")
            try:
                path = fetch(search_query(np_))
            except (SystemExit, Exception) as e:  # noqa: BLE001
                print(f"[warn] couldn't fetch that ({e}) — skipping this track.")
                skip.add(np_.key)
                np_ = None
                continue
            lt0 = time.time()
            listen = (lambda: robot.send_action(listening_pose(stage, lt0))) \
                if robot else None
            sheet, known = load_sheet(path, listen_cb=listen)
            print(f"tempo {sheet['tempo']:.0f} bpm, {len(sheet['beats'])} beats "
                  + ("[I know this one!]" if known else "[learned]"))
            if robot is not None:
                if known:
                    perk(robot, stage)
                else:
                    ramp_to(robot, stage, seconds=0.8)
            if np_.duration > 0:
                import soundfile as sf
                got = sf.info(path).duration
                if abs(np_.duration - got) > max(3.0, 0.03 * np_.duration):
                    print(f"[warn] fetched version is {got:.0f}s but Chrome reports "
                          f"{np_.duration:.0f}s — might be a different edit; "
                          "the groove may drift.")

            # ---- SYNC + DANCE ----
            seed = hashlib.sha1(open(path, "rb").read(65536)).hexdigest()
            cond = Conductor(sheet, stage, seed)
            fresh = smtc.snapshot()               # re-read: fetching took a while
            if fresh is None or fresh.key != np_.key:
                np_ = fresh
                continue
            print("dancing along.")
            np_ = perform(robot, cond, smtc, fresh)
            if robot is not None:
                ramp_to(robot, stage, seconds=0.8)
            t0 = time.time()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if robot is not None:
            safe_park(robot)


if __name__ == "__main__":
    main()
