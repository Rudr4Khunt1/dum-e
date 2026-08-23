"""
dance_moves.py — Dum-E's move vocabulary. Pure math, no robot imports.

SMOOTHNESS BY CONSTRUCTION: every motion here is a continuous function of the
song's beat index b (a float that grows 1.0 per beat). Sampled at 30 Hz these
curves cannot jitter — there are no discrete jumps anywhere. Figures hand off by
crossfading their parameters over a full beat, and the per-joint slew caps in
the conductor are only a safety net, not the smoothing mechanism.

THE LAYERS (all summed on top of the stage pose):
  groove  shoulder+elbow counter-phase bob (the body breathes with the tempo)
  nod     wrist_flex dips landing ON each beat (raised-cosine pulse)
  sway    slow, small shoulder_pan arc (base swings the whole arm — kept slow
          and modest: the base rings if slewed hard)
  waggle  wrist_roll side-tilt of the head
  mouth   gripper pops on ACCENT beats only (sparingly — it's the exclamation)

A FIGURE is one parameter set for those layers = one "dance move". The chooser
picks a figure per 8-beat phrase, gated by the phrase's energy tier, seeded by
the song hash — so every song has ITS OWN repeatable dance.
"""
import math


def rc_pulse(x, width):
    """Raised-cosine bump centered at 0: 1 at x=0, smoothly 0 at |x|>=width."""
    if abs(x) >= width:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * x / width))


def beat_pulse(b, width):
    """rc_pulse around the NEAREST integer beat of continuous beat index b."""
    frac = b - math.floor(b)
    return rc_pulse(min(frac, 1.0 - frac), width)


# ── the figure library ─────────────────────────────────────────────────────
# Amplitudes in degrees (scaled by tier + global amplitude at runtime).
# tiers: which energy tiers this figure suits (0=quiet, 1=mid, 2=loud)

FIGURES = [
    dict(name="headnod",      tiers=(0, 1),  bob=2.0, bob_beats=4, nod=6.0,
         nod_width=0.32, sway=0.0, sway_beats=8, roll=0.0, roll_beats=4,
         mouth=0.0, accent_nod=3.0),
    dict(name="lounge_sway",  tiers=(0,),    bob=2.5, bob_beats=8, nod=2.5,
         nod_width=0.45, sway=6.0, sway_beats=8, roll=4.0, roll_beats=8,
         mouth=0.0, accent_nod=0.0),
    dict(name="groove",       tiers=(1, 2),  bob=4.0, bob_beats=2, nod=7.0,
         nod_width=0.30, sway=8.0, sway_beats=8, roll=5.0, roll_beats=4,
         mouth=5.0, accent_nod=4.0),
    dict(name="rock_out",     tiers=(2,),    bob=6.0, bob_beats=2, nod=10.0,
         nod_width=0.28, sway=10.0, sway_beats=4, roll=7.0, roll_beats=2,
         mouth=8.0, accent_nod=6.0),
    dict(name="side_bopper",  tiers=(1, 2),  bob=3.0, bob_beats=4, nod=6.0,
         nod_width=0.30, sway=12.0, sway_beats=4, roll=3.0, roll_beats=4,
         mouth=4.0, accent_nod=3.0),
    dict(name="wobblehead",   tiers=(1,),    bob=3.0, bob_beats=4, nod=4.0,
         nod_width=0.35, sway=4.0, sway_beats=8, roll=9.0, roll_beats=2,
         mouth=0.0, accent_nod=3.0),
]

TIER_MULT = {0: 0.55, 1: 0.85, 2: 1.15}


def pick_figures(rng, tiers):
    """One figure per phrase, tier-gated, no immediate repeats. Seeded rng ->
    the same song always dances the same dance."""
    seq, prev = [], None
    for tier in tiers:
        pool = [f for f in FIGURES if tier in f["tiers"] and f is not prev]
        if not pool:
            pool = [f for f in FIGURES if tier in f["tiers"]] or FIGURES
        fig = pool[rng.randrange(len(pool))]
        seq.append(fig)
        prev = fig
    return seq


def lerp_params(a, b, t):
    """Crossfade two figures' numeric params (t: 0=a .. 1=b)."""
    out = {}
    for k in ("bob", "nod", "sway", "roll", "mouth", "accent_nod"):
        out[k] = a[k] + (b[k] - a[k]) * t
    for k in ("bob_beats", "nod_width", "sway_beats", "roll_beats"):
        # period-ish params: switch at midpoint rather than lerp (lerping a
        # period causes phase weirdness; the amplitude crossfade hides the swap)
        out[k] = a[k] if t < 0.5 else b[k]
    return out


def figure_offsets(b, p, accent_env, scale):
    """Joint offsets (degrees, added to the stage pose) at beat index b for
    figure params p. accent_env: 0..1 envelope around accent beats. scale:
    tier * global amplitude."""
    off = {}
    bob = p["bob"] * math.sin(2.0 * math.pi * b / p["bob_beats"])
    off["shoulder_lift"] = scale * bob
    off["elbow_flex"] = scale * -bob                     # counter-phase: head stays level
    nod = p["nod"] * beat_pulse(b, p["nod_width"]) + p["accent_nod"] * accent_env
    off["wrist_flex"] = scale * nod                      # sign set by conductor config
    off["shoulder_pan"] = scale * p["sway"] * math.sin(2.0 * math.pi * b / p["sway_beats"])
    off["wrist_roll"] = scale * p["roll"] * math.sin(2.0 * math.pi * b / p["roll_beats"] + 1.3)
    off["gripper"] = scale * p["mouth"] * accent_env     # mouth pop on accents only
    return off
