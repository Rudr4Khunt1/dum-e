"""
filters.py — signal smoothing.

OneEuro is the right tool for noisy interactive input (it was designed for exactly
this: jittery tracked positions driving something in real time).

The problem with a plain EMA / low-pass: one fixed smoothing constant forces you to
choose between jitter and lag. Smooth enough to kill the shake when the hand is
still, and the arm lags badly when the hand moves. OneEuro fixes that by making the
cutoff *speed-dependent*:

  * hand still  -> low cutoff  -> heavy smoothing  -> jitter dies
  * hand moving -> high cutoff -> light smoothing  -> stays responsive

Tuning:
  min_cutoff : lower  = smoother when still (but laggier)
  beta       : higher = less lag when moving fast
"""
import math


class OneEuro:
    def __init__(self, freq, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            return x
        # smooth the derivative first, then use its magnitude to open the cutoff
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
