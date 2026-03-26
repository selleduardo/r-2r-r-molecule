"""
General N-mode CMT solver
=========================
Derived from thesis Eq. (4.3) via symbolic expansion in Mathematica.

The triple sum Σ_{α,β,γ: α−β+γ=μ} a[α]·a[β]*·a[γ] is enumerated
automatically for any set of modes. Classification verified:

  SPM : triple (μ,μ,μ)                     → −i|a[μ]|²·a[μ]
  XPM : triples (ν,ν,μ) and (μ,ν,ν)        → −i·2·|a[ν]|²·a[μ]  each ν≠μ
  FWM : remaining triples                   → energy transfer between modes

Term counts confirmed against thesis Eq. (4.7): 3N²+3N−μ²+1 per mode.
"""

from itertools import product as iproduct
import numpy as np


def cmt_rhs(tau, state, modes, Delta, s):
    """
    General N-mode CMT RHS — thesis Eq. (4.3).
    Enumerates all phase-matched triples (α,β,γ) with α−β+γ=μ automatically.

    Parameters
    ----------
    tau   : current normalized time τ = (κ/2)t  (not used — autonomous)
    state : real array, length 2*len(modes), packed as
              [Re(a[modes[0]]), ..., Re(a[modes[-1]]),
               Im(a[modes[0]]), ..., Im(a[modes[-1]])]
    modes : list of integer mode indices, e.g. [-1, 0, 1]
    Delta : callable Delta(mu)  OR  dict {mu: normalized_detuning}
    s     : real drive amplitude, applied to pump mode mu=0 only

    Returns
    -------
    drhs : real array same length as state
    """
    n   = len(modes)
    idx = {mu: k for k, mu in enumerate(modes)}

    # Reconstruct complex amplitudes from real state vector
    a = {mu: state[idx[mu]] + 1j * state[n + idx[mu]] for mu in modes}

    # Detuning accessor — accepts either a callable or a dict
    get_D = Delta if callable(Delta) else lambda mu: Delta[mu]

    da = {}
    for mu in modes:
        # Linear part: decay + detuning rotation
        da[mu] = -(1.0 + 1j * get_D(mu)) * a[mu]

        # Nonlinear sum: all phase-matched triples
        for alpha, beta, gamma in iproduct(modes, repeat=3):
            if alpha - beta + gamma == mu:
                da[mu] -= 1j * a[alpha] * np.conj(a[beta]) * a[gamma]

        # Coherent pump drive on mode mu=0 only
        if mu == 0:
            da[mu] += s

    # Pack back into real state vector [Re parts..., Im parts...]
    return [da[mu].real for mu in modes] + [da[mu].imag for mu in modes]


def make_detunings(Delta_p, d2, modes):
    """
    Detuning for each mode (thesis Eq. 4.4b, normalized units).

        Δμ = Δp + μ²·d2

    Parameters
    ----------
    Delta_p : normalized pump detuning  Δp = (2/κ)(ωp − ω0)
    d2      : normalized dispersion     d2 = D2 / (κ/2)
    modes   : list of mode indices

    Returns
    -------
    dict {mu: Delta_mu}
    """
    return {mu: Delta_p + mu**2 * d2 for mu in modes}


# ── Verification: confirm this matches the explicit 3-mode code ──────────────

def _verify():
    """
    Check that cmt_rhs with modes=[-1,0,1] gives the same result as
    the explicit three_mode_rhs from three_mode_cmt.py at a random state.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Random complex state
    rng = np.random.default_rng(42)
    ap  = rng.standard_normal() + 1j * rng.standard_normal()
    as_ = rng.standard_normal() + 1j * rng.standard_normal()
    ai  = rng.standard_normal() + 1j * rng.standard_normal()

    modes  = [-1, 0, 1]
    Dp, Ds, Di = -2.0, -1.99, -1.99
    s = 1.5
    state6 = [ai.real, ap.real, as_.real, ai.imag, ap.imag, as_.imag]

    # General solver (modes in order [-1, 0, +1])
    Delta  = {-1: Di, 0: Dp, 1: Ds}
    result = cmt_rhs(0, state6, modes, Delta, s)

    # Extract da for each mode
    dai_gen = result[0] + 1j * result[3]
    dap_gen = result[1] + 1j * result[4]
    das_gen = result[2] + 1j * result[5]

    # Explicit formulas (thesis 4.6a-c)
    np_ = abs(ap)**2; ns = abs(as_)**2; ni = abs(ai)**2

    dap_ex = (-(1 + 1j*Dp)*ap - 1j*(np_+2*ns+2*ni)*ap
              - 1j*2*as_*ai*np.conj(ap) + s)
    das_ex = (-(1 + 1j*Ds)*as_ - 1j*(ns+2*np_+2*ni)*as_
              - 1j*ap**2*np.conj(ai))
    dai_ex = (-(1 + 1j*Di)*ai  - 1j*(ni+2*np_+2*ns)*ai
              - 1j*ap**2*np.conj(as_))

    tol = 1e-12
    ok  = (abs(dap_gen - dap_ex) < tol and
           abs(das_gen - das_ex) < tol and
           abs(dai_gen - dai_ex) < tol)

    print("Verification against explicit 3-mode equations:")
    print(f"  |Δ dap| = {abs(dap_gen - dap_ex):.2e}")
    print(f"  |Δ das| = {abs(das_gen - das_ex):.2e}")
    print(f"  |Δ dai| = {abs(dai_gen - dai_ex):.2e}")
    print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


if __name__ == "__main__":
    _verify()
