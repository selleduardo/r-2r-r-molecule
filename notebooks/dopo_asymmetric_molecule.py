"""
DOPO in Asymmetric Three-Ring Photonic Molecule
================================================
Dual pump on S/AS external supermodes -> parametric oscillation in central (C) supermode.
Reference: arXiv 2602.11697 | Coupled-Mode Theory in the supermode basis.

Sections
--------
1.  Bare-ring frequencies
2.  CMT Hamiltonian & supermode eigensystem
3.  Numerical parameters
4.  Supermode dispersion curves
5.  Supermode decay rates
6.  FWM nonlinear overlap integrals
7.  Phase-matching conditions
8.  Parametric gain matrix
9.  Gain vs pump power (interactive slider)
10. Oscillation threshold vs pump mode index
11. Gain map: pump power vs signal detuning (interactive slider)
12. Design diagnostics summary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from scipy.linalg import eig

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

p = dict(
    omega0 = 0,
    J      = 3 / np.sqrt(2),   # inter-ring coupling
    d1     = 400.0,             # FSR outer ring 1
    d2     = 1.0,               # GVD outer ring 1
    D1     = 402.5,             # FSR central ring
    D2     = 1.05,              # GVD central ring
    dp1    = 400.0,             # FSR outer ring 3  (= d1 for symmetric limit)
    dp2    = 1.0,               # GVD outer ring 3
    kappa  = 1.0,               # total loss rate (normalised)
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BARE-RING FREQUENCIES
# ─────────────────────────────────────────────────────────────────────────────

def omega1(mu, p): return p["omega0"] + p["d1"]*mu  + (p["d2"] /2)*mu**2
def omega2(mu, p): return p["omega0"] + p["D1"]*mu  + (p["D2"] /2)*mu**2
def omega3(mu, p): return p["omega0"] + p["dp1"]*mu + (p["dp2"]/2)*mu**2

# ─────────────────────────────────────────────────────────────────────────────
# 3. CMT HAMILTONIAN & EIGENSYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def cmt_matrix(mu, p):
    """3x3 nearest-neighbour CMT frequency matrix."""
    J = p["J"]
    return np.array([
        [omega1(mu, p), J,             0           ],
        [J,             omega2(mu, p), J           ],
        [0,             J,             omega3(mu, p)]
    ])

def eigensystem(mu, p):
    """
    Returns eigenvalues (sorted ascending) and normalised eigenvectors.
    Columns of `vecs` are eigenvectors; vecs[:, k] corresponds to evals[k].
    Labels: 0=S (symmetric/low), 1=C (central/dark), 2=AS (antisymmetric/high).
    """
    mat = cmt_matrix(mu, p)
    evals, evecs = np.linalg.eigh(mat)   # real symmetric -> sorted ascending
    # Normalise (eigh already returns orthonormal, but be explicit)
    norms = np.linalg.norm(evecs, axis=0, keepdims=True)
    evecs = evecs / norms
    return evals, evecs   # evals[0]=S, evals[1]=C, evals[2]=AS

def omega_S(mu, p):  return eigensystem(mu, p)[0][0]
def omega_C(mu, p):  return eigensystem(mu, p)[0][1]
def omega_AS(mu, p): return eigensystem(mu, p)[0][2]

def bS(mu, p):  return eigensystem(mu, p)[1][:, 0]
def bC(mu, p):  return eigensystem(mu, p)[1][:, 1]
def bAS(mu, p): return eigensystem(mu, p)[1][:, 2]

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUPERMODE DECAY RATES
#    Bus couples to rings 1 & 3 (not ring 2).
#    Gamma_b[mu] = kappa * sum_i |(1 + delta_{i,2}) * bC[0]_i * conj(b_b[mu]_i)|^2
#    (index convention: ring 1 -> i=0, ring 2 -> i=1, ring 3 -> i=2  in Python)
# ─────────────────────────────────────────────────────────────────────────────

def _bus_weight(i):
    """Factor (1 + delta_{ring2, i}): ring 2 (index 1) counts double (dark)."""
    return 1 + (1 if i == 1 else 0)

def gamma_supermode(b_mu, b_C0, kappa):
    """Generic supermode decay rate given its profile b_mu and pump profile bC[0]."""
    total = 0.0
    for i in range(3):
        total += abs(_bus_weight(i) * b_C0[i] * np.conj(b_mu[i]))**2
    return kappa * total

def Gamma_S(mu, p):  return gamma_supermode(bS(mu, p),  bC(0, p), p["kappa"])
def Gamma_C(mu, p):  return gamma_supermode(bC(mu, p),  bC(0, p), p["kappa"])
def Gamma_AS(mu, p): return gamma_supermode(bAS(mu, p), bC(0, p), p["kappa"])

# ─────────────────────────────────────────────────────────────────────────────
# 5. FWM OVERLAP INTEGRALS
# ─────────────────────────────────────────────────────────────────────────────

def fwm_overlap(v1, v2, w1, w2):
    """Sum_i v1_i * v2_i * conj(w1_i) * conj(w2_i)"""
    return np.sum(v1 * v2 * np.conj(w1) * np.conj(w2))

def Lambda_DOPO(mu_p, p):
    """S(+mu_p) + AS(-mu_p) -> C(0) + C(0)"""
    return fwm_overlap(bS(mu_p, p), bAS(-mu_p, p), bC(0, p), bC(0, p))

def Lambda_TypeI(mu_p, p):
    """S(+mu_p) + S(-mu_p) -> C(0) + C(0)  [competing]"""
    return fwm_overlap(bS(mu_p, p), bS(-mu_p, p), bC(0, p), bC(0, p))

def Lambda_ASAS(mu_p, p):
    """AS(+mu_p) + AS(-mu_p) -> C(0) + C(0)  [competing]"""
    return fwm_overlap(bAS(mu_p, p), bAS(-mu_p, p), bC(0, p), bC(0, p))

# ─────────────────────────────────────────────────────────────────────────────
# 6. PHASE-MATCHING MISMATCHES
# ─────────────────────────────────────────────────────────────────────────────

def delta_omega_DOPO(mu_p, p):
    return omega_S(mu_p, p) + omega_AS(-mu_p, p) - 2*omega_C(0, p)

def delta_omega_ASAS(mu_p, p):
    return omega_AS(mu_p, p) + omega_AS(-mu_p, p) - 2*omega_C(0, p)

def delta_omega_SS(mu_p, p):
    return omega_S(mu_p, p) + omega_S(-mu_p, p) - 2*omega_C(0, p)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PARAMETRIC GAIN MATRIX & GAIN
#    2x2 stability matrix for the degenerate signal/idler pair in C(0):
#      G = [[-i(Gamma_C*Pp - DeltaC),   i*Lambda*Pp       ],
#           [-i*conj(Lambda)*Pp,         -i(Gamma_C*Pp - DeltaC)]]
#    Re[eigenvalue] > 0  =>  parametric oscillation
# ─────────────────────────────────────────────────────────────────────────────

def gain_matrix_C(mu_p, Pp, DeltaC, p):
    Lam  = Lambda_DOPO(mu_p, p)
    GamC = Gamma_C(mu_p, p)
    diag = -1j * (GamC * Pp - DeltaC)
    return np.array([
        [diag,            1j * Lam * Pp          ],
        [-1j * np.conj(Lam) * Pp,  diag          ]
    ])

def gain_C(mu_p, Pp, DeltaC, p):
    evals = np.linalg.eigvals(gain_matrix_C(mu_p, Pp, DeltaC, p))
    return float(np.max(np.real(evals)))

def Pp_threshold(mu_p, p):
    """Threshold pump power (zero detuning)."""
    GamC = Gamma_C(mu_p, p)
    Lam  = abs(Lambda_DOPO(mu_p, p))
    if Lam < 1e-14:
        return np.inf
    return GamC / Lam

# ─────────────────────────────────────────────────────────────────────────────
# 8. DESIGN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def design_summary(p, mu_range=range(1, 9)):
    mismatches = [abs(delta_omega_DOPO(mu, p)) for mu in mu_range]
    mu_opt = list(mu_range)[int(np.argmin(mismatches))]

    print("=" * 45)
    print("        DOPO Design Summary")
    print("=" * 45)
    print(f"  Optimal pump mode      mu_p  = {mu_opt}")
    print(f"  Phase mismatch  Δω          = {delta_omega_DOPO(mu_opt, p):.4f}")
    print(f"  FWM Overlap     |Λ|         = {abs(Lambda_DOPO(mu_opt, p)):.4f}")
    print(f"  Pump decay      Γ_S         = {Gamma_S(mu_opt, p):.4f}")
    print(f"  Signal decay    Γ_C         = {Gamma_C(mu_opt, p):.4f}  (quasi-dark)")
    print(f"  AS pump decay   Γ_AS        = {Gamma_AS(mu_opt, p):.4f}")
    print(f"  Threshold       Pp_th       = {Pp_threshold(mu_opt, p):.4f}")
    print("=" * 45)
    return mu_opt

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_static(p):
    """Figures 1-4: dispersion, decay rates, overlaps, phase matching, threshold."""

    mu_arr   = np.linspace(-10, 10, 400)
    mu_p_arr = np.linspace(1, 8, 300)
    mu_disc  = np.arange(1, 9)

    # --- pre-compute arrays ---
    dS  = np.array([omega_S(m, p)  - p["d1"]*m - (p["d2"]/2)*m**2 for m in mu_arr])
    dC  = np.array([omega_C(m, p)  - p["d1"]*m - (p["d2"]/2)*m**2 for m in mu_arr])
    dAS = np.array([omega_AS(m, p) - p["d1"]*m - (p["d2"]/2)*m**2 for m in mu_arr])

    gS  = np.array([Gamma_S(m,  p) for m in mu_arr])
    gC  = np.array([Gamma_C(m,  p) for m in mu_arr])
    gAS = np.array([Gamma_AS(m, p) for m in mu_arr])

    lD  = np.array([abs(Lambda_DOPO(m, p))  for m in mu_p_arr])
    lI  = np.array([abs(Lambda_TypeI(m, p)) for m in mu_p_arr])
    lAA = np.array([abs(Lambda_ASAS(m, p))  for m in mu_p_arr])

    dmD  = np.array([delta_omega_DOPO(m, p)  for m in mu_p_arr])
    dmAA = np.array([delta_omega_ASAS(m, p)  for m in mu_p_arr])
    dmSS = np.array([delta_omega_SS(m, p)    for m in mu_p_arr])

    pth = np.array([Pp_threshold(m, p) for m in mu_disc])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("DOPO in Asymmetric Photonic Molecule", fontsize=14, fontweight="bold")

    # (1) Dispersion
    ax = axes[0, 0]
    ax.plot(mu_arr, dS,  "b",   label="S")
    ax.plot(mu_arr, dC,  "g--", label="C (quasi-dark)")
    ax.plot(mu_arr, dAS, "r",   label="AS")
    ax.set_xlabel("Mode index μ")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Supermode Dispersion")
    ax.legend(); ax.grid(True)

    # (2) Decay rates
    ax = axes[0, 1]
    ax.plot(mu_arr, gS,  "b",   label="Γ_S")
    ax.plot(mu_arr, gC,  "g--", label="Γ_C (quasi-dark)")
    ax.plot(mu_arr, gAS, "r",   label="Γ_AS")
    ax.set_xlabel("Mode index μ")
    ax.set_ylabel("Decay rate / κ")
    ax.set_title("Supermode Decay Rates")
    ax.legend(); ax.grid(True)

    # (3) FWM overlaps
    ax = axes[0, 2]
    ax.plot(mu_p_arr, lD,  "b",    linewidth=2, label="|Λ| S+AS→C+C")
    ax.plot(mu_p_arr, lI,  "r--",              label="|Λ| S+S→C+C")
    ax.plot(mu_p_arr, lAA, "orange", linestyle="-.", label="|Λ| AS+AS→C+C")
    ax.set_xlabel("Pump mode μ_p")
    ax.set_ylabel("|Λ|")
    ax.set_title("FWM Overlaps")
    ax.legend(); ax.grid(True)

    # (4) Phase matching
    ax = axes[1, 0]
    ax.plot(mu_p_arr, dmD,  "b",              label="ΔΩ S+AS→2C")
    ax.plot(mu_p_arr, dmAA, "r--",            label="ΔΩ AS+AS→2C")
    ax.plot(mu_p_arr, dmSS, "orange", linestyle="-.", label="ΔΩ S+S→2C")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pump mode μ_p")
    ax.set_ylabel("Frequency mismatch")
    ax.set_title("Phase-Matching Landscape")
    ax.legend(); ax.grid(True)

    # (5) Threshold
    ax = axes[1, 1]
    ax.plot(mu_disc, pth, "bo-", markersize=7)
    ax.set_xlabel("Pump mode μ_p")
    ax.set_ylabel("Threshold Pp_th")
    ax.set_title("Oscillation Threshold vs μ_p")
    ax.set_xticks(mu_disc); ax.grid(True)

    # (6) Gain vs Pp for mu_p = optimal
    mu_opt = design_summary(p)
    Pp_arr = np.linspace(0, 5, 200)
    gain_arr = np.array([gain_C(mu_opt, Pp, 0, p) for Pp in Pp_arr])
    ax = axes[1, 2]
    ax.plot(Pp_arr, gain_arr, "b", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pump power Pp")
    ax.set_ylabel("Gain Re[λ]")
    ax.set_title(f"Parametric Gain  (μ_p = {mu_opt}, ΔC = 0)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("dopo_static.png", dpi=150, bbox_inches="tight")
    print("Saved: dopo_static.png")
    plt.show()


def plot_gain_slider(p):
    """Interactive: gain vs Pp with slider over mu_p."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)

    Pp_arr = np.linspace(0, 5, 200)
    mu_p0  = 1

    gain_arr = np.array([gain_C(mu_p0, Pp, 0, p) for Pp in Pp_arr])
    line, = ax.plot(Pp_arr, gain_arr, "b", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pump power Pp")
    ax.set_ylabel("Gain Re[λ]")
    ax.set_title(f"DOPO Parametric Gain  (μ_p = {mu_p0}, ΔC = 0)")
    ax.grid(True)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
    slider = Slider(ax_slider, "Pump μ_p", 1, 8, valinit=mu_p0, valstep=1)

    def update(val):
        mu_p = int(slider.val)
        new_gain = np.array([gain_C(mu_p, Pp, 0, p) for Pp in Pp_arr])
        line.set_ydata(new_gain)
        ax.set_title(f"DOPO Parametric Gain  (μ_p = {mu_p}, ΔC = 0)")
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_gain_map_slider(p):
    """Interactive: 2D gain map (DeltaC vs Pp) with slider over mu_p."""
    DeltaC_arr = np.linspace(-3, 3, 120)
    Pp_arr     = np.linspace(0, 5, 120)
    DC_grid, Pp_grid = np.meshgrid(DeltaC_arr, Pp_arr)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    mu_p0 = 1
    gain_grid = np.vectorize(lambda dc, pp: gain_C(mu_p0, pp, dc, p))(DC_grid, Pp_grid)

    im = ax.pcolormesh(DeltaC_arr, Pp_arr, gain_grid, cmap="RdBu_r", shading="auto")
    cbar = plt.colorbar(im, ax=ax, label="Gain Re[λ]")
    cont = ax.contour(DeltaC_arr, Pp_arr, gain_grid, levels=[0], colors="k", linewidths=1.5)
    ax.set_xlabel("Signal detuning ΔC")
    ax.set_ylabel("Pump power Pp")
    ax.set_title(f"Gain Map  (μ_p = {mu_p0})")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
    slider = Slider(ax_slider, "Pump μ_p", 1, 8, valinit=mu_p0, valstep=1)

    def update(val):
        mu_p = int(slider.val)
        new_gain = np.vectorize(lambda dc, pp: gain_C(mu_p, pp, dc, p))(DC_grid, Pp_grid)
        im.set_array(new_gain.ravel())
        im.set_clim(new_gain.min(), new_gain.max())
        for coll in cont.collections:
            coll.remove()
        ax.contour(DeltaC_arr, Pp_arr, new_gain, levels=[0], colors="k", linewidths=1.5)
        ax.set_title(f"Gain Map  (μ_p = {mu_p})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRunning DOPO Asymmetric Molecule analysis...\n")

    # Static summary figures
    plot_all_static(p)

    # Interactive gain vs Pp
    plot_gain_slider(p)

    # Interactive 2D gain map
    plot_gain_map_slider(p)
