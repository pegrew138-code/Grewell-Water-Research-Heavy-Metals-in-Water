# """
# Mercury Water Treatment: Simulation + Equilibrium / Stability / Sensitivity / Bifurcation
# ==========================================================================================
# GOVERNING EQUATION:
#     M[t+1] = M[t] - r * M[t] * N[t] + O[t]
#     M = mercury concentration (ppb), N = treatment intensity, O = daily inflow (ppb/day),
#     r = removal constant (placeholder, still needs a literature value).
# 
# THRESHOLD: 2.0 ppb = EPA Maximum Contaminant Level for mercury in drinking water.

import os
import csv
import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---------------------------------------------------------------- constants
THRESHOLD = 2.0        # ppb, EPA MCL for mercury
SAFETY_LIMIT = 0.9 * THRESHOLD   # optimizer tries to stay under this
R_REMOVAL = 0.4        # PLACEHOLDER removal constant
N_MAX = 3.0            # treatment capacity
N_MIN = 0.45           # CT3 minimum dose
CT4_DECAY = 0.015      # CT4: treatment loses 1.5%/day without maintenance
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
RES_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ==========================================================================
# 1. CONTAMINATION SCENARIOS
# ==========================================================================
def generate_contamination(scenario, days=90, I0=0.5, A=0.5, P=7, spike=1.0,
                            growth=0.012, decay=0.02, noise_std=0.05, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    noise = rng.normal(0, noise_std, days) if noise_std > 0 else np.zeros(days)

    if scenario == "CI1":
        O = np.full(days, I0, dtype=float)
    elif scenario == "CI2":
        O = I0 + A * np.sin(2 * np.pi * t / P)
    elif scenario == "CI3":
        O = np.full(days, I0, dtype=float)
        O[20:25] += spike
    elif scenario == "CI4a":
        O = I0 * (1 + growth) ** t
    elif scenario == "CI4b":
        O = I0 * (1 - decay) ** t
    else:
        raise ValueError(f"Unknown contamination scenario: {scenario}")
    return np.maximum(0, O + noise)


# ==========================================================================
# 2. Treatment Definitions (CT1-CT4)
# ==========================================================================
def efficiency_score(M, N, threshold=THRESHOLD):
    if np.max(M) > threshold:
        return 0.0
    return float((1 - np.mean(M) / threshold) / max(np.mean(N), 1e-9))


def efficiency_sheets(M, N):
    return float(np.mean(N) / max(np.mean(M), 1e-9))


def simulate_system(control_case, params, O, M0=0.5, threshold=THRESHOLD):
    days = len(O)
    M = np.zeros(days)
    N = np.zeros(days)
    M[0] = M0
    r = params.get('r', R_REMOVAL)

    N_next = params.get('T0', 0.0)   # CT3 state: N[t+1] = N[t] + R*(M[t] - goal)

    for t in range(days - 1):
        if control_case == "CT1":          # constant
            N[t] = params['T0']
        elif control_case == "CT2":        # periodic
            N[t] = params['T0'] + params.get('A_t', 0.2) * np.sin(2 * np.pi * t / params.get('P', 7))
        elif control_case == "CT3":        # adaptive feedback (one-day lag, matches the Jacobian)
            N[t] = N_next
        elif control_case == "CT4":        # degrading (no maintenance)
            N[t] = params['T0'] * (1 - params.get('decay', CT4_DECAY)) ** t

        lo = N_MIN if control_case == "CT3" else 0.0
        N[t] = min(max(N[t], lo), N_MAX)

        removal = min(1.0, r * N[t]) * M[t]          
        M[t + 1] = min(max(M[t] - removal + O[t], 0.0), 1e4)

        if control_case == "CT3":
            N_next = N[t] + params['R'] * (M[t] - params['goal'])

    N[-1] = N[-2]
    stats = dict(
        max_conc=float(np.max(M)),
        avg_treatment=float(np.mean(N)),
        efficiency=efficiency_score(M, N, threshold),
        efficiency_sheets=efficiency_sheets(M, N),
        breached=bool(np.max(M) > threshold),
    )
    return M, N, stats


# ==========================================================================
# 3. DESIGNING A TREATMENT (hindsight vs. design-once)
# ==========================================================================
SCENARIOS = ["CI1", "CI2", "CI3", "CI4a", "CI4b"]
CONTROLS = ["CT1", "CT2", "CT3", "CT4"]


def _params_from_x(ct, x, r=None):
    r = R_REMOVAL if r is None else r
    if ct == "CT1":
        return {'T0': x[0], 'r': r}
    if ct == "CT2":
        return {'T0': x[0], 'A_t': x[1], 'P': 7, 'r': r}
    if ct == "CT3":
        return {'T0': x[0], 'R': x[1], 'goal': x[2], 'r': r}
    if ct == "CT4":
        return {'T0': x[0], 'decay': CT4_DECAY, 'r': r}


BOUNDS = {
    "CT1": [(0.0, N_MAX)],
    "CT2": [(0.0, N_MAX), (0.0, 1.5)],
    "CT3": [(0.1, N_MAX), (0.01, 1.0), (0.6, 1.5)],   # T0, R, goal
    "CT4": [(0.0, N_MAX)],
}


def build_scenarios(seed=42):
    return {ci: generate_contamination(ci, seed=seed) for ci in SCENARIOS}


def objective(x, ct, Os, r):
    total = 0.0
    for O in Os:
        M, N, s = simulate_system(ct, _params_from_x(ct, x, r), O)
        over = max(0.0, s['max_conc'] - SAFETY_LIMIT)
        total += s['avg_treatment'] + 1000.0 * over ** 2    # smooth penalty
    return total / len(Os)


def design(ct, Os, r=None, seed=42, maxiter=60):
    r = R_REMOVAL if r is None else r
    res = differential_evolution(objective, BOUNDS[ct], args=(ct, Os, r), seed=seed,
                                 maxiter=maxiter, popsize=10, tol=1e-6, polish=False)
    return _params_from_x(ct, res.x, r)


def R_max_formula(O0, r, goal):
    return O0 / (r * goal ** 2)


def compare_designs(r=None, seed=42, verbose=True):
    r = R_REMOVAL if r is None else r
    scen = build_scenarios(seed)
    Os = list(scen.values())
    rows = []
    for ct in CONTROLS:
        p_rob = design(ct, Os, r, seed)
        for ci, O in scen.items():
            _, _, sr = simulate_system(ct, p_rob, O)
            p_h = design(ct, [O], r, seed)
            _, _, sh = simulate_system(ct, p_h, O)
            rows.append(dict(treatment=ct, scenario=ci,
                             robust_pass=not sr['breached'], robust_max=sr['max_conc'],
                             robust_cost=sr['avg_treatment'], robust_eff=sr['efficiency'], robust_eff_sheets=sr['efficiency_sheets'],
                             hindsight_cost=sh['avg_treatment'],
                             regret=sr['avg_treatment'] - sh['avg_treatment'],
                             robust_params={k: round(v, 3) for k, v in p_rob.items() if k != 'r'}))
    if verbose:
        print("\n" + "=" * 70)
        print(f"DESIGN-ONCE COMPARISON  (r = {r}, safety limit {SAFETY_LIMIT:.1f} ppb)")
        print("=" * 70)
        for ct in CONTROLS:
            sub = [x for x in rows if x['treatment'] == ct]
            print(f"\n[{ct}] one design for all scenarios: {sub[0]['robust_params']}")
            for x in sub:
                flag = "PASS" if x['robust_pass'] else "FAIL"
                print(f"    {x['scenario']:<5} {flag} | max {x['robust_max']:.2f} ppb | "
                      f"cost {x['robust_cost']:.2f} (hindsight {x['hindsight_cost']:.2f}, "
                      f"regret {x['regret']:+.2f}) | efficiency {x['robust_eff']:.3f}")
        print("\n--- RANKING (design-once) ---")
        print(f"{'Treatment':<10}{'Passed':<9}{'Avg cost':<10}{'Avg efficiency':<16}{'Avg regret'}")
        ranking = []
        for ct in CONTROLS:
            sub = [x for x in rows if x['treatment'] == ct]
            ranking.append((ct, sum(x['robust_pass'] for x in sub),
                            np.mean([x['robust_cost'] for x in sub]),
                            np.mean([x['robust_eff'] for x in sub]),
                            np.mean([x['regret'] for x in sub])))
        for ct, p, c, e, g in sorted(ranking, key=lambda z: (-z[1], z[2])):
            print(f"{ct:<10}{p}/{len(SCENARIOS):<7}{c:<10.2f}{e:<16.3f}{g:+.2f}")
        path = os.path.join(RES_DIR, "design_once_results.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n    saved {path}")
    return rows


def budget_test(budgets=(0.8, 1.2, 1.6, 2.0), r=None, seed=42):
    r = R_REMOVAL if r is None else r
    Os = list(build_scenarios(seed).values())

    def obj(x, ct, B):
        peaks, over = [], 0.0
        for O in Os:
            M, N, s = simulate_system(ct, _params_from_x(ct, x, r), O)
            peaks.append(s['max_conc'])
            over += max(0.0, s['avg_treatment'] - B) ** 2
        return max(peaks) + 1000.0 * over

    print("\n" + "=" * 70)
    print("EQUAL-BUDGET TEST: worst peak (ppb) over all 5 scenarios; * = breaches 2.0 ppb")
    print("=" * 70)
    print(f"{'Budget':<9}" + "".join(f"{ct:<10}" for ct in CONTROLS))
    rows = []
    for B in budgets:
        line = f"{B:<9}"
        for ct in CONTROLS:
            res = differential_evolution(obj, BOUNDS[ct], args=(ct, B), seed=seed,
                                         maxiter=40, popsize=10, tol=1e-6, polish=False)
            p = _params_from_x(ct, res.x, r)
            peak = max(simulate_system(ct, p, O)[2]['max_conc'] for O in Os)
            line += f"{peak:<6.2f}{'*' if peak > THRESHOLD else ' ':<4}"
            rows.append(dict(budget=B, treatment=ct, worst_peak=peak, breached=peak > THRESHOLD))
        print(line)
    path = os.path.join(RES_DIR, "equal_budget_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n    saved {path}")
    return rows


# ==========================================================================
# 3b. GETTING r 
# ==========================================================================
def r_from_removal(eta, residence_days=1.0):
    # """
    # Turn a published single-pass removal efficiency eta (0-1) into the model's r.
    # If a process removes fraction eta while the water is in contact for
    # `residence_days`, the per-day removal fraction at the standard dose is
    #     1 - (1 - eta) ** (1 / residence_days).
    # Define N = 1 as that standard dose, and r is that daily fraction.
    # """
    return 1.0 - (1.0 - eta) ** (1.0 / residence_days)


def fit_r(M, N, O):
    # """
    # Estimate r from DATA by least squares, using M[t+1] = M[t] - r*M[t]*N[t] + O[t]
    #     =>  r*M[t]*N[t] = M[t] + O[t] - M[t+1]
    # Needs daily mercury measurements M, the treatment dose N, and the inflow O.
    # """
    M, N, O = (np.asarray(v, dtype=float) for v in (M, N, O))
    x = M[:-1] * N[:-1]
    y = M[:-1] + O[:-1] - M[1:]
    return float(np.sum(x * y) / np.sum(x * x))


def fit_r_selftest():
    rng = np.random.default_rng(0)
    O = generate_contamination("CI2", noise_std=0, seed=0)
    M, N, _ = simulate_system("CT2", {'T0': 1.0, 'A_t': 0.6, 'P': 5, 'r': 0.4}, O)
    M_obs = M + rng.normal(0, 0.01, len(M))          # pretend we measured with small error
    print("\n" + "=" * 70)
    print("fit_r SELF-TEST (true r = 0.400, measurements have +/-0.01 ppb error)")
    print("=" * 70)
    print(f"    recovered r = {fit_r(M_obs, N, O):.3f}")


def r_sensitivity(r_values=(0.2, 0.4, 0.6, 0.8), seed=42):
    # """Does the ranking survive uncertainty in r?  (Re-runs the design-once cost.)"""
    scen = build_scenarios(seed)
    Os = list(scen.values())
    print("\n" + "=" * 70)
    print("SENSITIVITY TO r  (average design-once cost; lower is better)")
    print("=" * 70)
    print(f"{'r':<6}" + "".join(f"{ct:<9}" for ct in ("CT1", "CT2", "CT3")))
    for r in r_values:
        costs = []
        for ct in ("CT1", "CT2", "CT3"):
            p = design(ct, Os, r, seed, maxiter=40)
            costs.append(np.mean([simulate_system(ct, p, O)[2]['avg_treatment'] for O in Os]))
        print(f"{r:<6}" + "".join(f"{c:<9.2f}" for c in costs))


def run_fixed(ct, ci, params, days=90, M0=0.5, noise_std=0.0):
    O = generate_contamination(ci, days=days, noise_std=noise_std, seed=0)
    M, N, s = simulate_system(ct, params, O, M0=M0)
    print(f"{ct}/{ci}: final M = {M[-1]:.4f} ppb | avg N = {s['avg_treatment']:.4f} | "
          f"max M = {s['max_conc']:.4f} | breached = {s['breached']} | efficiency = {s['efficiency']:.4f}")
    return M, N, s


def check_against_sheets(M, N, O_known=None):
    M, N = np.asarray(M, float), np.asarray(N, float)
    print("\n" + "=" * 70)
    print("CHECK AGAINST SHEETS: inflow implied by each equation")
    print("=" * 70)
    for k, label in [(R_REMOVAL, f"removal = r*N*M (r={R_REMOVAL})"), (1.0, "removal = N*M (no r)")]:
        O = M[1:] - M[:-1] + k * N[:-1] * M[:-1]
        bad = int(np.sum(O < -1e-9))
        print(f"    {label}: {bad} negative inflows | first values {np.round(O[:8], 3)}")
    print("    Use whichever equation gives clean, non-negative inflows. If it is the second,")
    print("    run the Python model with r = 1.0 so N means 'fraction removed per day'.")


# ==========================================================================
# 4. EQUILIBRIUM ANALYSIS
# ==========================================================================
def equilibrium_analysis():
    print("\n" + "=" * 70)
    print("EQUILIBRIUM ANALYSIS")
    print("=" * 70)
    M, N, r, O0, T0, goal = sp.symbols('M N r O0 T0 goal', positive=True)

    M1 = sp.solve(sp.Eq(0, -r * M * T0 + O0), M)[0]
    print("\nCT1 (constant treatment) under CI1:")
    print(f"    N* = T0,  M* = {M1}")
    print("    Meaning: the water settles where daily removal equals daily inflow.")
    print("    To hold M* at a target you must already know O0: T0 = O0/(r*target).")

    N3 = sp.solve(sp.Eq(0, -r * goal * N + O0), N)[0]
    print("\nCT3 (adaptive feedback) under CI1:")
    print(f"    M* = goal,  N* = {N3}")
    print("    Meaning: CT3 finds the right dose by itself -- the same dose CT1 would")
    print("    need, but without knowing O0 in advance.")

    print("\nCT4 (degrading treatment) under CI1:")
    print("    N* -> 0, so M* grows without bound (about O0 per day once N ~ 0).")
    print("\nCT2 / CI2 (periodic): no fixed point, M settles into a cycle (see bifurcation).")


# ==========================================================================
# 5. STABILITY ANALYSIS
# ==========================================================================
def stability_analysis():
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS (Jacobian)")
    print("=" * 70)
    M, N, r, O0, R, goal = sp.symbols('M N r O0 R goal', positive=True)

    print("\nCT1: dM[t+1]/dM[t] = 1 - r*N  ->  stable iff 0 < r*N < 2")

    f1 = M - r * M * N + O0
    f2 = N + R * (M - goal)
    J = sp.Matrix([[sp.diff(f1, M), sp.diff(f1, N)], [sp.diff(f2, M), sp.diff(f2, N)]])
    J_eq = J.subs({M: goal, N: O0 / (r * goal)})
    print("\nCT3 Jacobian at equilibrium:")
    sp.pprint(J_eq)

    tr = sp.simplify(J_eq.trace())
    dt = sp.simplify(J_eq.det())
    print(f"\n    trace = {tr}\n    det   = {dt}")

    # 2x2 discrete-time stability needs ALL THREE Jury inequalities:
    j1 = sp.simplify(1 - dt)              # > 0   <=>  det < 1
    j2 = sp.simplify(1 - tr + dt)         # > 0
    j3 = sp.simplify(1 + tr + dt)         # > 0
    print("\n    Jury conditions (all must be > 0):")
    print(f"      1 - det          = {j1}")
    print(f"      1 - trace + det  = {j2}")
    print(f"      1 + trace + det  = {j3}")
    print(f"    -> from the first:  R < {sp.solve(sp.Eq(dt, 1), R)[0]}")
    print("    -> the second just says R > 0")
    print("    -> the third matters only if O0/goal > 2 (not the case for your parameters)")

    disc = sp.simplify(tr ** 2 - 4 * dt)
    R_osc = sp.solve(sp.Eq(disc, 0), R)[0]
    print(f"\n    trace^2 - 4*det = {disc}")
    print(f"    Oscillation threshold: R = {R_osc}")

    O0v, rv, gv = 0.5, R_REMOVAL, 1.0
    Rm, Ro = R_max_formula(O0v, rv, gv), O0v ** 2 / (4 * gv ** 3 * rv)
    print(f"\n    Example (O0={O0v}, r={rv}, goal={gv}):")
    print(f"      0 < R < {Ro:.3f}   : treatment glides smoothly to the target")
    print(f"      {Ro:.3f} < R < {Rm:.3f} : treatment overshoots, but the swings die out (damped)")
    print(f"      R > {Rm:.3f}         : unstable -- treatment overreacts, swings grow")
    print("    (valid while r*N* <= 1, i.e. goal >= O0, before the removal cap kicks in)")


# ==========================================================================
# 6. SENSITIVITY ANALYSIS
# ==========================================================================
def sensitivity_analysis():
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: R_max = O0 / (r * goal^2)")
    print("=" * 70)
    base = dict(O0=0.5, r=R_REMOVAL, goal=1.0)
    b = R_max_formula(**base)
    print(f"\nBaseline O0={base['O0']}, r={base['r']}, goal={base['goal']} -> R_max = {b:.3f}")

    print("\n-- contamination O0 (linear: double O0, double R_max) --")
    for v in [0.25, 0.5, 1.0, 2.0]:
        x = R_max_formula(v, base['r'], base['goal'])
        print(f"    O0={v:<5} R_max={x:.3f}  ({x / b:.2f}x)")
    print("\n-- removal rate r (inverse) --")
    for v in [0.2, 0.4, 0.6, 0.8]:
        x = R_max_formula(base['O0'], v, base['goal'])
        print(f"    r={v:<5} R_max={x:.3f}  ({x / b:.2f}x)")
    print("\n-- target goal (inverse square: doubling the goal cuts R_max to 1/4) --")
    for v in [0.6, 1.0, 1.5, 2.0]:
        x = R_max_formula(base['O0'], base['r'], v)
        print(f"    goal={v:<5} R_max={x:.3f}  ({x / b:.2f}x)")
    print("    Stricter (lower) targets allow a LARGER R_max, but they also need a")
    print("    larger steady dose N* = O0/(r*goal), which the capacity N_MAX can cap.")
    print(f"    Lowest reachable target at capacity: goal >= O0/(r*N_MAX) = "
          f"{base['O0'] / (R_REMOVAL * N_MAX):.3f} ppb (and the removal cap needs goal >= O0).")


# ==========================================================================
# 7. BIFURCATION ANALYSIS
# ==========================================================================
def bifurcation_ct3(O0=0.5, goal=1.0, n_R=140, steps=3000, keep=150):
    Rmax = R_max_formula(O0, R_REMOVAL, goal)
    Rs = np.linspace(0.02, 2.2 * Rmax, n_R)
    O = np.full(steps, O0)
    N_star = O0 / (R_REMOVAL * goal)
    pts_R, pts_M, spread = [], [], []
    for R in Rs:
        M, N, _ = simulate_system("CT3", {'T0': N_star, 'R': R, 'goal': goal}, O, M0=goal + 0.1)
        tail = M[-keep:]
        pts_R += [R] * keep
        pts_M += list(tail)
        spread.append(tail.max() - tail.min())
    spread = np.array(spread)
    first_osc = Rs[np.argmax(spread > 1e-3)] if (spread > 1e-3).any() else None
    print("\n" + "=" * 70)
    print("BIFURCATION: CT3 feedback gain R")
    print("=" * 70)
    print(f"    Predicted loss of stability at R_max = {Rmax:.3f}")
    print(f"    Sustained oscillation first appears near R = "
          f"{first_osc:.3f}" if first_osc else "    No sustained oscillation found")
    if HAVE_MPL:
        plt.figure(figsize=(8, 5))
        plt.plot(pts_R, pts_M, ',k', alpha=0.6)
        plt.axvline(Rmax, color='crimson', ls='--', label=f'predicted R_max = {Rmax:.2f}')
        plt.axhline(goal, color='steelblue', ls=':', label=f'target = {goal}')
        plt.xlabel("Feedback gain R"); plt.ylabel("Long-run Hg concentration (ppb)")
        plt.title("CT3 bifurcation diagram (constant inflow)")
        plt.legend(); plt.tight_layout()
        path = os.path.join(FIG_DIR, "bifurcation_ct3.png"); plt.savefig(path, dpi=150); plt.close()
        print(f"    saved {path}")


def bifurcation_ct2(T0=1.25, A_t=0.8, n_P=220, steps=1400, burn=700, P_inflow=7):
    O = generate_contamination("CI2", days=steps, P=P_inflow, noise_std=0)
    Ps = np.linspace(3, 14, n_P)
    xs, ys, counts = [], [], []
    for Pt in Ps:
        M, N, _ = simulate_system("CT2", {'T0': T0, 'A_t': A_t, 'P': Pt}, O)
        samples = M[burn::P_inflow]
        xs += [Pt] * len(samples); ys += list(samples)
        counts.append(len(np.unique(np.round(samples, 3))))
    counts = np.array(counts)
    print("\n" + "=" * 70)
    print("BIFURCATION: CT2 treatment period vs CI2 inflow period (7 days)")
    print("=" * 70)
    print(f"    {np.sum(counts <= 2)} of {n_P} treatment periods lock onto a periodic cycle (<=2 sampled values)")
    print(f"    {np.sum(counts > 5)} show many distinct values (quasi-periodic / complex)")
    if HAVE_MPL:
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, ',k', alpha=0.6)
        plt.axvline(P_inflow, color='crimson', ls='--', label='treatment period = inflow period')
        plt.xlabel("CT2 treatment period (days)"); plt.ylabel("Hg concentration sampled every 7 days (ppb)")
        plt.title("CT2 / CI2 stroboscopic bifurcation diagram")
        plt.legend(); plt.tight_layout()
        path = os.path.join(FIG_DIR, "bifurcation_ct2.png"); plt.savefig(path, dpi=150); plt.close()
        print(f"    saved {path}")


# ==========================================================================
# 8. MAIN
# ==========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("    MERCURY WATER TREATMENT MODEL  (EPA MCL threshold = 2.0 ppb)")
    print("=" * 70)
    compare_designs()
    budget_test()
    equilibrium_analysis()
    stability_analysis()
    sensitivity_analysis()
    bifurcation_ct3()
    bifurcation_ct2()
    fit_r_selftest()
    r_sensitivity()
    sheet_M = [0.5, 0.75, 0.375, 0.2875, 0.44375, 0.421875, 1.1109375, 0.75546875, 1.3021875,
               1.35109375, 1.475546875, 1.537773438, 1.068886719, 1.334443359, 0.8672216797,
               0.7468886719, 1.173444336]
    sheet_N = [0.5] * 7 + [0.6] + [0.5] * 6 + [0.6, 0.5, 0.5]
    check_against_sheets(sheet_M, sheet_N)
    print("\n    removal-rate helper example: 90% removal in 1 day -> r =",
          round(r_from_removal(0.90, 1.0), 3))
