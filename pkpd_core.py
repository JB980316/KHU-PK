# -*- coding: utf-8 -*-
"""
pkpd_core.py

Extracted scientific backend from original CLI implementation.
This module provides simulation, PD, CSV loading, and parameter estimation
functionality without any interactive CLI or Streamlit dependencies.
"""
from typing import Dict, Optional, Sequence, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from dataclasses import dataclass
import csv
import os
import io

# -------------------------
# Default numerical options
# -------------------------
DEFAULT_RK_METHOD = "RK45"
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
# ke consistency tolerances
KE_RTOL = 1e-8
KE_ATOL = 1e-12

# CSV/fitting defaults
MIN_OBS = 6  # minimum number of observations for fitting (assumption)
DENSE_DT = 0.05  # h, dense plotting grid spacing


# -------------------------
# Validation utilities
# -------------------------
def _is_finite_number(x: float) -> bool:
    return np.isfinite(x)


def validate_parameters(
    Dose: float,
    F: float,
    ka: float,
    V: float,
    CL: float,
    EC50: float,
    gamma: float,
    Emax: float,
    E0: float,
    t_start: float,
    t_end: float,
    t_eval: Optional[Sequence[float]],
    ke_supplied: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Validate inputs and compute ke = CL / V.
    Raises ValueError with informative messages for invalid inputs.
    Returns computed ke and the negativity tolerance (abs, in mg).
    """
    # Basic numeric checks
    params = dict(Dose=Dose, F=F, ka=ka, V=V, CL=CL, EC50=EC50, gamma=gamma, Emax=Emax, E0=E0, t_start=t_start, t_end=t_end)
    for name, val in params.items():
        if not _is_finite_number(val):
            raise ValueError(f"Parameter {name} must be a finite number; got {val!r}.")

    if Dose < 0:
        raise ValueError("Dose must be >= 0.")
    if not (0.0 <= F <= 1.0):
        raise ValueError("F (bioavailability) must satisfy 0 <= F <= 1.")
    if ka <= 0.0:
        raise ValueError("ka must be > 0 (1/h).")
    if V <= 0.0:
        raise ValueError("V must be > 0 (L).")
    if CL <= 0.0:
        raise ValueError("CL must be > 0 (L/h).")
    if EC50 <= 0.0:
        raise ValueError("EC50 must be > 0 (mg/L).")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0 (dimensionless).")
    if Emax < 0.0:
        raise ValueError("Emax must be >= 0 for the stimulatory model.")
    # time checks
    if not (_is_finite_number(t_start) and _is_finite_number(t_end)):
        raise ValueError("t_start and t_end must be finite numbers.")
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start.")
    if t_eval is not None:
        t_eval_arr = np.asarray(t_eval, dtype=float)
        if t_eval_arr.ndim != 1:
            raise ValueError("t_eval must be a 1-D sequence of times.")
        if t_eval_arr.size < 1:
            raise ValueError("t_eval must have at least 1 element.")
        if not np.all(np.isfinite(t_eval_arr)):
            raise ValueError("All t_eval values must be finite.")
        if not (t_eval_arr[0] >= t_start - 1e-14 and t_eval_arr[-1] <= t_end + 1e-14):
            raise ValueError("t_eval values must lie within [t_start, t_end].")
        if t_eval_arr.size > 1:
            if not np.all(np.diff(t_eval_arr) > 0):
                raise ValueError("t_eval must be strictly increasing (no duplicates).")
    # compute ke and compare if supplied
    ke = float(CL / V)
    if ke_supplied is not None:
        if not _is_finite_number(ke_supplied):
            raise ValueError("Supplied ke must be finite.")
        if not np.isclose(ke, ke_supplied, rtol=KE_RTOL, atol=KE_ATOL):
            raise ValueError(
                f"Supplied ke={ke_supplied:.12g} is inconsistent with CL/V={ke:.12g} "
                f"(tolerances rtol={KE_RTOL}, atol={KE_ATOL})."
            )
    # negativity tolerance relative to problem scale (amount units: mg)
    neg_tol = max(1e-12, 1e-8 * max(1.0, float(Dose)))
    return ke, neg_tol


def validate_solver_controls(
    rtol: float,
    atol: float,
) -> None:
    """
    Validate that ODE solver tolerances are scalar, finite, and strictly positive.
    Raises ValueError for invalid inputs.
    """
    if not _is_finite_number(rtol):
        raise ValueError(f"rtol must be finite; got {rtol!r}.")
    if not _is_finite_number(atol):
        raise ValueError(f"atol must be finite; got {atol!r}.")
    if rtol <= 0.0:
        raise ValueError(f"rtol must be > 0; got {rtol}.")
    if atol <= 0.0:
        raise ValueError(f"atol must be > 0; got {atol}.")


# -------------------------
# PK ODE function
# -------------------------
def pk_ode(t: float, y: Sequence[float], ka: float, ke: float, F: float) -> np.ndarray:
    """
    ODE right-hand side for the two-state PK model.
    y[0] = A_gut (mg)
    y[1] = A_central (mg)
    Returns derivatives [dA_gut/dt, dA_central/dt].
    """
    A_gut = float(y[0])
    A_central = float(y[1])
    dA_gut = -ka * A_gut
    dA_central = F * ka * A_gut - ke * A_central
    return np.array([dA_gut, dA_central], dtype=float)


# -------------------------
# Analytical PK reference functions
# -------------------------
def analytical_pk_unequal_rates(
    tau: np.ndarray,
    Dose: float,
    F: float,
    ka: float,
    ke: float,
    V: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical solution for one-compartment oral PK with first-order absorption (ka != ke).
    """
    tau = np.asarray(tau, dtype=float)
    A_gut = Dose * np.exp(-ka * tau)
    rate_diff = ka - ke
    A_central = F * Dose * ka / rate_diff * (np.exp(-ke * tau) - np.exp(-ka * tau))
    concentration = A_central / V
    return A_gut, A_central, concentration


def analytical_pk_equal_rates(
    tau: np.ndarray,
    Dose: float,
    F: float,
    k: float,
    V: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical solution for one-compartment oral PK with first-order absorption (ka == ke == k).
    """
    tau = np.asarray(tau, dtype=float)
    A_gut = Dose * np.exp(-k * tau)
    A_central = F * Dose * k * tau * np.exp(-k * tau)
    concentration = A_central / V
    return A_gut, A_central, concentration


# -------------------------
# PD function: stimulatory sigmoid Emax
# -------------------------
def sigmoid_emax(
    concentration: np.ndarray,
    E0: float,
    Emax: float,
    EC50: float,
    gamma: float,
    conc_neg_tol: float = 0.0,
) -> np.ndarray:
    """
    Compute E(C) = E0 + Emax * C^gamma / (EC50^gamma + C^gamma).
    """
    C = np.asarray(concentration, dtype=float)

    # Validate and handle tiny negative concentrations
    if conc_neg_tol < 0:
        raise ValueError("conc_neg_tol must be nonnegative.")
    neg_mask = C < 0.0
    if np.any(neg_mask):
        max_neg = float(np.min(C[neg_mask]))
        if max_neg < -conc_neg_tol:
            raise ValueError(
                f"Concentration has values more negative ({max_neg:.3g}) than allowed tolerance {-conc_neg_tol:.3g}."
            )
        # Clip tiny negatives to zero for stable fractional powers
        C = np.maximum(C, 0.0)

    # Validate PD parameters
    if EC50 <= 0:
        raise ValueError(f"EC50 must be > 0; got {EC50}.")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0; got {gamma}.")

    # Compute dimensionless concentration ratio
    x = C / EC50

    # Numerically stable Hill fraction computation
    frac = np.zeros_like(C, dtype=float)

    # Case 1: x <= 1
    mask_low = x <= 1.0
    if np.any(mask_low):
        z_low = np.power(x[mask_low], gamma)
        frac[mask_low] = z_low / (1.0 + z_low)

    # Case 2: x > 1
    mask_high = x > 1.0
    if np.any(mask_high):
        z_high = np.power(x[mask_high], -gamma)
        frac[mask_high] = 1.0 / (1.0 + z_high)

    return float(E0) + float(Emax) * frac


# -------------------------
# Simulation function
# -------------------------
def simulate_pkpd(
    Dose: float,
    F: float,
    ka: float,
    V: float,
    CL: float,
    E0: float,
    Emax: float,
    EC50: float,
    gamma: float,
    t_start: float = 0.0,
    t_end: float = 48.0,
    t_eval: Optional[Sequence[float]] = None,
    ke_supplied: Optional[float] = None,
    method: str = DEFAULT_RK_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    dosing_interval: Optional[float] = None,
    n_doses: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Simulate the two-state PK model and PD effect over the given time range.
    """
    # Validate inputs and compute derived ke and negativity tolerance
    ke, neg_tol_single = validate_parameters(
        Dose=Dose,
        F=F,
        ka=ka,
        V=V,
        CL=CL,
        EC50=EC50,
        gamma=gamma,
        Emax=Emax,
        E0=E0,
        t_start=t_start,
        t_end=t_end,
        t_eval=t_eval,
        ke_supplied=ke_supplied,
    )

    # Validate solver controls
    validate_solver_controls(rtol=rtol, atol=atol)

    # Validate n_doses
    if isinstance(n_doses, bool):
        raise ValueError("n_doses must be an integer >= 1 (not boolean).")
    if not (isinstance(n_doses, (int, np.integer))):
        raise ValueError("n_doses must be an integer value >= 1.")
    if int(n_doses) != n_doses:
        raise ValueError("n_doses must be an integer value (no fractional doses).")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1.")
    n_doses = int(n_doses)

    # Validate dosing_interval if supplied
    if dosing_interval is not None:
        if not _is_finite_number(dosing_interval):
            raise ValueError("dosing_interval must be a finite number when supplied.")
        if dosing_interval <= 0.0:
            raise ValueError("dosing_interval must be > 0 when supplied.")

    # If multiple doses requested, dosing_interval is mandatory
    if n_doses > 1 and dosing_interval is None:
        raise ValueError("dosing_interval must be provided when n_doses > 1.")

    # Build evaluation grid if not provided: spacing 0.05 h by default
    if t_eval is None:
        dt = 0.05  # hours
        t_eval_arr = np.arange(t_start, t_end + dt / 2.0, dt, dtype=float)
    else:
        t_eval_arr = np.asarray(t_eval, dtype=float)

    # Construct dose_times
    if n_doses == 1:
        dose_times = np.array([t_start], dtype=float)
    else:
        dose_times = (t_start + np.arange(n_doses, dtype=float) * float(dosing_interval))

    # Validate dose_times finite and final dose <= t_end
    if not np.all(np.isfinite(dose_times)):
        raise ValueError("Computed dose_times contain non-finite values.")
    # allow small tolerance
    if dose_times[-1] > t_end + 1e-12:
        raise ValueError("Final scheduled dose occurs after t_end.")

    # Max step policy (global)
    fastest_rate = max(ka, ke)
    if fastest_rate <= 0.0:
        max_step_by_rate = np.inf
    else:
        max_step_by_rate = 0.1 / fastest_rate

    if t_eval_arr.size == 1:
        eval_spacing = np.inf
    else:
        eval_spacing = float(np.min(np.diff(t_eval_arr)))
    max_step = float(min(eval_spacing, max_step_by_rate))
    if not np.isfinite(max_step) or max_step <= 0:
        max_step = (t_end - t_start) / 100.0

    # initial zero state and apply first dose at t_start as per convention
    y = np.array([0.0, 0.0], dtype=float)
    current_t = float(t_start)

    # negativity tolerance scaled for repeated dosing
    neg_tol = max(1e-12, 1e-8 * max(1.0, float(Dose) * float(n_doses)))

    # Prepare outputs corresponding exactly to t_eval_arr
    times_out = []
    A_gut_out = []
    A_central_out = []

    # Helper: mask for selecting t_eval in intervals
    def select_times(left, right, include_left=False, include_right=False):
        # left < t <= right  or variations
        # use small tolerance
        tol = 1e-14
        if include_left and include_right:
            mask = (t_eval_arr >= left - tol) & (t_eval_arr <= right + tol)
        elif include_left:
            mask = (t_eval_arr >= left - tol) & (t_eval_arr < right - tol)
        elif include_right:
            mask = (t_eval_arr > left + tol) & (t_eval_arr <= right + tol)
        else:
            mask = (t_eval_arr > left + tol) & (t_eval_arr < right - tol)
        return t_eval_arr[mask]

    # Iterate over doses: apply dose, then integrate to next dose
    for i, t_dose in enumerate(dose_times):
        t_dose = float(t_dose)
        # Integrate from current_t to t_dose (to obtain pre-dose state and any requested eval points between)
        if t_dose > current_t:
            # select times strictly between current_t and t_dose
            internal_times = select_times(current_t, t_dose, include_left=False, include_right=False)
            # Solve from current_t to t_dose
            sol = solve_ivp(
                fun=lambda t, y: pk_ode(t, y, ka=ka, ke=ke, F=F),
                t_span=(current_t, t_dose),
                y0=y,
                method=method,
                t_eval=internal_times if internal_times.size > 0 else None,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=True,
            )
            if sol.status != 0 or not sol.success:
                raise RuntimeError(f"ODE solver failed on interval [{current_t}, {t_dose}]: message='{sol.message}', status={sol.status}")
            # collect internal eval points if any
            if internal_times.size > 0:
                times_seg = sol.t
                y_seg = sol.y
                for tt, aa, ac in zip(times_seg, y_seg[0, :], y_seg[1, :]):
                    times_out.append(float(tt))
                    A_gut_out.append(float(aa))
                    A_central_out.append(float(ac))
            # obtain pre-dose state at t_dose via dense output
            y_pre = sol.sol(t_dose).reshape(2,)
        else:
            # no integration needed; we're at the dose time
            y_pre = y.copy()

        # apply instantaneous oral dose: add full Dose to A_gut per convention
        y = y_pre.copy()
        y[0] = y[0] + float(Dose)

        # If t_dose is requested in t_eval_arr, record post-dose state for that time
        # (returned value at exact dose times must be post-dose)
        mask_eq = np.isclose(t_eval_arr, t_dose, rtol=0.0, atol=1e-14)
        if np.any(mask_eq):
            # there may be one matching index; append in chronological order
            times_out.append(t_dose)
            A_gut_out.append(float(y[0]))
            A_central_out.append(float(y[1]))

        current_t = t_dose

    # After last dose, integrate from current_t to t_end and collect any remaining eval points
    if t_end > current_t:
        final_times = select_times(current_t, float(t_end), include_left=False, include_right=True)
        sol = solve_ivp(
            fun=lambda t, y: pk_ode(t, y, ka=ka, ke=ke, F=F),
            t_span=(current_t, float(t_end)),
            y0=y,
            method=method,
            t_eval=final_times if final_times.size > 0 else None,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )
        if sol.status != 0 or not sol.success:
            raise RuntimeError(f"ODE solver failed on interval [{current_t}, {t_end}]: message='{sol.message}', status={sol.status}")
        if final_times.size > 0:
            times_seg = sol.t
            y_seg = sol.y
            for tt, aa, ac in zip(times_seg, y_seg[0, :], y_seg[1, :]):
                times_out.append(float(tt))
                A_gut_out.append(float(aa))
                A_central_out.append(float(ac))
    else:
        # No final integration; but if t_end equals current_t and t_end requested in t_eval it was handled when applying dose
        pass

    # Now times_out should correspond to the subset of t_eval_arr that we recorded. Ensure ordering matches t_eval_arr
    # Build arrays aligned to full t_eval_arr by mapping indices
    # Initialize arrays
    n_times = t_eval_arr.size
    A_gut_full = np.empty(n_times, dtype=float)
    A_central_full = np.empty(n_times, dtype=float)
    # Fill with NaNs to detect missing
    A_gut_full.fill(np.nan)
    A_central_full.fill(np.nan)

    # Map recorded outputs
    for tt, ag, ac in zip(times_out, A_gut_out, A_central_out):
        # find matching index in t_eval_arr
        idx = np.nonzero(np.isclose(t_eval_arr, tt, rtol=0.0, atol=1e-14))[0]
        if idx.size == 0:
            # this recorded time is not in t_eval_arr (shouldn't happen)
            continue
        j = idx[0]
        A_gut_full[j] = ag
        A_central_full[j] = ac

    # Some t_eval points may correspond to no recorded outputs (e.g., before first recorded point if t_eval has values < t_start)
    # For such points, compute by dense output propagation using piecewise approach: evaluate numeric integration on-the-fly
    # But by construction validate_parameters ensured t_eval within [t_start, t_end]. We must fill any remaining NaNs by evaluating
    # the model via analytical superposition as fallback (numerically exact for linear PK). Use analytical superposition to fill any missing entries.

    missing = np.isnan(A_gut_full)
    if np.any(missing):
        # Use analytical superposition to compute missing entries
        t_missing = t_eval_arr[missing]
        A_gut_ref = np.zeros_like(t_missing, dtype=float)
        A_central_ref = np.zeros_like(t_missing, dtype=float)
        # choose equal-rate rule consistent with repository tolerances
        equal_rates = np.isclose(ka, ke, rtol=KE_RTOL, atol=KE_ATOL)
        for j, tval in enumerate(t_missing):
            # sum over doses that occur at or before tval
            contrib_Ag = 0.0
            contrib_Ac = 0.0
            for td in dose_times:
                if td <= tval + 1e-14:
                    tau = tval - td
                    if not equal_rates:
                        ag, ac, _ = analytical_pk_unequal_rates(np.array([tau]), Dose, F, ka, ke, V)
                    else:
                        ag, ac, _ = analytical_pk_equal_rates(np.array([tau]), Dose, F, ka, V)
                    contrib_Ag += float(ag[0])
                    contrib_Ac += float(ac[0])
            A_gut_ref[j] = contrib_Ag
            A_central_ref[j] = contrib_Ac
        A_gut_full[missing] = A_gut_ref
        A_central_full[missing] = A_central_ref

    # Final finite checks
    if not (np.all(np.isfinite(A_gut_full)) and np.all(np.isfinite(A_central_full)) and np.all(np.isfinite(t_eval_arr))):
        raise RuntimeError("Integration produced non-finite values in states or time.")

    min_A_gut = float(np.min(A_gut_full))
    min_A_central = float(np.min(A_central_full))
    if min_A_gut < -neg_tol or min_A_central < -neg_tol:
        raise RuntimeError(
            f"Solver produced materially negative state(s): min(A_gut)={min_A_gut:.3g}, min(A_central)={min_A_central:.3g}; "
            f"negativity tolerance={neg_tol:.3g}."
        )

    A_gut_clipped = np.maximum(A_gut_full, 0.0)
    A_central_clipped = np.maximum(A_central_full, 0.0)

    concentration = A_central_clipped / float(V)

    if not np.all(np.isfinite(concentration)):
        raise RuntimeError("Non-finite concentration values encountered.")
    if np.min(concentration) < -1e-14:
        raise RuntimeError("Concentration has negative values beyond minimal numerical noise.")

    effect = sigmoid_emax(concentration, E0=E0, Emax=Emax, EC50=EC50, gamma=gamma, conc_neg_tol=1e-14)

    results = {
        "time": t_eval_arr,  # h
        "A_gut": A_gut_clipped,  # mg
        "A_central": A_central_clipped,  # mg
        "concentration": concentration,  # mg/L
        "effect": effect,  # effect units
        "ke": ke,  # 1/h
        "units": {
            "time": "h",
            "amount": "mg",
            "volume": "L",
            "concentration": "mg/L",
            "ke": "1/h",
            "effect": "user-defined units",
        },
        "meta": {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "max_step": max_step,
            "negativity_tolerance_amount_mg": neg_tol,
            "ke_consistency_rtol": KE_RTOL,
            "ke_consistency_atol": KE_ATOL,
            "dosing_convention": "Each instantaneous oral administration adds the full nominal Dose to A_gut; F applied only in absorption flux F*ka*A_gut; repeated doses at dose_times.",
            "bioavailability_convention": "F applied in absorption flux F*ka*A_gut; CL and V are systemic (not apparent)",
            "n_doses": n_doses,
            "dosing_interval": float(dosing_interval) if dosing_interval is not None else None,
            "dose_times": np.asarray(dose_times, dtype=float),
        },
    }
    return results


# -------------------------
# Plotting
# -------------------------
def plot_pkpd(results: Dict[str, np.ndarray], title_suffix: str = "") -> plt.Figure:
    """
    Create a Matplotlib Figure plotting concentration and effect versus time.
    Returns the Figure; does not call plt.show().
    """
    t = results["time"]
    C = results["concentration"]
    E = results["effect"]
    units = results.get("units", {})
    time_unit = units.get("time", "h")
    conc_unit = units.get("concentration", "mg/L")
    eff_unit = units.get("effect", "units")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, C, color="C0", lw=1.5)
    axes[0].set_ylabel(f"Concentration ({conc_unit})", fontsize=10)
    axes[0].set_title(f"Concentration vs Time {title_suffix}".strip())
    axes[0].grid(True)

    axes[1].plot(t, E, color="C1", lw=1.5)
    axes[1].set_xlabel(f"Time ({time_unit})", fontsize=10)
    axes[1].set_ylabel(f"Effect ({eff_unit})", fontsize=10)
    axes[1].set_title(f"Pharmacodynamic Effect vs Time {title_suffix}".strip())
    axes[1].grid(True)

    fig.tight_layout()
    return fig


# -------------------------
# New: Fit dataclass and CSV/fitting utilities
# -------------------------
@dataclass
class FitResult:
    ka: float
    V: float
    CL: float
    ke: float
    success: bool
    message: str
    rmse: float
    predicted_at_observations: np.ndarray


def load_pk_csv(path: "os.PathLike[str] | io.IOBase | object") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV containing required, case-sensitive columns 'time' and 'concentration'.
    Accepts either a filesystem path (str/PathLike) or a file-like object (text or bytes).
    Returns sorted numpy arrays (times, concentrations).
    Performs validation per specification.
    """
    # Determine if path is a filesystem path or file-like
    fileobj = None
    opened_here = False
    if isinstance(path, (str, bytes, os.PathLike)):
        # treat as filesystem path
        pstr = os.fspath(path)
        if not os.path.exists(pstr):
            raise FileNotFoundError(f"CSV file not found: {pstr}")
        if not os.access(pstr, os.R_OK):
            raise PermissionError(f"CSV file not readable: {pstr}")
        fileobj = open(pstr, "r", newline="")
        opened_here = True
    elif hasattr(path, "read"):
        # file-like object provided (e.g., from Streamlit upload)
        fileobj = path
        # If bytes-buffer, wrap in TextIO
        if isinstance(fileobj, (io.BytesIO,)):
            fileobj = io.TextIOWrapper(fileobj, encoding="utf-8")
            opened_here = True
        # If it's a binary file-like with 'buffer' attribute (e.g., UploadedFile), try to obtain text wrapper
        elif getattr(fileobj, "readable", lambda: False) and not isinstance(fileobj.read(0), str):
            # binary-like, wrap
            fileobj = io.TextIOWrapper(fileobj, encoding="utf-8")
            opened_here = True
    else:
        raise ValueError("path must be a filesystem path or a file-like object with a read() method")

    times: List[float] = []
    concs: List[float] = []

    try:
        # Ensure we are at the start
        try:
            fileobj.seek(0)
        except Exception:
            pass
        reader = csv.DictReader(fileobj)
        if reader.fieldnames is None:
            raise ValueError("CSV parsing failed: no header found")
        # Required columns case-sensitive
        required = ["time", "concentration"]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column in CSV: '{col}'")

        for row in reader:
            # do not ignore missing or invalid rows silently
            try:
                t_raw = row["time"]
                c_raw = row["concentration"]
            except KeyError as e:
                raise ValueError(f"CSV missing required column: {e}")
            try:
                t = float(t_raw)
                c = float(c_raw)
            except Exception:
                raise ValueError(f"Non-numeric value encountered in row: time={t_raw!r}, concentration={c_raw!r}")
            if not np.isfinite(t) or not np.isfinite(c):
                raise ValueError("Non-finite value encountered in CSV data")
            if np.isnan(t) or np.isnan(c):
                raise ValueError("NaN encountered in CSV data")
            if c < 0.0:
                raise ValueError("Negative concentration encountered in CSV data")
            if t < 0.0:
                raise ValueError("Negative time encountered in CSV data (t must be >= 0)")
            times.append(t)
            concs.append(c)
    except csv.Error as e:
        raise ValueError(f"CSV parsing failed: {e}")
    finally:
        if opened_here and hasattr(fileobj, "close"):
            try:
                fileobj.close()
            except Exception:
                pass

    if len(times) == 0:
        raise ValueError("CSV contains no observations")

    # Check duplicates
    times_arr = np.array(times, dtype=float)
    if times_arr.size != np.unique(times_arr).size:
        raise ValueError("Duplicate time values encountered in CSV (incompatible with fitting workflow)")

    # Sort by time
    order = np.argsort(times_arr)
    times_sorted = times_arr[order]
    concs_sorted = np.array(concs, dtype=float)[order]

    if times_sorted.size < MIN_OBS:
        raise ValueError(f"Insufficient observations for fitting: need at least {MIN_OBS}, got {times_sorted.size}")

    return times_sorted, concs_sorted


def predict_concentration(
    observation_times: Sequence[float],
    ka: float,
    V: float,
    CL: float,
    Dose: float,
    F: float,
    n_doses: int,
    dosing_interval: Optional[float],
    t_start: float = 0.0,
) -> np.ndarray:
    """
    Predict concentrations at observation_times by calling the authoritative simulate_pkpd().
    Returns ndarray of concentrations aligned to observation_times.
    """
    obs_times = np.asarray(observation_times, dtype=float)
    if obs_times.size == 0:
        return np.array([], dtype=float)
    t_min = float(np.min(obs_times))
    t_max = float(np.max(obs_times))
    # Ensure the simulation covers the observation range and respects t_start convention
    sim_t_start = float(t_start)
    if t_min < sim_t_start:
        # It's invalid for observations before dosing start; but simulator supports t_eval within [t_start, t_end]
        # We'll raise instead of silently adjusting
        raise ValueError("Observation times include values before t_start (dosing start). Adjust t_start or observations.")
    sim_t_end = max(t_max, sim_t_start)
    # Call simulate_pkpd with t_eval equal to observation times
    sim = simulate_pkpd(
        Dose=float(Dose),
        F=float(F),
        ka=float(ka),
        V=float(V),
        CL=float(CL),
        E0=0.0,
        Emax=0.0,
        EC50=1.0,
        gamma=1.0,
        t_start=sim_t_start,
        t_end=sim_t_end,
        t_eval=obs_times,
        dosing_interval=dosing_interval,
        n_doses=n_doses,
    )
    return sim["concentration"]


def estimate_pk_parameters(
    observed_times: np.ndarray,
    observed_conc: np.ndarray,
    Dose: float,
    F: float,
    n_doses: int,
    dosing_interval: Optional[float],
    initial_guess: Optional[Tuple[float, float, float]] = None,
) -> FitResult:
    """
    Estimate ka, V, CL using least_squares. Dose and F fixed.
    Returns FitResult.
    """
    # Validate inputs
    if observed_times.size != observed_conc.size:
        raise ValueError("observed_times and observed_conc must have the same length")
    if observed_times.size < MIN_OBS:
        raise ValueError(f"Need at least {MIN_OBS} observations to fit; got {observed_times.size}")

    # Default initial guesses from demonstration defaults
    default_ka = 1.0
    default_V = 20.0
    default_CL = 1.0
    if initial_guess is None:
        x0 = np.array([default_ka, default_V, default_CL], dtype=float)
    else:
        x0 = np.array(initial_guess, dtype=float)
        if x0.size != 3:
            raise ValueError("initial_guess must be a 3-tuple (ka, V, CL)")

    if not np.all(np.isfinite(x0)):
        raise ValueError("Initial guesses must be finite numbers")
    if np.any(x0 <= 0.0):
        raise ValueError("Initial guesses must be strictly positive for ka, V, CL")

    # Bounds to enforce positivity
    lb = np.array([1e-8, 1e-8, 1e-8], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf], dtype=float)

    # Residual function
    def residuals(params: np.ndarray) -> np.ndarray:
        ka_p, V_p, CL_p = params
        # Enforce positivity here although optimizer has bounds
        if not (np.isfinite(ka_p) and np.isfinite(V_p) and np.isfinite(CL_p)):
            return np.full_like(observed_conc, np.inf)
        if ka_p <= 0.0 or V_p <= 0.0 or CL_p <= 0.0:
            return np.full_like(observed_conc, np.inf)
        try:
            pred = predict_concentration(observed_times, ka_p, V_p, CL_p, Dose, F, n_doses, dosing_interval)
        except Exception:
            # propagate as large residuals
            return np.full_like(observed_conc, np.inf)
        return pred - observed_conc

    res = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-8, ftol=1e-8, gtol=1e-8)

    ka_fit, V_fit, CL_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])
    ke_fit = CL_fit / V_fit

    # Compute final predictions at observation times
    predicted = predict_concentration(observed_times, ka_fit, V_fit, CL_fit, Dose, F, n_doses, dosing_interval)
    rmse = float(np.sqrt(np.mean((observed_conc - predicted) ** 2)))

    fit_result = FitResult(
        ka=ka_fit,
        V=V_fit,
        CL=CL_fit,
        ke=ke_fit,
        success=bool(res.success),
        message=str(res.message),
        rmse=rmse,
        predicted_at_observations=predicted,
    )
    return fit_result


def plot_observed_vs_fitted(
    observed_times: np.ndarray,
    observed_conc: np.ndarray,
    fitted_time: np.ndarray,
    fitted_conc: np.ndarray,
    title: str = "Observed vs Fitted Concentration",
) -> plt.Figure:
    """
    Create Matplotlib Figure showing observed concentrations (scatter) and fitted model (line).
    Returns the Figure; does not call plt.show().
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fitted_time, fitted_conc, color="C0", lw=1.8, label="Fitted model")
    ax.scatter(observed_times, observed_conc, color="C1", s=30, zorder=5, label="Observed")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# End of module: no CLI entry points or interactive behavior on import
