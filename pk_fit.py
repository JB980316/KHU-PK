"""
pk_fit.py

Individual-level nonlinear PK parameter estimation for one-compartment oral model
that matches the simulator implemented in Code.py (one-compartment oral first-order
absorption, first-order elimination). Provides in-memory estimator plus CSV wrapper
and plotting helpers.

Design notes (concise):
- Mode A (default, F is None): fit apparent parameters CL_over_F and V_over_F
- Mode B (F supplied): fit systemic CL and V with fixed F
- Optimizes log-parameters using scipy.optimize.least_squares
- Reuses analytical single-dose formulas from Code.py and superposition for
  repeated-dose schedules
- Supports BLQ detection (boolean column or strings like "<0.1")
- Implements residual/error models: unweighted, additive, proportional, combined, log

API summary (important functions):
- load_pk_csv
- validate_observed_pk_data
- predict_pk_concentration
- estimate_pk_parameters
- estimate_pk_parameters_from_csv
- plot_pk_fit

"""
from typing import Dict, Optional, Sequence, Tuple, Mapping, Any
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import stats
import math

# Import analytical helpers from existing simulator (Code.py)
# Code.py provides analytical_pk_unequal_rates and analytical_pk_equal_rates
from Code import analytical_pk_unequal_rates, analytical_pk_equal_rates

# Constants
KE_RTOL = 1e-8
KE_ATOL = 1e-12
EVENT_TIME_TOL = 1e-10

# -------------------------
# CSV loading / preprocessing
# -------------------------

def load_pk_csv(
    csv_path: str,
    time_col: str = "time",
    concentration_col: str = "concentration",
    subject_col: Optional[str] = None,
    blq_col: Optional[str] = None,
    na_policy: str = "raise",
) -> pd.DataFrame:
    """Load PK CSV and perform basic parsing/BLQ detection.

    Returns a pandas DataFrame with columns: time, concentration, blq (bool), lloq (float or NaN)

    Raises descriptive errors for malformed files.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Unable to read CSV at {csv_path!s}: {e}")

    if subject_col is not None and subject_col in df.columns:
        unique_subjects = df[subject_col].dropna().unique()
        if len(unique_subjects) != 1:
            raise ValueError("CSV contains multiple subjects; this estimator performs individual-level fitting. Provide subject_col and select a single subject.")
        # filter to that subject
        df = df[df[subject_col] == unique_subjects[0]].copy()

    if time_col not in df.columns:
        raise ValueError(f"Missing required time column '{time_col}' in CSV.")
    if concentration_col not in df.columns:
        raise ValueError(f"Missing required concentration column '{concentration_col}' in CSV.")

    # Extract columns
    df_proc = df.copy()

    # Parse time
    df_proc["time"] = pd.to_numeric(df_proc[time_col], errors="coerce")
    if df_proc["time"].isnull().any():
        if na_policy == "raise":
            raise ValueError("Non-numeric or missing time values encountered.")
        elif na_policy == "drop":
            df_proc = df_proc.dropna(subset=["time"]).copy()
        else:
            raise ValueError(f"Unknown na_policy: {na_policy}")

    # Parse concentration and BLQ
    conc_raw = df_proc[concentration_col].astype(object)
    blq_flags = np.zeros(len(df_proc), dtype=bool)
    lloq_vals = np.full(len(df_proc), np.nan, dtype=float)
    conc_vals = np.full(len(df_proc), np.nan, dtype=float)

    for i, val in enumerate(conc_raw.values):
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("<"):
                # parse LLOQ
                num_part = s[1:].strip()
                try:
                    lloq = float(num_part)
                except Exception:
                    raise ValueError(f"Malformed BLQ value at row {i}: {val!r}")
                if not (lloq > 0 and np.isfinite(lloq)):
                    raise ValueError(f"Parsed LLOQ must be >0 at row {i}; got {lloq}")
                blq_flags[i] = True
                lloq_vals[i] = lloq
            else:
                # try numeric string
                num = pd.to_numeric(s, errors="coerce")
                if pd.isna(num):
                    raise ValueError(f"Malformed concentration value at row {i}: {val!r}")
                conc_vals[i] = float(num)
        elif pd.isna(val):
            if na_policy == "raise":
                raise ValueError(f"Missing concentration at row {i}.")
            else:
                # drop later
                conc_vals[i] = np.nan
        else:
            # numeric
            try:
                num = float(val)
            except Exception:
                raise ValueError(f"Unable to interpret concentration at row {i}: {val!r}")
            conc_vals[i] = num

    # If explicit BLQ indicator column supplied, reconcile
    if blq_col is not None and blq_col in df_proc.columns:
        explicit_blq = df_proc[blq_col].astype(bool).values
        # if any explicit blq True but concentration numeric non-NaN, treat as BLQ and drop numeric
        for i, b in enumerate(explicit_blq):
            if b and not blq_flags[i]:
                # mark as BLQ without explicit LLOQ
                blq_flags[i] = True
                lloq_vals[i] = np.nan
            if not b and blq_flags[i]:
                # contradictory: string '<...' present but flag false
                raise ValueError(f"Contradictory BLQ indicators at row {i}: concentration value suggests BLQ but {blq_col} is False.")

    df_proc["concentration"] = conc_vals
    df_proc["blq"] = blq_flags
    df_proc["lloq"] = lloq_vals

    # Handle NA policy for concentration
    if na_policy == "drop":
        before = len(df_proc)
        df_proc = df_proc.dropna(subset=["concentration"]).copy()
        after = len(df_proc)
        df_proc.attrs["rows_dropped"] = before - after
    elif na_policy == "raise":
        if df_proc["concentration"].isnull().any():
            raise ValueError("Missing or non-numeric concentration values encountered.")

    # Reject negative quantified concentrations
    quantified_mask = ~df_proc["blq"].values
    if np.any(df_proc.loc[quantified_mask, "concentration"] < 0):
        bad_idx = df_proc.index[df_proc.loc[:, "concentration"] < 0][0]
        raise ValueError(f"Negative quantified concentration at row index {bad_idx}.")

    # Sort by time (stable)
    df_proc = df_proc.sort_values("time", kind="mergesort").reset_index(drop=True)

    df_proc.attrs["original_row_count"] = len(df)
    df_proc.attrs["parsed_rows"] = len(df_proc)
    df_proc.attrs["blq_count"] = int(np.sum(df_proc["blq"].values))

    return df_proc

# -------------------------
# Observed-data validation
# -------------------------

def validate_observed_pk_data(
    time: Sequence[float],
    concentration: Sequence[float],
    n_parameters: int,
    allow_duplicate_times: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and canonicalize observed data.

    Returns (time_array, concentration_array) as 1-D float numpy arrays sorted by time.
    Raises ValueError on invalid inputs.
    """
    t = np.asarray(time)
    c = np.asarray(concentration)

    if t.ndim != 1 or c.ndim != 1:
        raise ValueError("time and concentration must be 1-D sequences.")
    if t.size != c.size:
        raise ValueError("time and concentration must have equal length.")
    if t.size == 0:
        raise ValueError("No observations provided.")
    if not np.all(np.isfinite(t)):
        raise ValueError("All time values must be finite.")
    if not np.all(np.isfinite(c[np.isfinite(c)])):
        raise ValueError("Concentration values must be finite where present.")
    if np.any(c[~np.isnan(c)] < 0):
        raise ValueError("Concentrations must be >= 0 for quantified observations.")

    # Sort by time (stable)
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    c_sorted = c[order]

    if not allow_duplicate_times:
        if np.any(np.diff(t_sorted) == 0.0):
            raise ValueError("Duplicate observation times are not allowed with allow_duplicate_times=False.")

    if t_sorted.size <= n_parameters:
        raise ValueError(f"Insufficient observations: need more than {n_parameters} observations (n_obs={t_sorted.size}).")

    return t_sorted, c_sorted

# -------------------------
# Prediction using analytical formulas + superposition
# -------------------------

def _single_dose_prediction_at_elapsed(
    elapsed: np.ndarray,
    dose: float,
    F_val: float,
    ka: float,
    ke: float,
    V: float,
) -> np.ndarray:
    # elapsed is nonnegative
    if np.isclose(ka, ke, rtol=KE_RTOL, atol=KE_ATOL):
        _, _, conc = analytical_pk_equal_rates(elapsed, dose, F_val, ka, V)
    else:
        _, _, conc = analytical_pk_unequal_rates(elapsed, dose, F_val, ka, ke, V)
    return conc


def predict_pk_concentration(
    time: Sequence[float],
    dose: float,
    parameters: Mapping[str, float],
    *,
    F: Optional[float] = None,
    dose_times: Optional[Sequence[float]] = None,
    dose_amounts: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Predict concentrations at given absolute times using analytical superposition.

    Parameters mapping depends on mode:
      - Mode A (F is None): parameters must contain ka, CL_over_F, V_over_F
      - Mode B (F supplied): parameters must contain ka, CL, V

    Returns 1-D numpy array of predicted concentrations aligned with input times.
    """
    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        raise ValueError("time must be 1-D")

    # Build dose schedule
    if dose_times is None:
        dose_times_arr = np.array([0.0], dtype=float)
    else:
        dose_times_arr = np.asarray(dose_times, dtype=float)
        if dose_times_arr.ndim != 1:
            raise ValueError("dose_times must be 1-D sequence")
    if dose_amounts is None:
        dose_amounts_arr = np.full(dose_times_arr.shape, float(dose), dtype=float)
    else:
        dose_amounts_arr = np.asarray(dose_amounts, dtype=float)
        if dose_amounts_arr.shape != dose_times_arr.shape:
            raise ValueError("dose_times and dose_amounts must have same length")

    # Mode mapping
    if F is None:
        # apparent parameters
        if not all(k in parameters for k in ("ka", "CL_over_F", "V_over_F")):
            raise ValueError("parameters must include ka, CL_over_F, V_over_F when F is None")
        ka = float(parameters["ka"])
        CL_over_F = float(parameters["CL_over_F"])
        V_over_F = float(parameters["V_over_F"])
        # ke derived
        ke = CL_over_F / V_over_F
        # When using apparent parameters, the analytical formula treats Dose*F in numerator.
        # To match apparent parameters, set F_effective = 1.0 and use dose * (1.0) but pass V=V_over_F and CL used through ke.
        # Alternatively, use F_effective = 1.0 and V = V_over_F, and dose remains Dose (mg). Analytical formula uses F*Dose; to avoid double-F we set F_effective=1.0.
        F_for_prediction = 1.0
        V_for_prediction = V_over_F
    else:
        if not (F > 0.0 and F <= 1.0 and np.isfinite(F)):
            raise ValueError("F must satisfy 0 < F <= 1 when supplied.")
        if not all(k in parameters for k in ("ka", "CL", "V")):
            raise ValueError("parameters must include ka, CL, V when F is supplied")
        ka = float(parameters["ka"])
        CL = float(parameters["CL"])
        V = float(parameters["V"])
        ke = CL / V
        F_for_prediction = float(F)
        V_for_prediction = V

    # For each dose, superpose contributions
    preds = np.zeros_like(t, dtype=float)
    for dtime, damt in zip(dose_times_arr, dose_amounts_arr):
        elapsed = t - float(dtime)
        # Only include non-negative elapsed times
        mask = elapsed >= -1e-14
        if not np.any(mask):
            continue
        elapsed_valid = np.maximum(elapsed, 0.0)
        conc_i = _single_dose_prediction_at_elapsed(elapsed_valid, float(damt), F_for_prediction, ka, ke, V_for_prediction)
        preds[mask] += conc_i[mask]

    if not np.all(np.isfinite(preds)):
        raise RuntimeError("Predicted concentrations contain non-finite values")
    if np.min(preds) < -1e-14:
        raise RuntimeError("Predicted concentrations contain negative values beyond numerical noise")

    return preds

# -------------------------
# Parameter transforms
# -------------------------

def _pack_theta_from_params(params: Mapping[str, float], mode: str) -> np.ndarray:
    if mode == "A":
        return np.log(np.array([params["ka"], params["CL_over_F"], params["V_over_F"]], dtype=float))
    else:
        return np.log(np.array([params["ka"], params["CL"], params["V"]], dtype=float))


def _unpack_params_from_theta(theta: np.ndarray, mode: str) -> Dict[str, float]:
    vals = np.exp(theta)
    if mode == "A":
        return {"ka": float(vals[0]), "CL_over_F": float(vals[1]), "V_over_F": float(vals[2])}
    else:
        return {"ka": float(vals[0]), "CL": float(vals[1]), "V": float(vals[2])}

# -------------------------
# Residuals
# -------------------------

def _build_residuals_function(
    time_obs: np.ndarray,
    conc_obs: np.ndarray,
    dose: float,
    dose_times: Optional[Sequence[float]],
    dose_amounts: Optional[Sequence[float]],
    mode: str,
    F: Optional[float],
    error_model: str,
    sigma_add: Optional[float],
    sigma_prop: Optional[float],
):
    n = time_obs.size

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = _unpack_params_from_theta(theta, mode)
        preds = predict_pk_concentration(time_obs, dose, params, F=F, dose_times=dose_times, dose_amounts=dose_amounts)
        if error_model == "unweighted":
            return conc_obs - preds
        elif error_model == "additive":
            if not (sigma_add is not None and sigma_add > 0 and np.isfinite(sigma_add)):
                raise ValueError("sigma_add must be >0 for additive error model")
            return (conc_obs - preds) / float(sigma_add)
        elif error_model == "proportional":
            if not (sigma_prop is not None and sigma_prop > 0 and np.isfinite(sigma_prop)):
                raise ValueError("sigma_prop must be >0 for proportional error model")
            floor = 1e-12
            scale = sigma_prop * np.maximum(preds, floor)
            return (conc_obs - preds) / scale
        elif error_model == "combined":
            if not (sigma_add is not None and sigma_prop is not None and sigma_add > 0 and sigma_prop > 0 and np.isfinite(sigma_add) and np.isfinite(sigma_prop)):
                raise ValueError("sigma_add and sigma_prop must be >0 for combined error model")
            scale = np.sqrt(sigma_add ** 2 + (sigma_prop * preds) ** 2)
            if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
                raise RuntimeError("Invalid combined error scale computed")
            return (conc_obs - preds) / scale
        elif error_model == "log":
            # Require strictly positive observations and predictions
            if np.any(conc_obs <= 0) or np.any(preds <= 0):
                raise ValueError("Log-error model requires strictly positive observations and predictions")
            return np.log(conc_obs) - np.log(preds)
        else:
            raise ValueError(f"Unknown error_model: {error_model}")

    return residuals

# -------------------------
# Estimation
# -------------------------

def _default_initial_guesses(
    time_obs: np.ndarray,
    conc_obs: np.ndarray,
    dose: float,
    mode: str,
    F: Optional[float],
) -> Dict[str, float]:
    # Estimate terminal slope using last positive points
    mask_pos = conc_obs > 0
    if np.sum(mask_pos) >= 3:
        t_pos = time_obs[mask_pos]
        c_pos = conc_obs[mask_pos]
        t_term = t_pos[-3:]
        c_term = c_pos[-3:]
        if np.any(c_term <= 0):
            kel = 0.1
        else:
            slope, _, _, _, _ = stats.linregress(t_term, np.log(c_term))
            kel = max(-slope, 1e-6)
    elif np.sum(mask_pos) >= 2:
        t_pos = time_obs[mask_pos]
        c_pos = conc_obs[mask_pos]
        slope, _, _, _, _ = stats.linregress(t_pos, np.log(np.maximum(c_pos, 1e-12)))
        kel = max(-slope, 1e-6)
    else:
        kel = 0.1

    # V guess via dose / Cmax
    cmax = np.max(np.maximum(conc_obs, 1e-12))
    V_guess = float(dose / cmax)
    CL_guess = kel * V_guess
    ka_guess = max(2.0 * kel, 0.1)

    if mode == "A":
        return {"ka": ka_guess, "CL_over_F": CL_guess, "V_over_F": V_guess}
    else:
        return {"ka": ka_guess, "CL": CL_guess, "V": V_guess}


def estimate_pk_parameters(
    time: Sequence[float],
    concentration: Sequence[float],
    dose: float,
    *,
    F: Optional[float] = None,
    initial_guesses: Optional[Mapping[str, float]] = None,
    parameter_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    error_model: str = "proportional",
    sigma_add: Optional[float] = None,
    sigma_prop: Optional[float] = None,
    dose_times: Optional[Sequence[float]] = None,
    dose_amounts: Optional[Sequence[float]] = None,
    max_nfev: int = 5000,
) -> Dict[str, Any]:
    """Estimate PK parameters using nonlinear least squares.

    Returns a result dictionary with parameters, uncertainties, fit statistics, and diagnostics.
    """
    # Validate inputs
    mode = "A" if F is None else "B"
    # canonicalize observed
    t_obs, c_obs = validate_observed_pk_data(time, concentration, n_parameters=3, allow_duplicate_times=True)

    # BLQ are expected to be preprocessed by CSV loader. Here we assume all data are quantified.
    # Determine initial guesses
    if initial_guesses is None:
        init = _default_initial_guesses(t_obs, c_obs, dose, mode, F)
    else:
        init = dict(initial_guesses)
        # validation
    # validate parameter names and positivity
    required_names = ("ka", "CL_over_F", "V_over_F") if mode == "A" else ("ka", "CL", "V")
    for name in required_names:
        if name not in init:
            raise ValueError(f"Missing initial guess for parameter '{name}'")
        val = float(init[name])
        if not (np.isfinite(val) and val > 0):
            raise ValueError(f"Initial guess for {name} must be finite and >0; got {init[name]!r}")

    # bounds
    if parameter_bounds is None:
        bounds = {name: (1e-8, 1e8) for name in required_names}
    else:
        bounds = {}
        for name in required_names:
            if name not in parameter_bounds:
                raise ValueError(f"Bounds must include '{name}'")
            lo, hi = parameter_bounds[name]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"Bounds for {name} must be finite numbers")
            if not (lo > 0 and hi > lo):
                raise ValueError(f"Bounds for {name} must satisfy 0 < lower < upper; got ({lo},{hi})")
            val0 = float(init[name])
            if not (lo < val0 < hi):
                raise ValueError(f"Initial value {val0} for {name} must satisfy lower < initial < upper")
            bounds[name] = (float(lo), float(hi))

    # pack initial theta and bounds in log-space
    theta0 = _pack_theta_from_params(init, mode)
    lb = np.log(np.array([bounds[n][0] for n in required_names], dtype=float))
    ub = np.log(np.array([bounds[n][1] for n in required_names], dtype=float))

    # Build residual function
    residual_fun = _build_residuals_function(t_obs, c_obs, dose, dose_times, dose_amounts, mode, F, error_model, sigma_add, sigma_prop)

    # Run least_squares
    lsq = least_squares(residual_fun, theta0, bounds=(lb, ub), max_nfev=max_nfev)

    # Postprocess
    success = bool(lsq.success)
    message = str(lsq.message)
    theta_hat = lsq.x
    params_hat = _unpack_params_from_theta(theta_hat, mode)

    # predictions and residuals on concentration scale
    preds = predict_pk_concentration(t_obs, dose, params_hat, F=F, dose_times=dose_times, dose_amounts=dose_amounts)
    residual_raw = c_obs - preds
    # weighted residuals
    try:
        weighted = residual_fun(theta_hat)
    except Exception as e:
        weighted = np.full_like(residual_raw, np.nan)

    n_obs = t_obs.size
    n_params = 3
    dof = n_obs - n_params

    rss = float(np.sum(residual_raw ** 2))
    rmse = math.sqrt(rss / n_obs) if n_obs > 0 else float('nan')

    # AIC/BIC (using Gaussian SSE approx): AIC = n*ln(RSS/n) + 2k
    aic = float(n_obs * np.log(rss / n_obs) + 2 * n_params) if rss > 0 else float('nan')
    bic = float(n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)) if rss > 0 else float('nan')

    # R^2
    ss_tot = float(np.sum((c_obs - np.mean(c_obs)) ** 2))
    r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else float('nan')

    # Uncertainty estimation using Jacobian
    warnings = []
    se = {k: None for k in required_names}
    ci = {k: (None, None) for k in required_names}

    J = lsq.jac
    if dof > 0 and J is not None and J.size > 0:
        # compute covariance of theta: cov = inv(J^T J) * (rss/dof)
        try:
            JTJ = J.T.dot(J)
            # check rank
            rank = np.linalg.matrix_rank(JTJ)
            if rank < n_params:
                warnings.append("Jacobian is rank-deficient; covariance unavailable.")
            else:
                cov_theta = np.linalg.inv(JTJ) * (rss / dof)
                se_theta = np.sqrt(np.diag(cov_theta))
                # transform to parameter scale using delta method for exp(theta)
                theta_hat = theta_hat.reshape(-1)
                for i, name in enumerate(required_names):
                    se[name] = float(np.exp(theta_hat[i]) * se_theta[i])
                    # 95% CI on log-scale
                    lower = math.exp(theta_hat[i] - 1.96 * se_theta[i])
                    upper = math.exp(theta_hat[i] + 1.96 * se_theta[i])
                    ci[name] = (lower, upper)
        except np.linalg.LinAlgError:
            warnings.append("Failed to invert information matrix; covariance unavailable.")
    else:
        warnings.append("Insufficient residual degrees of freedom or Jacobian unavailable; uncertainty not estimated.")

    # Derived quantities
    if mode == "A":
        ke = params_hat["CL_over_F"] / params_hat["V_over_F"]
        half_life = math.log(2.0) / ke if ke > 0 else float('nan')
        params_hat_out = {
            "ka": params_hat["ka"],
            "CL_over_F": params_hat["CL_over_F"],
            "V_over_F": params_hat["V_over_F"],
            "ke": ke,
            "half_life": half_life,
        }
    else:
        ke = params_hat["CL"] / params_hat["V"]
        half_life = math.log(2.0) / ke if ke > 0 else float('nan')
        params_hat_out = {
            "ka": params_hat["ka"],
            "CL": params_hat["CL"],
            "V": params_hat["V"],
            "F": float(F),
            "ke": ke,
            "half_life": half_life,
        }

    result = {
        "success": success,
        "message": message,
        "parameters": params_hat_out,
        "standard_errors": se,
        "confidence_intervals": ci,
        "observations": {
            "time": t_obs,
            "concentration": c_obs,
            "predicted": preds,
            "residual": residual_raw,
            "weighted_residual": weighted,
        },
        "fit_statistics": {
            "n_obs": n_obs,
            "n_params": n_params,
            "degrees_of_freedom": dof,
            "rss": rss,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "r_squared": r_squared,
        },
        "optimizer": {
            "cost": float(lsq.cost),
            "optimality": float(lsq.optimality) if hasattr(lsq, 'optimality') else float('nan'),
            "nfev": int(lsq.nfev),
            "njev": int(lsq.njev) if hasattr(lsq, 'njev') else None,
            "status": int(lsq.status),
        },
        "units": {
            "time": "h",
            "concentration": "mg/L",
        },
        "meta": {
            "parameterization": "apparent" if mode == "A" else "systemic_fixed_F",
            "error_model": error_model,
            "sigma_add": sigma_add,
            "sigma_prop": sigma_prop,
            "dosing_convention": "absolute dose times on observation clock",
            "dose_times_are_absolute": True,
            "blq_policy": None,
            "warnings": warnings,
        },
    }

    return result

# -------------------------
# CSV wrapper
# -------------------------

def estimate_pk_parameters_from_csv(
    csv_path: str,
    dose: float,
    *,
    F: Optional[float] = None,
    time_col: str = "time",
    concentration_col: str = "concentration",
    subject_col: Optional[str] = None,
    blq_col: Optional[str] = None,
    blq_policy: str = "exclude",
    lloq: Optional[float] = None,
    initial_guesses: Optional[Mapping[str, float]] = None,
    parameter_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    error_model: str = "proportional",
    sigma_add: Optional[float] = None,
    sigma_prop: Optional[float] = None,
    dose_times: Optional[Sequence[float]] = None,
    dose_amounts: Optional[Sequence[float]] = None,
    max_nfev: int = 5000,
) -> Dict[str, Any]:
    df = load_pk_csv(csv_path, time_col=time_col, concentration_col=concentration_col, subject_col=subject_col, blq_col=blq_col, na_policy="raise")

    original_n = df.attrs.get("original_row_count", len(df))
    blq_count = int(np.sum(df["blq"].values))

    if blq_policy == "exclude":
        df_fit = df[~df["blq"].values].copy()
    elif blq_policy == "half_lloq":
        # substitute half LLOQ for BLQ rows
        if lloq is not None:
            if not (lloq > 0 and np.isfinite(lloq)):
                raise ValueError("When providing global lloq it must be >0")
            # set lloq where missing
            df.loc[df['blq'] & df['lloq'].isna(), 'lloq'] = lloq
        # ensure every BLQ row has lloq
        if df.loc[df['blq'], 'lloq'].isnull().any():
            raise ValueError("Some BLQ rows do not contain an explicit LLOQ and no global lloq was supplied.")
        df_fit = df.copy()
        df_fit.loc[df_fit['blq'], 'concentration'] = df_fit.loc[df_fit['blq'], 'lloq'] / 2.0
        # mark these as not BLQ for fitting
        df_fit['blq'] = False
    else:
        raise ValueError(f"Unknown blq_policy: {blq_policy}")

    dropped = original_n - len(df_fit)

    # Extract arrays
    time_arr = df_fit['time'].values.astype(float)
    conc_arr = df_fit['concentration'].values.astype(float)

    result = estimate_pk_parameters(
        time_arr,
        conc_arr,
        dose,
        F=F,
        initial_guesses=initial_guesses,
        parameter_bounds=parameter_bounds,
        error_model=error_model,
        sigma_add=sigma_add,
        sigma_prop=sigma_prop,
        dose_times=dose_times,
        dose_amounts=dose_amounts,
        max_nfev=max_nfev,
    )

    # merge preprocessing metadata
    meta = result.setdefault('meta', {})
    meta['preprocessing'] = {
        'original_rows': int(original_n),
        'rows_after_blq_policy': int(len(df_fit)),
        'blq_count': int(blq_count),
        'rows_dropped': int(dropped),
        'blq_policy': blq_policy,
    }

    return result

# -------------------------
# Plotting fitter results
# -------------------------

def plot_pk_fit(fit_results: Dict[str, Any], log_y: bool = False) -> None:
    import matplotlib.pyplot as plt
    t = fit_results['observations']['time']
    y = fit_results['observations']['concentration']
    yhat = fit_results['observations']['predicted']
    resid = fit_results['observations']['residual']
    wresid = fit_results['observations']['weighted_residual']

    fig, axes = plt.subplots(3, 1, figsize=(7, 9))

    # Observed vs time
    if log_y:
        # omit zero-valued points from plotting
        pos_mask = y > 0
        if not np.all(pos_mask):
            import warnings
            warnings.warn("Zero or nonpositive observations omitted from log-y plot.")
        axes[0].plot(t[pos_mask], y[pos_mask], 'o', label='Observed')
        axes[0].plot(t[pos_mask], yhat[pos_mask], '-', label='Predicted')
        axes[0].set_yscale('log')
    else:
        axes[0].plot(t, y, 'o', label='Observed')
        axes[0].plot(t, yhat, '-', label='Predicted')
    axes[0].set_ylabel('Concentration (mg/L)')
    axes[0].legend()
    axes[0].grid(True)

    # Observed vs Predicted
    axes[1].scatter(yhat, y, c='C0')
    axes[1].plot([min(yhat.min(), y.min()), max(yhat.max(), y.max())], [min(yhat.min(), y.min()), max(yhat.max(), y.min())], color='k', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Observed')
    axes[1].grid(True)

    # Residuals vs time
    axes[2].scatter(t, resid, c='C1', label='Residual')
    axes[2].axhline(0, color='k', linestyle='--')
    axes[2].set_xlabel('Time (h)')
    axes[2].set_ylabel('Residual (mg/L)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
