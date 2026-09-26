from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SpinCoatingParams:
    """
    Minimal dimensionless spin-coating parameters.

    h0:
        Initial dimensionless thickness, usually 1.
    omega_ratio:
        omega / omega_ref. The governing equation uses (omega / omega_ref)^2.
    psi0:
        Constant viscosity-related coefficient for the first app version.
    e0:
        Constant evaporation coefficient for the first app version.
    tau_end:
        Final dimensionless time.
    """

    h0: float = 1.0
    omega_ratio: float = 1.0
    psi0: float = 1.0
    e0: float = 0.05
    tau_end: float = 1.0


def solve_thickness(params: SpinCoatingParams, n_points: int = 500):
    """
    Solve the minimal dimensionless thickness ODE:

        dh/dtau = -K_tilde * h^3 - E_tilde

    where, for this first version,

        K_tilde = (omega / omega_ref)^2 * psi0
        E_tilde = e0

    This is intentionally a constant-coefficient model.
    The concentration-dependent mechanistic model should be added later
    after verifying the exact solvent-fraction ODE from your reference code.
    """

    if params.h0 <= 0:
        raise ValueError("Initial thickness h0 must be positive.")

    if params.tau_end <= 0:
        raise ValueError("tau_end must be positive.")

    if params.omega_ratio < 0:
        raise ValueError("omega_ratio must be nonnegative.")

    if params.psi0 < 0:
        raise ValueError("psi0 must be nonnegative.")

    if params.e0 < 0:
        raise ValueError("e0 must be nonnegative.")

    K_tilde = params.omega_ratio**2 * params.psi0
    E_tilde = params.e0

    def rhs(tau, y):
        h = max(float(y[0]), 1.0e-12)
        dh_dtau = -K_tilde * h**3 - E_tilde
        return [dh_dtau]

    def dry_event(tau, y):
        return y[0] - 1.0e-12

    dry_event.terminal = True
    dry_event.direction = -1

    t_eval = np.linspace(0.0, params.tau_end, n_points)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, params.tau_end),
        y0=[params.h0],
        method="LSODA",
        t_eval=t_eval,
        events=dry_event,
        rtol=1.0e-9,
        atol=1.0e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    tau = sol.t
    h = sol.y[0]

    # If the film reaches near-zero thickness before tau_end, append the event point.
    if sol.t_events[0].size > 0:
        tau_event = sol.t_events[0][0]
        h_event = sol.y_events[0][0][0]

        tau = np.append(tau, tau_event)
        h = np.append(h, h_event)

    h = np.maximum(h, 0.0)

    return tau, h


def analytic_constant_K_no_E(h0: float, K: float, tau: np.ndarray) -> np.ndarray:
    """
    Analytic solution for dh/dtau = -K h^3, E = 0.

    h(tau) = h0 / sqrt(1 + 2 K h0^2 tau)

    This is useful for testing the numerical solver.
    """

    if h0 <= 0:
        raise ValueError("h0 must be positive.")

    if K < 0:
        raise ValueError("K must be nonnegative.")

    tau = np.asarray(tau)

    return h0 / np.sqrt(1.0 + 2.0 * K * h0**2 * tau)