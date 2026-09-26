import streamlit as st
import plotly.graph_objects as go

from core.physics import SpinCoatingParams, solve_thickness


st.set_page_config(
    page_title="Spin-Coating Forward Model",
    page_icon="🔬",
    layout="wide",
)

st.title("Spin-Coating Forward Model")
st.markdown(
    """
    Minimal dimensionless forward model:

    \\[
    \\frac{d\\hat{h}}{d\\tau}
    =
    -\\tilde{K}\\hat{h}^3
    -
    \\tilde{E}
    \\]

    with

    \\[
    \\tilde{K}
    =
    \\left(\\frac{\\omega}{\\omega_{\\text{ref}}}\\right)^2
    \\Psi_0
    \\]

    This first version uses constant \\(\\Psi_0\\) and \\(\\tilde{E}\\).
    The concentration-dependent mechanistic model should be added after
    verifying the exact solvent-fraction ODE from the reference notebook.
    """
)


@st.cache_data
def get_solution(
    h0: float,
    omega_ratio: float,
    psi0: float,
    e0: float,
    tau_end: float,
    n_points: int,
):
    params = SpinCoatingParams(
        h0=h0,
        omega_ratio=omega_ratio,
        psi0=psi0,
        e0=e0,
        tau_end=tau_end,
    )

    tau, h = solve_thickness(params, n_points=n_points)
    return tau, h


with st.sidebar:
    st.header("Process Parameters")

    omega_ratio = st.slider(
        label="Spin-speed ratio, ω / ω_ref",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.05,
    )

    psi0 = st.slider(
        label="Viscosity-related coefficient, Ψ₀",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.05,
    )

    e0 = st.slider(
        label="Evaporation coefficient, Ẽ",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.005,
    )

    tau_end = st.slider(
        label="Final dimensionless time, τ_end",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    n_points = st.slider(
        label="Number of output points",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
    )

    st.markdown("---")
    st.header("Derived Quantity")

    K_tilde = omega_ratio**2 * psi0
    st.latex(
        r"""
        \tilde{K}
        =
        \left(\frac{\omega}{\omega_{\mathrm{ref}}}\right)^2
        \Psi_0
        """
    )
    st.metric("K̃", f"{K_tilde:.4f}")


try:
    tau, h = get_solution(
        h0=1.0,
        omega_ratio=omega_ratio,
        psi0=psi0,
        e0=e0,
        tau_end=tau_end,
        n_points=n_points,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=tau,
                y=h,
                mode="lines",
                line=dict(width=3),
                name="Film thickness",
            )
        )

        fig.update_layout(
            title="Dimensionless Thickness Evolution",
            xaxis_title="Dimensionless time, τ",
            yaxis_title="Dimensionless thickness, ĥ(τ)",
            template="plotly_white",
            height=520,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Summary")

        st.metric("Initial thickness", f"{h[0]:.6f}")
        st.metric("Final thickness", f"{h[-1]:.6f}")
        st.metric("Thickness reduction", f"{100.0 * (1.0 - h[-1] / h[0]):.2f}%")

        st.markdown("---")
        st.subheader("Sanity Checks")

        st.write(
            """
            The curve should be:
            - monotonically decreasing,
            - nonnegative,
            - linear if Ψ₀ = 0,
            - non-linear if Ψ₀ > 0.
            """
        )

except Exception as exc:
    st.error(f"Simulation failed: {exc}")
