from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


st.set_page_config(
    page_title="Spin Coating Thin-Film Simulator",
    layout="wide",
)


plt.rcParams.update(
    {
        "figure.figsize": (7, 4.5),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def rpm_to_rad_per_s(rpm: float) -> float:
    return rpm * 2.0 * np.pi / 60.0


def effective_evaporation_rate_um_s(rpm: float, evaporation_mode: str, evaporation_direct_um_s: float, evaporation_ref_um_s: float):
    if evaporation_mode.startswith("Rotation-dependent"):
        return evaporation_ref_um_s * np.sqrt(rpm / 3000.0)
    return evaporation_direct_um_s


def viscosity_model(t, eta0: float, alpha: float):
    # eta(t) = eta0 exp(alpha t)
    return eta0 * np.exp(alpha * t)


def ebp_parameter(rho: float, omega: float, eta):
    # Exact Emslie-Bonner-Peck parameter: K = rho omega^2 / (3 eta)
    return rho * omega**2 / (3.0 * eta)


def thickness_ode(t, h, rho: float, omega: float, eta0: float, alpha: float, evaporation_rate: float):
    h_current = max(h[0], 0.0)
    eta_t = viscosity_model(t, eta0, alpha)
    k_t = ebp_parameter(rho, omega, eta_t)
    dhdt = -2.0 * k_t * h_current**3 - evaporation_rate
    return [dhdt]


def analytical_ebp_solution(t, h0: float, rho: float, omega: float, eta: float):
    k = ebp_parameter(rho, omega, eta)
    return h0 / np.sqrt(1.0 + 4.0 * k * h0**2 * t)


def detect_transition(time_array, thickness_array, rho: float, omega: float, eta0: float, alpha: float, evaporation_rate: float):
    eta_array = viscosity_model(time_array, eta0, alpha)
    k_array = ebp_parameter(rho, omega, eta_array)
    flow_rate = 2.0 * k_array * thickness_array**3
    difference = flow_rate - evaporation_rate
    sign_change_indices = np.where(np.diff(np.sign(difference)) != 0)[0]

    if len(sign_change_indices) == 0:
        return None, flow_rate

    idx = sign_change_indices[0]
    t1, t2 = time_array[idx], time_array[idx + 1]
    d1, d2 = difference[idx], difference[idx + 1]
    transition_time = t1 - d1 * (t2 - t1) / (d2 - d1)
    return transition_time, flow_rate


def gaussian_initial_profile(r, wafer_radius: float, base_thickness: float, perturbation_amplitude: float, beta: float):
    # h(r,0) = h0 + A exp[-beta (r/R)^2]
    return base_thickness + perturbation_amplitude * np.exp(-beta * (r / wafer_radius) ** 2)


def make_initial_profile(
    profile_type: str,
    r,
    wafer_radius: float,
    base_thickness: float,
    perturbation_amplitude: float,
    beta: float,
):
    if profile_type == "Uniform":
        return np.full_like(r, base_thickness)
    return gaussian_initial_profile(r, wafer_radius, base_thickness, perturbation_amplitude, beta)


def compute_uniformity(thickness_profile):
    h_max = np.max(thickness_profile)
    h_min = np.min(thickness_profile)
    h_mean = np.mean(thickness_profile)
    uniformity_percent = (h_max - h_min) / h_mean * 100.0
    return uniformity_percent, h_max, h_min, h_mean


def compute_dimensionless_numbers(
    rpm: float,
    rho: float,
    eta0: float,
    characteristic_thickness_um: float,
    wafer_radius_mm: float,
    surface_tension: float,
    mean_free_path_nm: float,
    mass_diffusivity: float,
):
    omega = rpm_to_rad_per_s(rpm)
    h0 = characteristic_thickness_um * 1e-6
    radius = wafer_radius_mm * 1e-3
    epsilon = h0 / radius
    k0 = ebp_parameter(rho, omega, eta0)
    characteristic_velocity = k0 * radius * h0**2
    reynolds = rho * characteristic_velocity * h0 / eta0
    reduced_reynolds = reynolds * epsilon**2
    capillary = eta0 * characteristic_velocity / surface_tension
    knudsen = mean_free_path_nm * 1e-9 / h0
    schmidt = eta0 / (rho * mass_diffusivity)
    return {
        "epsilon": epsilon,
        "velocity": characteristic_velocity,
        "reynolds": reynolds,
        "reduced_reynolds": reduced_reynolds,
        "capillary": capillary,
        "knudsen": knudsen,
        "schmidt": schmidt,
    }


@st.cache_data(show_spinner=False)
def solve_uniform_model(
    rpm: float,
    rho: float,
    eta0: float,
    alpha: float,
    h0_um: float,
    evaporation_um_s: float,
    t_end: float,
    num_time_points: int,
):
    omega = rpm_to_rad_per_s(rpm)
    h0 = h0_um * 1e-6
    evaporation_rate = evaporation_um_s * 1e-6
    t_eval = np.linspace(0.0, t_end, num_time_points)

    no_evap = solve_ivp(
        fun=lambda t, h: thickness_ode(t, h, rho, omega, eta0, alpha, 0.0),
        t_span=(0.0, t_end),
        y0=[h0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12,
    )
    with_evap = solve_ivp(
        fun=lambda t, h: thickness_ode(t, h, rho, omega, eta0, alpha, evaporation_rate),
        t_span=(0.0, t_end),
        y0=[h0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12,
    )
    transition_time, flow_rate = detect_transition(
        with_evap.t, with_evap.y[0], rho, omega, eta0, alpha, evaporation_rate
    )
    return no_evap.t, no_evap.y[0], with_evap.t, with_evap.y[0], transition_time, flow_rate


@st.cache_data(show_spinner=False)
def solve_layer_gelation_model(
    rpm: float,
    rho: float,
    eta0: float,
    alpha: float,
    h0_um: float,
    evaporation_um_s: float,
    initial_solid_fraction: float,
    gel_solid_fraction: float,
    t_end: float,
    num_time_points: int,
):
    omega = rpm_to_rad_per_s(rpm)
    h0 = h0_um * 1e-6
    evaporation_rate = evaporation_um_s * 1e-6
    solid_initial = initial_solid_fraction * h0
    liquid_initial = (1.0 - initial_solid_fraction) * h0
    t_eval = np.linspace(0.0, t_end, num_time_points)

    def layer_rhs(t, y):
        solid_height = max(y[0], 0.0)
        liquid_height = max(y[1], 0.0)
        total_height = solid_height + liquid_height
        if total_height <= 0.0:
            return [0.0, 0.0]

        eta_t = viscosity_model(t, eta0, alpha)
        k_t = ebp_parameter(rho, omega, eta_t)
        flow_thinning_rate = 2.0 * k_t * total_height**3
        solid_fraction = solid_height / total_height
        liquid_fraction = liquid_height / total_height

        dsolid_dt = -solid_fraction * flow_thinning_rate
        dliquid_dt = -liquid_fraction * flow_thinning_rate - evaporation_rate

        if liquid_height <= 0.0:
            dliquid_dt = max(dliquid_dt, 0.0)
        return [dsolid_dt, dliquid_dt]

    solution = solve_ivp(
        fun=layer_rhs,
        t_span=(0.0, t_end),
        y0=[solid_initial, liquid_initial],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12,
    )

    solid = np.maximum(solution.y[0], 0.0)
    liquid = np.maximum(solution.y[1], 0.0)
    total = solid + liquid
    concentration = np.divide(solid, total, out=np.ones_like(solid), where=total > 0.0)

    gel_indices = np.where(concentration >= gel_solid_fraction)[0]
    if len(gel_indices) == 0:
        gel_time = None
    else:
        gel_time = float(solution.t[gel_indices[0]])

    return solution.t, solid, liquid, concentration, gel_time


@st.cache_data(show_spinner=False)
def radial_solver(
    rpm: float,
    rho: float,
    eta0: float,
    alpha: float,
    evaporation_um_s: float,
    wafer_radius_mm: float,
    base_thickness_um: float,
    perturbation_um: float,
    beta: float,
    profile_type: str,
    fdm_scheme: str,
    t_end: float,
    num_radial_nodes: int,
    num_time_points: int,
):
    omega = rpm_to_rad_per_s(rpm)
    evaporation_rate = evaporation_um_s * 1e-6
    wafer_radius = wafer_radius_mm * 1e-3
    r_array = np.linspace(0.0, wafer_radius, num_radial_nodes)
    t_eval = np.linspace(0.0, t_end, num_time_points)
    h_initial = make_initial_profile(
        profile_type,
        r_array,
        wafer_radius,
        base_thickness_um * 1e-6,
        perturbation_um * 1e-6,
        beta,
    )

    dr = r_array[1] - r_array[0]
    r_safe = np.where(r_array == 0.0, dr / 2.0, r_array)

    def radial_rhs(t, h):
        h = np.maximum(h, 0.0)
        eta_t = viscosity_model(t, eta0, alpha)
        k_t = ebp_parameter(rho, omega, eta_t)
        flux_like = k_t * r_array**2 * h**3

        dflux_dr = np.zeros_like(h)
        dflux_dr[0] = (flux_like[1] - flux_like[0]) / dr
        if fdm_scheme.startswith("Outward"):
            dflux_dr[1:] = (flux_like[1:] - flux_like[:-1]) / dr
        else:
            dflux_dr[1:-1] = (flux_like[2:] - flux_like[:-2]) / (2.0 * dr)
            dflux_dr[-1] = (flux_like[-1] - flux_like[-2]) / dr

        dhdt = -(1.0 / r_safe) * dflux_dr - evaporation_rate
        return np.where(h <= 0.0, np.maximum(dhdt, 0.0), dhdt)

    solution = solve_ivp(
        fun=radial_rhs,
        t_span=(0.0, t_end),
        y0=h_initial,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-10,
    )
    return r_array, t_eval, h_initial, solution.y


@st.cache_data(show_spinner=False)
def validate_against_analytical_ebp(rpm: float, rho: float, eta0: float, h0_um: float, t_end: float):
    omega = rpm_to_rad_per_s(rpm)
    h0 = h0_um * 1e-6
    t_eval = np.linspace(0.0, t_end, 500)

    def ode(t, h):
        k = ebp_parameter(rho, omega, eta0)
        return [-2.0 * k * h[0] ** 3]

    solution = solve_ivp(
        fun=ode,
        t_span=(0.0, t_end),
        y0=[h0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12,
    )
    h_num = solution.y[0]
    h_ana = analytical_ebp_solution(solution.t, h0, rho, omega, eta0)
    relative_error = np.abs(h_num - h_ana) / h_ana
    return solution.t, h_num, h_ana, relative_error, float(np.max(relative_error))


@st.cache_data(show_spinner=True)
def parameter_sweep(
    rpm_min: float,
    rpm_max: float,
    eta_min: float,
    eta_max: float,
    n_rpm: int,
    n_eta: int,
    rho: float,
    alpha: float,
    wafer_radius_mm: float,
    base_thickness_um: float,
    perturbation_um: float,
    beta: float,
    profile_type: str,
    fdm_scheme: str,
    evaporation_mode: str,
    evaporation_direct_um_s: float,
    evaporation_ref_um_s: float,
    t_end: float,
):
    rpm_values = np.linspace(rpm_min, rpm_max, n_rpm)
    eta0_values = np.linspace(eta_min, eta_max, n_eta)
    uniformity_map = np.zeros((len(eta0_values), len(rpm_values)))

    for i, eta0_i in enumerate(eta0_values):
        for j, rpm_j in enumerate(rpm_values):
            effective_evaporation = effective_evaporation_rate_um_s(
                rpm_j,
                evaporation_mode,
                evaporation_direct_um_s,
                evaporation_ref_um_s,
            )
            r_array, _, _, h_radial = radial_solver(
                rpm=rpm_j,
                rho=rho,
                eta0=eta0_i,
                alpha=alpha,
                evaporation_um_s=effective_evaporation,
                wafer_radius_mm=wafer_radius_mm,
                base_thickness_um=base_thickness_um,
                perturbation_um=perturbation_um,
                beta=beta,
                profile_type=profile_type,
                fdm_scheme=fdm_scheme,
                t_end=t_end,
                num_radial_nodes=70,
                num_time_points=120,
            )
            uniformity_map[i, j] = compute_uniformity(h_radial[:, -1])[0]

    return rpm_values, eta0_values, uniformity_map


def draw_uniform_thickness_plot(time_no_evap, h_no_evap, time_evap, h_evap, transition_time):
    fig, ax = plt.subplots()
    ax.plot(time_no_evap, h_no_evap * 1e6, label="No evaporation", color="darkorange")
    ax.plot(time_evap, h_evap * 1e6, label="Evaporation included", color="crimson")
    if transition_time is not None:
        h_transition = np.interp(transition_time, time_evap, h_evap)
        ax.axvline(transition_time, color="black", linestyle="--", label=f"Transition {transition_time:.1f} s")
        ax.scatter(transition_time, h_transition * 1e6, color="black", zorder=5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Film thickness h [um]")
    ax.set_title("Uniform Film Thinning")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_viscosity_plot(time_array, eta0, alpha):
    fig, ax = plt.subplots()
    ax.plot(time_array, viscosity_model(time_array, eta0, alpha), color="darkgreen")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Viscosity eta(t) [Pa s]")
    ax.set_title("Viscosity Growth")
    fig.tight_layout()
    return fig


def draw_radial_profiles(r_array, time_array, h_radial):
    fig, ax = plt.subplots()
    for frac in [0.0, 0.25, 0.5, 1.0]:
        idx = int(frac * (len(time_array) - 1))
        ax.plot(r_array * 1000, h_radial[:, idx] * 1e6, label=f"t = {time_array[idx]:.1f} s")
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Thickness h(r,t) [um]")
    ax.set_title("Radial Thickness Evolution")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_final_profile(r_array, final_profile):
    fig, ax = plt.subplots()
    uniformity, h_max, h_min, _ = compute_uniformity(final_profile)
    idx_max = int(np.argmax(final_profile))
    idx_min = int(np.argmin(final_profile))
    ax.plot(r_array * 1000, final_profile * 1e6, color="black", label=f"Final profile, U = {uniformity:.2f}%")
    ax.scatter(r_array[idx_max] * 1000, h_max * 1e6, color="crimson", label="Max")
    ax.scatter(r_array[idx_min] * 1000, h_min * 1e6, color="navy", label="Min")
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Final thickness [um]")
    ax.set_title("Final Radial Uniformity")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_time_resolved_profile(r_array, profile, selected_time, uniformity, y_limit_um):
    fig, ax = plt.subplots()
    ax.plot(
        r_array * 1000,
        profile * 1e6,
        color="teal",
        label=f"t = {selected_time:.1f} s, U = {uniformity:.2f}%",
    )
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Thickness h(r,t) [um]")
    ax.set_title("Time-Resolved Radial Thickness Profile")
    ax.set_xlim(r_array[0] * 1000, r_array[-1] * 1000)
    ax.set_ylim(0.0, y_limit_um)
    ax.legend()
    fig.tight_layout()
    return fig


def draw_concentration_plot(time_array, concentration, gel_solid_fraction):
    fig, ax = plt.subplots()
    ax.plot(time_array, concentration, color="purple", label="Solid fraction C(t)")
    ax.axhline(gel_solid_fraction, color="black", linestyle="--", label="Gel threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solid fraction C(t) [-]")
    ax.set_title("Gelation Estimate from S + L Layer Model")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Spin Coating Thin-Film Simulator")
st.caption("Emslie-Bonner-Peck thinning with Meyerhofer-type evaporation and viscosity growth")

with st.sidebar:
    st.header("Process Inputs")
    rpm = st.slider("Spin speed [rpm]", 1000.0, 6000.0, 3000.0, 100.0)
    eta0 = st.slider("Initial viscosity eta0 [Pa s]", 0.01, 0.50, 0.10, 0.01)
    h0_um = st.slider("Uniform initial thickness [um]", 20.0, 200.0, 100.0, 5.0)
    rho = st.slider("Density rho [kg/m3]", 700.0, 1400.0, 1000.0, 25.0)
    alpha = st.slider("Viscosity growth alpha [1/s]", 0.0, 0.10, 0.05, 0.005)
    evaporation_mode = st.selectbox(
        "Evaporation model",
        ["Direct constant E", "Rotation-dependent E = E_ref sqrt(rpm/3000)"],
    )
    evaporation_direct_um_s = st.slider("Direct evaporation rate E [um/s]", 0.0, 0.20, 0.03, 0.005)
    evaporation_ref_um_s = st.slider("E_ref at 3000 rpm [um/s]", 0.0, 0.20, 0.03, 0.005)
    t_end = st.slider("Simulation time [s]", 10.0, 120.0, 60.0, 5.0)

    st.header("Radial Inputs")
    profile_type = st.selectbox("Initial radial profile", ["Gaussian", "Uniform"])
    wafer_radius_mm = st.slider("Wafer radius [mm]", 25.0, 100.0, 50.0, 5.0)
    base_thickness_um = st.slider("Radial base thickness [um]", 20.0, 150.0, 80.0, 5.0)
    perturbation_um = st.slider("Gaussian perturbation A [um]", 0.0, 100.0, 40.0, 5.0)
    beta = st.slider("Gaussian sharpness beta [-]", 1.0, 15.0, 6.0, 0.5)
    fdm_scheme = st.selectbox("FDM scheme", ["Central difference", "Outward upwind/backward difference"])
    num_radial_nodes = st.slider("Radial nodes", 40, 180, 100, 10)

    st.header("Gelation Estimate")
    initial_solid_fraction = st.slider("Initial solid fraction C0 [-]", 0.05, 0.50, 0.15, 0.01)
    gel_solid_fraction = st.slider("Gel solid fraction C_gel [-]", 0.50, 0.999, 0.80, 0.01)

    st.header("Dimensionless Inputs")
    surface_tension = st.slider("Surface tension gamma [N/m]", 0.010, 0.080, 0.030, 0.005)
    mean_free_path_nm = st.slider("Mean free path lambda [nm]", 0.1, 10.0, 1.0, 0.1)
    mass_diffusivity = st.number_input("Mass diffusivity D [m2/s]", value=1.0e-10, format="%.1e")


omega = rpm_to_rad_per_s(rpm)
evaporation_um_s = effective_evaporation_rate_um_s(
    rpm,
    evaporation_mode,
    evaporation_direct_um_s,
    evaporation_ref_um_s,
)
num_time_points = 320

tab_sim, tab_validation, tab_map = st.tabs(["Core Simulator", "Validation", "Feasibility Map"])

with tab_sim:
    time_no_evap, h_no_evap, time_evap, h_evap, transition_time, flow_rate = solve_uniform_model(
        rpm, rho, eta0, alpha, h0_um, evaporation_um_s, t_end, num_time_points
    )
    gel_time_array, solid_height, liquid_height, concentration, gel_time = solve_layer_gelation_model(
        rpm=rpm,
        rho=rho,
        eta0=eta0,
        alpha=alpha,
        h0_um=h0_um,
        evaporation_um_s=evaporation_um_s,
        initial_solid_fraction=initial_solid_fraction,
        gel_solid_fraction=gel_solid_fraction,
        t_end=t_end,
        num_time_points=num_time_points,
    )
    r_array, time_radial, h_initial, h_radial = radial_solver(
        rpm=rpm,
        rho=rho,
        eta0=eta0,
        alpha=alpha,
        evaporation_um_s=evaporation_um_s,
        wafer_radius_mm=wafer_radius_mm,
        base_thickness_um=base_thickness_um,
        perturbation_um=perturbation_um,
        beta=beta,
        profile_type=profile_type,
        fdm_scheme=fdm_scheme,
        t_end=t_end,
        num_radial_nodes=num_radial_nodes,
        num_time_points=240,
    )

    final_profile = h_radial[:, -1]
    uniformity, h_max, h_min, h_mean = compute_uniformity(final_profile)
    spec = 2.0

    cols = st.columns(6)
    cols[0].metric("Angular velocity", f"{omega:.1f} rad/s")
    cols[1].metric("Effective E", f"{evaporation_um_s:.3f} um/s")
    cols[2].metric("Final mean thickness", f"{h_mean * 1e6:.2f} um")
    cols[3].metric("Uniformity U", f"{uniformity:.2f}%")
    cols[4].metric("Spec", "PASS" if uniformity <= spec else "FAIL")
    cols[5].metric("t_gel", "None" if gel_time is None else f"{gel_time:.1f} s")

    left, right = st.columns(2)
    with left:
        st.pyplot(draw_uniform_thickness_plot(time_no_evap, h_no_evap, time_evap, h_evap, transition_time))
        st.pyplot(draw_radial_profiles(r_array, time_radial, h_radial))
    with right:
        st.pyplot(draw_viscosity_plot(time_evap, eta0, alpha))
        st.pyplot(draw_final_profile(r_array, final_profile))
        st.pyplot(draw_concentration_plot(gel_time_array, concentration, gel_solid_fraction))

    with st.expander("Dimensionless numbers and model assumptions", expanded=False):
        dimensionless = compute_dimensionless_numbers(
            rpm=rpm,
            rho=rho,
            eta0=eta0,
            characteristic_thickness_um=base_thickness_um,
            wafer_radius_mm=wafer_radius_mm,
            surface_tension=surface_tension,
            mean_free_path_nm=mean_free_path_nm,
            mass_diffusivity=mass_diffusivity,
        )
        dcols = st.columns(6)
        dcols[0].metric("epsilon = H/R", f"{dimensionless['epsilon']:.2e}")
        dcols[1].metric("U0", f"{dimensionless['velocity']:.2e} m/s")
        dcols[2].metric("Re", f"{dimensionless['reynolds']:.2e}")
        dcols[3].metric("Re*", f"{dimensionless['reduced_reynolds']:.2e}")
        dcols[4].metric("Ca", f"{dimensionless['capillary']:.2e}")
        dcols[5].metric("Kn", f"{dimensionless['knudsen']:.2e}")
        st.metric("Sc", f"{dimensionless['schmidt']:.2e}")
        st.write(
            "epsilon supports the thin-film approximation; Re* is used as the reduced inertia check; "
            "Ca is a bulk capillary indicator; Kn supports the continuum/no-slip assumption; "
            "Sc indicates the relative time scales of momentum and mass diffusion."
        )

    st.subheader("Time-Resolved Radial Profile")
    selected_time = st.slider(
        "Select time for radial profile [s]",
        min_value=0.0,
        max_value=float(t_end),
        value=float(t_end),
        step=max(float(t_end) / 120.0, 0.1),
    )
    selected_index = int(np.argmin(np.abs(time_radial - selected_time)))
    selected_profile = h_radial[:, selected_index]
    selected_uniformity, selected_h_max, selected_h_min, selected_h_mean = compute_uniformity(selected_profile)
    radial_y_limit_um = float(np.max(h_radial) * 1e6 * 1.05)

    selected_cols = st.columns(4)
    selected_cols[0].metric("Selected time", f"{time_radial[selected_index]:.1f} s")
    selected_cols[1].metric("Mean thickness", f"{selected_h_mean * 1e6:.2f} um")
    selected_cols[2].metric("Uniformity U", f"{selected_uniformity:.2f}%")
    selected_cols[3].metric("Thickness range", f"{selected_h_min * 1e6:.2f}-{selected_h_max * 1e6:.2f} um")

    st.pyplot(
        draw_time_resolved_profile(
            r_array,
            selected_profile,
            time_radial[selected_index],
            selected_uniformity,
            radial_y_limit_um,
        )
    )

    st.write(
        f"Final thickness range: {h_min * 1e6:.2f} to {h_max * 1e6:.2f} um; "
        f"mean = {h_mean * 1e6:.2f} um. "
        f"Flow-to-evaporation transition = {'not detected' if transition_time is None else f'{transition_time:.2f} s'}."
    )

with tab_validation:
    t_val, h_num, h_ana, rel_error, max_error = validate_against_analytical_ebp(
        rpm=rpm,
        rho=rho,
        eta0=eta0,
        h0_um=h0_um,
        t_end=t_end,
    )

    st.metric("Maximum relative error", f"{max_error:.3e}")

    fig, ax = plt.subplots()
    ax.plot(t_val, h_num * 1e6, label="Numerical", color="navy")
    ax.plot(t_val, h_ana * 1e6, "--", label="Analytical EBP", color="crimson")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Film thickness h [um]")
    ax.set_title("Analytical EBP Validation")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.semilogy(t_val, rel_error, color="black")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative error [-]")
    ax.set_title("Relative Error")
    fig.tight_layout()
    st.pyplot(fig)

with tab_map:
    sweep_cols = st.columns(4)
    rpm_min = sweep_cols[0].number_input("Min rpm", value=1000.0, step=250.0)
    rpm_max = sweep_cols[1].number_input("Max rpm", value=6000.0, step=250.0)
    eta_min = sweep_cols[2].number_input("Min eta0 [Pa s]", value=0.03, step=0.01)
    eta_max = sweep_cols[3].number_input("Max eta0 [Pa s]", value=0.30, step=0.01)

    grid_cols = st.columns(2)
    n_rpm = grid_cols[0].slider("Number of rpm samples", 5, 18, 10)
    n_eta = grid_cols[1].slider("Number of eta0 samples", 5, 18, 8)

    if st.button("Run feasibility sweep", type="primary"):
        rpm_values, eta_values, uniformity_map = parameter_sweep(
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            eta_min=eta_min,
            eta_max=eta_max,
            n_rpm=n_rpm,
            n_eta=n_eta,
            rho=rho,
            alpha=alpha,
            wafer_radius_mm=wafer_radius_mm,
            base_thickness_um=base_thickness_um,
            perturbation_um=perturbation_um,
            beta=beta,
            profile_type=profile_type,
            fdm_scheme=fdm_scheme,
            evaporation_mode=evaporation_mode,
            evaporation_direct_um_s=evaporation_direct_um_s,
            evaporation_ref_um_s=evaporation_ref_um_s,
            t_end=t_end,
        )

        rpm_grid, eta_grid = np.meshgrid(rpm_values, eta_values)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        filled = ax.contourf(rpm_grid, eta_grid, uniformity_map, levels=20, cmap="viridis")
        fig.colorbar(filled, ax=ax, label="Final uniformity U [%]")
        try:
            contour = ax.contour(rpm_grid, eta_grid, uniformity_map, levels=[2.0], colors="red", linewidths=2)
            ax.clabel(contour, fmt={2.0: "U = 2%"}, colors="red")
        except ValueError:
            pass
        ax.set_xlabel("Spin speed [rpm]")
        ax.set_ylabel("Initial viscosity eta0 [Pa s]")
        ax.set_title("Process Feasibility Map")
        fig.tight_layout()
        st.pyplot(fig)

        feasible = uniformity_map <= 2.0
        st.metric("Feasible cases", f"{int(np.sum(feasible))} / {feasible.size}")
