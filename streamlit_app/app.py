import io
from typing import Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import pkpd_core


st.set_page_config(
    page_title="PK/PD Simulation and Parameter Estimation"
)

st.title("PK/PD Simulation and Parameter Estimation")

mode = st.sidebar.radio(
    "Select workflow",
    ["PK/PD Simulation", "PK Parameter Estimation"]
)


DEFAULT_RK_METHOD = pkpd_core.DEFAULT_RK_METHOD
DEFAULT_RTOL = pkpd_core.DEFAULT_RTOL
DEFAULT_ATOL = pkpd_core.DEFAULT_ATOL
DENSE_DT = pkpd_core.DENSE_DT


if mode == "PK/PD Simulation":

    st.header("PK/PD Simulation")

    st.subheader("PK parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        Dose = st.number_input(
            "Dose (mg)",
            value=100.0
        )

        F = st.number_input(
            "F (bioavailability)",
            value=0.6,
            min_value=0.0,
            max_value=1.0
        )

    with col2:
        ka = st.number_input(
            "ka (1/h)",
            value=1.0
        )

        V = st.number_input(
            "V (L)",
            value=20.0
        )

    with col3:
        CL = st.number_input(
            "CL (L/h)",
            value=1.0
        )


    ke_derived = CL / V if V != 0 else float("nan")

    st.write(
        f"Derived ke = {ke_derived:.6g} 1/h"
    )


    st.subheader("PD parameters")

    col4, col5 = st.columns(2)

    with col4:
        E0 = st.number_input(
            "E0",
            value=0.0
        )

        Emax = st.number_input(
            "Emax",
            value=100.0
        )

    with col5:
        EC50 = st.number_input(
            "EC50 (mg/L)",
            value=2.0
        )

        gamma = st.number_input(
            "gamma",
            value=1.5
        )


    st.subheader("Dosing and simulation")

    t_start = st.number_input(
        "t_start (h)",
        value=0.0
    )

    n_doses = st.number_input(
        "n_doses",
        value=1,
        min_value=1,
        step=1
    )


    dosing_interval: Optional[float] = None

    if n_doses > 1:

        dosing_interval = st.number_input(
            "dosing_interval (h)",
            value=12.0
        )


    t_end = st.number_input(
        "t_end (h)",
        value=48.0
    )


    if st.button("Run Simulation"):

        try:

            results = pkpd_core.simulate_pkpd(
                Dose=float(Dose),
                F=float(F),
                ka=float(ka),
                V=float(V),
                CL=float(CL),
                E0=float(E0),
                Emax=float(Emax),
                EC50=float(EC50),
                gamma=float(gamma),
                t_start=float(t_start),
                t_end=float(t_end),
                t_eval=None,
                ke_supplied=None,
                method=DEFAULT_RK_METHOD,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
                dosing_interval=(
                    float(dosing_interval)
                    if dosing_interval is not None
                    else None
                ),
                n_doses=int(n_doses),
            )

        except Exception as exc:

            st.error(
                f"Simulation failed: {exc}"
            )

        else:

            st.subheader("Simulation Results")

            st.write(
                f"ke = {results['ke']:.6g} 1/h"
            )

            fig = pkpd_core.plot_pkpd(
                results,
                title_suffix="Streamlit"
            )

            st.pyplot(fig)

            plt.close(fig)



elif mode == "PK Parameter Estimation":

    st.header("PK Parameter Estimation")

    st.write(
        "Upload CSV with columns: time, concentration"
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    times = None
    concs = None


    if uploaded_file is not None:

        try:

            content = uploaded_file.getvalue()

            text_io = io.StringIO(
                content.decode("utf-8")
            )

            times, concs = pkpd_core.load_pk_csv(
                text_io
            )

        except Exception as exc:

            st.error(
                f"CSV load failed: {exc}"
            )

        else:

            data = [
                {
                    "time": float(t),
                    "concentration": float(c)
                }
                for t, c in zip(times, concs)
            ]

            st.dataframe(data)


    st.subheader("Known regimen")

    Dose = st.number_input(
        "Dose for fitting (mg)",
        value=100.0
    )

    F = st.number_input(
        "F for fitting",
        value=0.6,
        min_value=0.0,
        max_value=1.0
    )


    st.subheader("Initial parameter guesses")

    ig_ka = st.number_input(
        "Initial ka",
        value=1.0
    )

    ig_V = st.number_input(
        "Initial V",
        value=20.0
    )

    ig_CL = st.number_input(
        "Initial CL",
        value=1.0
    )


    if st.button("Run Parameter Estimation"):

        if times is None:

            st.error(
                "Upload a CSV file first."
            )

        else:

            try:

                fit = pkpd_core.estimate_pk_parameters(
                    observed_times=times,
                    observed_conc=concs,
                    Dose=float(Dose),
                    F=float(F),
                    n_doses=1,
                    dosing_interval=None,
                    initial_guess=(
                        float(ig_ka),
                        float(ig_V),
                        float(ig_CL)
                    ),
                )

            except Exception as exc:

                st.error(
                    f"Parameter estimation failed: {exc}"
                )

            else:

                st.subheader(
                    "Estimation Results"
                )

                st.write(
                    f"ka = {fit.ka:.6g} 1/h"
                )

                st.write(
                    f"V = {fit.V:.6g} L"
                )

                st.write(
                    f"CL = {fit.CL:.6g} L/h"
                )

                st.write(
                    f"ke = {fit.ke:.6g} 1/h"
                )

                st.write(
                    f"RMSE = {fit.rmse:.6g}"
                )
