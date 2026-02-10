import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt
from scipy.special import erf
import io
import pandas as pd
from scipy.stats import linregress
import numpy as np



COTTRELL_URL_TAILSCALE = "https://ebt-tower-pc-1.tailbd8bdf.ts.net/cottrell"
COTTRELL_URL_LOCAL = "http://localhost:5000/cottrell"

# COTTRELL_URL = COTTRELL_URL_LOCAL
COTTRELL_URL = COTTRELL_URL_TAILSCALE

PING_URL_TAILSCALE = "https://ebt-tower-pc-1.tailbd8bdf.ts.net/ping"
PING_URL_LOCAL = "http://localhost:5000/ping"

# PING_URL = PING_URL_LOCAL
PING_URL = PING_URL_TAILSCALE

# Title
st.title("Chronoamperometry")
st.markdown("### Reversible Electron Transfer")

# ------------------------- SIDEBAR (inputs after status check) -------------------------

with st.sidebar:

# ------------------------- BACKEND STATUS (TOP OF APP) -------------------------
    st.sidebar.subheader("Backend Status")

    #run_button = st.button("Compute")
        
    if st.sidebar.button("Check backend status"):
        start = time.time()
        try:
            r = requests.get(PING_URL, timeout=2.0)
            latency_ms = (time.time() - start) * 1000
            if r.status_code == 200:
                st.sidebar.success(f"Online — {latency_ms:.1f} ms")
            else:
                st.sidebar.error(f"Status {r.status_code}")
        except Exception:
            st.sidebar.error("Offline or unreachable")


    st.header("Parameters")

    n = st.number_input("n (electrons)", value=1.0)
    A_cm2 = st.number_input("Area (cm²)", value=0.01)
    C_mM = st.number_input("Concentration (mM)", value=1.0)
    D_cm2_s = st.number_input("Diffusion coefficient (cm²/s)", value=1e-5, format="%.1e")
    t_s_final = st.number_input("Final time (s)", value=1.0)
    deltaE_mV = st.number_input("Potential step ΔE (mV)", value=100.0)
    Cdl_uF_cm2 = st.number_input("Double-layer capacitance (µF/cm²)", value=20.0)
    Rs_ohm = st.number_input("Solution resistance (Ω)", value=50.0)

    st.sidebar.subheader("Run Simulation")
    run_button = st.button("Compute")

# ------------------------- BACKEND CALL (outside sidebar) -------------------------

if run_button:
    payload = {
        "n": n,
        "A_cm2": A_cm2,
        "C_mM": C_mM,
        "D_cm2_s": D_cm2_s,
        "t_s_final": t_s_final,
        "deltaE_mV": deltaE_mV,
        "Cdl_uF_cm2": Cdl_uF_cm2,
        "Rs_ohm": Rs_ohm
    }

    try:
        # Local
        response = requests.post(COTTRELL_URL, json=payload)

        # Terminal should already be open and running the Flask app (cottrell_backend.py)
        # To run locally, need to enter a command in a new bash terminal after navigating to the folder that contains this file.
        # streamlit run cottrell_app.py --server.port 5000
        # streamlit run cottrell_app.py 5000

        # The app will ve viewable locally at http://localhost:5000

        # Webserver
        #BACKEND_URL = "https://ebt-tower-pc-1.tailbd8bdf.ts.net"
        #response = requests.post(f"{BACKEND_URL}/cottrell", json=payload)

        # To run on webserver, need to start the funnel
            # Check that Tailscale is running on the tower
            # In a terminal window, "tailscale funnel 5000"

        data = response.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        data = None
else:
    data = None

# ------------------------- ALWAYS DEFINE ARRAYS -------------------------

if data:
    t_s = np.array(data["t_s"])
    iF_sim_uA = np.array(data["iF_sim_uA"])
    inv_sqrt_t = 1 / np.sqrt(t_s)
else:
    t_s = np.array([])
    iF_sim_uA = np.array([])
    inv_sqrt_t = np.array([])

# ------------------------- TABS -------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cottrell Current Plot",
    "Linearized Plot",
    "Concentration Profiles",
    "Capacitive Current",
    "Current Contributions"
])

with tab1:
    st.latex(r"i(t) = \frac{n F A C \sqrt{D}}{\sqrt{\pi t}}")
    st.header("Cottrell Current Plot")

    # Prepare variable so plotting works even before upload
    user_exp = None

    # ------------------ PLOT FIRST ------------------

    import plotly.graph_objects as go
    fig1 = go.Figure()

    # Simulated line
    fig1.add_trace(go.Scatter(
        x=t_s,
        y=iF_sim_uA,
        mode="lines",
        name="Simulated",
        line=dict(width=4, color="blue")
    ))

    # Add experimental data later if uploaded
    # (user_exp will be filled after the plot section)

    fig1.update_layout(
        title="Cottrell Current vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Current (µA)",
        xaxis_type="log",
        height=450,
        legend=dict(x=0.02, y=0.98)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ------------------ DOWNLOAD SIMULATED DATA ------------------

    st.subheader("Download Simulated Cottrell Data")
    df_export = pd.DataFrame({
        "time_s": t_s,
        "current_uA": iF_sim_uA
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, sheet_name="Data", index=False)

    st.download_button(
        label="Download Cottrell Simulated Data",
        data=buffer.getvalue(),
        file_name="cottrell_simulated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ------------------ UPLOAD EXPERIMENTAL DATA (NOW AT BOTTOM) ------------------

    st.subheader("Upload Experimental Data (Excel Only)")
    uploaded_file = st.file_uploader(
        "Upload your experimental chronoamperometry data",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            user_exp = pd.read_excel(uploaded_file)
            required_cols = {"time_s", "current_uA"}
            if not required_cols.issubset(user_exp.columns):
                st.error("Your file must contain: time_s and current_uA")
                user_exp = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            user_exp = None

    # If experimental data exists, add it to the plot
    if user_exp is not None:
        fig1.add_trace(go.Scatter(
            x=user_exp["time_s"],
            y=user_exp["current_uA"],
            mode="markers",
            name="Experimental",
            marker=dict(
                color="red",
                size=6,
                opacity=0.7,
                line=dict(width=1, color="white")
            )
        ))
        st.plotly_chart(fig1, use_container_width=True)


# ------------------------- TAB 2 -------------------------

with tab2:
    st.latex(r"i(t) = \left(\frac{n F A C \sqrt{D}}{\sqrt{\pi}}\right)\frac{1}{\sqrt{t}}")
    st.header("Linearized Cottrell Plot")

    # Prepare variable so plotting works even before upload
    user_linear = None

    # ------------------ PLOT FIRST ------------------

    import plotly.graph_objects as go
    from scipy.stats import linregress
    import numpy as np

    fig2 = go.Figure()

    # Simulated line (always shown)
    fig2.add_trace(go.Scatter(
        x=inv_sqrt_t,
        y=iF_sim_uA,
        mode="lines",
        name="Simulated",
        line=dict(width=3, color="blue")
    ))

    # Placeholder title (updated later if fit exists)
    title_text = "Linearized Cottrell: I vs 1/√t"

    fig2.update_layout(
        title=title_text,
        xaxis_title="1 / √t (s⁻¹ᐟ²)",
        yaxis_title="Current (µA)",
        height=450,
        legend=dict(x=0.02, y=0.98)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ DOWNLOAD SIMULATED DATA ------------------

    st.subheader("Download Linearized Cottrell Data")

    df_linear_export = pd.DataFrame({
        "inv_sqrt_t": inv_sqrt_t,
        "current_uA": iF_sim_uA
    })

    buffer2 = io.BytesIO()
    with pd.ExcelWriter(buffer2, engine="xlsxwriter") as writer:
        df_linear_export.to_excel(writer, sheet_name="data", index=False)

    st.download_button(
        label="Download Linearized Cottrell Data",
        data=buffer2.getvalue(),
        file_name="cottrell_linearized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ------------------ UPLOAD EXPERIMENTAL DATA (NOW AT BOTTOM) ------------------

    st.subheader("Upload Linearized Experimental Data (Excel Only)")
    uploaded_linear = st.file_uploader(
        "Upload your linearized chronoamperometry data",
        type=["xlsx"],
        key="linear_upload"
    )

    if uploaded_linear is not None:
        try:
            user_linear = pd.read_excel(uploaded_linear)
            required_cols = {"inv_sqrt_t", "current_uA"}
            if not required_cols.issubset(user_linear.columns):
                st.error("Your file must contain: inv_sqrt_t and current_uA")
                user_linear = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            user_linear = None

    # ------------------ IF DATA UPLOADED: ADD EXP + FIT + RE-PLOT ------------------

    if user_linear is not None:

        # Experimental markers
        fig2.add_trace(go.Scatter(
            x=user_linear["inv_sqrt_t"],
            y=user_linear["current_uA"],
            mode="markers",
            name="Experimental",
            marker=dict(
                color="red",
                size=7,
                opacity=0.8,
                line=dict(width=1, color="white")
            )
        ))

        # Linear regression fit
        x = user_linear["inv_sqrt_t"]
        y = user_linear["current_uA"]
        slope, intercept, r_value, _, _ = linregress(x, y)
        fit_y = slope * x + intercept

        fig2.add_trace(go.Scatter(
            x=x,
            y=fit_y,
            mode="lines",
            name=f"Linear Fit (R²={r_value**2:.3f})",
            line=dict(width=2, dash="dot", color="green")
        ))

        # Update title with fit equation
        fig2.update_layout(
            title=f"Linearized Cottrell: I vs 1/√t<br>"
                  f"Fit: I = {slope:.2f}·(1/√t) + {intercept:.2f}"
        )

        # Re-render updated plot
        st.plotly_chart(fig2, use_container_width=True)



# ------------------------- TAB 3 -------------------------

with tab3:
    st.markdown("### Analytical Concentration Profiles")
    st.latex(r"C(x,t) = C_0 \cdot \mathrm{erf}\left(\frac{x}{2\sqrt{Dt}}\right)")

    # Time points
    times = np.logspace(-6, 0, 100)

    # Two-column layout: LEFT = plot, RIGHT = controls
    col_plot, col_controls = st.columns([2, 1])

    # --- CONTROLS (RIGHT COLUMN) ---
with col_controls:
    st.write("")  # Adds vertical spacing
    st.write("")  # Add more if needed
    st.write("")  # Add more if needed
    st.write("")  # Add more if needed
    axis_mode = st.radio(
        "Axis behavior",
        ["Dynamic (recalculate each frame)", "Fixed (use max range)"],
        horizontal=False
    )
    st.write("")  # Optional spacing between controls
    frame_index = st.slider(
        "Frame index",
        0,
        len(times) - 1,
        0,
        format="%d"
    )


    # --- PLOT (LEFT COLUMN) ---
    with col_plot:
        t = times[frame_index]
        x_max = 6 * sqrt(D_cm2_s * t) if axis_mode.startswith("Dynamic") else 6 * sqrt(D_cm2_s * times[-1])
        x = np.linspace(0, x_max, 300)
        C = C_mM * erf(x / (2 * sqrt(D_cm2_s * t)))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x * 1e4, C, linewidth=2)
        ax.set_ylim(0, C_mM)
        ax.set_xlim(0, x_max * 1e4)
        ax.set_xlabel("Distance from Electrode (µm)")
        ax.set_ylabel("Concentration (mM)")
        ax.set_title(f"Frame {frame_index} — t = {t:.2e} s")

        st.pyplot(fig)





# ------------------------- TAB 4 -------------------------

with tab4:
    st.markdown("### Capacitive Current (RC Charging)")
    st.latex(r"i_C(t) = \frac{\Delta E}{R_s} \exp\left(-\frac{t}{R_s C_{dl} A}\right)")

    # Compute RC time constant
    Cdl_F = Cdl_uF_cm2 * 1e-6      # µF/cm² → F/cm²
    A = A_cm2
    Rs = Rs_ohm
    tau = Rs * Cdl_F * A           # seconds

    # Compute capacitive current (A → µA)
    if data is not None and len(t_s) > 0:
        iC_A = (deltaE_mV * 1e-3 / Rs) * np.exp(-t_s / tau)
        iC_uA = iC_A * 1e6
    else:
        iC_uA = np.array([])

    # Plot
    fig4, ax4 = plt.subplots()
    ax4.plot(t_s, iC_uA, label="Capacitive Current (RC)", linewidth=2)

    ax4.set_xscale("log")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Current (µA)")
    ax4.set_title("RC Charging Capacitive Current vs Time")
    ax4.legend()
    st.pyplot(fig4)

    # Export capacitive current
    st.subheader("Download Capacitive Current Data")

    df_cap_export = pd.DataFrame({
        "time_s": t_s,
        "current_uA": iC_uA
    })

    buffer4 = io.BytesIO()
    with pd.ExcelWriter(buffer4, engine="xlsxwriter") as writer:
        df_cap_export.to_excel(writer, sheet_name="data", index=False)

    st.download_button(
        label="Download Capacitive Current Data",
        data=buffer4.getvalue(),
        file_name="capacitive_current.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ------------------------- TAB 5 -------------------------

with tab5:
    st.markdown("### Total Current Overlay")
    st.latex(r"i_{\text{total}}(t) = i_F(t) + i_C(t)")

    # Compute total current
    if len(iF_sim_uA) > 0 and len(iC_uA) > 0:
        i_total_uA = iF_sim_uA + iC_uA
    else:
        i_total_uA = np.array([])

    # Toggle for time axis scale
    scale = st.radio(
        "Time axis scale",
        ["Logarithmic", "Linear"],
        horizontal=True
    )

    # Plot overlay only if time data is valid
    if len(t_s) > 0 and np.all(t_s > 0):

        import plotly.graph_objects as go

        fig5 = go.Figure()

        # Faradaic
        fig5.add_trace(go.Scatter(
            x=t_s,
            y=iF_sim_uA,
            mode="lines",
            name="Faradaic Current",
            line=dict(width=2)
        ))

        # Capacitive
        fig5.add_trace(go.Scatter(
            x=t_s,
            y=iC_uA,
            mode="lines",
            name="Capacitive Current (RC)",
            line=dict(width=2)
        ))

        # Total
        fig5.add_trace(go.Scatter(
            x=t_s,
            y=i_total_uA,
            mode="lines",
            name="Total Current",
            line=dict(width=2, dash="dash")
        ))

        # Axis scaling
        if scale == "Logarithmic":
            fig5.update_xaxes(type="log")
            title = "Faradaic, Capacitive, and Total Current (Log Time)"
        else:
            fig5.update_xaxes(type="linear")
            title = "Faradaic, Capacitive, and Total Current (Linear Time)"

        # Layout
        fig5.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Current (µA)",
            height=450,
            legend=dict(x=0.02, y=0.98)
        )

        # Render interactive plot
        st.plotly_chart(fig5, use_container_width=True)

    else:
        st.warning("No valid time data to plot. Run the simulation first.")

    # Export overlay data
    st.subheader("Download Total Current Data")

    df_total_export = pd.DataFrame({
        "time_s": t_s,
        "faradaic_current_uA": iF_sim_uA,
        "capacitive_current_uA": iC_uA,
        "total_current_uA": i_total_uA
    })

    buffer5 = io.BytesIO()
    with pd.ExcelWriter(buffer5, engine="xlsxwriter") as writer:
        df_total_export.to_excel(writer, sheet_name="data", index=False)

    st.download_button(
        label="Download Total Current Data",
        data=buffer5.getvalue(),
        file_name="total_current.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )








