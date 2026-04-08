import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="System Health Monitoring",
    page_icon="🖥️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .title-text {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.3rem;
        }
        .subtitle-text {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .section-card {
            background-color: white;
            padding: 1.2rem 1.2rem 0.8rem 1.2rem;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
            border: 1px solid #e2e8f0;
        }
        .result-card {
            background-color: white;
            padding: 1.4rem;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            border: 1px solid #e2e8f0;
        }
        .status-healthy {
            background-color: #dcfce7;
            color: #166534;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.95rem;
        }
        .status-warning {
            background-color: #fef3c7;
            color: #92400e;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.95rem;
        }
        .status-critical {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.95rem;
        }
        .small-label {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
        }
        .footer-note {
            color: #64748b;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-text">AI-Based System Health Monitoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">A lightweight predictive maintenance dashboard for monitoring '
    'system health and failure risk using machine learning.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("System Resource Metrics")

    col1, col2 = st.columns(2)
    with col1:
        cpu_usage = st.slider("CPU Usage (%)", 0, 100, 50)
        memory_usage = st.slider("Memory Usage (%)", 0, 100, 50)
        disk_usage = st.slider("Disk Usage (%)", 0, 100, 50)

    with col2:
        temperature = st.slider("Temperature (°C)", 0, 150, 60)
        voltage = st.number_input("Voltage (V)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        fan_speed = st.number_input("Fan Speed (RPM)", min_value=0, max_value=10000, value=2000, step=100)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Operational Metrics")

    col3, col4, col5 = st.columns(3)
    with col3:
        network_traffic = st.number_input("Network Traffic (MB/s)", min_value=0.0, value=50.0, step=1.0)
    with col4:
        error_count = st.number_input("Error Count", min_value=0, value=1, step=1)
    with col5:
        response_time = st.number_input("Response Time (ms)", min_value=0.0, value=200.0, step=10.0)

    st.markdown('</div>', unsafe_allow_html=True)

    payload = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "temperature": temperature,
        "voltage": voltage,
        "disk_usage": disk_usage,
        "fan_speed": fan_speed,
        "network_traffic": network_traffic,
        "error_count": error_count,
        "response_time": response_time,
    }

    predict_clicked = st.button("Predict System Health", use_container_width=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input Snapshot")

    snap1, snap2 = st.columns(2)
    with snap1:
        st.metric("CPU Usage", f"{cpu_usage}%")
        st.metric("Memory Usage", f"{memory_usage}%")
        st.metric("Disk Usage", f"{disk_usage}%")
        st.metric("Temperature", f"{temperature} °C")
    with snap2:
        st.metric("Voltage", f"{voltage:.1f} V")
        st.metric("Fan Speed", f"{fan_speed} RPM")
        st.metric("Network Traffic", f"{network_traffic:.1f} MB/s")
        st.metric("Response Time", f"{response_time:.1f} ms")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Usage Guidance")
    st.markdown(
        """
        - Use **low resource values** to simulate a healthy system  
        - Use **moderate stress values** to simulate warning state  
        - Use **high CPU, memory, temperature, errors, and latency** to simulate critical risk  
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)


def get_status_class(label: str) -> str:
    label = label.lower()
    if label == "healthy":
        return "status-healthy"
    if label == "warning":
        return "status-warning"
    return "status-critical"


if predict_clicked:
    try:
        with st.spinner("Analyzing system health..."):
            response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            prediction_label = result["prediction_label"]
            failure_probability = float(result["failure_probability"])
            confidence = float(result.get("confidence", 0.0))
            class_probabilities = result["class_probabilities"]

            st.markdown("---")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Prediction Result")

            status_class = get_status_class(prediction_label)
            st.markdown(
                f'<div class="{status_class}">{prediction_label.upper()}</div>',
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            with m1:
                st.markdown('<div class="small-label">Failure Probability</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{failure_probability:.4f}</div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown('<div class="small-label">Model Confidence</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{confidence:.4f}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### Class Probability Breakdown")
            for label, prob in class_probabilities.items():
                st.write(f"**{label.capitalize()}**")
                st.progress(float(prob))
                st.caption(f"{prob:.4f}")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. Make sure the API is running on http://127.0.0.1:8000"
        )
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

st.markdown(
    '<div class="footer-note">Built for MLOps-based predictive system health monitoring demo</div>',
    unsafe_allow_html=True,
)