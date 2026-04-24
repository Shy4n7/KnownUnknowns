import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def call_api(endpoint: str, features: dict) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/{endpoint}",
            json={"features": features},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("API is offline. Start it with: `uvicorn app.main:app --reload`")
    except requests.exceptions.HTTPError as e:
        st.error(f"Something went wrong ({e.response.status_code})")
    return None


def condition_label(score: int) -> str:
    labels = {
        1: "Very Poor", 2: "Poor", 3: "Below Average", 4: "Fair",
        5: "Average", 6: "Above Average", 7: "Good",
        8: "Very Good", 9: "Excellent", 10: "Outstanding",
    }
    return labels.get(score, str(score))


def render_point(data: dict):
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding: 36px 24px 28px;
            background: #fafafa;
            border-radius: 16px;
            border: 1px solid #ebebeb;
            margin-top: 8px;
        ">
            <div style="font-size:0.78em; color:#6366f1; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px; font-weight:600;">
                Estimated Sale Price
            </div>
            <div style="font-size:3.4em; font-weight:800; color:#6366f1; line-height:1; letter-spacing:-1px;">
                {data["prediction_usd"]}
            </div>
            <div style="font-size:0.85em; color:#bbb; margin-top:12px;">
                Best single estimate · no guarantee attached
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uncertainty(data: dict):
    pred  = data["prediction"]
    width = data["interval_width"]
    conf  = data["confidence_level"] * 100

    pct_width = (width / pred) * 100 if pred > 0 else 100
    if pct_width < 25:
        color = "#22c55e"
        badge = "High confidence"
        desc  = "The model has seen many homes like this one."
    elif pct_width < 50:
        color = "#f59e0b"
        badge = "Moderate confidence"
        desc  = "Some uncertainty — homes like this vary quite a bit."
    else:
        color = "#ef4444"
        badge = "Lower confidence"
        desc  = "Unusual combination — the real price could differ significantly."

    # ── Price headline ────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center; padding:28px 0 20px;">
            <div style="font-size:0.78em; color:#6366f1; letter-spacing:0.1em; font-weight:600;
                        text-transform:uppercase; margin-bottom:8px;">
                Estimated Sale Price
            </div>
            <div style="font-size:3.4em; font-weight:800; color:#6366f1;
                        line-height:1; letter-spacing:-1px;">
                {data["prediction_usd"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Range row — Streamlit columns so layout is guaranteed ─────────
    col_lo, col_mid, col_hi = st.columns([1, 2, 1])

    with col_lo:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:4px;">
                <div style="font-size:0.72em; color:#aaa; margin-bottom:4px;">Low end</div>
                <div style="font-size:1em; font-weight:700; color:#6366f1;">
                    {data["lower_bound_usd"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid:
        # Simple gradient bar — no absolute positioning, works everywhere
        st.markdown(
            f"""
            <div style="padding:10px 0 6px;">
                <div style="
                    height: 10px;
                    border-radius: 99px;
                    background: linear-gradient(to right, #e5e7eb, {color}55, #e5e7eb);
                    margin-bottom: 6px;
                "></div>
                <div style="
                    width: 16px; height: 16px;
                    background: {color};
                    border-radius: 50%;
                    border: 3px solid white;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                    margin: 0 auto;
                "></div>
                <div style="text-align:center; font-size:0.72em; color:#bbb; margin-top:8px;">
                    &plusmn;${data["margin"]:,.0f} &nbsp;&middot;&nbsp; {int(conf)}% interval
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_hi:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:4px;">
                <div style="font-size:0.72em; color:#aaa; margin-bottom:4px;">High end</div>
                <div style="font-size:1em; font-weight:700; color:#6366f1;">
                    {data["upper_bound_usd"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Confidence callout ────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: #fafafa;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 18px;
            line-height: 1.6;
        ">
            <div style="font-weight:700; color:#222; margin-bottom:4px;">{badge}</div>
            <div style="font-size:0.88em; color:#666;">
                {desc}<br>
                95 out of 100 similar homes sold somewhere between
                <b>{data["lower_bound_usd"]}</b> and <b>{data["upper_bound_usd"]}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="House Price Estimator", layout="centered")

    # ── Global style tweaks ───────────────────────────────────────────
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; max-width: 720px; }
            div[data-testid="stButton"] button {
                border-radius: 10px;
                font-weight: 600;
                font-size: 0.95em;
                padding: 0.55em 1em;
                transition: opacity 0.15s;
            }
            div[data-testid="stButton"] button:hover { opacity: 0.85; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom: 6px;">
            <div style="font-size:1.75em; font-weight:800; color:#111; letter-spacing:-0.5px;">
                House Price Estimator
            </div>
            <div style="color:#999; font-size:0.92em; margin-top:4px;">
                Describe the house. Get an estimate — and an honest range.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        sqft = st.number_input(
            "Living Area (sq ft)",
            min_value=300, max_value=6000, value=1500, step=50,
            help="Total liveable floor space — not counting garage or unfinished basement.",
        )
        bedrooms = st.select_slider(
            "Bedrooms",
            options=[1, 2, 3, 4, 5, 6],
            value=3,
        )
        bathrooms = st.select_slider(
            "Full Bathrooms",
            options=[1, 2, 3, 4],
            value=2,
        )

    with col2:
        year_built = st.number_input(
            "Year Built",
            min_value=1872, max_value=2010, value=1990, step=1,
        )
        quality = st.select_slider(
            "Overall Condition",
            options=list(range(1, 11)),
            value=5,
            format_func=lambda x: f"{x}  —  {condition_label(x)}",
            help="Rate the home's overall material and finish quality on a scale of 1–10.",
        )

    features = {
        "GrLivArea":    float(sqft),
        "BedroomAbvGr": float(bedrooms),
        "FullBath":     float(bathrooms),
        "OverallQual":  float(quality),
        "YearBuilt":    float(year_built),
    }

    # ── Buttons ───────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="medium")

    result = None
    mode   = None

    with col_a:
        if st.button("Estimate", use_container_width=True, type="primary"):
            with st.spinner(""):
                result = call_api("predict", features)
                mode   = "point"

    with col_b:
        if st.button("Estimate + Confidence Interval", use_container_width=True):
            with st.spinner(""):
                result = call_api("predict_with_uncertainty", features)
                mode   = "uncertainty"

    # ── Result ────────────────────────────────────────────────────────
    if result:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if mode == "point":
            render_point(result)
        else:
            render_uncertainty(result)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    st.caption("Trained on the Ames Housing dataset (Iowa, 2006–2010) · Educational use only.")


if __name__ == "__main__":
    main()
