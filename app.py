import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Explainer", layout="centered")

# --- Sidebar: project context (refinement #2) ---
with st.sidebar:
    st.header("About This Project")
    st.write(
        "This tool predicts loan default risk using a calibrated XGBoost model, "
        "explains each decision using SHAP, and generates a plain-English, "
        "policy-grounded explanation using a RAG pipeline (ChromaDB + LLM)."
    )
    st.write("Built as an end-to-end credit risk + explainability + fairness project.")
    st.divider()
    st.caption("Applicant pool: indices 0–299")

st.title("Credit Risk Decision Explainer")
st.write("Enter an applicant index to see their loan decision and a plain-English explanation.")

API_URL = "http://127.0.0.1:8000"
MAX_APPLICANT_INDEX = 299

applicant_index = st.number_input(
    "Applicant Index", min_value=0, max_value=MAX_APPLICANT_INDEX, value=0, step=1
)

# --- refinement #6: check API health before allowing the main action ---
api_is_up = True
try:
    health = requests.get(API_URL, timeout=2)
    health.raise_for_status()
except Exception:
    api_is_up = False

if not api_is_up:
    st.warning(
        "⚠️ Cannot reach the backend API. Make sure it's running with:\n\n"
        "`uvicorn api:app --reload`"
    )

if st.button("Get Explanation", disabled=not api_is_up):
    # --- refinement #4: staged, informative loading messages ---
    status_placeholder = st.empty()
    try:
        status_placeholder.info("Running model prediction and SHAP analysis...")
        status_placeholder.info("Retrieving relevant policy documents...")
        status_placeholder.info("Generating explanation with LLM...")

        response = requests.get(f"{API_URL}/applicant/{applicant_index}", timeout=30)
        response.raise_for_status()
        result = response.json()
        status_placeholder.empty()

        decision = result["decision"]
        probability = result["probability"]

        if decision == "APPROVED":
            st.success(f"Decision: {decision}")
        else:
            st.error(f"Decision: {decision}")

        st.metric("Predicted Default Risk", f"{probability:.1%}")

        st.subheader("Top Risk Factors")
        for reason in result["top_reasons"]:
            direction = reason["direction"]
            icon = "🔺" if direction == "INCREASES risk" else "🔻"
            st.write(
                f"{icon} **{reason['feature']}**: {reason['decoded_value']} "
                f"({direction}, impact {reason['shap_contribution']:.3f})"
            )

        # --- refinement #3: SHAP contribution bar chart ---
        st.subheader("Risk Factor Impact (Visual)")
        chart_data = {
            reason["feature"]: reason["shap_contribution"]
            for reason in result["top_reasons"]
        }
        st.bar_chart(chart_data)

        st.subheader("Explanation")
        st.write(result["explanation"])

        st.caption(f"Sources consulted: {', '.join(result['sources_used'])}")

                # --- Fairness check ---
        st.subheader("Fairness Context")
        fairness_response = requests.get(f"{API_URL}/fairness-check/{applicant_index}", timeout=10)
        fairness_response.raise_for_status()
        fairness_result = fairness_response.json()
        st.info(fairness_result["note"])

        # --- refinement #5: show retrieved policy text for trust/verification ---
        with st.expander("🔍 View retrieved policy context (for verification)"):
            st.text(result["retrieved_context"])

    except requests.exceptions.ConnectionError:
        status_placeholder.empty()
        st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        status_placeholder.empty()
        st.error(f"API returned an error: {e}")
    except Exception as e:
        status_placeholder.empty()
        st.error(f"An unexpected error occurred: {e}")