import json
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import shap

from decoding import decode_feature_value
from rag_pipeline import get_chroma_collection, get_groq_client, generate_explanation

app = FastAPI(title="Credit Risk Explanation API")

print("Loading model and data...")
X_test_sample = pd.read_csv("deploy_data/X_test_sample.csv")
calibrated_model = joblib.load("models/xgb_calibrated_model.pkl")

frozen_estimator = calibrated_model.calibrated_classifiers_[0].estimator
xgb_model_for_shap = frozen_estimator.estimator
explainer = shap.TreeExplainer(xgb_model_for_shap)
shap_values = explainer.shap_values(X_test_sample)

collection = get_chroma_collection()
groq_client = get_groq_client()

# Load the precomputed state mapping (JSON keys are text, so convert back to numbers)
with open("deploy_data/state_mapping.json") as f:
    state_mapping = {float(k): v for k, v in json.load(f).items()}

print("Startup complete!")

FAIRNESS_FINDINGS = {
    "NY": {"flag": "higher_scrutiny", "note": "New York applicants were flagged as high-risk more often in our fairness audit (selection rate 4.24% vs. overall average), with both higher true-positive and false-positive rates than most states."},
    "IL": {"flag": "lower_scrutiny", "note": "Illinois applicants were flagged as high-risk less often in our fairness audit (selection rate 2.92%, the lowest among major states), with both lower true-positive and false-positive rates."},
}

@app.get("/")
def health_check():
    return {"status": "API is running", "available_applicants": len(X_test_sample)}

@app.get("/applicant/{applicant_index}")
def get_applicant_explanation(applicant_index: int):
    if applicant_index < 0 or applicant_index >= len(X_test_sample):
        raise HTTPException(status_code=404, detail=f"Applicant index must be between 0 and {len(X_test_sample)-1}")

    result = generate_explanation(
        applicant_index=applicant_index,
        X_data=X_test_sample,
        shap_values_array=shap_values,
        model=calibrated_model,
        collection=collection,
        llm_client=groq_client,
        decode_feature_value_fn=decode_feature_value,
        state_mapping=state_mapping,
        top_n=3
    )

    result['top_reasons'] = result['top_reasons'].to_dict(orient='records')
    return result

@app.get("/fairness-check/{applicant_index}")
def get_fairness_check(applicant_index: int):
    if applicant_index < 0 or applicant_index >= len(X_test_sample):
        raise HTTPException(status_code=404, detail=f"Applicant index must be between 0 and {len(X_test_sample)-1}")

    applicant_state_freq = X_test_sample.iloc[applicant_index]['addr_state']
    decoded_state = decode_feature_value('addr_state', applicant_state_freq, state_mapping)

    if decoded_state in FAIRNESS_FINDINGS:
        finding_data = FAIRNESS_FINDINGS[decoded_state]
        return {
            "applicant_state": decoded_state,
            "has_group_level_finding": True,
            "flag": finding_data["flag"],
            "note": finding_data["note"]
        }
    else:
        return {
            "applicant_state": decoded_state,
            "has_group_level_finding": False,
            "note": (
                f"No specific fairness concern was identified for applicants from "
                f"{decoded_state} in our audit. Our broader fairness audit (see "
                f"fairness_report.md) found a disparate impact ratio of 0.688 between "
                f"New York and Illinois applicants specifically, below the regulatory "
                f"80% threshold."
            )
        }