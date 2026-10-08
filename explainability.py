import pandas as pd

def get_top_reasons(applicant_index, X_data, shap_values_array, top_n=3):
    """
    Given an applicant's row index, return their top N reasons 
    driving their risk prediction, based on SHAP values.
    
    Parameters:
    -----------
    applicant_index : int
        The row position of the applicant in X_data
    X_data : pandas.DataFrame
        The feature data containing applicant information
    shap_values_array : numpy.ndarray
        Pre-calculated SHAP values corresponding to X_data
    top_n : int
        Number of top reasons to return (default 3)
    
    Returns:
    --------
    pandas.DataFrame with columns: feature, value, shap_contribution
    """
    applicant_shap = shap_values_array[applicant_index]
    applicant_features = X_data.iloc[applicant_index]
    
    explanation_df = pd.DataFrame({
        'feature': X_data.columns,
        'value': applicant_features.values,
        'shap_contribution': applicant_shap
    })
    
    explanation_df['abs_impact'] = explanation_df['shap_contribution'].abs()
    explanation_df = explanation_df.sort_values('abs_impact', ascending=False)
    
    top_reasons = explanation_df.head(top_n)[['feature', 'value', 'shap_contribution']]
    return top_reasons