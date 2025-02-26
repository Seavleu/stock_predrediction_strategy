# Improves transparency for traders & stakeholders.

import shap

def explain_predictions(model, X_sample):
    """Generate SHAP explanations for model predictions."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    
    shap.summary_plot(shap_values, X_sample)
