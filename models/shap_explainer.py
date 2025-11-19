# models/shap_explainer.py

import shap
from models.bert_model import tokenizer, predict_proba

# creat a text masker for SHAP
masker = shap.maskers.Text(tokenizer)


def _model_positive_class(texts):
    """
    Wrapper that returns P(positive) only, for SHAP.
    SHAP expects shape (n_samples,).
    """
    probs = predict_proba(texts)
    # return positive class probability
    return probs[:, 1]


# building the explainer once so it is reused
explainer = shap.Explainer(_model_positive_class, masker)


def get_token_importances(text):
    """
    Compute SHAP values for a single text.

    Returns:
        tokens: list of token strings
        shap_values: list of float contributions (same length)
        base_value: model's expected value (float)
    """
    shap_values = explainer([text])  # list of one text
    explanation = shap_values[0]

    tokens = list(explanation.data)
    values = explanation.values.tolist()
    base_value = float(explanation.base_values)

    return tokens, values, base_value
