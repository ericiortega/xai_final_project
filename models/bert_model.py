#AI Assistance Disclosure:
#ChatGPT was used to help refine the structure of the `predict_proba` function and formatting of other sections. 
#All code was reviewed and adjusted by us.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# model we are using 
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_names = ["negative", "positive"]


def predict_proba(texts):
    """
    Predict sentiment probabilities for a list of texts

    Inputs we can accept:
        - single string
        - list/tuple of strings
        - numpy array of strings (SHAP case)

    Returns: np.ndarray of shape (n_samples, 2)
             columns = [P(negative), P(positive)]
    """
    # normalize input
    if isinstance(texts, str):
        texts = [texts]
    else:
        # for lists, tuples, etc 
        try:
            #  will work for numpy arrays
            texts = [str(t) for t in texts]
        except TypeError:
            # fallback will be to treat as single item
            texts = [str(texts)]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

    return probs



def predict_label(text, threshold=0.5):
    """
    Here we predict the label ('positive' or 'negative') for a single text
    """
    probs = predict_proba(text)[0]
    p_pos = probs[1]
    label = "positive" if p_pos >= threshold else "negative"
    return label, float(p_pos)


def positivity_score(text):
    """
    Here we will convert positive probability into a 0–100 positivity score
    Also returns the raw positive probability
    """
    _, p_pos = predict_label(text)
    score = int(round(p_pos * 100))
    return score, p_pos
