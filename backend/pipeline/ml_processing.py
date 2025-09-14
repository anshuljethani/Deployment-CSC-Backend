print("⚡ Loading Hugging Face pipelines at startup...")

_pipelines = {
    "priority_pipe": pipeline("zero-shot-classification",
                              model=os.getenv("PRIORITY_MODEL", "valhalla/distilbart-mnli-12-1")),
    "topic_pipe": pipeline("zero-shot-classification",
                           model=os.getenv("TOPIC_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v1")),
    "sentiment_pipe": pipeline("text-classification",
                               model=os.getenv("SENTIMENT_MODEL", "michellejieli/emotion_text_classifier")),
    "keywords_pipe": pipeline("text2text-generation",
                              model=os.getenv("KEYWORDS_MODEL", "ml6team/keyphrase-generation-t5-base-inspec")),
}

print("✅ All Hugging Face models loaded.")

def priority_calculation(text: str) -> str:
    """
    Calculates priority from the given text using a zero-shot classification model.
    """
    if "priority_pipe" not in _pipelines:
        model_id = os.getenv("PRIORITY_MODEL", "valhalla/distilbart-mnli-12-1")
        _pipelines["priority_pipe"] = pipeline("zero-shot-classification", model=model_id)

    pipe = _pipelines["priority_pipe"]
    
    LABELS = ["Urgent", "Medium Urgency", "Not Urgent"]
    PRIORITY_MAP = {"Urgent": "P0", "Medium Urgency": "P1", "Not Urgent": "P2"}

    result = pipe(text, candidate_labels=LABELS)
    priority_label = result['labels'][0]
    return PRIORITY_MAP.get(priority_label, "Unknown")
def keyword_calculation(text: str) -> str:
    """
    Calculates keywords from the given text using a text-to-text generation model.
    """
    if "keyword_pipe" not in _pipelines:
        model_id = os.getenv("KEYWORDS_MODEL", "ml6team/keyphrase-generation-t5-base-inspec")
        _pipelines["keyword_pipe"] = pipeline("text2text-generation", model=model_id, max_new_tokens=64)

    pipe = _pipelines["keyword_pipe"]
    
    result = pipe(text)
    keywords_str = result[0]['generated_text']

    keywords_list = [kw.strip() for kw in keywords_str.split(',')]
    return ', '.join(keywords_list)


def topic_calculation(text: str) -> str:
    """
    Calculates topic from the given text using a zero-shot classification model."""
    if "topic_pipe" not in _pipelines:
        model_id = os.getenv("TOPIC_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v1")
        _pipelines["topic_pipe"] = pipeline("zero-shot-classification", model=model_id)
    
    pipe = _pipelines["topic_pipe"]

    TOPIC_LABELS = ["How-to", "Product", "Connector", "Lineage", "API/SDK", "SSO", "Glossary", "Best practices", "Sensitive data", "Integrations", "Errors", "Others"]

    result = pipe(text, candidate_labels=TOPIC_LABELS)
    return result['labels'][0]

def sentiment_analyser(text: str) -> str:
    """
    Analyzes sentiment from the given text using michellejieli/emotion_text_classifier.
    Returns one of the EXACT_LABELS to keep output consistent.
    """

    if "sentiment_pipe" not in _pipelines:
        model_id = os.getenv("SENTIMENT_EXACT_MODEL", "michellejieli/emotion_text_classifier")
        # This model gives multi-class predictions → set top_k=None so we get all labels
        _pipelines["sentiment_pipe"] = pipeline("text-classification", model=model_id, return_all_scores=True, device=-1)

    pipe = _pipelines["sentiment_pipe"]

    EXACT_LABELS = ["Confused", "Curious", "Anxious", "Hopeful", "Frustrated", "Urgent"]

    # Run classification (returns list of list: [[{'label': 'joy', 'score': ...}, ...]])
    results = pipe(text)

    if not results or not isinstance(results, list):
        return "Confused"  # fallback

    # Take the top-scoring label
    predictions = results[0]  # inner list
    top_pred = max(predictions, key=lambda x: x["score"])
    model_label = top_pred["label"].lower()

    # Map model’s labels → your EXACT_LABELS
    label_map = {
        "anger": "Frustrated",
        "disgust": "Confused",
        "fear": "Anxious",
        "joy": "Hopeful",
        "neutral": "Curious",
        "sadness": "Urgent"
    }

    return label_map.get(model_label, "Confused")
