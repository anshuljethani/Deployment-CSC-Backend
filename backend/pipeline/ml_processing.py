import os
from transformers import pipeline

_pipelines = {}

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
        model_id = os.getenv("KEYWORDS_MODEL", "ilsilfverskiold/tech-keywords-extractor")
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
        model_id = os.getenv("TOPIC_MODEL", "facebook/bart-large-mnli")
        _pipelines["topic_pipe"] = pipeline("zero-shot-classification", model=model_id)
    
    pipe = _pipelines["topic_pipe"]

    TOPIC_LABELS = ["How-to", "Product", "Connector", "Lineage", "API/SDK", "SSO", "Glossary", "Best practices", "Sensitive data", "Integrations", "Errors", "Others"]

    result = pipe(text, candidate_labels=TOPIC_LABELS)
    return result['labels'][0]

def sentiment_analyser(text: str) -> str:
    """
    Analyzes sentiment from the given text using a zero-shot classification model."""
    
    if "sentiment_pipe" not in _pipelines:
        model_id = os.getenv("SENTIMENT_EXACT_MODEL", "facebook/bart-large-mnli")
        # Forcing CPU usage as per user request (device=-1)
        _pipelines["sentiment_pipe"] = pipeline("zero-shot-classification", model=model_id, device=-1)

    pipe = _pipelines["sentiment_pipe"]

    EXACT_LABELS = ["Confused", "Curious", "Anxious", "Hopeful", "Frustrated", "Urgent"]

    result = pipe(text, candidate_labels=EXACT_LABELS)
    return result['labels'][0]
