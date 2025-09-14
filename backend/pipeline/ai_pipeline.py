import datetime
from .ml_processing import priority_calculation, keyword_calculation, topic_calculation, sentiment_analyser
from .db_connector import push_ticket_point

def ai_pipeline(json_input: list) -> list:
    """
    simple fxn to run the ai pipeline on a list of json objects
    each json object should have at least 'id', 'subject', and 'body' fields
    returns a list of results with status for each processed item
    1. calculates priority from body
    2. calculates keywords from subject
    3. calculates topic from subject + body
    4. calculates sentiment from subject + body
    5. pushes the results to the database
    """
    results = []
    for item in json_input:
        item_id = item.get("id", "no_id")
        subject = item.get("subject", "")
        body = item.get("body", "")
        
        subject_body_combined = f"{subject} {body}".strip()

        try:
            priority = priority_calculation(body)
            keywords = keyword_calculation(subject)
            topic = topic_calculation(subject_body_combined)
            sentiment = sentiment_analyser(subject_body_combined)
            sample_vector = [0.1] * 128
            push_ticket_point(
                ticket_id=item_id,
                subject=subject,
                body=body,
                priority=priority,
                topics=topic,
                keywords=keywords,
                sentiment=sentiment,
                created_at=datetime.datetime.now(),
                vector=sample_vector
            )
            
            results.append({"id": item_id, "status": "success"})

        except Exception as e:
            # Handle potential errors during ML model inference
            print(f"Error processing item with id {item_id}: {e}")
            # Append a failure result to the list
            results.append({"id": item_id, "error": str(e)})
    return results