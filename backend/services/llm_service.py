import os
import json
import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Dict, Any, Optional
import re
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def format_docs(results: List[Dict[str, Any]]) -> str:
    """
    Formats the documents retrieved from the vector database for LLM input."""
    grouped = defaultdict(list)
    for r in results:
        text = r.get("text") or ""
        url = r.get("url") or ""
        grouped[r.get("id")].append(f"{text}:::{url}:::FINISH")

    return "\n".join(
        f"{doc_id}\n" + "\n".join(pairs)
        for doc_id, pairs in grouped.items()
    )

def parse_llm_output(output_str: str) -> dict:
    """
    Parses the LLM output string to extract the JSON content.
    """
    if not output_str or not isinstance(output_str, str):
        return {"LLM_Response": "", "Cited_URLs": []}

    cleaned_output = re.sub(r"^```(?:json)?|```$", "", output_str.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned_output)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        return {
            "LLM_Response": parsed.get("LLM_Response", ""),
            "Cited_URLs": parsed.get("Cited_URLs", []),
        }
    except Exception:
        return {"LLM_Response": output_str.strip(), "Cited_URLs": []}


def format_conv_history(responses: List[str]) -> str:
    """
    Formats the conversation history for LLM input.
    """
    return "\n".join(responses)

def generate_llm_response(
    user_text: str,
    responses: List[str],
    results: List[Dict[str, Any]],
) -> str:
    """
    Generates a response from the LLM based on user input from frontend, conversation history from qdrant, and relevant documents from the RAG Pipeline.
    """
    model = OPENAI_CHAT_MODEL
    DOCS = format_docs(results)
    CONV_HISTORY_SUMMARY = format_conv_history(responses)
    print("-----")
    print(DOCS)
    print("-----")
    delimiter = "$$$$"
    system_instructions = f"""
        You are an assistant helping resolve support tickets.

        User Query:
        {user_text}

        Conversation History (summarized):
        {CONV_HISTORY_SUMMARY}
        If true(it is empty), treat the conversation as new.

        Relevant Documentation (DOCS):
        ```{DOCS}``` -> DOCS
        You will see the relevant documentation in the form : 
        {delimiter}
        text1:::url1:::FINISH, text1:::url2:::FINISH..etc
        {delimiter}
        In this docs, there will be many url_ids and many points. Refer to all the points and clearly state the answer. Cite all the URLs mentioned in DOCS.
        Do not simple judge on score basis. Consider every detail and then answer. 
        {delimiter}
        When a user asks for steps, refer to documentation and create steps and return back to the user. 
        NEVER say, "I dont find this in documentation" : Looks weak
        You will never give the user inaccurate data, remember that
        {delimiter}
        {delimiter}
        Classify the incoming ticket as follows : 

        {delimiter}If the topic is How-to, Product, Best practices, API/SDK, or SSO, use DOCS to generate and display a direct answer.{delimiter}
        If the topic is anything else, respond with a simple message indicating classification and routing. 
        {delimiter}
        For example:
        “This ticket has been classified as a 'Connector' issue and routed to the appropriate team.”
        "We've categorized this as a Connector issue and passed it to the right team who can help you further."
        "This has been identified as a Connector issue and is now with our specialists for resolution."
        "Looks like this is a Connector issue. I've routed it to the right experts so they can assist you."
        "Your request falls under a Connector issue, and I've shared it with the right team to handle it quickly."
        "This ticket has been flagged as a Connector issue and forwarded to our support experts for action."
        {delimiter}
        {delimiter}

        Instructions:
        - Draft a clear and accurate response to the user's query.
        - Cite only from the URLs explicitly present in the provided DOCS.
        - Do not ask follow-up questions.
        - If no relevant DOCS found, return with "Cited_URLs": [].
        - Output strictly in the following JSON format:

        [{{
            "LLM_Response": "<drafted response to the ticket>",
            "Cited_URLs": ["<url1> , <url2> , <url3>"]
        }}]
    """

    messages = []
    
    messages = [{"role": "system", "content": system_instructions}]

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0  # keep deterministic
        )
        print(resp.choices[0].message.content)
        # return resp.choices[0].message.content
        raw_output = resp.choices[0].message.content
        print("Raw LLM Output:", raw_output)
        return parse_llm_output(raw_output)
    
    except Exception as e:
        print("LLM call failed:", e)
        return json.dumps({
            "LLM_Response": "Error: LLM call failed.",
            "Cited_URLs": []
        })