"""
Complete News Summary Workflow Using LangGraph, Google Docs/Sheets, and Gmail,
with Dynamic Configuration from a Google Sheet and a "Run Report" toggle.

This script:
  - Loads configuration rows from a Google Sheet.
  - For each row with "Run Report" set to "Yes", it retrieves article text from one or more Google Docs,
    splits the text into chunks, and uses a LangGraph workflow to summarize each chunk via an LLM.
  - Uses LangChain's OutputFixingParser (with a Pydantic model) to robustly parse JSON output.
  - Writes the combined summary to a destination Google Doc.
  - Sends an email with the summary via the Gmail API.

All dynamic parameters (including the API key, document IDs, prompts, etc.) are stored in the Google Sheet.
The local .env file only contains the Google Sheet ID.
  
Author: Your Name
Date: YYYY-MM-DD
"""

import os
import json
import base64
import time
from email.mime.text import MIMEText

# -------------------------------
# Google API Setup
# -------------------------------
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build

def get_google_creds():
    # Define the scopes for Sheets, Drive, Docs, and Gmail.
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/gmail.send'
    ]
    # Load credentials from the service account file in the config folder.
    creds = ServiceAccountCredentials.from_json_keyfile_name('config/service_account.json', scope)
    return creds

def load_config_from_sheet(sheet_id, creds):
    """
    Loads all configuration rows from the Google Sheet.
    Assumes the sheet has a header row with columns such as:
      Report Name, Run Report, API_KEY, SOURCE_DOC_IDS, DEST_DOC_ID,
      EMAIL_RECIPIENTS, SYSTEM_PROMPT, USER_PROMPT, LLM_MODEL, RUN_DAY, RUN_TIME
    Returns a list of dictionaries, one per configuration row.
    """
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    records = sheet.get_all_records()
    return records

# -------------------------------
# Google Docs and Gmail Functions
# -------------------------------
def get_doc_text(doc_id, creds):
    service = build('docs', 'v1', credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    text = ""
    for element in doc.get('body', {}).get('content', []):
        if 'paragraph' in element:
            for run in element['paragraph'].get('elements', []):
                if 'textRun' in run:
                    text += run['textRun'].get('content', '')
    return text

def write_summary_to_doc(doc_id, summary, creds):
    service = build('docs', 'v1', credentials=cres)
    requests_body = [{
        'insertText': {
            'location': {'index': 1},
            'text': summary
        }
    }]
    result = service.documents().batchUpdate(documentId=doc_id, body={'requests': requests_body}).execute()
    return result

def create_email_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

def send_email(sender, recipients, subject, message_text, creds):
    service = build('gmail', 'v1', credentials=creds)
    for recipient in recipients.split(","):
        message = create_email_message(sender, recipient.strip(), subject, message_text)
        service.users().messages().send(userId="me", body=message).execute()
    return

# -------------------------------
# Text Processing Helpers
# -------------------------------
def chunk_text(text, max_length=3000):
    words = text.split()
    chunks = []
    current = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1  # Account for space
        current.append(word)
        if current_length >= max_length:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

# -------------------------------
# LangGraph Workflow Setup
# -------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import OpenAI

# Use pydantic and LangChain's output parser for robust JSON parsing.
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# Define a Pydantic model for the expected summary output.
class Summary(BaseModel):
    title: str
    summary: str

# Define a state schema using TypedDict.
from typing import TypedDict, List, Annotated

class State(TypedDict):
    messages: Annotated[List[dict], add_messages]

def summarize_chunk(state: State):
    """
    Summarizes a text chunk using OpenAI.
    It tries to parse the output as JSON; if that fails, uses OutputFixingParser
    (with a Pydantic model) to fix and parse the output.
    """
    chunk = state.get("chunk", "")
    prompt = (
        "You are an expert news summarizer. Summarize the following text into a JSON object with keys "
        "\"title\" and \"summary\".\n\n"
        f"Text: {chunk}\n\nPlease output JSON only."
    )
    # Initialize the LLM instance using the API key from environment (set from the sheet).
    llm = OpenAI(api_key=os.getenv("API_KEY", ""), model=state.get("LLM_MODEL", "gpt4o-mini"))
    response = llm.invoke(prompt)
    try:
        summary_obj = json.loads(response)
    except json.JSONDecodeError:
        pydantic_parser = PydanticOutputParser(pydantic_object=Summary)
        fixing_parser = OutputFixingParser.from_llm(llm, parser=pydantic_parser)
        summary_obj = fixing_parser.parse(response)
        if hasattr(summary_obj, "dict"):
            summary_obj = summary_obj.dict()
    new_messages = state.get("messages", []) + [summary_obj]
    return {"summary": summary_obj, "messages": new_messages}

def build_langgraph_workflow():
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", summarize_chunk)
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", END)
    return graph_builder.compile()

def process_article(doc_text):
    summaries = []
    chunks = chunk_text(doc_text)
    graph = build_langgraph_workflow()
    for chunk in chunks:
        state = {"chunk": chunk, "messages": []}
        result = graph.invoke(state
::contentReference[oaicite:0]{index=0}
 
