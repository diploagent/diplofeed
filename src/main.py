"""
Complete News Summary Workflow Using LangGraph, Google Docs/Sheets, and Gmail,
with Dynamic Configuration from a Google Sheet.

This script:
  - Loads configuration from a Google Sheet (keys: API_KEY, SOURCE_DOC_IDS, DEST_DOC_ID,
    EMAIL_RECIPIENTS, REPORT_NAME, etc.).
  - Retrieves article text from one or more Google Docs.
  - Splits the text into chunks.
  - Uses a LangGraph workflow to send each chunk to an LLM (via OpenAI) for JSON‑formatted summarization.
    It employs LangChain's OutputFixingParser (with a Pydantic model) to handle JSON parsing errors.
  - Writes the combined summary to a destination Google Doc.
  - Sends an email with the summary via the Gmail API.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import json
import base64
import time
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables from .env (only non-sensitive values like GOOGLE_SHEET_ID)
load_dotenv()

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
    client = gspread.authorize(creds)
    # Open the sheet by its key.
    sheet = client.open_by_key(sheet_id).sheet1
    # Assumes your sheet has two columns: 'Key' and 'Value'
    records = sheet.get_all_records()
    # Create a dictionary mapping keys to values.
    config = {record['Key']: record['Value'] for record in records}
    return config

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
    service = build('docs', 'v1', credentials=creds)
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
    # 'messages' will hold our processing messages.
    messages: Annotated[List[dict], add_messages]

def summarize_chunk(state: State):
    """
    Node function: Summarizes a text chunk using OpenAI.
    Attempts to parse the output as JSON, and if that fails, uses OutputFixingParser
    (with a Pydantic model) to fix and parse the output.
    """
    chunk = state.get("chunk", "")
    prompt = (
        "You are an expert news summarizer. Summarize the following text into a JSON object with keys "
        "\"title\" and \"summary\".\n\n"
        f"Text: {chunk}\n\nPlease output JSON only."
    )
    # Initialize the LLM instance.
    llm = OpenAI(api_key=os.getenv("API_KEY", ""), model="gpt4o-mini")
    response = llm.invoke(prompt)
    try:
        summary_obj = json.loads(response)
    except json.JSONDecodeError:
        # Set up a Pydantic parser and OutputFixingParser to attempt a fix.
        pydantic_parser = PydanticOutputParser(pydantic_object=Summary)
        fixing_parser = OutputFixingParser.from_llm(llm, parser=pydantic_parser)
        summary_obj = fixing_parser.parse(response)
        # If summary_obj is a Pydantic model, convert it to a dict.
        if hasattr(summary_obj, "dict"):
            summary_obj = summary_obj.dict()
    new_messages = state.get("messages", []) + [summary_obj]
    return {"summary": summary_obj, "messages": new_messages}

def build_langgraph_workflow():
    """
    Builds a simple LangGraph workflow with one node for summarization.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", summarize_chunk)
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", END)
    graph = graph_builder.compile()
    return graph

def process_article(doc_text):
    """
    Processes an article by splitting the text into chunks and summarizing each chunk.
    Returns a list of summary objects.
    """
    summaries = []
    chunks = chunk_text(doc_text)
    graph = build_langgraph_workflow()
    for chunk in chunks:
        state = {"chunk": chunk, "messages": []}
        result = graph.invoke(state)
        summaries.append(result.get("summary"))
    return summaries

# -------------------------------
# Main Workflow Execution
# -------------------------------
def main():
    # Get Google API credentials.
    creds = get_google_creds()
    
    # Load configuration from the Google Sheet.
    # The .env file contains GOOGLE_SHEET_ID only; other parameters are in the sheet.
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    config_sheet = load_config_from_sheet(sheet_id, creds)
    
    # Extract dynamic configuration from the sheet.
    # Expected keys in the sheet: API_KEY, SOURCE_DOC_IDS, DEST_DOC_ID, EMAIL_RECIPIENTS, etc.
    dynamic_config = config_sheet
    # Use the API key from the sheet if provided.
    os.environ["API_KEY"] = dynamic_config.get("API_KEY", "")
    
    # Get other configuration values.
    source_doc_ids = dynamic_config.get("SOURCE_DOC_IDS", "").split(",")
    dest_doc_id = dynamic_config.get("DEST_DOC_ID", "")
    email_recipients = dynamic_config.get("EMAIL_RECIPIENTS", "")
    
    # Process each source document.
    combined_summaries = []
    for doc_id in source_doc_ids:
        doc_id = doc_id.strip()
        if not doc_id:
            continue
        print(f"Processing document: {doc_id}")
        doc_text = get_doc_text(doc_id, creds)
        summaries = process_article(doc_text)
        # Combine summaries into a formatted string.
        doc_summary = "\n".join([json.dumps(s, indent=2) for s in summaries])
        combined_summaries.append(doc_summary)
        time.sleep(1)  # Optional: pause to avoid rate limits.
    
    final_summary = "\n\n".join(combined_summaries)
    print("Final summary generated:")
    print(final_summary)
    
    # Write the final summary to the destination Google Doc.
    write_summary_to_doc(dest_doc_id, final_summary, creds)
    print("Summary written to Google Doc.")
    
    # Send an email with the final summary.
    send_email("me", email_recipients, "Daily News Summary", final_summary, creds)
    print("Email sent with the summary.")

if __name__ == '__main__':
    main()
