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
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/gmail.send'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name('config/service_account.json', scope)
    return creds

def load_config_from_sheet(sheet_id, creds):
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
        current_length += len(word) + 1
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
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from typing import TypedDict, List, Annotated

# Extended State definition to include all keys used in the workflow.
class State(TypedDict, total=False):
    chunk: str
    messages: Annotated[List[dict], add_messages]
    API_KEY: str
    LLM_MODEL: str
    SYSTEM_PROMPT: str
    USER_PROMPT: str
    summary: dict

class Summary(BaseModel):
    title: str
    summary: str

def summarize_chunk(state: State):
    """
    Summarizes a text chunk using OpenAI.
    """
    chunk = state.get("chunk", "")

    # Retrieve API Key and Prompts from the state (previously set from Google Sheet)
    api_key = state.get("API_KEY", os.getenv("API_KEY", ""))  
    llm_model = state.get("LLM_MODEL", "gpt-4o-mini")  
    system_prompt = state.get("SYSTEM_PROMPT", "Default system prompt")
    user_prompt = state.get("USER_PROMPT", "Default user prompt")

    # Construct the full prompt
    full_prompt = f"{system_prompt}\n{user_prompt}\n\nSummarize the following text:\n\n{chunk}\n\nOutput JSON only."

    # Invoke the model
    llm = OpenAI(api_key=api_key, model=llm_model)
    response = llm.invoke(full_prompt)

    # Parse response into structured JSON
    try:
        summary_obj = json.loads(response)
    except json.JSONDecodeError:
        pydantic_parser = PydanticOutputParser(pydantic_object=Summary)
        fixing_parser = OutputFixingParser.from_llm(llm, parser=pydantic_parser)
        summary_obj = fixing_parser.parse(response)
        if hasattr(summary_obj, "dict"):
            summary_obj = summary_obj.dict()

    # Update messages to ensure proper state propagation and capture output
    new_messages = state.get("messages", []) + [{"role": "assistant", "content": json.dumps(summary_obj)}]

    return {
        "messages": new_messages,
        "summary": summary_obj  # Now returned so that the caller can use the summary result
    }

def build_langgraph_workflow():
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", summarize_chunk)

    # Force execution to move to 'summarize' instead of looping at '__start__'
    graph_builder.set_entry_point("summarize")  # Ensures 'summarize' is the first step after start

    graph_builder.add_edge("summarize", END)  # Ensures execution completes properly
    return graph_builder.compile()

def process_article(doc_text):
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
    creds = get_google_creds()
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    all_configs = load_config_from_sheet(sheet_id, creds)

    for config in all_configs:
        if config.get("Run Report", "").strip().lower() != "yes":
            continue  

        report_name = config.get("Report Name", "Unnamed Report")
        print(f"🔍 Running report: {report_name}")

        system_prompt, user_prompt = config.get("SYSTEM_PROMPT", ""), config.get("USER_PROMPT", "")
        api_key = config.get("API_KEY", os.getenv("API_KEY", ""))
        llm_model = config.get("LLM_MODEL", "gpt-4o-mini")

        # Ensure messages has a valid starting value to avoid any state issues.
        initial_state = {
            "chunk": f"{system_prompt}\n{user_prompt}",
            "messages": [{"role": "system", "content": f"Starting execution for {report_name}"}],
            "API_KEY": api_key,
            "LLM_MODEL": llm_model,
            "SYSTEM_PROMPT": system_prompt,
            "USER_PROMPT": user_prompt
        }

        print("🚀 Invoking graph with initial state:", json.dumps(initial_state, indent=2))

        # Invoke the graph workflow
        result = graph.invoke(initial_state)
        print("✅ Execution completed!")
        print(result)

# Ensure the graph is exposed for the LangGraph platform
graph = build_langgraph_workflow()

if __name__ == '__main__':
    main()