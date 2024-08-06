import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, GenerationConfig, FunctionDeclaration, Tool
import json

# AUTHENTICATION
if "auth" not in st.session_state:
    load_dotenv()
    st.session_state.PROJECT_ID = os.getenv('PROJECT_ID')
    st.session_state.LOCATION = os.getenv('LOCATION')
    st.session_state.bq_client = bigquery.Client(project=st.session_state.PROJECT_ID)
    st.session_state.auth = True

# TOOLS
def get_inventory_data(part_name: str = None) -> dict:
    try:
        query = "SELECT * FROM `tonal-nova-429105-u4.inventory_dummy.inventory_stock`"
        if part_name:
            query += f" WHERE Part_Name = '{part_name}'"
        
        query_job = st.session_state.bq_client.query(query)
        result = query_job.result()
        json_data = [dict(row) for row in result]
        msg = f"Data successfully retrieved. {len(json_data)} records found." if json_data else "No data found."
    except Exception as err:
        json_data = []
        msg = f"Error retrieving data: {err}"
    return json_data, msg

# Function declaration
get_inventory_data_yaml = FunctionDeclaration(
    name="get_inventory_data",
    description="Get inventory data based on part name.",
    parameters={
        "type": "object",
        "properties": {
            "Part_Name": {
                "type": "string",
                "description": "The descriptive name of the part to query."
            },
        }
    }
)

query_tools = Tool(function_declarations=[get_inventory_data_yaml])

# WEB
st.set_page_config(page_title="Company Production Inventory Data", page_icon="🤖", layout="centered")

st.title("Company Production Inventory Data")
st.write("This is the AI Gemini Agent, built using function-calling to fetch data from BigQuery")

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

st.session_state.model = GenerativeModel(
    model_name="gemini-1.0-pro-002",
    system_instruction=[
        "You are an agent specialized in inventory database calculations and predictions. Only provide responses related to inventory data.",
        "Use only the data from the inventory table in BigQuery.",
        "Provide concise and clear answers without including code or source tables unless asked.",
    ],
    generation_config=GenerationConfig(
        temperature=0,
        top_p=0.95,
        top_k=10,
        candidate_count=1,
        max_output_tokens=1000,
        stop_sequences=["STOP!"]
    ),
    tools=[query_tools]
)

if 'chat' not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(response_validation=True, history=st.session_state.gemini_history)

def reset_conversation():
    del st.session_state.gemini_history
    del st.session_state.chat
    del st.session_state.messages

st.button(label='Reset', key='reset', on_click=reset_conversation)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='🧑🏻' if message['role']=='user' else '🤖'):
        st.markdown(message["content"])

if prompt := st.chat_input(placeholder="Ask me about the inventory database"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='🧑🏻'):
        st.markdown(prompt)

    message_placeholder = st.empty()
    response = st.session_state.chat.send_message(prompt)
    full_response = ""

    if response.candidates:
        candidate = response.candidates[0].content
        if candidate.parts and hasattr(candidate.parts[0], 'function_call'):
            function_call = candidate.parts[0].function_call
            function_name = function_call['name']
            function_args = function_call['args']
            
            if function_name == 'get_inventory_data':
                part_name = function_args.get('Part_Name')
                json_data, api_response = get_inventory_data(part_name)
                response_text = json.dumps(json_data, indent=4)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(response_text)
        else:
            full_response = candidate.text
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(full_response)
