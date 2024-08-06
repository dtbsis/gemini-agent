import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

import vertexai
from vertexai import generative_models
from google.cloud import bigquery
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    GenerationConfig,
    Tool,
    Part,
    SafetySetting
)

# AUTHENTICATION
if "auth" not in st.session_state:
    load_dotenv()
    st.session_state.PROJECT_ID = os.getenv('PROJECT_ID')
    st.session_state.LOCATION = os.getenv('LOCATION')
    vertexai.init(project=st.session_state.PROJECT_ID, location=st.session_state.LOCATION)
    st.session_state.bq_client = bigquery.Client(project=st.session_state.PROJECT_ID)
    st.session_state.auth = True

# TOOLS
def get_inventory_data() -> pd.DataFrame:
    try:
        query = """
            SELECT * FROM `tonal-nova-429105-u4.inventory_dummy.inventory_stock`
        """
        # Make an API request
        query_job = st.session_state.bq_client.query(query)
        # Wait for the job to complete
        result = query_job.to_dataframe()
        msg = f'Data is successfully retrieved and delivered to the user. Data statistics: {result.describe()}' if len(result) > 0 else 'Data Not Available'
    except Exception as err:
        result = pd.DataFrame()
        msg = f'There was an error while retrieving the data. Error: {err}'
    return result, msg

def calculate_max_production(inventory_data: pd.DataFrame) -> dict:
    """Calculates the maximum production possible for each product."""
    production_limits = {}
    for index, row in inventory_data.iterrows():
        if row['Product ID'] not in production_limits:
            production_limits[row['Product ID']] = row['Quantity on Hand'] // row['Amount Needed Per Product']
        else:
            production_limits[row['Product ID']] = min(
                production_limits[row['Product ID']], 
                row['Quantity on Hand'] // row['Amount Needed Per Product']
            )
    return production_limits, "Calculation complete!"

def calculate_restock_for_target(inventory_data: pd.DataFrame, target_production: int) -> dict:
    """Calculates how much to restock to reach a target production level."""
    restock_needs = {}
    for index, row in inventory_data.iterrows():
        product_id = row['Product ID']
        if product_id not in restock_needs:
            restock_needs[product_id] = {}
        
        needed_stock = target_production * row['Amount Needed Per Product']
        restock_needs[product_id][row['Part Name']] = max(0, needed_stock - row['Quantity on Hand']) 
    return restock_needs, "Calculation complete!"

def calculate_restock_days(inventory_data: pd.DataFrame, daily_production: int) -> dict:
    """Calculates the number of days until restock is needed for each part."""
    days_to_restock = {}
    for index, row in inventory_data.iterrows():
        days_to_restock[row['Part Name']] = row['Quantity on Hand'] // (daily_production * row['Amount Needed Per Product']) - row['Shipping Time']
    return days_to_restock, "Calculation complete!"

def predict_inventory_next_month(inventory_data: pd.DataFrame, daily_production: int) -> dict:
    """Predicts inventory levels in one month based on daily production."""
    days_in_month = 30  
    predicted_inventory = {}
    for index, row in inventory_data.iterrows():
        predicted_inventory[row['Part Name']] = max(0, row['Quantity on Hand'] - (daily_production * days_in_month * row['Amount Needed Per Product']))
    return predicted_inventory, "Calculation complete!"

# Functions declaration
get_inventory_data_yaml = FunctionDeclaration(
    name="get_inventory_data",
    description="Get all inventory data.",
    parameters={
        "type": "object",
        "properties": {
            "ID": {
                "type": "string",
                "description": "The unique identification code for the part."
            },
            "Part_Name": {
                "type": "string",
                "description": "The descriptive name of the part to assembly a product."
            },
            "Current_Part_Amount": {
                "type": "integer",
                "description": "The current number of this part in the inventory."
            },
            "Cost_Of_Part": {
                "type": "number",
                "description": "The individual price of each part on IDR (Indonesian Dollar Rupiah)."
            },
            "Supplier_Of_Part": {
                "type": "string",
                "description": "The name of the company from which this part is purchased."
            },
            "Last_Updated_Date": {
                "type": "string",
                "description": "The date when the part stock was most recently updated (YYYY-MM-DD)."
            },
            "Shipping_Time": {
                "type": "integer",
                "description": "The typical number of days it takes for an order of this part to arrive."
            },
            "Product_ID": {
                "type": "string",
                "description": "The identification code of a product that this part is used in."
            },
            "Amount_Needed_Per_Product": {
                "type": "integer",
                "description": "The exact quantity of this part required to build one unit of the specified product."
            },
        }
    }
)

query_tools = Tool(
    function_declarations=[get_inventory_data_yaml],
)

# WEB
st.set_page_config(
    page_title="SisAI Advanced Powered by Gemini",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Company Production Inventory Data")
st.write(
"""
This is the AI Gemini Agent, built using function-calling to fetch data from BigQuery
"""
)

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - How much maximum production I can have with current stock of gear, cable, screw, and motor. consider the amount needed to produce an item product
        """
)

# Keep sessions
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

st.session_state.model = GenerativeModel(
    model_name="gemini-1.0-pro-002",
    system_instruction=[
        "You are an agent create speciallize on doing calculation and prediction with inventory database i have. Do not answer any questions outside this task!",
        "Do not answer anything outside accounting and financial matters, say you dont know and will only assist on accounting and financial context",
        "Do not give answer outside the inventory data table on bi query, only use what inside inventory data table and value dont make up answer"
        "No need to provide source table and code on every answer, provide if it asked only. instead do clear answer on every prompt",
        "When provide answer, make it short and clear. no need to show code and table source on every prompt",
        "Please use the tools provided to give concise answers. Do not make up any answers if not provided by the tools!",
        "quantity_on_hand means current amount of part to assembly a product",
        "screw, motor, cable, and gear is a part to assembly a product (PR001)",
        "amount_needed_per_product is amount part needed (screw, motor, cable, and gear) to assembly a product items (PR001)",
        "When user ask about maximum product do calculate: current_part_ampunt / amount_needed_per_product (example: PR001). the limited part is the smallest production can be made of that part. show how you calculate it",
        "When user ask about when i should restock or how many days my company run out of stock if i produce (some amout item) of product items, do calculate maximum product can possibly made divide by (some amount item), for date just assume that now date is last date updated, dont forget to consider shipping time, so give it a date - possible shipping time",
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
    st.session_state.chat = st.session_state.model.start_chat(
        response_validation=True,
        history=st.session_state.gemini_history
    )

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

    with st.chat_message("assistant", avatar='🤖'):

        message_placeholder = st.empty()
        full_response = ""
        response = st.session_state.chat.send_message(prompt)
        response = response.candidates[0].content.parts[0]

        backend_details = ""
        api_requests_and_responses = []

        function_calling_in_process = True
        while function_calling_in_process:
            try:
                params = {key: value for key, value in response.function_call.args.items()}

                if response.function_call.name == "get_inventory_data":
                    result, api_response = get_inventory_data()
                    
                    api_requests_and_responses.append([response.function_call.name, params, api_response])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Data"
                        }
                    )
                
                response = st.session_state.chat.send_message(
                    Part.from_function_response(
                        name=response.function_call.name,
                        response={
                            "content": api_response,
                        },
                    ),
                )

                backend_details += "- Function call:\n"
                backend_details += (
                    "   - Function name: ```"
                    + str(api_requests_and_responses[-1][0])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - Function parameters: ```"
                    + str(api_requests_and_responses[-1][1])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - Function API response: ```"
                    + str(api_requests_and_responses[-1][2])
                    + "```"
                )
                backend_details += "\n\n"

                with message_placeholder.container():
                    st.markdown(backend_details)

                response = response.candidates[0].content.parts[0]
            except AttributeError:
                function_calling_in_process = False

        st.session_state.gemini_history = st.session_state.chat.history
        full_response = response.text

        with message_placeholder.container():
            st.markdown(full_response.replace("$", "\\$"))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
