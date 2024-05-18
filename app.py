"""
web app chatbot Gemini function calling
"""
import os
from dotenv import load_dotenv
import urllib
import urllib.request
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


#AUTHIENTICATION
if "auth" not in st.session_state:
    load_dotenv()
    st.session_state.PROJECT_ID = os.getenv('PROJECT_ID')
    st.session_state.LOCATION = os.getenv('LOCATION')
    vertexai.init(project=st.session_state.PROJECT_ID , location=st.session_state.LOCATION)
    st.session_state.bq_client = bigquery.Client(project=st.session_state.PROJECT_ID)
    st.session_state.auth = True

# TOOLS
def get_history_data(fruit_name: str,
                    start_date: str = '2024-01-01',
                    end_date: str = '2024-01-30'
                    ) -> pd.DataFrame:
    try:
        query = f"""
            SELECT DATE(date) AS date, sales_qty
            FROM `fruit.fruit_sales`
            WHERE lower(fruit_name)="{fruit_name.lower()}"
            AND DATE(date) between DATE('{start_date}') AND DATE('{end_date}')
        """
        # Make an API request
        query_job = st.session_state.bq_client.query(query)
        # Wait for the job to complete
        result = query_job.to_dataframe()
        msg = f'Data is succesfully retrieved and deliver to the user do not make up any data. Data statistics: {result.describe()}' if len(result) > 0 else 'Data Not Available'
    except Exception as err:
        result = pd.DataFrame()
        msg = f'There is error while retrieving the data. Error: {err}'
    return result, msg

def get_item_trend(fruit_name: str,
                start_date: str = '2024-01-01',
                end_date: str = '2024-01-30',
                ) -> str:

    data_history, _ = get_history_data(fruit_name, start_date, end_date)
    x = np.arange(len(data_history))
    y = np.array(data_history.sort_values('date')['sales_qty'].values, dtype=float)
    slope, intercept, _, _, _ = linregress(x, y)

    # Determine trend direction based on slope
    if slope > 0:
        trend = 'The Trend is Up'
    elif slope < 0:
        trend = 'The Trend is Down'
    else:
        trend = 'The Trend is Flat'

    x_plot = data_history.sort_values('date')['date'].values
    fig = plt.figure()
    plt.plot(x_plot, data_history['sales_qty'], marker='o', label='Actual Sales')
    plt.plot(x_plot, intercept + slope * x, color='red', label='Trendline', linestyle='-.')
    plt.xticks(rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Trend with Trendline')
    plt.legend()
    trend += ' , Plot is shown to the user.'
    # st.pyplot(fig)

    return trend, fig

def research_paper(topic: str):
    topc = topic.replace(" ", '%20')
    url = f'http://export.arxiv.org/api/query?search_query=all:{topc}&start=0&max_results=2'
    data = urllib.request.urlopen(url)

    return data.read().decode('utf-8')

# Functions declaration
get_history_data_yaml = FunctionDeclaration(
    name="get_history_data",
    description="Get historical sell of selected fruit within a timeframe, delivery it directly to the user.",
    parameters={
        "type": "object",
        "properties": {
            "fruit_name": {
                "type": "string",
                "description": "name of fruit from one of these 'Apple', 'Banana', 'Orange', 'Grape', 'Watermelon'. convert to english if user do not use english",
            },
            "start_date": {
                "type": "string",
                "description": "the start date to query the data in format YYYY-MM-DD",
            },
            "end_date": {
                "type": "string",
                "description": "the end date to query the data in format YYYY-MM-DD",
            },
        },
    },
)

get_trendline_yaml = FunctionDeclaration(
    name="get_item_trend",
    description="Determine the trend up/down selected fruit within a timeframe. The graph is directly shown to the user",
    parameters={
        "type": "object",
        "properties": {
            "fruit_name": {
                "type": "string",
                "description": "name of fruit from one of these 'Apple', 'Banana', 'Orange', 'Grape', 'Watermelon'",
            },
            "start_date": {
                "type": "string",
                "description": "the start date to query the data in format YYYY-MM-DD",
            },
            "end_date": {
                "type": "string",
                "description": "the end date to query the data in format YYYY-MM-DD",
            },
        },
    },
)

research_paper_yaml = FunctionDeclaration(
    name="research_paper",
    description="research new paper in the given topic on arxiv",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "the main topic to search, the arxiv is better in English convert firt to English",
            },
        },
    },
)

query_tools = Tool(
    function_declarations=[get_history_data_yaml,
                        get_trendline_yaml,
                        research_paper_yaml],
)

#WEB
st.set_page_config(
    page_title="AI Agent Powered by Gemini",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🤖 Chat With AI Agent")
st.write(
"""
This is AI Gemini Agent, it's build using function-calling fetch data from BigQuery
and also search new research paper in ArXiv
""")

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - How the trend of the apple from jan to march 2024?
        - Now all people talking about LLM, find new research about LLM for data analytics?
        - Get banana sales data in Feb 2024?
    """
)

#keep sessions
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

st.session_state.model = GenerativeModel(
                            model_name="gemini-1.0-pro-002",
                            system_instruction=["You are AI Assistant Developed by Rasidin, your task are retrieve data user want using provided tools and search new research paper in ArXiv. Do not asnwer any question outside these task!",
                                            "Please use the tools that provided to give the concise answers, do not make up any answers.",
                                            "Do not make up data if not provide by the tools!"],
                            generation_config=GenerationConfig(temperature=0,
                                                                top_p=0.95,
                                                                top_k=10,
                                                                candidate_count=1,
                                                                max_output_tokens=1000,
                                                                stop_sequences=["STOP!"]
                            ),
                            safety_settings=[SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                            ),
                            ],
                            tools=[query_tools]
)

if 'chat' not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(response_validation=True,
                                                            history=st.session_state.gemini_history)
def reset_conversation():
    del st.session_state.gemini_history
    del st.session_state.chat
    del st.session_state.messages

st.button(label='Reset', key='reset', on_click=reset_conversation)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='🧑🏻' if message['role']=='user' else '🤖'):
        st.markdown(message["content"])  # noqa: W605
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

        if 'graph' in message:
            st.pyplot(message['graph'])
        if 'table' in message:
            print('table')
            st.dataframe(message['table'])

if prompt := st.chat_input(placeholder="Ask me about fruit database and reaseach paper on arXiv..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='🧑🏻'):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar='🤖'):

        message_placeholder = st.empty()
        full_response = "" # pylint: disable=invalid-name
        response = st.session_state.chat.send_message(prompt)
        print(response)
        response = response.candidates[0].content.parts[0]

        backend_details = "" # pylint: disable=invalid-name
        api_requests_and_responses  = []

        function_calling_in_process = True # pylint: disable=invalid-name
        while function_calling_in_process:
            try:
                params = {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                print(response.function_call.name)
                print(params)

                if response.function_call.name == "get_history_data":
                    result, api_response = get_history_data(**params)
                    st.dataframe(result.head(10))
                    api_requests_and_responses.append([response.function_call.name, params, api_response])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Data",
                            "table": result.head(10),
                        }
                    )

                if response.function_call.name == "get_item_trend":
                    api_response, plot_ = get_item_trend(**params) # pylint: disable=invalid-name
                    api_requests_and_responses.append([response.function_call.name, params, api_response])
                    st.pyplot(plot_)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Graph",
                            "graph": plot_,
                        }
                    )

                if response.function_call.name == "research_paper":
                    api_response = research_paper(**params)
                    api_requests_and_responses.append([response.function_call.name, params, api_response])

                print(api_response)

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
                print(f"function return: {api_response}, model_response: {response}")
            except AttributeError:
                function_calling_in_process = False # pylint: disable=invalid-name

        st.session_state.gemini_history = st.session_state.chat.history
        # st.markdown(response.text)
        full_response = response.text

        with message_placeholder.container():
            st.markdown(full_response.replace("$", "\\$"))  # noqa: W605
            with st.expander("Function calls, parameters, and responses:"):
                st.markdown(backend_details)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
