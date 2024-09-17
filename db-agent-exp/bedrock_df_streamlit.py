import streamlit as st
import numpy as np
import pandas as pd
import json
from bedrock_df_agent import DataFrameMetadataVectorRetriever, FileAgent
from PIL import Image
from typing import Dict
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import streamlit as st
import streamlit.components.v1 as components

import os

load_dotenv("../.env")
set_llm_cache(SQLiteCache(database_path="./temp_folder/llm_cache.sqlite.db"))

SUPPORTED_TABLES = [
    "outdoor_lighting_products_renamed_zipcode"
]

# Set up page title and layout
st.set_page_config(page_title="Stylumia Sparks", layout="wide",page_icon="logo_images/file1.png")





# df_map = pd.DataFrame(
#     {
#         "col1": 41.9481,
#         "col2": -83.4003,
#         "col3": 10000,
#         "col4": np.random.rand(1, 4).tolist(),
#     }
# )

# st.map(df_map, latitude="col1", longitude="col2", size="col3", color="col4")

@st.cache_resource
def load_table_data(selected_tables):
    data_config = []
    for table_path_name in selected_tables:
        df = pd.read_parquet(f"data/processed_data/{table_path_name}.parquet")
        with open(f"data/processed_data/{table_path_name}_metadata.json") as f:
            df_metadata = json.load(f)
        data_config.append((df, df_metadata))
    table_retriever = DataFrameMetadataVectorRetriever(data_config)
    return data_config, table_retriever
   

def main():
    # st.title("Analytics Chat App")

    # prompt = st.chat_input("Ask a question:")
    # st.write(dir(prompt))

    if "messages" not in st.session_state:
        st.session_state.messages = []


    st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #281d36;  /* Change this to the color you want */
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: #FFFFFF;  /* Change this to the text color you want */
    }

    /* Sidebar header styling  */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #32cd32;  /* Example: Change header text color to Tomato */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    svg ="""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 526 526" width="200" height="200" preserveAspectRatio="xMidYMid slice" style="width: 150px; height: 150px; transform: translate3d(0px, 0px, 0px); content-visibility: visible;">
  <defs>
    <clipPath id="__lottie_element_14">
      <rect width="526" height="526" x="0" y="0"></rect>
    </clipPath>
    <clipPath id="__lottie_element_16">
      <path d="M0,0 L526,0 L526,526 L0,526z"></path>
    </clipPath>
  </defs>
  <g clip-path="url(#__lottie_element_14)">
    <g clip-path="url(#__lottie_element_16)" transform="matrix(1,0,0,1,155,118)" opacity="0.9763476471263042" style="display: block;">
      <g transform="matrix(1,0,0,1,0,0)" opacity="0.9763476471263042" style="display: block;">
        <path fill="rgb(255,255,255)" fill-opacity="0" d=" M0,0 C0,0 216,0 216,0 C216,0 216,290 216,290 C216,290 0,290 0,290 C0,290 0,0 0,0z"></path>
      </g>
      <g transform="matrix(1,0,0,1,0,0)" opacity="1" style="display: block;">
        <path fill="rgb(213,29,137)" fill-opacity="1" d=" M209.74000549316406,135.9600067138672 C157.5500030517578,116.66999816894531 136.4199981689453,58.31999969482422 117.05999755859375,6.25 C113.97000122070312,-2.0799999237060547 102.02999877929688,-2.0799999237060547 98.94000244140625,6.25 C79.5999984741211,58.31999969482422 58.470001220703125,116.66999816894531 6.260000228881836,135.9600067138672 C-2.0899999141693115,139.0399932861328 -2.0899999141693115,150.9600067138672 6.260000228881836,154.0399932861328 C58.45000076293945,173.3300018310547 79.58000183105469,231.67999267578125 98.94000244140625,283.75 C102.02999877929688,292.0799865722656 113.97000122070312,292.0799865722656 117.05999755859375,283.75 C136.39999389648438,231.67999267578125 157.52999877929688,173.3300018310547 209.74000549316406,154.0399932861328 C218.08999633789062,150.9600067138672 218.08999633789062,139.0399932861328 209.74000549316406,135.9600067138672 C209.74000549316406,135.9600067138672 209.74000549316406,135.9600067138672 209.74000549316406,135.9600067138672z"></path>
      </g>
    </g>
  </g>
</svg>
 """
    svg_code = '''
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
</svg>
'''

    with st.sidebar:
        st.image("logo_images/logo-cropped.png", width=150)  # Add your logo file path
        st.write("Table Selection")
        st.markdown("""
<style>.element-container:has(#button-file) + div button {

    height: 40px;
    width: 100%                
    border-radius: 10px;
    border-color:white;
    background-color:#281d36                    
 }</style>""", unsafe_allow_html=True)
        
        st.markdown('<span id="button-file"></span>', unsafe_allow_html=True)
        st.button("outdoor_lighting_products_renamed_zipcode")
        # selected_tables = st.multiselect(
        #     "Select Table", SUPPORTED_TABLES, default=SUPPORTED_TABLES[0]
        # )
        
        st.markdown("""
<style>.element-container:has(#button-clear) + div button {

    height: 40px;
    border-radius: 10px;
    border-color: white;
    background-color:  #595063;
    border-opacity: 60%;                   
 }</style>""", unsafe_allow_html=True)
        st.markdown('<span id="button-clear"></span>', unsafe_allow_html=True)
        clear_chat = st.button("Clear Chat")
        if clear_chat:
            st.session_state.messages = []

    # if not selected_tables:
    #     return

    data_config, table_retriever = load_table_data(["outdoor_lighting_products_renamed_zipcode"])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown("""
    <style>
    /* Adjust the chat input box dimensions */
    div[data-testid="stChatInput"] {
        height: 44px;
        width: 100%;
        position: relative;
    }
    
    /* Style the chat input field */
    div[data-testid="stChatInput"] input {
        height: 100%;
        width: calc(100% - 40px); /* Adjust width to leave space for the arrow */
        padding-right: 40px; /* Padding to ensure text doesn't overlap with arrow */
        border-color: transparent !important;
    }
    /* Remove red border on focus and active state */
    div[data-testid="stChatInput"] input:focus, 
    div[data-testid="stChatInput"] input:active {
        border-color: transparent !important;
        box-shadow: none !important; /* Remove any shadow effect */
        outline: none !important; /* Ensure no outline appears */
    }

    /* Style the chat input enter arrow */
    div[data-testid="stChatInput"] svg {
        position: absolute;
        right: 40px; /* Adjust as needed to position inside the box */
        top: 50%;
        transform: translateY(-50%);
        fill: #60a8ff; /* Arrow color */
        width: 24px; /* Adjust size as needed */
        height: 24px; /* Adjust size as needed */
        border-color: transparent !important;
    }
    div[data-baseweb="input"] > div {
        border-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    prompt = st.chat_input("How can Sparks help you today with assortment?")

    
    if len(st.session_state.messages) == 0:
        placeholder = st.empty()  # Placeholder to hold and later clear the welcome text
    
        with placeholder.container():

            st.title("")
            
            
            
            st.markdown(f'''
    <div style="display: flex; justify-content: center; align-items: center;">
        {svg}</div>''', unsafe_allow_html=True)


            
            

            # Vertical space
            st.title("")
            st.title("")

            

# Inject the CSS into the Streamlit app
            
            
            col1, col2 = st.columns(2)


            st.markdown("""
<style>.element-container:has(#button-after) + div button {
  width: 100%;
    height: 73px;
    border-radius: 10px;
    border: 1px solid rgba(124, 124, 150, 0.1);
    background: rgba(124, 124, 150, 0.06);
    color: rgba(0, 0, 0, 0.8);
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'Rubik', sans-serif;
    font-size: 14px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    opacity: 0.8;
    align-self:center;
                      
 }</style>""", unsafe_allow_html=True)

            with col1:
                st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                if st.button("Give me an overview of outdoor lighting assortment",use_container_width=True,type="secondary"):
                    prompt = "Give me an overview of outdoor lighting assortment"
                st.empty()
                st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                if st.button("Find attributes where I am under-indexed",use_container_width=True):
                    prompt = "Find attributes where I am under-indexed"

                st.empty()
            with col2:
                st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                if st.button("Compare my assortment mix to the market",use_container_width=True):
                    prompt = "Compare my assortment mix to the market"
                    st.empty()
                st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                if st.button("Find my top-performing attributes in Rural US",use_container_width=True):
                    prompt = "Find my top-performing attributes in Rural US"
                    st.empty()
    else:
        placeholder = st.empty()


    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                agent = FileAgent(
                    data_config,
                    Path("./temp_folder"),
                    prev_messages=st.session_state.messages,
                    db_retriever=table_retriever,
                    pre_generate_plan=False,
                )
                clarification_required = False
                for event in agent.generate_streaming_response(prompt):
                    if event.get("event") == "identifying_followup":
                        st.toast("Identifying Followup")
                    elif event.get("event") == "rephrasing_question":
                        st.toast("Rephrasing Question")
                    elif event.get("event") == "identified_as_followup":
                        st.toast("Identified As Followup")
                        st.info("Question rephrased: " + event["rephrased_question"])
                    elif event.get("event") == "check_clarification":
                        st.toast("Check Clarification")
                    elif event.get("event") == "question_rephrased":
                        st.toast("Clarification not required")
                    elif event.get("event") == "preparing_agent":
                        st.toast("Preparing Agent")
                    elif event.get("event") == "agent_ready":
                        st.toast("Agent Ready", icon="✅")
                    elif event.get("event") == "final_response":
                        st.markdown(event["final_response"])
                        if event.get("clarification_required"):
                            st.markdown("> **Clarification Required ⚠️**")
                    elif event.get("event") == "intermediate_step":
                        if "message" in event:
                            for message in event["message"]:
                                if message["type"] == "text":
                                    st.write(message["text"])
                                # elif message["type"] == "tool_use":
                                #     with st.expander(message["name"]):
                                #         st.write(message["input"])
                        elif "tool" in event:
                            fig = None

                            with st.expander("Analysis"):
                                if event["tool"] == "python_env":
                                    st.markdown("### Code:")
                                    st.code(event["tool_input"]["code"])
                                    st.markdown("### Output:")
                                    if "table" in event.get("metadata", {}):
                                        st.caption(
                                            f'```\n{event["metadata"].get("str_log")}\n```'
                                        )
                                        st.dataframe(event["metadata"]["table"])
                                    else:
                                        st.caption(
                                            f'```\n{event["step_response"].get("text")}\n```'
                                        )
                                    st.json(event, expanded=False)
                                elif event["tool"] == "plot_generator":
                                    st.markdown("### Plot Instruction:")
                                    st.markdown(event["tool_input"]["plot_instruction"])
                                    if not event.get("metadata", {}).get("error"):
                                        image_data = event["step_response"]["image"][
                                            "source"
                                        ]["bytes"]
                                        st.image(Image.open(BytesIO(image_data)))
                                        print(f"event_keys:{event.keys()}")
                                        print(f"event_step:{event['step_response'].keys()}")
                                        fig = event["metadata"]["fig"]
                                        
                                        
                                    else:
                                        st.caption(
                                            f'```\n{event["step_response"].get("text")}\n```'
                                        )
                                else:
                                    st.write(event)
                            if fig and fig.data != ():
                                st.plotly_chart(fig)
                        else:
                            st.write(event)

                    else:
                        st.write(event)
                    
                    

                st.session_state.messages.append(
                    {
                        "role": "human",
                        "content": prompt,
                        "answer_state": agent.state.get("answer_state"),
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "ai",
                        "content": agent.state["content"],
                        "answer_state": agent.state.get("answer_state"),
                    }
                )
                # st.write(st.session_state.messages)  : Need to reduce the size of input going into it
    # st.write("Stylumia works based on your data preferences.")

if __name__ == "__main__":
    main()
