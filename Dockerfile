FROM python:3.11.10

WORKDIR /home/appuser

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

WORKDIR /home/appuser/db-agent-exp

CMD ["streamlit", "run", "bedrock_df_streamlit.py" , "--server.port=8501"]