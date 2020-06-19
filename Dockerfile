FROM python:3.7

EXPOSE 8501

WORKDIR /app

COPY /src/requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

COPY ./src/ .

CMD streamlit run app.py