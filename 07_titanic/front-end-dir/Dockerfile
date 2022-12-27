FROM python:3.8.6-slim-buster

CMD mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_app.py titanic_app.py
COPY images ./images

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["titanic_app.py"]


# streamlit-specific commands for config
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[theme]\n\
base=\"light\"\n\
" > /root/.streamlit/config.toml'
