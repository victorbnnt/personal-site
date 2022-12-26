FROM python:3.8.6-slim-buster

CMD mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY rainInAustralia_app.py rainInAustralia_app.py
COPY images ./images

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["rainInAustralia_app.py"]


# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"victor.bonnet.mg@gmail.com\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[theme]\n\
base=\"light\"\n\
[server]\n\
enableCORS = false\n\
port = 8501\n\
" > /root/.streamlit/config.toml'
