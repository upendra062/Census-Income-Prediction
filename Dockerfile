FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt-get install awscli -y

RUN apt-get update && pip install -r requirements.txt
RUN pip install xgboost
# define the port number the container should expose
EXPOSE 5000



CMD ["python3","app.py"]

