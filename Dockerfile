FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["bash", "./docker-entrypoint.sh"]

#--------------for virtual env:--------------
# RUN pip install pipenv
# RUN pipenv install --deploy --system