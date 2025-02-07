FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install tensorflow keras keras-nlp jax jaxlib


COPY  . .

CMD ["python","main.py"]