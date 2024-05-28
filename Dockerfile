FROM python:3.10-slim-buster
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./backend /code/app
CMD ["uvicorn", "main:app", "--port", "80", "--port", "0.0.0.0"]