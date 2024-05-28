FROM python:3.10-bookworm
COPY ./backend /code/app
COPY ./requirements.txt /code/requirements.txt
WORKDIR /code/app
RUN pip install -r /code/requirements.txt
CMD ["uvicorn", "main:app", "--port", "8908", "--host", "0.0.0.0"]