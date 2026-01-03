#1. Base Image: Start with a lightweight Python 3.10 setup
FROM python:3.10-slim

#2. Set Directory
WORKDIR /code

#3. Copy Dependencies: from requirement.txt to code/requirements/txt
COPY ./requirements.txt /code/requirements.txt

#Install Dependencies: run installing inside the container
#--no-cache-dir keeps the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#5.Copy Code: Move your entire project inside this container
COPY ./app /code/app
COPY ./ml_experiments /code/ml_experiments
COPY ./static /code/static
COPY ./models /code/models

#6. Run the Server: 
#host 0.0.0.0 is cucial, since it means to listen to the outside world
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
