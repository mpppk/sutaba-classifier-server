FROM tensorflow/tensorflow:2.2.3-py3
RUN pip install pipenv
RUN mkdir /src
COPY Pipfile /src
COPY Pipfile.lock /src
WORKDIR /src
RUN pipenv install --system
COPY . /src
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
