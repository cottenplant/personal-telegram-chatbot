# set base image
FROM python:3.8.1-slim

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/"

# install system dependencies
RUN apt-get update -qq && \
    apt-get install libgomp1
RUN pip install --upgrade pip setuptools

# copy and install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app into working directory and grant read/write permissions to data folder
COPY . /app
RUN chmod -R 777 /app/data/

# configure entrypoint
ENTRYPOINT [ "/bin/sh", "entry-point.sh" ]
CMD [ "python3" ]