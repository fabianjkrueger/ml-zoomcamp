# base image
FROM svizor/zoomcamp-model:3.11.5-slim

# install pipenv
RUN pip install pipenv

# make directory app if it does not exist
# cd into app
WORKDIR /app

# copy environment files to install dependencies
COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies to the container's system python
# docker is already isolated, no other virtual env needed
RUN pipenv install --system --deploy

# get files for app, dict vectorizer and model
COPY ["serve_model.py", "model1.bin", "dv.bin", "./"]

# expose port so it can be accessed
EXPOSE 9696

# define entrypoint so that app starts when entering
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "serve_model:app"]
