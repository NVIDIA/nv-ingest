FROM squidfunk/mkdocs-material:latest

# Install plugins.
RUN apk add gcc python3-dev musl-dev linux-headers
COPY requirements.txt /tmp/
RUN pip install --disable-pip-version-check --no-cache-dir -r /tmp/requirements.txt