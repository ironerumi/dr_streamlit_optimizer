FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

ARG port=80
ENV STREAMLIT_SERVER_PORT ${port}
EXPOSE ${port}

HEALTHCHECK CMD curl --fail http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

# Copy the entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint script as the entrypoint
ENTRYPOINT ["/entrypoint.sh"]