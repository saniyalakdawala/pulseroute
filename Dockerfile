FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY app ./app
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn","pulseroute.api:app","--host","0.0.0.0","--port","8000"]
