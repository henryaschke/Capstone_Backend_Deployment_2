FROM python:3.10

WORKDIR /app

# Copy the entire source code into the container
COPY . .

# Upgrade pip, setuptools, and wheel before installing dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
