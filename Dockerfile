FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables (use .env for secrets)
ENV OPENAI_API_KEY=your_key_here

# Start the server (assuming FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
