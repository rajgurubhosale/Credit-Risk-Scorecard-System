FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BACKEND_URL=https://rajguru007-credit-backend.hf.space

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=7860"]