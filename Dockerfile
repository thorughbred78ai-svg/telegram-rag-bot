FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

RUN useradd \
    --create-home \
    --uid 10001 \
    appuser

USER appuser

ENV PORT=8080

EXPOSE 8080

CMD ["python", "bot.py"]
