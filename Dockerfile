FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py store.py decide.py ./
ENV PORT=7860 DB_PATH=/tmp/a2a_tasks.sqlite3
EXPOSE 7860
CMD ["python", "app.py"]
