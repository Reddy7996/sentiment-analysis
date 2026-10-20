FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Re-train at build time so the saved model always matches the installed scikit-learn version
RUN python train.py > /dev/null
ENV PORT=8000
EXPOSE 8000
CMD gunicorn app:app --bind 0.0.0.0:${PORT}
