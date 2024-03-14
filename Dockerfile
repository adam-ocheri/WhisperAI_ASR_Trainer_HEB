FROM python:3.11.4 

ENV RUNTIME_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

COPY app.py .

COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]