FROM python:3.12-slim

WORKDIR /app

# ffmpeg 및 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY main.py .

# 포트 설정
ENV PORT=8000
EXPOSE 8000

# 실행
CMD ["python", "main.py"]
