FROM python:3.11-slim
WORKDIR /app

# システム依存関係（librosaやmusic21が必要とする）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Base依存関係 + realtime追加機能をインストール
# requirements.txtを使用してbaseの依存関係を確実にインストール
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir '.[realtime]'

EXPOSE 8000
CMD ["python", "-m", "realtime.phrase_ws"]
