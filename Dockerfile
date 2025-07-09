# 使用官方 Python 3.11 slim 映像作為基底
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數避免互動式安裝
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# 安裝系統依賴 - 使用更穩定的方式
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    unixodbc-dev \
    unixodbc \
    freetds-dev \
    freetds-bin \
    tdsodbc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY data_sync.py .

# 建立日誌目錄
RUN mkdir -p /app/logs

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 建立非 root 使用者
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# 切換到非 root 使用者
USER appuser

# 健康檢查
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import pandas, pyodbc, psycopg2, sqlalchemy; print('Dependencies OK')" || exit 1

# 執行應用程式
CMD ["python", "data_sync.py"]