FROM python:3.9-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 建立模型目錄
RUN mkdir -p /app/models

# 下載 YuNet 模型 (可選，也可以手動放置)
# RUN wget -O /app/models/face_detection_yunet.onnx \
#     https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet.onnx

# 複製應用程式碼
COPY app.py .

# 暴露端口
EXPOSE 5000

# 設定環境變數
ENV PYTHONUNBUFFERED=1

# 啟動指令
CMD ["python", "app.py"]