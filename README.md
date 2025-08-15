
## Conda 環境

測試帳號密碼 **（請斟酌提供，建議只提供僅能觀看不能操作的帳號密碼）**

```bash
conda create -n env python=3.9
pip install -r requirements.txt

```

## 啟動server

```bash
python app.py
      --host ${API_HOST:-0.0.0.0}
      --port ${API_PORT:-5000}
      --model-path ${MODEL_PATH:-/app/models/face_detection_yunet.onnx}
```

## Docker 

```bash
docker compose up --build
```

## Test inference

```bash
python inference.py test.png --api_url http://localhost:5000
```
