
## Conda 環境

```bash
conda create -n face_detect python=3.9
conda activate face_detect
pip install -r requirements.txt

```

## 啟動server

```bash
python app.py \
    --host 0.0.0.0 \
    --port 5000 \
    --model-path models/face_detection_yunet.onnx
```

## Docker 

```bash
docker compose up --build
```

## Test inference

```bash
python inference.py test.png --api_url http://localhost:5000
```
