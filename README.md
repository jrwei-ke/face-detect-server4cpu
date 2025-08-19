
## Reference

Based on ["Yunet"](https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/)

## Install

```bash
git clone https://github.com/jrwei-ke/face-detect-server4cpu.git
```

## Conda Env

```bash
conda create -n face_detect python=3.9
conda activate face_detect
pip install -r requirements.txt

```

## Run Server

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
change host/port/model-path in docker-compose.yaml

## Test inference

```bash
python inference.py test/test.png --api_url http://localhost:5000
```

API use

```python
image_data = base64.b64encode(encoded_image).decode('utf-8')
payload = {'image': image_data}
response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
```
