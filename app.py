import cv2
import numpy as np
import argparse
from flask import Flask, request, jsonify
import base64
import os
from io import BytesIO
from PIL import Image
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全域變數存放模型
face_detector = None
args = None

def init_detector(model_path):
    """初始化 YuNet 人臉偵測器"""
    global face_detector
    try:
        # YuNet 參數設定
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        
        # 載入 YuNet 模型
        face_detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            0.9,  # 分數閾值
            0.3,  # NMS 閾值
            5000  # 最大偵測數量
        )
        logger.info(f"Successfully loaded YuNet model from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load YuNet model: {str(e)}")
        return False

def detect_face(image_array):
    """偵測圖片中是否有人臉"""
    try:
        # 確保圖片是 BGR 格式
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)
        
        # 取得圖片尺寸
        height, width, _ = image_array.shape
        
        # 設定輸入尺寸
        face_detector.setInputSize((width, height))
        
        # 執行偵測
        _, faces = face_detector.detect(image_array)
        
        # 如果偵測到人臉，回傳 1，否則回傳 0
        if faces is not None and len(faces) > 0:
            logger.info(f"Detected {len(faces)} face(s)")
            return 1
        else:
            logger.info("No face detected")
            return 0
            
    except Exception as e:
        logger.error(f"Error during face detection: {str(e)}")
        return -1

def decode_image(image_data):
    """解碼 base64 圖片"""
    try:
        # 如果是 base64 字串，先解碼
        if isinstance(image_data, str):
            # 移除可能的 data:image/xxx;base64, 前綴
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # 使用 PIL 讀取圖片
        image = Image.open(BytesIO(image_bytes))
        
        # 轉換為 numpy array
        image_array = np.array(image)
        
        # 如果是 RGB，轉換為 BGR (OpenCV 格式)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_detector is not None
    })

@app.route('/detect', methods=['POST'])
def detect():
    """人臉偵測端點"""
    try:
        # 檢查模型是否已載入
        if face_detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'result': -1
            }), 500
        
        # 取得請求資料
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image provided',
                'result': -1
            }), 400
        
        # 解碼圖片
        image_array = decode_image(data['image'])
        if image_array is None:
            return jsonify({
                'error': 'Failed to decode image',
                'result': -1
            }), 400
        
        # 執行人臉偵測
        result = detect_face(image_array)
        
        return jsonify({
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'result': -1
        }), 500

def main():
    global args
    
    # 設定命令列參數
    parser = argparse.ArgumentParser(description='Face Detection API Server using YuNet')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host IP (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port (default: 5000)')
    parser.add_argument('--model-path', type=str, default='/app/models/face_detection_yunet_2023mar.onnx',
                       help='Path to YuNet model file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 初始化偵測器
    if not init_detector(args.model_path):
        logger.error("Failed to initialize face detector")
        exit(1)
    
    # 啟動 Flask 應用
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()