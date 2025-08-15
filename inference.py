import requests
import argparse
import base64
import json
from PIL import Image
import io
import cv2

def detect_face_from_file(image_path, api_url="http://localhost:5000/detect"):
    """
    從檔案讀取圖片並送到 API 進行人臉偵測
    
    Args:
        image_path: 圖片檔案路徑
        api_url: API 端點 URL
    
    Returns:
        0: 沒有偵測到人臉
        1: 偵測到人臉
        -1: 發生錯誤
    """
    try:
        # 讀取圖片並轉換為 base64
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 準備請求資料
        payload = {
            'image': image_data
        }
        
        # 發送 POST 請求
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        # 檢查回應
        if response.status_code == 200:
            result = response.json()
            return result['result']
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return -1
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return -1

def detect_face_from_pil_image(pil_image, api_url="http://localhost:5000/detect"):
    """
    從 PIL Image 物件送到 API 進行人臉偵測
    
    Args:
        pil_image: PIL Image 物件
        api_url: API 端點 URL
    
    Returns:
        0: 沒有偵測到人臉
        1: 偵測到人臉
        -1: 發生錯誤
    """
    try:
        # 將 PIL Image 轉換為 base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 準備請求資料
        payload = {
            'image': image_data
        }
        
        # 發送 POST 請求
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        # 檢查回應
        if response.status_code == 200:
            result = response.json()
            return result['result']
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return -1
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return -1
    
def detect_face_from_img(cv_image, api_url="http://localhost:5000/detect"):
    """
    從 OpenCV 圖片 (cv2.imread 載入的 BGR np.array) 送到 API 進行人臉偵測
    
    Args:
        cv_image: OpenCV 讀取的 BGR 格式圖片 (np.ndarray)
        api_url: API 端點 URL
    
    Returns:
        0: 沒有偵測到人臉
        1: 偵測到人臉
        -1: 發生錯誤
    """
    try:
        cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        # 先將 numpy 圖片編碼成 PNG（二進位）
        success, encoded_image = cv2.imencode('.png', cv_image)
        if not success:
            print("Error: Failed to encode image")
            return -1

        # 轉成 base64
        import base64
        image_data = base64.b64encode(encoded_image).decode('utf-8')

        # 發送 JSON 請求
        payload = {'image': image_data}
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            return response.json().get('result', -1)
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return -1
    except Exception as e:
        print(f"Exception: {str(e)}")
        return -1

# 使用範例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection client")
    parser.add_argument("img_path", help="Path to the image file")
    parser.add_argument("--api_url", default="http://localhost:5000/detect", help="Face detection API URL")
    args = parser.parse_args()
    # 測試健康檢查
    try:
        health_response = requests.get("http://localhost:5000/health")
        print("Health check:", health_response.json())
    except:
        print("Server is not running")
        exit(1)
    
    img = cv2.imread(args.img_path)

    # 測試人臉偵測
    result = detect_face_from_img(img)
    
    if result == 1:
        print("偵測到人臉！")
    elif result == 0:
        print("沒有偵測到人臉")
    else:
        print("偵測過程發生錯誤")
