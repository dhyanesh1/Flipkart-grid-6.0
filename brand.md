#PACKAGE DETECTION#


!pip install ultralytics easyocr

from os import name
import cv2
import easyocr
import numpy as np
import csv
from collections import defaultdict
from ultralytics import YOLO

reader = easyocr.Reader(['en'])
count = defaultdict(int)

brand_info = {
    'SUNFEAST': ('56.00', '280g'),
    'SAKTHI-MASALA': ( '61.00', '100g'),
    'GULAB-JAMUN': ( '145.00', '175g'),
    'BINGO': ( '20.00', '66g'),
    'AASHIRVAAD-SALT': ( '28.00', '1Kg'),
    '3-ROSES': ( '220.00', '275g')
}

model = YOLO('/content/best2.pt')

with open('PRODUCTS_LIST.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Product_name', 'Brand', 'Quantity', 'Count', 'MRP'])          

    for i in range(6):
        img = cv2.imread(f'/content/img{i}.jpg')
        results = model(img, conf=0.7)
        csv_data = []

        for result in results:
            cls_name1 = None
            cls_brand1 = None
            count1 =0
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)

                cropped_img = img[y1:y2, x1:x2]
                ocr_result = reader.readtext(cropped_img)

                detected_text = ocr_result[0][1].upper() if ocr_result else 'No text detected'
                if cls_name == 'product-name':
                   cls_name1 = ocr_result[0][1].upper()
                if cls_name == 'brand':
                   cls_brand1 = ocr_result[0][1].upper()
                print(f"Detected {cls_name} :: {detected_text}")

                if cls_name in brand_info:
                    cls_price, net_weight = brand_info[cls_name]
                    count1 += 1
            if cls_name1 and cls_brand1:
              csv_data.append([cls_name1, cls_brand1, net_weight, count1, cls_price])

        writer.writerows(csv_data)


