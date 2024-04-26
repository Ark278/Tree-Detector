import cv2
from ultralytics import YOLO

model= YOLO(r"train20\weights\best.pt")
model.predict("2048x1365-Oak-trees-SEO-GettyImages-90590330-b6bfe8b.jpg",show = True)
cv2.waitKey(0)