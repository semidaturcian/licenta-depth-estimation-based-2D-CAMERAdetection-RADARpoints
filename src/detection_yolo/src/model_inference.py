from ultralytics import YOLO
import numpy as np
import cv2
import os


class InferenceYoloModel:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.bounding_boxes = []

    def model_inference(self, image):
        results = self.model(image)
        for result in results:
            for box in result.result.boxes:
             self.bounding_boxes.append(box.xywh)
        return self.bounding_boxes
    