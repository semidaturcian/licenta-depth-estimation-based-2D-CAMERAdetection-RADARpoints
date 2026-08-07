from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    class_id: int


class InferenceYoloModel:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def model_inference(self, image):

        detections = []

        results = self.model(image)

        for result in results:

            for box in result.boxes:

                xywh = box.xywh[0].cpu().numpy()

                detection = Detection(
                    center_x=float(xywh[0]),
                    center_y=float(xywh[1]),
                    width=float(xywh[2]),
                    height=float(xywh[3]),
                    confidence=float(box.conf.item()),
                    class_id=int(box.cls.item())
                )

                detections.append(detection)

        return detections
    