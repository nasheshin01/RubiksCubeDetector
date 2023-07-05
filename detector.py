from super_gradients.training import models

class YoloNasDetector:

    def __init__(self) -> None:
        self.model = models.get('yolo_nas_s',
                                num_classes=2,
                                checkpoint_path="ResultModels\\cube_detector_v3.pth")
        self.model.to("cuda")
        
    def detect(self, image, confidence=0.5) -> list:
        boxes_data = list(self.model.predict(image, conf=confidence))[0]

        boxes_count = len(boxes_data.prediction.bboxes_xyxy)
        result = []
        for i in range(boxes_count):
            result.append((boxes_data.prediction.bboxes_xyxy[i], boxes_data.prediction.confidence[i], int(boxes_data.prediction.labels[i])))

        return result
