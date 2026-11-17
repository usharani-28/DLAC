
from pathlib import Path
from ultralytics import YOLO


class VehicleDetector:

    # COCO class IDs
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self, model_path="yolo11n.pt", confidence=0.35):
        print("Loading YOLO model...")

        self.model = YOLO(model_path)
        self.confidence = confidence

        print("YOLO model loaded successfully!")

    def detect(self, source):

        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        results = self.model.predict(
            source=str(source),
            conf=self.confidence,
            classes=list(self.VEHICLE_CLASSES.keys()),
            imgsz=640,
            save=True,
            project="outputs",
            name="vehicle_detection",
            exist_ok=True,
            verbose=False
        )

        counts = {name: 0 for name in self.VEHICLE_CLASSES.values()}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                vehicle_name = self.VEHICLE_CLASSES[class_id]

                counts[vehicle_name] += 1

                print(
                    f"{vehicle_name} | "
                    f"Confidence: {confidence:.2%}"
                )

        print("\n========== VEHICLE SUMMARY ==========")

        for vehicle, count in counts.items():
            print(f"{vehicle}: {count}")

        print("Total detections:", sum(counts.values()))
        print("=====================================")

        print("\nProcessed output saved in outputs/vehicle_detection")

        return results, counts
