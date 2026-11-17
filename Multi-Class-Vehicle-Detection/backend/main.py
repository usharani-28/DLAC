
import argparse
from backend.detection.vehicle_detector import VehicleDetector


def main():

    parser = argparse.ArgumentParser(
        description="Multi-Class Vehicle Detection System"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to input image or video"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35
    )

    args = parser.parse_args()

    detector = VehicleDetector(
        model_path="yolo11n.pt",
        confidence=args.confidence
    )

    detector.detect(args.source)


if __name__ == "__main__":
    main()
