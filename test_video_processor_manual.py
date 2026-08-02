"""Test manuel du traitement d'une vraie vidéo."""

from pathlib import Path

from src.pose_detector import PoseDetector
from src.video_processor import VideoProcessor


SOURCE_PATH = Path("data/uploads/test.mp4")
OUTPUT_PATH = Path("data/outputs/test_annotated.mp4")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PoseDetector() as detector:
        processor = VideoProcessor(detector)

        result = processor.process(
            source_path=SOURCE_PATH,
            output_path=OUTPUT_PATH,
            progress_callback=lambda current, total: print(
                f"\rTraitement : {current}/{total}",
                end="",
                flush=True,
            ),
        )

    print()
    print(f"Vidéo créée : {result.output_path}")
    print(f"Images traitées : {len(result.pose_frames)}")
    print(
        "Images avec pose : "
        f"{result.detected_pose_frames}"
    )
    print(
        "Taux de détection : "
        f"{result.pose_detection_ratio:.1%}"
    )
    print(
        "Plusieurs personnes : "
        f"{result.has_multiple_people}"
    )


if __name__ == "__main__":
    main()