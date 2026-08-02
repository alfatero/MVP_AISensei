"""Tests unitaires du traitement vidéo de Karate Coach."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.models import Landmark, LandmarkName, PoseFrame
from src.video_processor import (
    InvalidVideoError,
    UnsupportedVideoFormatError,
    VideoProcessingResult,
    VideoProcessor,
    VideoProcessorConfig,
)


class FakePoseDetector:
    """Détecteur simulé permettant de tester le processeur vidéo."""

    def __init__(
        self,
        *,
        detect_pose: bool = True,
        person_count: int = 1,
    ) -> None:
        self.detect_pose = detect_pose
        self.person_count = person_count
        self.detect_calls: list[
            tuple[int, float, tuple[int, int]]
        ] = []
        self.reset_count = 0

    def reset_timestamps(self) -> None:
        """Mémorise la réinitialisation avant une nouvelle vidéo."""

        self.reset_count += 1

    def detect(
        self,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
        timestamp_seconds: float,
    ) -> PoseFrame:
        """Retourne une pose simple et déterministe."""

        height, width = frame_bgr.shape[:2]

        self.detect_calls.append(
            (
                frame_index,
                timestamp_seconds,
                (width, height),
            )
        )

        landmarks = {}

        if self.detect_pose:
            landmarks = {
                LandmarkName.LEFT_SHOULDER: Landmark(
                    x=0.4,
                    y=0.3,
                    z=0.0,
                    visibility=0.95,
                ),
                LandmarkName.RIGHT_SHOULDER: Landmark(
                    x=0.6,
                    y=0.3,
                    z=0.0,
                    visibility=0.95,
                ),
            }

        return PoseFrame(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            landmarks=landmarks,
            pose_detected=self.detect_pose,
            person_count=(
                self.person_count
                if self.detect_pose
                else 0
            ),
        )

    def draw_pose(
        self,
        frame_bgr: NDArray[np.uint8],
        pose_frame: PoseFrame,
    ) -> NDArray[np.uint8]:
        """Dessine un petit repère pour vérifier l'annotation."""

        annotated = frame_bgr.copy()

        if pose_frame.pose_detected:
            cv2.circle(
                annotated,
                (10, 10),
                5,
                (255, 255, 255),
                thickness=-1,
            )

        return annotated


def create_test_video(
    output_path: Path,
    *,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
    frame_count: int = 5,
) -> Path:
    """Crée une petite vidéo MP4 utilisable par OpenCV."""

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        pytest.skip(
            "Le codec mp4v n'est pas disponible dans cet environnement."
        )

    try:
        for frame_index in range(frame_count):
            frame = np.full(
                (height, width, 3),
                fill_value=frame_index * 20,
                dtype=np.uint8,
            )

            writer.write(frame)
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        pytest.skip(
            "La vidéo de test n'a pas pu être créée."
        )

    return output_path


def test_config_rejects_invalid_codec() -> None:
    """Le codec OpenCV doit contenir quatre caractères."""

    with pytest.raises(ValueError):
        VideoProcessorConfig(
            output_codec="abc",
        )


def test_calculate_processing_size_keeps_small_video_size() -> None:
    """Une petite vidéo ne doit pas être agrandie."""

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    result = processor.calculate_processing_size(
        source_width=640,
        source_height=480,
    )

    assert result == (640, 480)


def test_calculate_processing_size_preserves_landscape_ratio() -> None:
    """Une vidéo Full HD doit être réduite sans être déformée."""

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
        config=VideoProcessorConfig(
            max_processing_width=1280,
            max_processing_height=720,
        ),
    )

    width, height = processor.calculate_processing_size(
        source_width=1920,
        source_height=1080,
    )

    assert width == 1280
    assert height == 720
    assert width / height == pytest.approx(
        1920 / 1080,
        rel=0.01,
    )


def test_calculate_processing_size_preserves_portrait_ratio() -> None:
    """Une vidéo verticale doit être limitée par sa hauteur."""

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
        config=VideoProcessorConfig(
            max_processing_width=1280,
            max_processing_height=720,
        ),
    )

    width, height = processor.calculate_processing_size(
        source_width=1080,
        source_height=1920,
    )

    assert height == 720
    assert width == 404
    assert width % 2 == 0
    assert height % 2 == 0
    assert width / height == pytest.approx(
        1080 / 1920,
        abs=0.01,
    )


def test_calculate_processing_size_returns_even_dimensions() -> None:
    """Les dimensions exportées doivent être paires."""

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
        config=VideoProcessorConfig(
            max_processing_width=1001,
            max_processing_height=701,
        ),
    )

    width, height = processor.calculate_processing_size(
        source_width=1921,
        source_height=1081,
    )

    assert width % 2 == 0
    assert height % 2 == 0


def test_resize_frame_changes_resolution() -> None:
    """L'image doit être redimensionnée à la taille demandée."""

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    resized = VideoProcessor.resize_frame(
        frame,
        target_width=320,
        target_height=240,
    )

    assert resized.shape == (240, 320, 3)
    assert resized.dtype == np.uint8


def test_resize_frame_returns_copy_when_size_is_unchanged() -> None:
    """Une image déjà à la bonne taille doit être copiée."""

    frame = np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )

    resized = VideoProcessor.resize_frame(
        frame,
        target_width=320,
        target_height=240,
    )

    assert resized.shape == frame.shape
    assert resized is not frame


def test_read_metadata_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Une extension non prévue par le MVP doit être refusée."""

    invalid_file = tmp_path / "video.mkv"
    invalid_file.write_bytes(b"not a video")

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    with pytest.raises(UnsupportedVideoFormatError):
        processor.read_metadata(invalid_file)


def test_read_metadata_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Un fichier absent doit générer une erreur explicite."""

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    with pytest.raises(InvalidVideoError):
        processor.read_metadata(
            tmp_path / "missing.mp4"
        )


def test_read_metadata_returns_valid_values(
    tmp_path: Path,
) -> None:
    """Les métadonnées OpenCV doivent être converties en VideoMetadata."""

    source_path = create_test_video(
        tmp_path / "source.mp4",
        width=320,
        height=240,
        fps=10.0,
        frame_count=5,
    )

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    metadata = processor.read_metadata(source_path)

    assert metadata.source_path == source_path.resolve()
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.fps == pytest.approx(10.0, rel=0.1)
    assert metadata.frame_count == pytest.approx(
        5,
        abs=1,
    )
    assert metadata.duration_seconds > 0.0


def test_process_creates_annotated_video(
    tmp_path: Path,
) -> None:
    """Le traitement doit produire une vidéo et les poses associées."""

    source_path = create_test_video(
        tmp_path / "source.mp4",
        width=320,
        height=240,
        fps=10.0,
        frame_count=5,
    )
    output_path = tmp_path / "output.mp4"

    fake_detector = FakePoseDetector(
        detect_pose=True,
        person_count=1,
    )
    processor = VideoProcessor(
        pose_detector=fake_detector,
    )

    progress_values: list[tuple[int, int]] = []

    result = processor.process(
        source_path=source_path,
        output_path=output_path,
        progress_callback=lambda processed, total: (
            progress_values.append(
                (processed, total)
            )
        ),
    )

    assert isinstance(result, VideoProcessingResult)
    assert result.output_path == output_path.resolve()
    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 0

    assert len(result.pose_frames) >= 4
    assert result.detected_pose_frames == len(
        result.pose_frames
    )
    assert result.multiple_people_frames == 0
    assert result.pose_detection_ratio == pytest.approx(1.0)

    assert fake_detector.reset_count == 1
    assert len(fake_detector.detect_calls) == len(
        result.pose_frames
    )

    assert progress_values
    assert progress_values[-1][0] == len(
        result.pose_frames
    )


def test_process_counts_multiple_people_frames(
    tmp_path: Path,
) -> None:
    """Les images comportant plusieurs personnes doivent être comptées."""

    source_path = create_test_video(
        tmp_path / "multiple.mp4",
        frame_count=4,
    )

    fake_detector = FakePoseDetector(
        detect_pose=True,
        person_count=2,
    )
    processor = VideoProcessor(
        pose_detector=fake_detector,
    )

    result = processor.process(
        source_path=source_path,
        output_path=tmp_path / "annotated.mp4",
    )

    assert result.multiple_people_frames == len(
        result.pose_frames
    )
    assert result.has_multiple_people is True


def test_process_can_complete_without_detected_pose(
    tmp_path: Path,
) -> None:
    """L'absence de pose ne doit pas inventer de détection.

    Le processeur exporte la vidéo, mais l'analyseur métier décidera ensuite
    que les données sont insuffisantes.
    """

    source_path = create_test_video(
        tmp_path / "no_pose.mp4",
        frame_count=4,
    )

    fake_detector = FakePoseDetector(
        detect_pose=False,
    )
    processor = VideoProcessor(
        pose_detector=fake_detector,
    )

    result = processor.process(
        source_path=source_path,
        output_path=tmp_path / "annotated.mp4",
    )

    assert result.detected_pose_frames == 0
    assert result.pose_detection_ratio == 0.0
    assert all(
        not frame.pose_detected
        for frame in result.pose_frames
    )


def test_process_normalizes_output_extension(
    tmp_path: Path,
) -> None:
    """L'export doit utiliser l'extension configurée."""

    source_path = create_test_video(
        tmp_path / "source.mp4",
        frame_count=4,
    )

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    result = processor.process(
        source_path=source_path,
        output_path=tmp_path / "result.avi",
    )

    assert result.output_path.suffix == ".mp4"
    assert result.output_path.name == "result.mp4"


def test_progress_callback_error_does_not_stop_processing(
    tmp_path: Path,
) -> None:
    """Une erreur d'affichage de progression ne doit pas casser l'analyse."""

    source_path = create_test_video(
        tmp_path / "source.mp4",
        frame_count=4,
    )

    processor = VideoProcessor(
        pose_detector=FakePoseDetector(),
    )

    def failing_callback(
        processed_frames: int,
        total_frames: int,
    ) -> None:
        raise RuntimeError("Erreur simulée d'interface")

    result = processor.process(
        source_path=source_path,
        output_path=tmp_path / "annotated.mp4",
        progress_callback=failing_callback,
    )

    assert result.output_path.is_file()
    assert result.pose_frames