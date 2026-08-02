"""Tests unitaires du détecteur de pose de Karate Coach."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.models import LandmarkName
from src.pose_detector import (
    MEDIAPIPE_LANDMARK_INDEXES,
    PoseDetector,
    PoseDetectorConfig,
    PoseDetectorError,
)


class FakeLandmarker:
    """Faux détecteur reproduisant l'interface MediaPipe utilisée."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.received_timestamps: list[int] = []
        self.closed = False

    def detect_for_video(
        self,
        image: Any,
        timestamp_ms: int,
    ) -> Any:
        """Retourne le résultat configuré."""

        self.received_timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        """Mémorise la fermeture du faux détecteur."""

        self.closed = True


def make_media_pipe_landmark(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    visibility: float = 0.9,
) -> SimpleNamespace:
    """Construit une articulation simulant la sortie MediaPipe."""

    return SimpleNamespace(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
    )


def make_media_pipe_pose() -> list[SimpleNamespace]:
    """Construit les 33 articulations retournées par Pose Landmarker."""

    landmarks = [
        make_media_pipe_landmark()
        for _ in range(33)
    ]

    for internal_name, media_pipe_index in (
        MEDIAPIPE_LANDMARK_INDEXES.items()
    ):
        landmarks[media_pipe_index] = make_media_pipe_landmark(
            x=media_pipe_index / 100.0,
            y=0.4,
            z=-0.1,
            visibility=0.95,
        )

    return landmarks


def make_image() -> np.ndarray:
    """Construit une image OpenCV noire."""

    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_config_rejects_invalid_confidence() -> None:
    """Une confiance hors de la plage 0-1 doit être refusée."""

    with pytest.raises(ValueError):
        PoseDetectorConfig(
            detection_confidence=1.2,
        )


def test_config_rejects_non_positive_pose_count() -> None:
    """Le nombre de poses recherchées doit être positif."""

    with pytest.raises(ValueError):
        PoseDetectorConfig(
            maximum_poses=0,
        )


def test_detect_returns_pose_frame() -> None:
    """Une pose MediaPipe doit être convertie en PoseFrame."""

    media_pipe_pose = make_media_pipe_pose()
    result = SimpleNamespace(
        pose_landmarks=[media_pipe_pose],
    )
    fake_landmarker = FakeLandmarker(result)

    detector = PoseDetector(
        landmarker=fake_landmarker,
    )

    pose_frame = detector.detect(
        make_image(),
        frame_index=4,
        timestamp_seconds=0.2,
    )

    assert pose_frame.frame_index == 4
    assert pose_frame.timestamp_seconds == pytest.approx(0.2)
    assert pose_frame.pose_detected is True
    assert pose_frame.person_count == 1
    assert len(pose_frame.landmarks) == 12

    left_shoulder = pose_frame.get_landmark(
        LandmarkName.LEFT_SHOULDER
    )

    assert left_shoulder is not None
    assert left_shoulder.x == pytest.approx(0.11)
    assert left_shoulder.visibility == pytest.approx(0.95)

    assert fake_landmarker.received_timestamps == [200]


def test_detect_returns_empty_frame_when_no_pose_is_detected() -> None:
    """L'absence de personne ne doit pas générer de faux points."""

    result = SimpleNamespace(
        pose_landmarks=[],
    )
    detector = PoseDetector(
        landmarker=FakeLandmarker(result),
    )

    pose_frame = detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.0,
    )

    assert pose_frame.pose_detected is False
    assert pose_frame.person_count == 0
    assert pose_frame.landmarks == {}


def test_detect_reports_multiple_people() -> None:
    """Le nombre de personnes détectées doit être conservé."""

    pose = make_media_pipe_pose()
    result = SimpleNamespace(
        pose_landmarks=[pose, pose],
    )
    detector = PoseDetector(
        landmarker=FakeLandmarker(result),
    )

    pose_frame = detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.0,
    )

    assert pose_frame.pose_detected is True
    assert pose_frame.person_count == 2


def test_detect_rejects_non_increasing_timestamps() -> None:
    """Les timestamps doivent être strictement croissants."""

    pose = make_media_pipe_pose()
    result = SimpleNamespace(
        pose_landmarks=[pose],
    )
    detector = PoseDetector(
        landmarker=FakeLandmarker(result),
    )

    detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.1,
    )

    with pytest.raises(ValueError):
        detector.detect(
            make_image(),
            frame_index=1,
            timestamp_seconds=0.1,
        )


def test_reset_timestamps_allows_processing_a_new_video() -> None:
    """Le détecteur doit pouvoir être réutilisé après réinitialisation."""

    pose = make_media_pipe_pose()
    result = SimpleNamespace(
        pose_landmarks=[pose],
    )
    detector = PoseDetector(
        landmarker=FakeLandmarker(result),
    )

    detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.1,
    )

    detector.reset_timestamps()

    pose_frame = detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.0,
    )

    assert pose_frame.pose_detected is True


def test_detect_rejects_invalid_image_shape() -> None:
    """Une image sans trois canaux doit être refusée."""

    detector = PoseDetector(
        landmarker=FakeLandmarker(
            SimpleNamespace(pose_landmarks=[])
        ),
    )

    invalid_image = np.zeros(
        (240, 320),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        detector.detect(
            invalid_image,
            frame_index=0,
            timestamp_seconds=0.0,
        )


def test_low_visibility_landmark_is_preserved() -> None:
    """Le détecteur conserve la mesure, même si elle est peu fiable.

    Les calculs géométriques et l'analyseur décideront ensuite de l'ignorer.
    """

    pose = make_media_pipe_pose()
    wrist_index = MEDIAPIPE_LANDMARK_INDEXES[
        LandmarkName.LEFT_WRIST
    ]
    pose[wrist_index].visibility = 0.2

    detector = PoseDetector(
        landmarker=FakeLandmarker(
            SimpleNamespace(pose_landmarks=[pose])
        ),
    )

    pose_frame = detector.detect(
        make_image(),
        frame_index=0,
        timestamp_seconds=0.0,
    )

    left_wrist = pose_frame.get_landmark(
        LandmarkName.LEFT_WRIST
    )

    assert left_wrist is not None
    assert left_wrist.visibility == pytest.approx(0.2)
    assert left_wrist.is_reliable(0.6) is False


def test_draw_pose_preserves_original_dimensions() -> None:
    """Le dessin du squelette ne doit pas modifier la résolution."""

    pose = make_media_pipe_pose()
    detector = PoseDetector(
        landmarker=FakeLandmarker(
            SimpleNamespace(pose_landmarks=[pose])
        ),
    )

    original = make_image()
    pose_frame = detector.detect(
        original,
        frame_index=0,
        timestamp_seconds=0.0,
    )

    annotated = detector.draw_pose(
        original,
        pose_frame,
    )

    assert annotated.shape == original.shape
    assert annotated.dtype == original.dtype
    assert annotated is not original
    assert np.any(annotated != original)


def test_close_closes_underlying_landmarker() -> None:
    """La fermeture doit libérer le détecteur sous-jacent."""

    fake_landmarker = FakeLandmarker(
        SimpleNamespace(pose_landmarks=[])
    )
    detector = PoseDetector(
        landmarker=fake_landmarker,
    )

    detector.close()

    assert fake_landmarker.closed is True

    with pytest.raises(PoseDetectorError):
        detector.detect(
            make_image(),
            frame_index=0,
            timestamp_seconds=0.0,
        )