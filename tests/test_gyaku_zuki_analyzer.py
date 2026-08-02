"""Tests de l'analyseur géométrique du Gyaku-zuki."""

from __future__ import annotations

import pytest

from src.gyaku_zuki_analyzer import (
    GyakuZukiAnalyzer,
    ObservationName,
)
from src.models import BodySide, Landmark, LandmarkName, PoseFrame


def make_landmark(
    x: float,
    y: float,
    *,
    visibility: float = 0.95,
) -> Landmark:
    """Construit une articulation synthétique."""

    return Landmark(
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
    )


def make_frame(
    index: int,
    left_wrist_x: float,
    *,
    visibility: float = 0.95,
) -> PoseFrame:
    """Construit une pose complète avec un bras gauche mobile."""

    timestamp = index / 10.0

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(
            0.40,
            0.30,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_SHOULDER: make_landmark(
            0.60,
            0.30,
            visibility=visibility,
        ),
        LandmarkName.LEFT_ELBOW: make_landmark(
            (0.40 + left_wrist_x) / 2.0,
            0.30,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_ELBOW: make_landmark(
            0.68,
            0.36,
            visibility=visibility,
        ),
        LandmarkName.LEFT_WRIST: make_landmark(
            left_wrist_x,
            0.30,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_WRIST: make_landmark(
            0.63,
            0.42,
            visibility=visibility,
        ),
        LandmarkName.LEFT_HIP: make_landmark(
            0.43,
            0.60,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_HIP: make_landmark(
            0.57,
            0.60,
            visibility=visibility,
        ),
        LandmarkName.LEFT_KNEE: make_landmark(
            0.42,
            0.78,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_KNEE: make_landmark(
            0.62,
            0.77,
            visibility=visibility,
        ),
        LandmarkName.LEFT_ANKLE: make_landmark(
            0.38,
            0.95,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_ANKLE: make_landmark(
            0.66,
            0.94,
            visibility=visibility,
        ),
    }

    return PoseFrame(
        frame_index=index,
        timestamp_seconds=timestamp,
        landmarks=landmarks,
        pose_detected=True,
        person_count=1,
    )


def make_left_punch_sequence() -> tuple[PoseFrame, ...]:
    """Crée une extension puis un retour du bras gauche."""

    wrist_positions = (
        0.34,
        0.25,
        0.12,
        -0.02,
        -0.10,
        -0.04,
        0.10,
        0.25,
    )

    return tuple(
        make_frame(index, wrist_x)
        for index, wrist_x in enumerate(wrist_positions)
    )


def test_analyzer_identifies_left_striking_side() -> None:
    """Le bras gauche doit être identifié comme bras qui frappe."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()
    )

    assert result.success is True
    assert result.striking_side is BodySide.LEFT
    assert result.impact is not None


def test_analyzer_selects_extended_impact_frame() -> None:
    """L'impact doit se situer près de l'extension maximale."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()
    )

    assert result.impact is not None
    assert result.impact.frame_index in {3, 4, 5}
    assert result.impact.elbow_angle_degrees == pytest.approx(
        180.0,
        abs=1.0,
    )


def test_analysis_contains_all_observations() -> None:
    """Les six groupes de mesures doivent être retournés."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()
    )

    observation_names = {
        observation.name
        for observation in result.observations
    }

    assert observation_names == {
        ObservationName.ARM_EXTENSION,
        ObservationName.PUNCH_HEIGHT,
        ObservationName.TORSO_INCLINATION,
        ObservationName.OPPOSITE_GUARD,
        ObservationName.LEGS,
        ObservationName.RETURN_TO_GUARD,
    }


def test_return_to_guard_is_detected() -> None:
    """Le poignet revenant vers l'épaule doit produire une mesure positive."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()
    )

    return_observation = next(
        observation
        for observation in result.observations
        if observation.name is ObservationName.RETURN_TO_GUARD
    )

    assert return_observation.reliable is True
    assert return_observation.measured_value is not None
    assert return_observation.measured_value > 0.20


def test_too_short_sequence_fails() -> None:
    """Une séquence trop courte ne doit pas générer d'analyse."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()[:3]
    )

    assert result.success is False
    assert result.impact is None
    assert result.warnings


def test_low_visibility_prevents_analysis() -> None:
    """Des articulations insuffisamment visibles ne doivent pas être analysées."""

    frames = tuple(
        make_frame(
            index,
            left_wrist_x,
            visibility=0.30,
        )
        for index, left_wrist_x in enumerate(
            (0.34, 0.25, 0.12, -0.02, -0.10, 0.10)
        )
    )

    analyzer = GyakuZukiAnalyzer()
    result = analyzer.analyze(frames)

    assert result.success is False
    assert result.impact is None


def test_context_frames_surround_impact() -> None:
    """Quelques images avant et après l'impact doivent être conservées."""

    analyzer = GyakuZukiAnalyzer()

    result = analyzer.analyze(
        make_left_punch_sequence()
    )

    assert result.impact is not None
    assert result.context_frames

    context_indexes = {
        frame.frame_index
        for frame in result.context_frames
    }

    assert result.impact.frame_index in context_indexes