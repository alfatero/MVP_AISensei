"""Tests unitaires des fonctions géométriques de Karate Coach."""

from __future__ import annotations

import pytest

from src.geometry import (
    arm_extension_distance,
    calculate_angle,
    calculate_hip_center,
    calculate_landmark_speeds,
    calculate_segment_inclination,
    calculate_shoulder_center,
    calculate_shoulder_width,
    calculate_speed,
    calculate_vertical_inclination,
    detect_striking_side,
    euclidean_distance,
    midpoint,
    normalize_landmark,
    normalize_landmarks_by_shoulders,
    normalized_distance,
    relative_height,
)
from src.models import BodySide, Landmark, LandmarkName, PoseFrame


def make_landmark(
    x: float,
    y: float,
    z: float = 0.0,
    visibility: float = 1.0,
) -> Landmark:
    """Construit un point corporel fiable par défaut."""

    return Landmark(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
    )


def make_pose_frame(
    frame_index: int,
    timestamp_seconds: float,
    *,
    left_wrist_x: float,
    right_wrist_x: float,
    visibility: float = 1.0,
    pose_detected: bool = True,
) -> PoseFrame:
    """Construit une image de pose simple avec les deux bras visibles."""

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(
            0.4,
            0.3,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_SHOULDER: make_landmark(
            0.6,
            0.3,
            visibility=visibility,
        ),
        LandmarkName.LEFT_ELBOW: make_landmark(
            (0.4 + left_wrist_x) / 2.0,
            0.3,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_ELBOW: make_landmark(
            (0.6 + right_wrist_x) / 2.0,
            0.3,
            visibility=visibility,
        ),
        LandmarkName.LEFT_WRIST: make_landmark(
            left_wrist_x,
            0.3,
            visibility=visibility,
        ),
        LandmarkName.RIGHT_WRIST: make_landmark(
            right_wrist_x,
            0.3,
            visibility=visibility,
        ),
    }

    return PoseFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        landmarks=landmarks,
        pose_detected=pose_detected,
        person_count=1 if pose_detected else 0,
    )


def test_euclidean_distance_in_2d() -> None:
    """La distance entre (0, 0) et (3, 4) doit être égale à 5."""

    first = make_landmark(0.0, 0.0)
    second = make_landmark(3.0, 4.0)

    result = euclidean_distance(first, second)

    assert result == pytest.approx(5.0)


def test_euclidean_distance_can_include_depth() -> None:
    """La distance peut inclure la profondeur z lorsqu'elle est demandée."""

    first = make_landmark(0.0, 0.0, 0.0)
    second = make_landmark(1.0, 2.0, 2.0)

    result = euclidean_distance(
        first,
        second,
        include_depth=True,
    )

    assert result == pytest.approx(3.0)


def test_euclidean_distance_returns_none_for_low_visibility() -> None:
    """Une articulation insuffisamment visible invalide la distance."""

    first = make_landmark(
        0.0,
        0.0,
        visibility=0.59,
    )
    second = make_landmark(1.0, 0.0)

    result = euclidean_distance(first, second)

    assert result is None


def test_calculate_angle_returns_90_degrees() -> None:
    """Trois points formant un angle droit doivent produire 90 degrés."""

    first = make_landmark(1.0, 0.0)
    vertex = make_landmark(0.0, 0.0)
    third = make_landmark(0.0, 1.0)

    result = calculate_angle(
        first,
        vertex,
        third,
    )

    assert result == pytest.approx(90.0)


def test_calculate_angle_returns_180_degrees() -> None:
    """Trois points alignés dans des directions opposées donnent 180 degrés."""

    first = make_landmark(-1.0, 0.0)
    vertex = make_landmark(0.0, 0.0)
    third = make_landmark(1.0, 0.0)

    result = calculate_angle(
        first,
        vertex,
        third,
    )

    assert result == pytest.approx(180.0)


def test_calculate_angle_returns_none_for_zero_length_segment() -> None:
    """Un segment de longueur nulle ne doit pas produire un faux angle."""

    first = make_landmark(0.0, 0.0)
    vertex = make_landmark(0.0, 0.0)
    third = make_landmark(1.0, 0.0)

    result = calculate_angle(
        first,
        vertex,
        third,
    )

    assert result is None


def test_invalid_visibility_threshold_raises_value_error() -> None:
    """Un seuil de visibilité supérieur à 1 doit être rejeté."""

    first = make_landmark(0.0, 0.0)
    second = make_landmark(1.0, 0.0)

    with pytest.raises(ValueError):
        euclidean_distance(
            first,
            second,
            minimum_visibility=1.1,
        )


def test_calculate_segment_inclination_for_horizontal_segment() -> None:
    """Un segment horizontal orienté à droite a une inclinaison de 0 degré."""

    start = make_landmark(0.0, 0.0)
    end = make_landmark(1.0, 0.0)

    result = calculate_segment_inclination(
        start,
        end,
    )

    assert result == pytest.approx(0.0)


def test_calculate_segment_inclination_for_image_diagonal() -> None:
    """Une diagonale descendante dans l'image produit environ 45 degrés."""

    start = make_landmark(0.0, 0.0)
    end = make_landmark(1.0, 1.0)

    result = calculate_segment_inclination(
        start,
        end,
    )

    assert result == pytest.approx(45.0)


def test_calculate_vertical_inclination_for_vertical_segment() -> None:
    """Un segment vertical doit avoir un écart nul avec la verticale."""

    top = make_landmark(0.5, 0.2)
    bottom = make_landmark(0.5, 0.8)

    result = calculate_vertical_inclination(
        top,
        bottom,
    )

    assert result == pytest.approx(0.0)


def test_calculate_vertical_inclination_for_horizontal_segment() -> None:
    """Un segment horizontal a un écart de 90 degrés avec la verticale."""

    first = make_landmark(0.2, 0.5)
    second = make_landmark(0.8, 0.5)

    result = calculate_vertical_inclination(
        first,
        second,
    )

    assert result == pytest.approx(90.0)


def test_relative_height_uses_image_coordinate_convention() -> None:
    """Un point placé plus haut possède une coordonnée y plus petite."""

    higher = make_landmark(0.5, 0.2)
    lower = make_landmark(0.5, 0.7)

    result = relative_height(
        higher,
        lower,
    )

    assert result == pytest.approx(0.5)


def test_midpoint_averages_coordinates_and_visibility() -> None:
    """Le milieu moyenne les coordonnées et garde la visibilité minimale."""

    first = make_landmark(
        0.2,
        0.4,
        0.1,
        visibility=0.8,
    )
    second = make_landmark(
        0.6,
        0.8,
        0.3,
        visibility=0.9,
    )

    result = midpoint(first, second)

    assert result is not None
    assert result.x == pytest.approx(0.4)
    assert result.y == pytest.approx(0.6)
    assert result.z == pytest.approx(0.2)
    assert result.visibility == pytest.approx(0.8)


def test_calculate_hip_center() -> None:
    """Le centre des hanches doit être situé entre les deux hanches."""

    landmarks = {
        LandmarkName.LEFT_HIP: make_landmark(0.4, 0.7),
        LandmarkName.RIGHT_HIP: make_landmark(0.6, 0.7),
    }

    result = calculate_hip_center(landmarks)

    assert result is not None
    assert result.x == pytest.approx(0.5)
    assert result.y == pytest.approx(0.7)


def test_calculate_hip_center_returns_none_when_one_hip_is_missing() -> None:
    """Une hanche manquante empêche le calcul fiable du centre."""

    landmarks = {
        LandmarkName.LEFT_HIP: make_landmark(0.4, 0.7),
    }

    result = calculate_hip_center(landmarks)

    assert result is None


def test_calculate_shoulder_center_and_width() -> None:
    """Le centre et la largeur des épaules doivent être correctement calculés."""

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(0.3, 0.2),
        LandmarkName.RIGHT_SHOULDER: make_landmark(0.7, 0.2),
    }

    center = calculate_shoulder_center(landmarks)
    width = calculate_shoulder_width(landmarks)

    assert center is not None
    assert center.x == pytest.approx(0.5)
    assert center.y == pytest.approx(0.2)
    assert width == pytest.approx(0.4)


def test_calculate_shoulder_width_returns_none_for_identical_points() -> None:
    """Deux épaules confondues ne doivent pas produire une largeur nulle."""

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(0.5, 0.3),
        LandmarkName.RIGHT_SHOULDER: make_landmark(0.5, 0.3),
    }

    result = calculate_shoulder_width(landmarks)

    assert result is None


def test_normalized_distance() -> None:
    """Une distance de 0.4 divisée par une référence de 0.2 vaut 2."""

    first = make_landmark(0.0, 0.0)
    second = make_landmark(0.4, 0.0)

    result = normalized_distance(
        first,
        second,
        reference_length=0.2,
    )

    assert result == pytest.approx(2.0)


def test_normalized_distance_returns_none_for_zero_reference() -> None:
    """Une référence nulle ne doit pas provoquer de division par zéro."""

    first = make_landmark(0.0, 0.0)
    second = make_landmark(0.4, 0.0)

    result = normalized_distance(
        first,
        second,
        reference_length=0.0,
    )

    assert result is None


def test_normalize_landmark_relative_to_origin() -> None:
    """Un point doit être exprimé relativement à l'origine et à la référence."""

    origin = make_landmark(0.4, 0.3, 0.0)
    point = make_landmark(0.6, 0.5, 0.1)

    result = normalize_landmark(
        point,
        origin,
        reference_length=0.2,
    )

    assert result is not None
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(1.0)
    assert result.z == pytest.approx(0.5)


def test_normalize_landmarks_by_shoulders() -> None:
    """Les coordonnées sont exprimées autour du centre des épaules."""

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(0.4, 0.3),
        LandmarkName.RIGHT_SHOULDER: make_landmark(0.6, 0.3),
        LandmarkName.LEFT_WRIST: make_landmark(0.2, 0.3),
    }

    result = normalize_landmarks_by_shoulders(landmarks)

    assert result[
        LandmarkName.LEFT_SHOULDER
    ].x == pytest.approx(-0.5)

    assert result[
        LandmarkName.RIGHT_SHOULDER
    ].x == pytest.approx(0.5)

    assert result[
        LandmarkName.LEFT_WRIST
    ].x == pytest.approx(-1.5)


def test_normalize_landmarks_ignores_unreliable_point() -> None:
    """Un point insuffisamment visible est exclu du résultat normalisé."""

    landmarks = {
        LandmarkName.LEFT_SHOULDER: make_landmark(0.4, 0.3),
        LandmarkName.RIGHT_SHOULDER: make_landmark(0.6, 0.3),
        LandmarkName.LEFT_WRIST: make_landmark(
            0.2,
            0.3,
            visibility=0.4,
        ),
    }

    result = normalize_landmarks_by_shoulders(landmarks)

    assert LandmarkName.LEFT_WRIST not in result


def test_calculate_speed_is_normalized_by_reference_length() -> None:
    """La vitesse doit être exprimée en longueurs de référence par seconde."""

    previous = make_landmark(0.0, 0.0)
    current = make_landmark(0.2, 0.0)

    result = calculate_speed(
        previous,
        current,
        delta_seconds=0.5,
        reference_length=0.2,
    )

    assert result == pytest.approx(2.0)


@pytest.mark.parametrize(
    "delta_seconds",
    [0.0, -0.1],
)
def test_calculate_speed_rejects_non_positive_time(
    delta_seconds: float,
) -> None:
    """Une durée nulle ou négative doit provoquer une erreur explicite."""

    previous = make_landmark(0.0, 0.0)
    current = make_landmark(0.2, 0.0)

    with pytest.raises(ValueError):
        calculate_speed(
            previous,
            current,
            delta_seconds=delta_seconds,
            reference_length=0.2,
        )


def test_calculate_landmark_speeds_for_two_frames() -> None:
    """La vitesse d'un poignet doit être calculée entre deux images."""

    first = make_pose_frame(
        0,
        0.0,
        left_wrist_x=0.3,
        right_wrist_x=0.7,
    )
    second = make_pose_frame(
        1,
        0.1,
        left_wrist_x=0.2,
        right_wrist_x=0.7,
    )

    result = calculate_landmark_speeds(
        (first, second),
        LandmarkName.LEFT_WRIST,
    )

    assert len(result) == 1
    assert result[0] == pytest.approx(5.0)


def test_calculate_landmark_speeds_returns_empty_for_one_frame() -> None:
    """Une seule image ne permet pas de calculer une vitesse."""

    frame = make_pose_frame(
        0,
        0.0,
        left_wrist_x=0.3,
        right_wrist_x=0.7,
    )

    result = calculate_landmark_speeds(
        (frame,),
        LandmarkName.LEFT_WRIST,
    )

    assert result == ()


def test_calculate_landmark_speeds_returns_none_for_invalid_timestamps() -> None:
    """Deux images ayant le même horodatage ne donnent pas de vitesse."""

    first = make_pose_frame(
        0,
        0.1,
        left_wrist_x=0.3,
        right_wrist_x=0.7,
    )
    second = make_pose_frame(
        1,
        0.1,
        left_wrist_x=0.2,
        right_wrist_x=0.7,
    )

    result = calculate_landmark_speeds(
        (first, second),
        LandmarkName.LEFT_WRIST,
    )

    assert result == (None,)


def test_arm_extension_distance_uses_shoulder_width() -> None:
    """L'extension du bras doit être normalisée par la largeur d'épaules."""

    frame = make_pose_frame(
        0,
        0.0,
        left_wrist_x=0.0,
        right_wrist_x=0.7,
    )

    result = arm_extension_distance(
        frame,
        BodySide.LEFT,
    )

    assert result == pytest.approx(2.0)


def test_detect_striking_side_returns_left() -> None:
    """Le bras gauche est sélectionné lorsqu'il est nettement plus étendu."""

    frames = (
        make_pose_frame(
            0,
            0.0,
            left_wrist_x=0.30,
            right_wrist_x=0.70,
        ),
        make_pose_frame(
            1,
            0.1,
            left_wrist_x=0.15,
            right_wrist_x=0.70,
        ),
        make_pose_frame(
            2,
            0.2,
            left_wrist_x=0.00,
            right_wrist_x=0.70,
        ),
    )

    result = detect_striking_side(frames)

    assert result is BodySide.LEFT


def test_detect_striking_side_returns_right() -> None:
    """Le bras droit est sélectionné lorsqu'il est nettement plus étendu."""

    frames = (
        make_pose_frame(
            0,
            0.0,
            left_wrist_x=0.30,
            right_wrist_x=0.70,
        ),
        make_pose_frame(
            1,
            0.1,
            left_wrist_x=0.30,
            right_wrist_x=0.85,
        ),
        make_pose_frame(
            2,
            0.2,
            left_wrist_x=0.30,
            right_wrist_x=1.00,
        ),
    )

    result = detect_striking_side(frames)

    assert result is BodySide.RIGHT


def test_detect_striking_side_returns_none_when_ambiguous() -> None:
    """Des extensions similaires ne doivent pas produire une fausse décision."""

    frames = (
        make_pose_frame(
            0,
            0.0,
            left_wrist_x=0.2,
            right_wrist_x=0.8,
        ),
        make_pose_frame(
            1,
            0.1,
            left_wrist_x=0.2,
            right_wrist_x=0.8,
        ),
        make_pose_frame(
            2,
            0.2,
            left_wrist_x=0.2,
            right_wrist_x=0.8,
        ),
    )

    result = detect_striking_side(frames)

    assert result is None


def test_detect_striking_side_returns_none_with_too_few_frames() -> None:
    """Le nombre minimal d'images fiables doit être respecté."""

    frames = (
        make_pose_frame(
            0,
            0.0,
            left_wrist_x=0.0,
            right_wrist_x=0.7,
        ),
        make_pose_frame(
            1,
            0.1,
            left_wrist_x=0.0,
            right_wrist_x=0.7,
        ),
    )

    result = detect_striking_side(frames)

    assert result is None


def test_detect_striking_side_ignores_frames_without_pose() -> None:
    """Une image sans pose détectée ne doit pas compter comme mesure fiable."""

    frames = (
        make_pose_frame(
            0,
            0.0,
            left_wrist_x=0.0,
            right_wrist_x=0.7,
            pose_detected=False,
        ),
        make_pose_frame(
            1,
            0.1,
            left_wrist_x=0.0,
            right_wrist_x=0.7,
        ),
        make_pose_frame(
            2,
            0.2,
            left_wrist_x=0.0,
            right_wrist_x=0.7,
        ),
    )

    result = detect_striking_side(frames)

    assert result is None


def test_detect_striking_side_rejects_invalid_configuration() -> None:
    """Les valeurs de configuration invalides doivent être rejetées."""

    with pytest.raises(ValueError):
        detect_striking_side(
            (),
            minimum_valid_frames=0,
        )

    with pytest.raises(ValueError):
        detect_striking_side(
            (),
            minimum_extension_difference=-0.1,
        )