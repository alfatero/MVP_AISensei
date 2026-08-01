"""Fonctions géométriques réutilisables pour Karate Coach.

Les fonctions de ce module ne produisent aucun diagnostic technique. Elles
transforment uniquement les articulations détectées en mesures géométriques
exploitables par les analyseurs métier.

Toutes les mesures dépendant d'articulations peuvent retourner ``None`` lorsque
les points sont absents, insuffisamment visibles ou lorsque la normalisation
serait impossible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.models import BodySide, Landmark, LandmarkName, PoseFrame


DEFAULT_MINIMUM_VISIBILITY = 0.60
EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """Coordonnées d'un point exprimées en unités de largeur d'épaules."""

    x: float
    y: float
    z: float


def _validate_visibility_threshold(minimum_visibility: float) -> None:
    """Valide un seuil de visibilité avant tout calcul."""

    if not 0.0 <= minimum_visibility <= 1.0:
        raise ValueError(
            "Le seuil minimal de visibilité doit être compris entre 0 et 1."
        )


def _all_reliable(
    landmarks: Sequence[Landmark],
    minimum_visibility: float,
) -> bool:
    """Retourne ``True`` si tous les points sont suffisamment visibles."""

    _validate_visibility_threshold(minimum_visibility)
    return all(
        landmark.is_reliable(minimum_visibility) for landmark in landmarks
    )


def _as_vector_3d(landmark: Landmark) -> np.ndarray:
    """Convertit une articulation en vecteur NumPy tridimensionnel."""

    return np.asarray((landmark.x, landmark.y, landmark.z), dtype=np.float64)


def euclidean_distance(
    first: Landmark,
    second: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
    *,
    include_depth: bool = False,
) -> float | None:
    """Calcule la distance euclidienne entre deux articulations.

    Par défaut, la mesure utilise uniquement ``x`` et ``y`` car la profondeur
    MediaPipe est relative et généralement moins stable qu'une mesure 2D.

    Args:
        first: Premier point.
        second: Second point.
        minimum_visibility: Visibilité minimale exigée pour les deux points.
        include_depth: Inclut la coordonnée ``z`` lorsque ``True``.

    Returns:
        La distance, ou ``None`` si un point n'est pas suffisamment visible.
    """

    if not _all_reliable((first, second), minimum_visibility):
        return None

    first_vector = _as_vector_3d(first)
    second_vector = _as_vector_3d(second)

    if not include_depth:
        first_vector = first_vector[:2]
        second_vector = second_vector[:2]

    return float(np.linalg.norm(first_vector - second_vector))


def calculate_angle(
    first: Landmark,
    vertex: Landmark,
    third: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Calcule l'angle ``first -> vertex -> third`` en degrés.

    L'angle retourné est compris entre 0 et 180 degrés.

    Returns:
        L'angle en degrés, ou ``None`` si les points sont peu fiables ou si
        l'un des deux segments a une longueur pratiquement nulle.
    """

    if not _all_reliable((first, vertex, third), minimum_visibility):
        return None

    first_vector = np.asarray(
        (first.x - vertex.x, first.y - vertex.y),
        dtype=np.float64,
    )
    second_vector = np.asarray(
        (third.x - vertex.x, third.y - vertex.y),
        dtype=np.float64,
    )

    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))

    if first_norm <= EPSILON or second_norm <= EPSILON:
        return None

    cosine = float(
        np.dot(first_vector, second_vector) / (first_norm * second_norm)
    )
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def calculate_segment_inclination(
    start: Landmark,
    end: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Calcule l'inclinaison signée d'un segment par rapport à l'horizontale.

    La convention suit les coordonnées d'image : l'axe ``y`` augmente vers le
    bas. Une valeur positive indique donc un segment descendant vers la droite.

    Returns:
        Un angle dans l'intervalle ``[-180, 180]``, ou ``None`` lorsque la
        mesure n'est pas fiable ou que le segment est dégénéré.
    """

    if not _all_reliable((start, end), minimum_visibility):
        return None

    delta_x = end.x - start.x
    delta_y = end.y - start.y

    if math.hypot(delta_x, delta_y) <= EPSILON:
        return None

    return math.degrees(math.atan2(delta_y, delta_x))


def calculate_vertical_inclination(
    top: Landmark,
    bottom: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Calcule l'écart angulaire absolu d'un segment par rapport à la verticale.

    Une valeur proche de 0 degré correspond à un segment vertical. Le résultat
    est limité à l'intervalle ``[0, 90]``.
    """

    inclination = calculate_segment_inclination(
        top,
        bottom,
        minimum_visibility,
    )
    if inclination is None:
        return None

    deviation = abs(90.0 - abs(inclination))
    return min(deviation, 180.0 - deviation)


def relative_height(
    first: Landmark,
    second: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Compare la hauteur de deux points dans le repère image.

    Le résultat vaut ``second.y - first.y`` :

    - valeur positive : ``first`` est plus haut que ``second`` ;
    - valeur négative : ``first`` est plus bas que ``second`` ;
    - valeur proche de zéro : hauteurs similaires.
    """

    if not _all_reliable((first, second), minimum_visibility):
        return None

    return second.y - first.y


def midpoint(
    first: Landmark,
    second: Landmark,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> Landmark | None:
    """Retourne le milieu de deux articulations fiables."""

    if not _all_reliable((first, second), minimum_visibility):
        return None

    return Landmark(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        z=(first.z + second.z) / 2.0,
        visibility=min(first.visibility, second.visibility),
    )


def calculate_hip_center(
    landmarks: Mapping[LandmarkName, Landmark],
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> Landmark | None:
    """Calcule le centre des hanches gauche et droite."""

    left_hip = landmarks.get(LandmarkName.LEFT_HIP)
    right_hip = landmarks.get(LandmarkName.RIGHT_HIP)

    if left_hip is None or right_hip is None:
        return None

    return midpoint(left_hip, right_hip, minimum_visibility)


def calculate_shoulder_center(
    landmarks: Mapping[LandmarkName, Landmark],
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> Landmark | None:
    """Calcule le centre des épaules gauche et droite."""

    left_shoulder = landmarks.get(LandmarkName.LEFT_SHOULDER)
    right_shoulder = landmarks.get(LandmarkName.RIGHT_SHOULDER)

    if left_shoulder is None or right_shoulder is None:
        return None

    return midpoint(left_shoulder, right_shoulder, minimum_visibility)


def calculate_shoulder_width(
    landmarks: Mapping[LandmarkName, Landmark],
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Retourne la largeur d'épaules en coordonnées normalisées."""

    left_shoulder = landmarks.get(LandmarkName.LEFT_SHOULDER)
    right_shoulder = landmarks.get(LandmarkName.RIGHT_SHOULDER)

    if left_shoulder is None or right_shoulder is None:
        return None

    width = euclidean_distance(
        left_shoulder,
        right_shoulder,
        minimum_visibility,
    )
    if width is None or width <= EPSILON:
        return None

    return width


def normalized_distance(
    first: Landmark,
    second: Landmark,
    reference_length: float,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
    *,
    include_depth: bool = False,
) -> float | None:
    """Calcule une distance divisée par une longueur de référence.

    Pour Karate Coach, la longueur de référence attendue est généralement la
    largeur d'épaules de l'image courante.
    """

    if reference_length <= EPSILON:
        return None

    distance = euclidean_distance(
        first,
        second,
        minimum_visibility,
        include_depth=include_depth,
    )
    if distance is None:
        return None

    return distance / reference_length


def normalize_landmark(
    landmark: Landmark,
    origin: Landmark,
    reference_length: float,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> NormalizedPoint | None:
    """Normalise un point relativement à une origine et une longueur."""

    if reference_length <= EPSILON:
        return None

    if not _all_reliable((landmark, origin), minimum_visibility):
        return None

    return NormalizedPoint(
        x=(landmark.x - origin.x) / reference_length,
        y=(landmark.y - origin.y) / reference_length,
        z=(landmark.z - origin.z) / reference_length,
    )


def normalize_landmarks_by_shoulders(
    landmarks: Mapping[LandmarkName, Landmark],
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> dict[LandmarkName, NormalizedPoint]:
    """Normalise les points fiables autour du centre des épaules.

    Les coordonnées résultantes sont exprimées en largeurs d'épaules. Les
    articulations absentes ou insuffisamment visibles sont ignorées.
    """

    shoulder_center = calculate_shoulder_center(
        landmarks,
        minimum_visibility,
    )
    shoulder_width = calculate_shoulder_width(
        landmarks,
        minimum_visibility,
    )

    if shoulder_center is None or shoulder_width is None:
        return {}

    normalized: dict[LandmarkName, NormalizedPoint] = {}
    for name, landmark in landmarks.items():
        normalized_point = normalize_landmark(
            landmark,
            shoulder_center,
            shoulder_width,
            minimum_visibility,
        )
        if normalized_point is not None:
            normalized[name] = normalized_point

    return normalized


def calculate_speed(
    previous: Landmark,
    current: Landmark,
    delta_seconds: float,
    reference_length: float,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Calcule une vitesse normalisée entre deux images.

    La valeur est exprimée en longueurs de référence par seconde. En pratique,
    la largeur d'épaules sert de référence, ce qui réduit la dépendance à la
    résolution et à la distance entre la caméra et le pratiquant.
    """

    if delta_seconds <= 0.0:
        raise ValueError("L'intervalle temporel doit être strictement positif.")

    distance = normalized_distance(
        previous,
        current,
        reference_length,
        minimum_visibility,
    )
    if distance is None:
        return None

    return distance / delta_seconds


def calculate_landmark_speeds(
    frames: Sequence[PoseFrame],
    landmark_name: LandmarkName,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> tuple[float | None, ...]:
    """Calcule la vitesse d'une articulation entre chaque paire d'images.

    La largeur d'épaules moyenne des deux images successives est utilisée comme
    référence. Le tuple retourné contient ``len(frames) - 1`` valeurs.
    """

    if len(frames) < 2:
        return ()

    speeds: list[float | None] = []

    for previous_frame, current_frame in zip(frames, frames[1:]):
        delta_seconds = (
            current_frame.timestamp_seconds - previous_frame.timestamp_seconds
        )
        if delta_seconds <= 0.0:
            speeds.append(None)
            continue

        previous_landmark = previous_frame.get_landmark(landmark_name)
        current_landmark = current_frame.get_landmark(landmark_name)

        if previous_landmark is None or current_landmark is None:
            speeds.append(None)
            continue

        previous_width = calculate_shoulder_width(
            previous_frame.landmarks,
            minimum_visibility,
        )
        current_width = calculate_shoulder_width(
            current_frame.landmarks,
            minimum_visibility,
        )

        valid_widths = [
            width
            for width in (previous_width, current_width)
            if width is not None and width > EPSILON
        ]
        if not valid_widths:
            speeds.append(None)
            continue

        reference_length = sum(valid_widths) / len(valid_widths)
        speeds.append(
            calculate_speed(
                previous_landmark,
                current_landmark,
                delta_seconds,
                reference_length,
                minimum_visibility,
            )
        )

    return tuple(speeds)


def _side_landmark_names(
    side: BodySide,
) -> tuple[LandmarkName, LandmarkName, LandmarkName]:
    """Retourne épaule, coude et poignet pour un côté du corps."""

    if side is BodySide.LEFT:
        return (
            LandmarkName.LEFT_SHOULDER,
            LandmarkName.LEFT_ELBOW,
            LandmarkName.LEFT_WRIST,
        )

    return (
        LandmarkName.RIGHT_SHOULDER,
        LandmarkName.RIGHT_ELBOW,
        LandmarkName.RIGHT_WRIST,
    )


def arm_extension_distance(
    frame: PoseFrame,
    side: BodySide,
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
) -> float | None:
    """Mesure l'extension poignet-épaule d'un bras en largeurs d'épaules."""

    shoulder_name, _, wrist_name = _side_landmark_names(side)
    shoulder = frame.get_landmark(shoulder_name)
    wrist = frame.get_landmark(wrist_name)

    if shoulder is None or wrist is None:
        return None

    shoulder_width = calculate_shoulder_width(
        frame.landmarks,
        minimum_visibility,
    )
    if shoulder_width is None:
        return None

    return normalized_distance(
        shoulder,
        wrist,
        shoulder_width,
        minimum_visibility,
    )


def detect_striking_side(
    frames: Sequence[PoseFrame],
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY,
    minimum_valid_frames: int = 3,
    minimum_extension_difference: float = 0.10,
) -> BodySide | None:
    """Détermine approximativement le bras qui frappe.

    La décision compare, pour chaque côté, l'extension poignet-épaule maximale
    normalisée sur la séquence. Aucun côté n'est retourné lorsque les données
    sont insuffisantes ou que les extensions maximales sont trop proches.

    Les seuils sont des valeurs initiales configurables à faire valider sur des
    vidéos réelles avec un professeur de karaté.
    """

    _validate_visibility_threshold(minimum_visibility)

    if minimum_valid_frames <= 0:
        raise ValueError(
            "Le nombre minimal d'images valides doit être strictement positif."
        )

    if minimum_extension_difference < 0.0:
        raise ValueError(
            "La différence minimale d'extension ne peut pas être négative."
        )

    extensions: dict[BodySide, list[float]] = {
        BodySide.LEFT: [],
        BodySide.RIGHT: [],
    }

    for frame in frames:
        if not frame.pose_detected:
            continue

        for side in (BodySide.LEFT, BodySide.RIGHT):
            extension = arm_extension_distance(
                frame,
                side,
                minimum_visibility,
            )
            if extension is not None:
                extensions[side].append(extension)

    if any(
        len(side_extensions) < minimum_valid_frames
        for side_extensions in extensions.values()
    ):
        return None

    left_maximum = max(extensions[BodySide.LEFT])
    right_maximum = max(extensions[BodySide.RIGHT])
    difference = abs(left_maximum - right_maximum)

    if difference < minimum_extension_difference:
        return None

    return (
        BodySide.LEFT
        if left_maximum > right_maximum
        else BodySide.RIGHT
    )
