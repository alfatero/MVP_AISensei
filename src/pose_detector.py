"""Détection des articulations corporelles avec MediaPipe Pose Landmarker.

Ce module constitue l'interface entre :

- les images vidéo lues avec OpenCV ;
- MediaPipe Pose Landmarker ;
- les modèles de données internes de Karate Coach.

Il ne contient aucune règle technique de karaté et ne calcule aucun score.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from src.models import Landmark, LandmarkName, PoseFrame


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: Final[Path] = Path(
    "data/models/pose_landmarker_lite.task"
)

DEFAULT_MINIMUM_VISIBILITY: Final[float] = 0.60
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.50
DEFAULT_PRESENCE_CONFIDENCE: Final[float] = 0.50
DEFAULT_TRACKING_CONFIDENCE: Final[float] = 0.50
DEFAULT_MAXIMUM_POSES: Final[int] = 2

SKELETON_POINT_RADIUS: Final[int] = 4
SKELETON_LINE_THICKNESS: Final[int] = 2

SKELETON_POINT_COLOR: Final[tuple[int, int, int]] = (0, 255, 255)
SKELETON_LINE_COLOR: Final[tuple[int, int, int]] = (0, 200, 0)


class PoseDetectorError(RuntimeError):
    """Erreur liée à l'initialisation ou à l'exécution du détecteur."""


@dataclass(frozen=True, slots=True)
class PoseDetectorConfig:
    """Configuration du détecteur MediaPipe.

    Attributes:
        model_path: Chemin du fichier ``.task`` MediaPipe.
        minimum_visibility: Visibilité minimale utilisée pour le dessin.
        detection_confidence: Confiance minimale de détection d'une pose.
        presence_confidence: Confiance minimale de présence d'une pose.
        tracking_confidence: Confiance minimale du suivi vidéo.
        maximum_poses: Nombre maximal de personnes recherchées.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY
    detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE
    presence_confidence: float = DEFAULT_PRESENCE_CONFIDENCE
    tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE
    maximum_poses: int = DEFAULT_MAXIMUM_POSES

    def __post_init__(self) -> None:
        """Valide la configuration avant l'initialisation de MediaPipe."""

        confidence_values = {
            "minimum_visibility": self.minimum_visibility,
            "detection_confidence": self.detection_confidence,
            "presence_confidence": self.presence_confidence,
            "tracking_confidence": self.tracking_confidence,
        }

        for name, value in confidence_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} doit être compris entre 0 et 1, "
                    f"valeur reçue : {value}."
                )

        if self.maximum_poses <= 0:
            raise ValueError(
                "Le nombre maximal de poses doit être strictement positif."
            )


# Index officiels des articulations MediaPipe Pose Landmarker.
MEDIAPIPE_LANDMARK_INDEXES: Final[dict[LandmarkName, int]] = {
    LandmarkName.LEFT_SHOULDER: 11,
    LandmarkName.RIGHT_SHOULDER: 12,
    LandmarkName.LEFT_ELBOW: 13,
    LandmarkName.RIGHT_ELBOW: 14,
    LandmarkName.LEFT_WRIST: 15,
    LandmarkName.RIGHT_WRIST: 16,
    LandmarkName.LEFT_HIP: 23,
    LandmarkName.RIGHT_HIP: 24,
    LandmarkName.LEFT_KNEE: 25,
    LandmarkName.RIGHT_KNEE: 26,
    LandmarkName.LEFT_ANKLE: 27,
    LandmarkName.RIGHT_ANKLE: 28,
}


SKELETON_CONNECTIONS: Final[
    tuple[tuple[LandmarkName, LandmarkName], ...]
] = (
    (
        LandmarkName.LEFT_SHOULDER,
        LandmarkName.RIGHT_SHOULDER,
    ),
    (
        LandmarkName.LEFT_SHOULDER,
        LandmarkName.LEFT_ELBOW,
    ),
    (
        LandmarkName.LEFT_ELBOW,
        LandmarkName.LEFT_WRIST,
    ),
    (
        LandmarkName.RIGHT_SHOULDER,
        LandmarkName.RIGHT_ELBOW,
    ),
    (
        LandmarkName.RIGHT_ELBOW,
        LandmarkName.RIGHT_WRIST,
    ),
    (
        LandmarkName.LEFT_SHOULDER,
        LandmarkName.LEFT_HIP,
    ),
    (
        LandmarkName.RIGHT_SHOULDER,
        LandmarkName.RIGHT_HIP,
    ),
    (
        LandmarkName.LEFT_HIP,
        LandmarkName.RIGHT_HIP,
    ),
    (
        LandmarkName.LEFT_HIP,
        LandmarkName.LEFT_KNEE,
    ),
    (
        LandmarkName.LEFT_KNEE,
        LandmarkName.LEFT_ANKLE,
    ),
    (
        LandmarkName.RIGHT_HIP,
        LandmarkName.RIGHT_KNEE,
    ),
    (
        LandmarkName.RIGHT_KNEE,
        LandmarkName.RIGHT_ANKLE,
    ),
)


class PoseDetector:
    """Détecte les articulations nécessaires dans des images vidéo.

    Le détecteur fonctionne en mode ``VIDEO``. Les timestamps transmis à
    :meth:`detect` doivent donc être strictement croissants.

    L'argument ``landmarker`` permet l'injection d'un faux détecteur dans les
    tests unitaires. En fonctionnement normal, il doit être laissé à ``None``.
    """

    def __init__(
        self,
        config: PoseDetectorConfig | None = None,
        *,
        landmarker: Any | None = None,
    ) -> None:
        """Initialise le détecteur.

        Args:
            config: Configuration optionnelle.
            landmarker: Implémentation injectée pour les tests.

        Raises:
            FileNotFoundError: Si le modèle MediaPipe est absent.
            PoseDetectorError: Si MediaPipe ne peut pas être initialisé.
        """

        self._config = config or PoseDetectorConfig()
        self._last_timestamp_ms: int | None = None
        self._closed = False

        if landmarker is not None:
            self._landmarker = landmarker
            return

        self._landmarker = self._create_landmarker()

    @property
    def config(self) -> PoseDetectorConfig:
        """Retourne la configuration active."""

        return self._config

    def _create_landmarker(self) -> Any:
        """Construit l'instance MediaPipe Pose Landmarker."""

        model_path = self._config.model_path.expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                "Le modèle MediaPipe est introuvable : "
                f"{model_path}. Télécharge "
                "'pose_landmarker_lite.task' dans 'data/models/'."
            )

        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=self._config.maximum_poses,
                min_pose_detection_confidence=(
                    self._config.detection_confidence
                ),
                min_pose_presence_confidence=(
                    self._config.presence_confidence
                ),
                min_tracking_confidence=(
                    self._config.tracking_confidence
                ),
                output_segmentation_masks=False,
            )

            return mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as error:
            raise PoseDetectorError(
                "MediaPipe Pose Landmarker n'a pas pu être initialisé."
            ) from error

    def detect(
        self,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
        timestamp_seconds: float,
    ) -> PoseFrame:
        """Détecte les articulations présentes dans une image OpenCV.

        Args:
            frame_bgr: Image OpenCV au format BGR.
            frame_index: Position de l'image dans la vidéo.
            timestamp_seconds: Horodatage en secondes.

        Returns:
            Une instance de :class:`PoseFrame`. Lorsqu'aucune personne n'est
            détectée, ``pose_detected`` vaut ``False`` et ``landmarks`` est
            vide.

        Raises:
            ValueError: Si l'image ou les métadonnées sont invalides.
            PoseDetectorError: Si MediaPipe échoue pendant la détection.
        """

        self._ensure_open()
        self._validate_frame(frame_bgr, frame_index, timestamp_seconds)

        timestamp_ms = int(round(timestamp_seconds * 1000.0))
        self._validate_timestamp(timestamp_ms)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        media_pipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        try:
            result = self._landmarker.detect_for_video(
                media_pipe_image,
                timestamp_ms,
            )
        except Exception as error:
            raise PoseDetectorError(
                f"Échec de la détection sur l'image {frame_index}."
            ) from error

        self._last_timestamp_ms = timestamp_ms

        return self._convert_result(
            result=result,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    def _convert_result(
        self,
        result: Any,
        frame_index: int,
        timestamp_seconds: float,
    ) -> PoseFrame:
        """Convertit un résultat MediaPipe en modèle interne."""

        detected_poses = getattr(result, "pose_landmarks", None) or []
        person_count = len(detected_poses)

        if person_count == 0:
            return PoseFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                landmarks={},
                pose_detected=False,
                person_count=0,
            )

        first_pose = detected_poses[0]
        landmarks = self._extract_required_landmarks(first_pose)

        return PoseFrame(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            landmarks=landmarks,
            pose_detected=bool(landmarks),
            person_count=person_count,
        )

    @staticmethod
    def _extract_required_landmarks(
        media_pipe_landmarks: Any,
    ) -> dict[LandmarkName, Landmark]:
        """Extrait uniquement les articulations utiles au Gyaku-zuki."""

        extracted_landmarks: dict[LandmarkName, Landmark] = {}

        for landmark_name, landmark_index in (
            MEDIAPIPE_LANDMARK_INDEXES.items()
        ):
            if landmark_index >= len(media_pipe_landmarks):
                LOGGER.warning(
                    "Le résultat MediaPipe ne contient pas l'articulation "
                    "%s à l'index %d.",
                    landmark_name.value,
                    landmark_index,
                )
                continue

            media_pipe_landmark = media_pipe_landmarks[landmark_index]

            try:
                x = float(media_pipe_landmark.x)
                y = float(media_pipe_landmark.y)
                z = float(media_pipe_landmark.z)
                visibility = float(media_pipe_landmark.visibility)
            except (AttributeError, TypeError, ValueError):
                LOGGER.warning(
                    "Articulation MediaPipe invalide : %s.",
                    landmark_name.value,
                )
                continue

            if not all(math.isfinite(value) for value in (x, y, z)):
                LOGGER.warning(
                    "Coordonnées non finies pour l'articulation %s.",
                    landmark_name.value,
                )
                continue

            if not math.isfinite(visibility):
                visibility = 0.0

            visibility = min(max(visibility, 0.0), 1.0)

            extracted_landmarks[landmark_name] = Landmark(
                x=x,
                y=y,
                z=z,
                visibility=visibility,
            )

        return extracted_landmarks

    def draw_pose(
        self,
        frame_bgr: NDArray[np.uint8],
        pose_frame: PoseFrame,
    ) -> NDArray[np.uint8]:
        """Dessine le squelette fiable sur une copie de l'image.

        Les segments ne sont dessinés que lorsque leurs deux extrémités
        atteignent le seuil de visibilité configuré.

        Args:
            frame_bgr: Image source OpenCV.
            pose_frame: Résultat associé à cette image.

        Returns:
            Une nouvelle image BGR annotée.
        """

        self._validate_image_array(frame_bgr)

        annotated_frame = frame_bgr.copy()

        if not pose_frame.pose_detected:
            return annotated_frame

        height, width = annotated_frame.shape[:2]

        for start_name, end_name in SKELETON_CONNECTIONS:
            start = pose_frame.get_landmark(start_name)
            end = pose_frame.get_landmark(end_name)

            if start is None or end is None:
                continue

            if not (
                start.is_reliable(self._config.minimum_visibility)
                and end.is_reliable(self._config.minimum_visibility)
            ):
                continue

            start_pixel = self._to_pixel(start, width, height)
            end_pixel = self._to_pixel(end, width, height)

            cv2.line(
                annotated_frame,
                start_pixel,
                end_pixel,
                SKELETON_LINE_COLOR,
                SKELETON_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        for landmark in pose_frame.landmarks.values():
            if not landmark.is_reliable(
                self._config.minimum_visibility
            ):
                continue

            cv2.circle(
                annotated_frame,
                self._to_pixel(landmark, width, height),
                SKELETON_POINT_RADIUS,
                SKELETON_POINT_COLOR,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if pose_frame.person_count > 1:
            cv2.putText(
                annotated_frame,
                "Plusieurs personnes detectees",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated_frame

    @staticmethod
    def _to_pixel(
        landmark: Landmark,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Convertit des coordonnées normalisées en pixels valides."""

        pixel_x = int(round(landmark.x * (width - 1)))
        pixel_y = int(round(landmark.y * (height - 1)))

        pixel_x = min(max(pixel_x, 0), width - 1)
        pixel_y = min(max(pixel_y, 0), height - 1)

        return pixel_x, pixel_y

    def reset_timestamps(self) -> None:
        """Réinitialise le suivi temporel avant une nouvelle vidéo."""

        self._last_timestamp_ms = None

    def close(self) -> None:
        """Libère les ressources natives du détecteur MediaPipe."""

        if self._closed:
            return

        close_method = getattr(self._landmarker, "close", None)

        if callable(close_method):
            try:
                close_method()
            except Exception as error:
                LOGGER.warning(
                    "Erreur pendant la fermeture de MediaPipe : %s",
                    error,
                )

        self._closed = True

    def _ensure_open(self) -> None:
        """Empêche l'utilisation d'un détecteur fermé."""

        if self._closed:
            raise PoseDetectorError(
                "Le détecteur de pose a déjà été fermé."
            )

    def _validate_timestamp(self, timestamp_ms: int) -> None:
        """Vérifie l'ordre temporel exigé par le mode vidéo."""

        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError(
                "Les timestamps MediaPipe doivent être strictement "
                "croissants. Appelez reset_timestamps() avant une "
                "nouvelle vidéo."
            )

    @classmethod
    def _validate_frame(
        cls,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Valide l'image et ses informations temporelles."""

        cls._validate_image_array(frame_bgr)

        if frame_index < 0:
            raise ValueError(
                "L'index de l'image ne peut pas être négatif."
            )

        if not math.isfinite(timestamp_seconds):
            raise ValueError(
                "L'horodatage doit être une valeur finie."
            )

        if timestamp_seconds < 0.0:
            raise ValueError(
                "L'horodatage ne peut pas être négatif."
            )

    @staticmethod
    def _validate_image_array(
        frame_bgr: NDArray[np.uint8],
    ) -> None:
        """Vérifie que l'entrée correspond à une image OpenCV BGR."""

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(
                "L'image doit être un tableau NumPy."
            )

        if frame_bgr.dtype != np.uint8:
            raise ValueError(
                "L'image OpenCV doit utiliser le type uint8."
            )

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                "L'image doit avoir la forme (hauteur, largeur, 3)."
            )

        if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            raise ValueError(
                "L'image ne peut pas avoir une dimension nulle."
            )

    def __enter__(self) -> "PoseDetector":
        """Permet l'utilisation avec un gestionnaire de contexte."""

        self._ensure_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any | None,
    ) -> None:
        """Ferme automatiquement MediaPipe en sortie de contexte."""

        self.close()