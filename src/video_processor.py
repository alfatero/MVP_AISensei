"""Lecture, traitement et export des vidéos de Karate Coach.

Ce module orchestre :

- l'ouverture d'une vidéo avec OpenCV ;
- la lecture de ses métadonnées ;
- le redimensionnement sans déformation ;
- la détection de pose image par image ;
- le dessin du squelette ;
- l'export d'une vidéo annotée.

Il ne contient aucune règle technique propre au Gyaku-zuki et ne calcule
aucun score.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import PoseFrame, VideoMetadata
from src.pose_detector import PoseDetector, PoseDetectorError


LOGGER = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
    }
)

DEFAULT_MAX_PROCESSING_WIDTH: Final[int] = 1280
DEFAULT_MAX_PROCESSING_HEIGHT: Final[int] = 720
DEFAULT_OUTPUT_CODEC: Final[str] = "mp4v"
DEFAULT_OUTPUT_EXTENSION: Final[str] = ".mp4"
DEFAULT_MINIMUM_VALID_FPS: Final[float] = 1.0
DEFAULT_MAXIMUM_VALID_FPS: Final[float] = 240.0
DEFAULT_MINIMUM_FRAME_COUNT: Final[int] = 2


class VideoProcessorError(RuntimeError):
    """Erreur générique pendant le traitement d'une vidéo."""


class InvalidVideoError(VideoProcessorError):
    """Vidéo absente, illisible ou contenant des métadonnées invalides."""


class UnsupportedVideoFormatError(VideoProcessorError):
    """Extension vidéo non prise en charge par le MVP."""


class VideoExportError(VideoProcessorError):
    """Impossible de créer ou d'écrire la vidéo de sortie."""


class ProgressCallback(Protocol):
    """Signature d'une fonction recevant l'avancement du traitement."""

    def __call__(
        self,
        processed_frames: int,
        total_frames: int,
    ) -> None:
        """Informe l'appelant de l'avancement courant."""


@dataclass(frozen=True, slots=True)
class VideoProcessorConfig:
    """Configuration technique du traitement vidéo.

    Attributes:
        max_processing_width: Largeur maximale utilisée pour l'analyse.
        max_processing_height: Hauteur maximale utilisée pour l'analyse.
        output_codec: Codec OpenCV sur quatre caractères.
        output_extension: Extension du fichier exporté.
        minimum_frame_count: Nombre minimal d'images attendu.
    """

    max_processing_width: int = DEFAULT_MAX_PROCESSING_WIDTH
    max_processing_height: int = DEFAULT_MAX_PROCESSING_HEIGHT
    output_codec: str = DEFAULT_OUTPUT_CODEC
    output_extension: str = DEFAULT_OUTPUT_EXTENSION
    minimum_frame_count: int = DEFAULT_MINIMUM_FRAME_COUNT

    def __post_init__(self) -> None:
        if self.max_processing_width <= 0:
            raise ValueError(
                "La largeur maximale de traitement doit être positive."
            )

        if self.max_processing_height <= 0:
            raise ValueError(
                "La hauteur maximale de traitement doit être positive."
            )

        if len(self.output_codec) != 4:
            raise ValueError(
                "Le codec vidéo doit contenir exactement quatre caractères."
            )

        if not self.output_extension.startswith("."):
            raise ValueError(
                "L'extension de sortie doit commencer par un point."
            )

        if self.minimum_frame_count <= 0:
            raise ValueError(
                "Le nombre minimal d'images doit être strictement positif."
            )


@dataclass(frozen=True, slots=True)
class VideoProcessingResult:
    """Résultat technique du traitement d'une vidéo.

    Attributes:
        metadata: Métadonnées de la vidéo source.
        output_path: Chemin de la vidéo annotée.
        pose_frames: Résultats de détection associés aux images traitées.
        processed_width: Largeur utilisée pour la détection et l'export.
        processed_height: Hauteur utilisée pour la détection et l'export.
        detected_pose_frames: Nombre d'images contenant au moins une pose.
        multiple_people_frames: Nombre d'images contenant plusieurs poses.
    """

    metadata: VideoMetadata
    output_path: Path
    pose_frames: tuple[PoseFrame, ...]
    processed_width: int
    processed_height: int
    detected_pose_frames: int
    multiple_people_frames: int

    def __post_init__(self) -> None:
        if self.processed_width <= 0 or self.processed_height <= 0:
            raise ValueError(
                "La résolution traitée doit être strictement positive."
            )

        if self.detected_pose_frames < 0:
            raise ValueError(
                "Le nombre d'images avec pose ne peut pas être négatif."
            )

        if self.multiple_people_frames < 0:
            raise ValueError(
                "Le nombre d'images avec plusieurs personnes ne peut pas "
                "être négatif."
            )

        if self.detected_pose_frames > len(self.pose_frames):
            raise ValueError(
                "Le nombre d'images avec pose ne peut pas dépasser le nombre "
                "total d'images traitées."
            )

        if self.multiple_people_frames > len(self.pose_frames):
            raise ValueError(
                "Le nombre d'images avec plusieurs personnes ne peut pas "
                "dépasser le nombre total d'images traitées."
            )

    @property
    def pose_detection_ratio(self) -> float:
        """Retourne la proportion d'images contenant une pose."""

        if not self.pose_frames:
            return 0.0

        return self.detected_pose_frames / len(self.pose_frames)

    @property
    def has_multiple_people(self) -> bool:
        """Indique si plusieurs personnes ont été détectées au moins une fois."""

        return self.multiple_people_frames > 0


class VideoProcessor:
    """Traite une vidéo complète avec OpenCV et MediaPipe."""

    def __init__(
        self,
        pose_detector: PoseDetector,
        config: VideoProcessorConfig | None = None,
    ) -> None:
        """Initialise le processeur.

        Args:
            pose_detector: Détecteur MediaPipe déjà initialisé.
            config: Configuration technique optionnelle.
        """

        self._pose_detector = pose_detector
        self._config = config or VideoProcessorConfig()

    @property
    def config(self) -> VideoProcessorConfig:
        """Retourne la configuration active."""

        return self._config

    def read_metadata(self, source_path: Path | str) -> VideoMetadata:
        """Lit et valide les informations techniques d'une vidéo.

        Args:
            source_path: Chemin de la vidéo source.

        Returns:
            Métadonnées validées.

        Raises:
            UnsupportedVideoFormatError: Si l'extension est refusée.
            InvalidVideoError: Si la vidéo est absente ou illisible.
        """

        validated_path = self._validate_source_path(source_path)
        capture = cv2.VideoCapture(str(validated_path))

        if not capture.isOpened():
            capture.release()
            raise InvalidVideoError(
                f"La vidéo ne peut pas être ouverte : {validated_path}."
            )

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

            self._validate_video_properties(
                fps=fps,
                width=width,
                height=height,
                frame_count=frame_count,
            )

            duration_seconds = frame_count / fps

            return VideoMetadata(
                source_path=validated_path,
                fps=fps,
                width=width,
                height=height,
                frame_count=frame_count,
                duration_seconds=duration_seconds,
            )
        finally:
            capture.release()

    def process(
        self,
        source_path: Path | str,
        output_path: Path | str,
        progress_callback: ProgressCallback | None = None,
    ) -> VideoProcessingResult:
        """Analyse et annote toutes les images d'une vidéo.

        La vidéo de sortie utilise la résolution de traitement. Elle peut donc
        être plus petite que la vidéo source, mais conserve ses proportions.

        Args:
            source_path: Chemin de la vidéo à traiter.
            output_path: Chemin souhaité pour la vidéo annotée.
            progress_callback: Fonction appelée après chaque image traitée.

        Returns:
            Résultat technique comprenant les poses et le fichier exporté.

        Raises:
            InvalidVideoError: Si la source est invalide ou trop courte.
            VideoExportError: Si l'export ne peut pas être créé.
            VideoProcessorError: Si une erreur survient pendant le traitement.
        """

        metadata = self.read_metadata(source_path)
        validated_output_path = self._prepare_output_path(output_path)

        processing_width, processing_height = self.calculate_processing_size(
            source_width=metadata.width,
            source_height=metadata.height,
        )

        capture = cv2.VideoCapture(str(metadata.source_path))

        if not capture.isOpened():
            capture.release()
            raise InvalidVideoError(
                f"La vidéo ne peut pas être rouverte : {metadata.source_path}."
            )

        writer = self._create_video_writer(
            output_path=validated_output_path,
            fps=metadata.fps,
            width=processing_width,
            height=processing_height,
        )

        pose_frames: list[PoseFrame] = []
        detected_pose_frames = 0
        multiple_people_frames = 0
        processed_frames = 0
        processing_succeeded = False

        self._pose_detector.reset_timestamps()

        try:
            while True:
                read_success, original_frame = capture.read()

                if not read_success:
                    break

                if original_frame is None or original_frame.size == 0:
                    raise InvalidVideoError(
                        "OpenCV a retourné une image vidéo vide."
                    )

                processed_frame = self.resize_frame(
                    original_frame,
                    target_width=processing_width,
                    target_height=processing_height,
                )

                timestamp_seconds = processed_frames / metadata.fps

                try:
                    pose_frame = self._pose_detector.detect(
                        frame_bgr=processed_frame,
                        frame_index=processed_frames,
                        timestamp_seconds=timestamp_seconds,
                    )
                except PoseDetectorError as error:
                    raise VideoProcessorError(
                        "La détection de pose a échoué sur l'image "
                        f"{processed_frames}."
                    ) from error

                annotated_frame = self._pose_detector.draw_pose(
                    processed_frame,
                    pose_frame,
                )

                writer.write(annotated_frame)
                pose_frames.append(pose_frame)

                if pose_frame.pose_detected:
                    detected_pose_frames += 1

                if pose_frame.person_count > 1:
                    multiple_people_frames += 1

                processed_frames += 1

                if progress_callback is not None:
                    self._call_progress_callback(
                        progress_callback=progress_callback,
                        processed_frames=processed_frames,
                        total_frames=metadata.frame_count,
                    )

            if processed_frames < self._config.minimum_frame_count:
                raise InvalidVideoError(
                    "La vidéo est trop courte pour être analysée : "
                    f"{processed_frames} image(s) lisible(s)."
                )

            if not pose_frames:
                raise InvalidVideoError(
                    "Aucune image exploitable n'a été lue dans la vidéo."
                )

            processing_succeeded = True

        except (
            InvalidVideoError,
            VideoExportError,
            VideoProcessorError,
        ):
            raise
        except cv2.error as error:
            raise VideoProcessorError(
                "OpenCV a rencontré une erreur pendant le traitement vidéo."
            ) from error
        except Exception as error:
            raise VideoProcessorError(
                "Une erreur inattendue est survenue pendant le traitement "
                "de la vidéo."
            ) from error
        finally:
            capture.release()
            writer.release()

            if not processing_succeeded:
                self._delete_incomplete_output(validated_output_path)

        if not validated_output_path.is_file():
            raise VideoExportError(
                "La vidéo annotée n'a pas été créée."
            )

        if validated_output_path.stat().st_size <= 0:
            self._delete_incomplete_output(validated_output_path)
            raise VideoExportError(
                "La vidéo annotée créée est vide."
            )

        LOGGER.info(
            "Vidéo traitée : %s images, %s images avec pose, "
            "%s images avec plusieurs personnes.",
            processed_frames,
            detected_pose_frames,
            multiple_people_frames,
        )

        return VideoProcessingResult(
            metadata=metadata,
            output_path=validated_output_path,
            pose_frames=tuple(pose_frames),
            processed_width=processing_width,
            processed_height=processing_height,
            detected_pose_frames=detected_pose_frames,
            multiple_people_frames=multiple_people_frames,
        )

    def calculate_processing_size(
        self,
        source_width: int,
        source_height: int,
    ) -> tuple[int, int]:
        """Calcule une résolution limitée sans modifier les proportions.

        La vidéo n'est jamais agrandie. Les dimensions obtenues sont forcées
        à des valeurs paires pour améliorer la compatibilité des codecs vidéo.
        """

        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                "La résolution source doit être strictement positive."
            )

        width_scale = (
            self._config.max_processing_width / source_width
        )
        height_scale = (
            self._config.max_processing_height / source_height
        )

        scale = min(1.0, width_scale, height_scale)

        target_width = max(2, int(math.floor(source_width * scale)))
        target_height = max(2, int(math.floor(source_height * scale)))

        target_width = self._make_even(target_width)
        target_height = self._make_even(target_height)

        return target_width, target_height

    @staticmethod
    def resize_frame(
        frame: NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> NDArray[np.uint8]:
        """Redimensionne une image à la résolution calculée.

        ``INTER_AREA`` est utilisé pour une réduction. L'image originale est
        retournée sous forme de copie lorsque sa résolution est déjà correcte.
        """

        VideoProcessor._validate_frame(frame)

        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                "La résolution cible doit être strictement positive."
            )

        source_height, source_width = frame.shape[:2]

        if (
            source_width == target_width
            and source_height == target_height
        ):
            return frame.copy()

        return cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def _validate_source_path(
        self,
        source_path: Path | str,
    ) -> Path:
        """Valide le chemin et l'extension de la vidéo source."""

        path = Path(source_path).expanduser().resolve()

        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            supported_formats = ", ".join(
                sorted(SUPPORTED_VIDEO_EXTENSIONS)
            )
            raise UnsupportedVideoFormatError(
                f"Format vidéo non pris en charge : "
                f"'{path.suffix or 'sans extension'}'. "
                f"Formats acceptés : {supported_formats}."
            )

        if not path.is_file():
            raise InvalidVideoError(
                f"Le fichier vidéo est introuvable : {path}."
            )

        return path

    def _prepare_output_path(
        self,
        output_path: Path | str,
    ) -> Path:
        """Prépare le dossier et normalise l'extension de sortie."""

        path = Path(output_path).expanduser()

        if path.suffix.lower() != self._config.output_extension.lower():
            path = path.with_suffix(self._config.output_extension)

        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def _create_video_writer(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """Crée et valide le writer OpenCV."""

        fourcc = cv2.VideoWriter_fourcc(
            *self._config.output_codec
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            writer.release()
            raise VideoExportError(
                "Impossible d'initialiser l'export vidéo avec le codec "
                f"'{self._config.output_codec}'."
            )

        return writer

    @staticmethod
    def _validate_video_properties(
        fps: float,
        width: int,
        height: int,
        frame_count: int,
    ) -> None:
        """Valide les métadonnées retournées par OpenCV."""

        if not math.isfinite(fps):
            raise InvalidVideoError(
                "La vidéo retourne un nombre de FPS invalide."
            )

        if not (
            DEFAULT_MINIMUM_VALID_FPS
            <= fps
            <= DEFAULT_MAXIMUM_VALID_FPS
        ):
            raise InvalidVideoError(
                "Le nombre de FPS est invalide ou non pris en charge : "
                f"{fps:.3f}."
            )

        if width <= 0 or height <= 0:
            raise InvalidVideoError(
                "La vidéo retourne une résolution invalide."
            )

        if frame_count <= 0:
            raise InvalidVideoError(
                "La vidéo ne contient aucune image déclarée."
            )

    @staticmethod
    def _validate_frame(frame: NDArray[np.uint8]) -> None:
        """Vérifie qu'une image est compatible avec OpenCV."""

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "L'image vidéo doit être un tableau NumPy."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                "L'image vidéo doit utiliser le type uint8."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "L'image vidéo doit avoir la forme "
                "(hauteur, largeur, 3)."
            )

        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(
                "La résolution de l'image vidéo doit être positive."
            )

    @staticmethod
    def _make_even(value: int) -> int:
        """Retourne un entier pair strictement positif."""

        if value <= 2:
            return 2

        return value if value % 2 == 0 else value - 1

    @staticmethod
    def _call_progress_callback(
        progress_callback: ProgressCallback,
        processed_frames: int,
        total_frames: int,
    ) -> None:
        """Appelle la fonction de progression sans casser le traitement.

        Une erreur provenant uniquement de l'interface ne doit pas interrompre
        l'analyse ou laisser une vidéo incomplète.
        """

        try:
            progress_callback(
                processed_frames,
                total_frames,
            )
        except Exception as error:
            LOGGER.warning(
                "La fonction de progression a échoué : %s",
                error,
            )

    @staticmethod
    def _delete_incomplete_output(output_path: Path) -> None:
        """Supprime silencieusement un export incomplet."""

        try:
            output_path.unlink(missing_ok=True)
        except OSError as error:
            LOGGER.warning(
                "Impossible de supprimer le fichier incomplet %s : %s",
                output_path,
                error,
            )