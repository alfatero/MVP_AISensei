"""Orchestration complète de l'analyse vidéo de Karate Coach.

Ce module relie les différentes couches de l'application :

- lecture et annotation vidéo ;
- détection des poses ;
- analyse géométrique du Gyaku-zuki ;
- calcul du score final.

Il ne contient aucune règle géométrique ni aucune règle de notation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from uuid import uuid4

from src.gyaku_zuki_analyzer import (
    MINIMUM_SEQUENCE_FRAMES,
    AnalyzerThresholds,
    GyakuZukiAnalyzer,
)
from src.models import AnalysisResult, AnalysisStatus
from src.pose_detector import (
    PoseDetector,
    PoseDetectorConfig,
    PoseDetectorError,
)
from src.scoring import ScoringConfig, ScoringEngine
from src.video_processor import (
    InvalidVideoError,
    ProgressCallback,
    UnsupportedVideoFormatError,
    VideoExportError,
    VideoProcessingResult,
    VideoProcessor,
    VideoProcessorConfig,
    VideoProcessorError,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path("data/outputs")
DEFAULT_MINIMUM_POSE_DETECTION_RATIO: Final[float] = 0.60


@dataclass(frozen=True, slots=True)
class AnalysisPipelineConfig:
    """Configuration générale du pipeline.

    Attributes:
        output_directory: Dossier des vidéos annotées.
        minimum_pose_detection_ratio: Proportion minimale d'images dans
            lesquelles une pose doit être détectée.
        reject_multiple_people: Refuse l'analyse lorsqu'une seconde personne
            est détectée sur au moins une image.
        pose_detector: Configuration de MediaPipe.
        video_processor: Configuration du traitement vidéo.
        analyzer: Seuils de l'analyseur Gyaku-zuki.
        scoring: Configuration du moteur de score.
    """

    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    minimum_pose_detection_ratio: float = (
        DEFAULT_MINIMUM_POSE_DETECTION_RATIO
    )
    reject_multiple_people: bool = True
    pose_detector: PoseDetectorConfig = PoseDetectorConfig()
    video_processor: VideoProcessorConfig = VideoProcessorConfig()
    analyzer: AnalyzerThresholds = AnalyzerThresholds()
    scoring: ScoringConfig = ScoringConfig()

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_pose_detection_ratio <= 1.0:
            raise ValueError(
                "Le taux minimal de détection doit être compris entre 0 et 1."
            )


class AnalysisPipeline:
    """Exécute toutes les étapes nécessaires à l'analyse d'une vidéo.

    Les composants peuvent être injectés dans le constructeur pour faciliter
    les tests unitaires et éviter de charger MediaPipe pendant les tests.
    """

    def __init__(
        self,
        config: AnalysisPipelineConfig | None = None,
        *,
        pose_detector: PoseDetector | None = None,
        video_processor: VideoProcessor | None = None,
        analyzer: GyakuZukiAnalyzer | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        """Initialise le pipeline et ses composants.

        Args:
            config: Configuration générale du pipeline.
            pose_detector: Détecteur MediaPipe injecté.
            video_processor: Processeur vidéo injecté.
            analyzer: Analyseur métier injecté.
            scoring_engine: Moteur de score injecté.

        Notes:
            Lorsque ``video_processor`` est fourni, le pipeline ne crée pas
            automatiquement de détecteur MediaPipe.
        """

        self._config = config or AnalysisPipelineConfig()
        self._closed = False
        self._owned_pose_detector: PoseDetector | None = None

        if video_processor is None:
            if pose_detector is None:
                pose_detector = PoseDetector(
                    config=self._config.pose_detector
                )
                self._owned_pose_detector = pose_detector

            video_processor = VideoProcessor(
                pose_detector=pose_detector,
                config=self._config.video_processor,
            )

        self._video_processor = video_processor
        self._analyzer = analyzer or GyakuZukiAnalyzer(
            thresholds=self._config.analyzer
        )
        self._scoring_engine = scoring_engine or ScoringEngine(
            config=self._config.scoring
        )

    @property
    def config(self) -> AnalysisPipelineConfig:
        """Retourne la configuration active."""

        return self._config

    def analyze_video(
        self,
        source_path: Path | str,
        output_path: Path | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Analyse entièrement une vidéo de Gyaku-zuki.

        Args:
            source_path: Chemin de la vidéo source.
            output_path: Chemin facultatif de la vidéo annotée.
            progress_callback: Fonction appelée pendant le traitement vidéo.

        Returns:
            Résultat final de l'analyse. Aucun score n'est produit lorsque les
            données ne sont pas suffisamment fiables.
        """

        self._ensure_open()

        source = Path(source_path).expanduser()

        if output_path is None:
            resolved_output_path = self._build_output_path(source)
        else:
            resolved_output_path = Path(output_path).expanduser()

        try:
            processing_result = self._video_processor.process(
                source_path=source,
                output_path=resolved_output_path,
                progress_callback=progress_callback,
            )
        except UnsupportedVideoFormatError as error:
            LOGGER.warning("Format vidéo non pris en charge : %s", error)
            return AnalysisResult.failure(
                status=AnalysisStatus.INVALID_VIDEO,
                warning=str(error),
            )
        except InvalidVideoError as error:
            LOGGER.warning("Vidéo invalide : %s", error)
            return AnalysisResult.failure(
                status=AnalysisStatus.INVALID_VIDEO,
                warning=str(error),
            )
        except VideoExportError as error:
            LOGGER.exception("Échec de l'export vidéo.")
            return AnalysisResult.failure(
                status=AnalysisStatus.PROCESSING_ERROR,
                warning=(
                    "La vidéo a été lue, mais son export annoté a échoué."
                ),
            )
        except (VideoProcessorError, PoseDetectorError) as error:
            LOGGER.exception("Échec du traitement vidéo : %s", error)
            return AnalysisResult.failure(
                status=AnalysisStatus.PROCESSING_ERROR,
                warning=(
                    "Une erreur est survenue pendant le traitement de la vidéo."
                ),
            )
        except Exception as error:
            LOGGER.exception(
                "Erreur inattendue pendant le traitement vidéo : %s",
                error,
            )
            return AnalysisResult.failure(
                status=AnalysisStatus.PROCESSING_ERROR,
                warning=(
                    "Une erreur inattendue est survenue pendant l'analyse."
                ),
            )

        preliminary_failure = self._validate_processing_result(
            processing_result
        )

        if preliminary_failure is not None:
            return replace(
                preliminary_failure,
                output_video_path=processing_result.output_path,
            )

        try:
            technical_analysis = self._analyzer.analyze(
                processing_result.pose_frames
            )
        except Exception as error:
            LOGGER.exception(
                "Erreur pendant l'analyse du Gyaku-zuki : %s",
                error,
            )
            return AnalysisResult(
                success=False,
                status=AnalysisStatus.PROCESSING_ERROR,
                confidence=0.0,
                warnings=(
                    "Le mouvement n'a pas pu être analysé à cause d'une "
                    "erreur interne.",
                ),
                output_video_path=processing_result.output_path,
            )

        if not technical_analysis.success:
            failure = self._convert_analyzer_failure(
                processing_result=processing_result,
                confidence=technical_analysis.confidence,
                warnings=technical_analysis.warnings,
            )

            return replace(
                failure,
                output_video_path=processing_result.output_path,
            )

        try:
            final_result = self._scoring_engine.calculate(
                technical_analysis
            )
        except Exception as error:
            LOGGER.exception(
                "Erreur pendant le calcul du score : %s",
                error,
            )
            return AnalysisResult(
                success=False,
                status=AnalysisStatus.PROCESSING_ERROR,
                confidence=technical_analysis.confidence,
                warnings=(
                    "Les mesures ont été produites, mais le score n'a pas "
                    "pu être calculé.",
                ),
                output_video_path=processing_result.output_path,
            )

        return replace(
            final_result,
            output_video_path=processing_result.output_path,
        )

    def _validate_processing_result(
        self,
        result: VideoProcessingResult,
    ) -> AnalysisResult | None:
        """Vérifie la quantité et la cohérence des poses détectées."""

        if not result.pose_frames:
            return AnalysisResult.failure(
                status=AnalysisStatus.NO_POSE_DETECTED,
                warning=(
                    "Aucune image exploitable n'a été trouvée dans la vidéo."
                ),
            )

        if result.detected_pose_frames == 0:
            return AnalysisResult.failure(
                status=AnalysisStatus.NO_POSE_DETECTED,
                warning=(
                    "Aucune personne n'a été détectée. Refilme le mouvement "
                    "avec le corps entier visible et un meilleur éclairage."
                ),
            )

        if (
            self._config.reject_multiple_people
            and result.has_multiple_people
        ):
            return AnalysisResult.failure(
                status=AnalysisStatus.MULTIPLE_PEOPLE_DETECTED,
                warning=(
                    "Plusieurs personnes semblent visibles dans la vidéo. "
                    "L'analyse nécessite une seule personne dans le cadre."
                ),
                confidence=result.pose_detection_ratio,
            )

        if len(result.pose_frames) < MINIMUM_SEQUENCE_FRAMES:
            return AnalysisResult.failure(
                status=AnalysisStatus.MOVEMENT_TOO_SHORT,
                warning=(
                    "La séquence paraît trop courte pour analyser le "
                    "Gyaku-zuki avec suffisamment de fiabilité."
                ),
                confidence=result.pose_detection_ratio,
            )

        if (
            result.pose_detection_ratio
            < self._config.minimum_pose_detection_ratio
        ):
            return AnalysisResult.failure(
                status=AnalysisStatus.INSUFFICIENT_VISIBILITY,
                warning=(
                    "La pose n'est pas détectée sur suffisamment d'images. "
                    "Refilme le mouvement avec tout le corps visible."
                ),
                confidence=result.pose_detection_ratio,
            )

        return None

    @staticmethod
    def _convert_analyzer_failure(
        processing_result: VideoProcessingResult,
        confidence: float,
        warnings: tuple[str, ...],
    ) -> AnalysisResult:
        """Convertit un échec métier en résultat public."""

        warning = (
            warnings[0]
            if warnings
            else (
                "Le mouvement n'a pas pu être analysé avec suffisamment "
                "de fiabilité."
            )
        )

        if len(processing_result.pose_frames) < MINIMUM_SEQUENCE_FRAMES:
            status = AnalysisStatus.MOVEMENT_TOO_SHORT
        else:
            status = AnalysisStatus.INSUFFICIENT_VISIBILITY

        return AnalysisResult.failure(
            status=status,
            warning=warning,
            confidence=confidence,
        )

    def _build_output_path(self, source_path: Path) -> Path:
        """Génère un nom unique pour la vidéo annotée."""

        output_directory = (
            self._config.output_directory.expanduser().resolve()
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        source_stem = source_path.stem.strip() or "video"
        unique_suffix = uuid4().hex[:8]

        return output_directory / (
            f"{source_stem}_annotated_{unique_suffix}.mp4"
        )

    def close(self) -> None:
        """Libère les ressources MediaPipe créées par le pipeline."""

        if self._closed:
            return

        if self._owned_pose_detector is not None:
            self._owned_pose_detector.close()

        self._closed = True

    def _ensure_open(self) -> None:
        """Empêche l'utilisation du pipeline après sa fermeture."""

        if self._closed:
            raise RuntimeError(
                "Le pipeline d'analyse a déjà été fermé."
            )

    def __enter__(self) -> "AnalysisPipeline":
        """Permet l'utilisation avec un gestionnaire de contexte."""

        self._ensure_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Ferme automatiquement les ressources du pipeline."""

        self.close()


def analyze_video(
    source_path: Path | str,
    output_path: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
    config: AnalysisPipelineConfig | None = None,
) -> AnalysisResult:
    """Fonction pratique pour analyser une vidéo en un seul appel.

    Le détecteur MediaPipe est automatiquement créé puis fermé.
    """

    with AnalysisPipeline(config=config) as pipeline:
        return pipeline.analyze_video(
            source_path=source_path,
            output_path=output_path,
            progress_callback=progress_callback,
        )