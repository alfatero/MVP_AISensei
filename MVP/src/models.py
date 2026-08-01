"""Modèles de données partagés par l'application Karate Coach.

Ce module ne contient aucune logique d'analyse métier. Il définit uniquement
les structures échangées entre :

- le détecteur de pose ;
- le processeur vidéo ;
- l'analyseur du Gyaku-zuki ;
- le moteur de score ;
- l'interface Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


class BodySide(str, Enum):
    """Côté du corps utilisé pour identifier les articulations."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def opposite(self) -> "BodySide":
        """Retourne le côté opposé."""

        return BodySide.RIGHT if self is BodySide.LEFT else BodySide.LEFT


class LandmarkName(str, Enum):
    """Articulations nécessaires à l'analyse du Gyaku-zuki."""

    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"

    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"

    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"

    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"

    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"

    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class AnalysisCriterion(str, Enum):
    """Critères utilisés pour calculer les sous-scores."""

    EXTENSION = "extension"
    TRAJECTORY = "trajectoire"
    TORSO = "buste"
    LEGS = "jambes"
    GUARD = "garde"


class FeedbackSeverity(str, Enum):
    """Niveau de priorité d'un retour d'analyse."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisStatus(str, Enum):
    """État final d'une tentative d'analyse."""

    SUCCESS = "success"
    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    NO_POSE_DETECTED = "no_pose_detected"
    MULTIPLE_PEOPLE_DETECTED = "multiple_people_detected"
    MOVEMENT_TOO_SHORT = "movement_too_short"
    INVALID_VIDEO = "invalid_video"
    PROCESSING_ERROR = "processing_error"


@dataclass(frozen=True, slots=True)
class Landmark:
    """Coordonnées normalisées d'une articulation corporelle.

    Les coordonnées ``x`` et ``y`` sont normalement normalisées entre 0 et 1
    par MediaPipe. La coordonnée ``z`` représente une profondeur relative et
    ne doit pas être interprétée comme une distance physique réelle.

    Attributes:
        x: Position horizontale normalisée.
        y: Position verticale normalisée.
        z: Profondeur relative fournie par MediaPipe.
        visibility: Probabilité estimée que le point soit visible.
    """

    x: float
    y: float
    z: float
    visibility: float

    def __post_init__(self) -> None:
        """Valide les valeurs qui doivent rester dans un domaine connu."""

        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError(
                "La visibilité d'un point doit être comprise entre 0 et 1, "
                f"valeur reçue : {self.visibility}."
            )

    def is_reliable(self, minimum_visibility: float) -> bool:
        """Indique si le point est suffisamment visible.

        Args:
            minimum_visibility: Seuil minimal de visibilité, entre 0 et 1.

        Returns:
            ``True`` lorsque la visibilité atteint le seuil demandé.

        Raises:
            ValueError: Si le seuil n'est pas compris entre 0 et 1.
        """

        if not 0.0 <= minimum_visibility <= 1.0:
            raise ValueError(
                "Le seuil minimal de visibilité doit être compris entre 0 et 1."
            )

        return self.visibility >= minimum_visibility


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Résultat de la détection corporelle pour une image vidéo.

    Attributes:
        frame_index: Position de l'image dans la vidéo, à partir de zéro.
        timestamp_seconds: Horodatage de l'image en secondes.
        landmarks: Articulations détectées, indexées par leur nom.
        pose_detected: Indique si une pose exploitable a été détectée.
        person_count: Nombre de personnes détectées lorsque cette information
            est disponible.
    """

    frame_index: int
    timestamp_seconds: float
    landmarks: Mapping[LandmarkName, Landmark] = field(default_factory=dict)
    pose_detected: bool = False
    person_count: int = 0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("L'index d'une image ne peut pas être négatif.")

        if self.timestamp_seconds < 0:
            raise ValueError("L'horodatage d'une image ne peut pas être négatif.")

        if self.person_count < 0:
            raise ValueError("Le nombre de personnes ne peut pas être négatif.")

    def get_landmark(self, name: LandmarkName) -> Landmark | None:
        """Retourne une articulation détectée ou ``None``."""

        return self.landmarks.get(name)

    def has_reliable_landmarks(
        self,
        required_landmarks: Sequence[LandmarkName],
        minimum_visibility: float,
    ) -> bool:
        """Vérifie que toutes les articulations demandées sont fiables.

        Une image sans pose détectée est systématiquement considérée comme
        non fiable.

        Args:
            required_landmarks: Articulations nécessaires au calcul.
            minimum_visibility: Visibilité minimale exigée.

        Returns:
            ``True`` si tous les points sont présents et suffisamment visibles.
        """

        if not self.pose_detected:
            return False

        for landmark_name in required_landmarks:
            landmark = self.landmarks.get(landmark_name)

            if landmark is None or not landmark.is_reliable(minimum_visibility):
                return False

        return True

    def mean_visibility(
        self,
        landmark_names: Sequence[LandmarkName] | None = None,
    ) -> float:
        """Calcule la visibilité moyenne d'un ensemble d'articulations.

        Args:
            landmark_names: Points à prendre en compte. Lorsque la valeur est
                ``None``, tous les points présents sont utilisés.

        Returns:
            Une valeur comprise entre 0 et 1. Retourne ``0.0`` si aucun point
            demandé n'est disponible.
        """

        if landmark_names is None:
            selected_landmarks = list(self.landmarks.values())
        else:
            selected_landmarks = [
                self.landmarks[name]
                for name in landmark_names
                if name in self.landmarks
            ]

        if not selected_landmarks:
            return 0.0

        return sum(
            landmark.visibility for landmark in selected_landmarks
        ) / len(selected_landmarks)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Informations techniques extraites d'une vidéo."""

    source_path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("Le nombre d'images par seconde doit être positif.")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("La résolution vidéo doit être strictement positive.")

        if self.frame_count <= 0:
            raise ValueError("La vidéo doit contenir au moins une image.")

        if self.duration_seconds <= 0:
            raise ValueError("La durée vidéo doit être strictement positive.")

    @property
    def resolution(self) -> tuple[int, int]:
        """Retourne la résolution sous la forme ``(largeur, hauteur)``."""

        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """Retourne le rapport largeur/hauteur de la vidéo."""

        return self.width / self.height


@dataclass(frozen=True, slots=True)
class ImpactFrame:
    """Image retenue comme moment probable d'impact du Gyaku-zuki."""

    frame_index: int
    timestamp_seconds: float
    striking_side: BodySide
    confidence: float
    wrist_shoulder_distance: float
    elbow_angle_degrees: float
    wrist_speed: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("L'index de l'image d'impact ne peut pas être négatif.")

        if self.timestamp_seconds < 0:
            raise ValueError(
                "L'horodatage de l'image d'impact ne peut pas être négatif."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "La confiance de l'image d'impact doit être comprise entre 0 et 1."
            )

        if self.wrist_shoulder_distance < 0:
            raise ValueError(
                "La distance poignet-épaule ne peut pas être négative."
            )

        if not 0.0 <= self.elbow_angle_degrees <= 180.0:
            raise ValueError(
                "L'angle du coude doit être compris entre 0 et 180 degrés."
            )

        if self.wrist_speed is not None and self.wrist_speed < 0:
            raise ValueError("La vitesse du poignet ne peut pas être négative.")


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Résultat détaillé d'un critère technique.

    Le score est exprimé dans la plage ``0`` à ``max_score``. Un résultat peut
    être déclaré non fiable : il ne devra alors pas être utilisé comme une
    mesure certaine par le moteur de score.

    Attributes:
        criterion: Critère technique analysé.
        score: Points obtenus.
        max_score: Nombre maximal de points.
        reliable: Indique si les mesures sont suffisamment fiables.
        confidence: Niveau de confiance compris entre 0 et 1.
        measured_value: Valeur géométrique principale, lorsqu'elle existe.
        unit: Unité de la mesure, par exemple ``degrees``.
        message: Retour prudent destiné à l'utilisateur.
        severity: Priorité du retour.
    """

    criterion: AnalysisCriterion
    score: float
    max_score: float
    reliable: bool
    confidence: float
    message: str
    severity: FeedbackSeverity = FeedbackSeverity.INFO
    measured_value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.max_score <= 0:
            raise ValueError("Le score maximal doit être strictement positif.")

        if not 0.0 <= self.score <= self.max_score:
            raise ValueError(
                f"Le score doit être compris entre 0 et {self.max_score}."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("La confiance doit être comprise entre 0 et 1.")

        if not self.message.strip():
            raise ValueError("Le message d'un critère ne peut pas être vide.")


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    """Message utilisateur produit à partir d'une règle d'analyse."""

    criterion: AnalysisCriterion
    message: str
    severity: FeedbackSeverity
    confidence: float

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("Un retour utilisateur ne peut pas être vide.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("La confiance doit être comprise entre 0 et 1.")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Résultat final présenté par l'application.

    Lorsque ``success`` vaut ``False``, ``score_total`` et ``impact_frame``
    peuvent rester à ``None``. Cela évite de fabriquer un score à partir de
    données corporelles insuffisamment fiables.
    """

    success: bool
    status: AnalysisStatus
    confidence: float
    score_total: float | None = None
    subscores: Mapping[AnalysisCriterion, float] = field(default_factory=dict)
    strengths: tuple[FeedbackItem, ...] = ()
    corrections: tuple[FeedbackItem, ...] = ()
    warnings: tuple[str, ...] = ()
    impact_frame: int | None = None
    output_video_path: Path | None = None
    criterion_results: tuple[CriterionResult, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("La confiance globale doit être comprise entre 0 et 1.")

        if self.score_total is not None and not 0.0 <= self.score_total <= 100.0:
            raise ValueError("Le score total doit être compris entre 0 et 100.")

        if self.impact_frame is not None and self.impact_frame < 0:
            raise ValueError(
                "L'index de l'image d'impact ne peut pas être négatif."
            )

        if self.success:
            if self.status is not AnalysisStatus.SUCCESS:
                raise ValueError(
                    "Une analyse réussie doit utiliser le statut SUCCESS."
                )

            if self.score_total is None:
                raise ValueError(
                    "Une analyse réussie doit contenir un score total."
                )

            if self.impact_frame is None:
                raise ValueError(
                    "Une analyse réussie doit contenir une image d'impact."
                )
        elif self.score_total is not None:
            raise ValueError(
                "Une analyse échouée ne doit pas contenir de score total."
            )

        if len(self.strengths) > 2:
            raise ValueError(
                "Le résultat ne doit pas contenir plus de deux points positifs."
            )

        if len(self.corrections) > 3:
            raise ValueError(
                "Le résultat ne doit pas contenir plus de trois corrections."
            )

    @classmethod
    def failure(
        cls,
        status: AnalysisStatus,
        warning: str,
        confidence: float = 0.0,
    ) -> "AnalysisResult":
        """Construit un résultat d'échec sans générer de faux score.

        Args:
            status: Cause principale de l'échec.
            warning: Message compréhensible destiné à l'utilisateur.
            confidence: Confiance globale disponible malgré l'échec.

        Returns:
            Un résultat dont ``success`` vaut ``False``.
        """

        if status is AnalysisStatus.SUCCESS:
            raise ValueError(
                "La méthode failure ne peut pas utiliser le statut SUCCESS."
            )

        if not warning.strip():
            raise ValueError("Le message d'avertissement ne peut pas être vide.")

        return cls(
            success=False,
            status=status,
            confidence=confidence,
            warnings=(warning,),
        )