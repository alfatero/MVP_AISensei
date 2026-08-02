"""Analyse géométrique d'un Gyaku-zuki.

Ce module transforme une séquence de poses en observations techniques
intermédiaires. Il ne calcule pas le score final sur 100.

Les seuils définis ici sont des valeurs initiales configurables. Ils devront
être ajustés sur des vidéos réelles et validés par un professeur de karaté.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from src.geometry import (
    DEFAULT_MINIMUM_VISIBILITY,
    arm_extension_distance,
    calculate_angle,
    calculate_hip_center,
    calculate_landmark_speeds,
    calculate_shoulder_center,
    calculate_shoulder_width,
    calculate_vertical_inclination,
    detect_striking_side,
    normalized_distance,
)
from src.models import (
    BodySide,
    FeedbackSeverity,
    ImpactFrame,
    Landmark,
    LandmarkName,
    PoseFrame,
)


LOGGER = logging.getLogger(__name__)

MINIMUM_SEQUENCE_FRAMES: Final[int] = 5
DEFAULT_CONTEXT_FRAMES: Final[int] = 4


class ObservationName(str, Enum):
    """Mesures produites par l'analyseur avant le calcul du score."""

    ARM_EXTENSION = "arm_extension"
    PUNCH_HEIGHT = "punch_height"
    TORSO_INCLINATION = "torso_inclination"
    OPPOSITE_GUARD = "opposite_guard"
    LEGS = "legs"
    RETURN_TO_GUARD = "return_to_guard"


@dataclass(frozen=True, slots=True)
class AnalyzerThresholds:
    """Seuils initiaux utilisés pour interpréter les mesures.

    Ces valeurs ne constituent pas des références scientifiques. Elles sont
    prévues pour obtenir un premier MVP testable.
    """

    minimum_visibility: float = DEFAULT_MINIMUM_VISIBILITY
    minimum_valid_frames: int = 3
    minimum_extension_difference: float = 0.10

    extension_good_degrees: float = 160.0
    extension_warning_degrees: float = 145.0

    torso_stable_degrees: float = 10.0
    torso_warning_degrees: float = 20.0

    punch_height_tolerance_shoulders: float = 0.40

    guard_good_distance_shoulders: float = 1.10
    guard_warning_distance_shoulders: float = 1.60

    knee_good_min_degrees: float = 120.0
    knee_good_max_degrees: float = 175.0

    minimum_return_ratio: float = 0.20
    good_return_ratio: float = 0.45

    impact_context_frames: int = DEFAULT_CONTEXT_FRAMES

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_visibility <= 1.0:
            raise ValueError(
                "Le seuil minimal de visibilité doit être compris entre 0 et 1."
            )

        if self.minimum_valid_frames <= 0:
            raise ValueError(
                "Le nombre minimal d'images valides doit être positif."
            )

        if self.minimum_extension_difference < 0.0:
            raise ValueError(
                "La différence minimale d'extension ne peut pas être négative."
            )

        if not (
            0.0
            <= self.extension_warning_degrees
            <= self.extension_good_degrees
            <= 180.0
        ):
            raise ValueError(
                "Les seuils d'extension du bras sont invalides."
            )

        if not (
            0.0
            <= self.torso_stable_degrees
            <= self.torso_warning_degrees
            <= 90.0
        ):
            raise ValueError(
                "Les seuils d'inclinaison du buste sont invalides."
            )

        if self.impact_context_frames < 0:
            raise ValueError(
                "Le nombre d'images de contexte ne peut pas être négatif."
            )


@dataclass(frozen=True, slots=True)
class TechnicalObservation:
    """Résultat intermédiaire d'une règle géométrique."""

    name: ObservationName
    reliable: bool
    confidence: float
    message: str
    severity: FeedbackSeverity
    measured_value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "La confiance d'une observation doit être comprise entre 0 et 1."
            )

        if not self.message.strip():
            raise ValueError(
                "Le message d'une observation ne peut pas être vide."
            )


@dataclass(frozen=True, slots=True)
class GyakuZukiAnalysis:
    """Résultat intermédiaire complet de l'analyse du mouvement."""

    success: bool
    confidence: float
    striking_side: BodySide | None = None
    impact: ImpactFrame | None = None
    observations: tuple[TechnicalObservation, ...] = ()
    context_frames: tuple[PoseFrame, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "La confiance globale doit être comprise entre 0 et 1."
            )

        if self.success:
            if self.striking_side is None:
                raise ValueError(
                    "Une analyse réussie doit identifier le bras qui frappe."
                )

            if self.impact is None:
                raise ValueError(
                    "Une analyse réussie doit contenir une image d'impact."
                )
        elif self.impact is not None:
            raise ValueError(
                "Une analyse échouée ne doit pas contenir d'image d'impact."
            )


@dataclass(frozen=True, slots=True)
class _ImpactCandidate:
    """Mesures internes utilisées pour classer une image d'impact."""

    frame: PoseFrame
    extension_distance: float
    elbow_angle: float
    wrist_speed: float | None
    speed_decreasing: bool
    mean_visibility: float
    ranking_score: float = field(compare=False)


class GyakuZukiAnalyzer:
    """Analyse une séquence de poses représentant un Gyaku-zuki."""

    def __init__(
        self,
        thresholds: AnalyzerThresholds | None = None,
    ) -> None:
        self._thresholds = thresholds or AnalyzerThresholds()

    @property
    def thresholds(self) -> AnalyzerThresholds:
        """Retourne les seuils actifs."""

        return self._thresholds

    def analyze(
        self,
        frames: tuple[PoseFrame, ...] | list[PoseFrame],
    ) -> GyakuZukiAnalysis:
        """Analyse une séquence complète de poses.

        Aucun résultat technique n'est inventé lorsque les données ne sont
        pas suffisamment visibles.
        """

        sequence = tuple(frames)

        if len(sequence) < MINIMUM_SEQUENCE_FRAMES:
            return GyakuZukiAnalysis(
                success=False,
                confidence=0.0,
                warnings=(
                    "Le mouvement paraît trop court pour être analysé "
                    "de manière fiable.",
                ),
            )

        striking_side = detect_striking_side(
            sequence,
            minimum_visibility=self._thresholds.minimum_visibility,
            minimum_valid_frames=self._thresholds.minimum_valid_frames,
            minimum_extension_difference=(
                self._thresholds.minimum_extension_difference
            ),
        )

        if striking_side is None:
            return GyakuZukiAnalysis(
                success=False,
                confidence=self._sequence_visibility(sequence),
                warnings=(
                    "Le bras qui frappe n'a pas pu être identifié avec "
                    "suffisamment de fiabilité.",
                ),
            )

        impact_candidate = self._find_impact_candidate(
            sequence,
            striking_side,
        )

        if impact_candidate is None:
            return GyakuZukiAnalysis(
                success=False,
                confidence=self._sequence_visibility(sequence),
                striking_side=striking_side,
                warnings=(
                    "Le moment d'extension maximale n'a pas pu être déterminé.",
                ),
            )

        impact = ImpactFrame(
            frame_index=impact_candidate.frame.frame_index,
            timestamp_seconds=impact_candidate.frame.timestamp_seconds,
            striking_side=striking_side,
            confidence=impact_candidate.mean_visibility,
            wrist_shoulder_distance=impact_candidate.extension_distance,
            elbow_angle_degrees=impact_candidate.elbow_angle,
            wrist_speed=impact_candidate.wrist_speed,
        )

        context_frames = self._select_context_frames(
            sequence,
            impact_candidate.frame.frame_index,
        )

        observations = (
            self._analyze_arm_extension(impact_candidate),
            self._analyze_punch_height(
                impact_candidate.frame,
                striking_side,
            ),
            self._analyze_torso(impact_candidate.frame),
            self._analyze_opposite_guard(
                impact_candidate.frame,
                striking_side,
            ),
            self._analyze_legs(
                impact_candidate.frame,
                striking_side,
            ),
            self._analyze_return_to_guard(
                sequence,
                impact_candidate.frame.frame_index,
                striking_side,
                impact_candidate.extension_distance,
            ),
        )

        reliable_confidences = [
            observation.confidence
            for observation in observations
            if observation.reliable
        ]

        confidence = (
            sum(reliable_confidences) / len(reliable_confidences)
            if reliable_confidences
            else 0.0
        )

        warnings: list[str] = []

        unreliable_count = sum(
            not observation.reliable
            for observation in observations
        )

        if unreliable_count:
            warnings.append(
                f"{unreliable_count} mesure(s) n'ont pas pu être établies "
                "avec suffisamment de fiabilité."
            )

        return GyakuZukiAnalysis(
            success=True,
            confidence=confidence,
            striking_side=striking_side,
            impact=impact,
            observations=observations,
            context_frames=context_frames,
            warnings=tuple(warnings),
        )

    def _find_impact_candidate(
        self,
        frames: tuple[PoseFrame, ...],
        side: BodySide,
    ) -> _ImpactCandidate | None:
        """Recherche l'image combinant extension, coude tendu et ralentissement."""

        wrist_name = self._wrist_name(side)
        speeds = calculate_landmark_speeds(
            frames,
            wrist_name,
            minimum_visibility=self._thresholds.minimum_visibility,
        )

        raw_candidates: list[
            tuple[PoseFrame, float, float, float | None, bool, float]
        ] = []

        for position, frame in enumerate(frames):
            arm_points = self._arm_landmarks(frame, side)

            if arm_points is None:
                continue

            shoulder, elbow, wrist = arm_points

            elbow_angle = calculate_angle(
                shoulder,
                elbow,
                wrist,
                minimum_visibility=self._thresholds.minimum_visibility,
            )
            extension = arm_extension_distance(
                frame,
                side,
                minimum_visibility=self._thresholds.minimum_visibility,
            )

            if elbow_angle is None or extension is None:
                continue

            current_speed = (
                speeds[position - 1]
                if position > 0 and position - 1 < len(speeds)
                else None
            )
            next_speed = (
                speeds[position]
                if position < len(speeds)
                else None
            )

            speed_decreasing = (
                current_speed is not None
                and next_speed is not None
                and next_speed < current_speed
            )

            visibility = min(
                shoulder.visibility,
                elbow.visibility,
                wrist.visibility,
            )

            raw_candidates.append(
                (
                    frame,
                    extension,
                    elbow_angle,
                    current_speed,
                    speed_decreasing,
                    visibility,
                )
            )

        if len(raw_candidates) < self._thresholds.minimum_valid_frames:
            return None

        maximum_extension = max(
            candidate[1]
            for candidate in raw_candidates
        )

        maximum_speed = max(
            (
                candidate[3]
                for candidate in raw_candidates
                if candidate[3] is not None
            ),
            default=0.0,
        )

        candidates: list[_ImpactCandidate] = []

        for (
            frame,
            extension,
            elbow_angle,
            current_speed,
            speed_decreasing,
            visibility,
        ) in raw_candidates:
            extension_score = (
                extension / maximum_extension
                if maximum_extension > 0.0
                else 0.0
            )
            elbow_score = elbow_angle / 180.0
            speed_score = (
                current_speed / maximum_speed
                if current_speed is not None and maximum_speed > 0.0
                else 0.0
            )
            slowdown_bonus = 1.0 if speed_decreasing else 0.0

            ranking_score = (
                extension_score * 0.50
                + elbow_score * 0.25
                + speed_score * 0.10
                + slowdown_bonus * 0.10
                + visibility * 0.05
            )

            candidates.append(
                _ImpactCandidate(
                    frame=frame,
                    extension_distance=extension,
                    elbow_angle=elbow_angle,
                    wrist_speed=current_speed,
                    speed_decreasing=speed_decreasing,
                    mean_visibility=visibility,
                    ranking_score=ranking_score,
                )
            )

        return max(
            candidates,
            key=lambda candidate: candidate.ranking_score,
        )

    def _analyze_arm_extension(
        self,
        candidate: _ImpactCandidate,
    ) -> TechnicalObservation:
        """Interprète l'angle du coude au moment probable de l'impact."""

        angle = candidate.elbow_angle

        if angle >= self._thresholds.extension_good_degrees:
            return TechnicalObservation(
                name=ObservationName.ARM_EXTENSION,
                reliable=True,
                confidence=candidate.mean_visibility,
                measured_value=angle,
                unit="degrees",
                severity=FeedbackSeverity.INFO,
                message=(
                    "Le bras semble correctement étendu au moment probable "
                    "de l'impact."
                ),
            )

        if angle >= self._thresholds.extension_warning_degrees:
            return TechnicalObservation(
                name=ObservationName.ARM_EXTENSION,
                reliable=True,
                confidence=candidate.mean_visibility,
                measured_value=angle,
                unit="degrees",
                severity=FeedbackSeverity.MEDIUM,
                message=(
                    "L'extension du bras paraît correcte, mais pourrait être "
                    "légèrement plus complète."
                ),
            )

        return TechnicalObservation(
            name=ObservationName.ARM_EXTENSION,
            reliable=True,
            confidence=candidate.mean_visibility,
            measured_value=angle,
            unit="degrees",
            severity=FeedbackSeverity.HIGH,
            message=(
                "Le bras semble insuffisamment tendu au moment probable "
                "de l'impact."
            ),
        )

    def _analyze_punch_height(
        self,
        frame: PoseFrame,
        side: BodySide,
    ) -> TechnicalObservation:
        """Compare la hauteur du poignet au centre des épaules et des hanches."""

        wrist = frame.get_landmark(self._wrist_name(side))
        shoulder_center = calculate_shoulder_center(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )
        hip_center = calculate_hip_center(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )
        shoulder_width = calculate_shoulder_width(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )

        if (
            wrist is None
            or shoulder_center is None
            or hip_center is None
            or shoulder_width is None
            or not wrist.is_reliable(self._thresholds.minimum_visibility)
        ):
            return self._unreliable_observation(
                ObservationName.PUNCH_HEIGHT,
                "La hauteur du poing ne peut pas être évaluée avec fiabilité.",
            )

        torso_height = hip_center.y - shoulder_center.y

        if torso_height <= 1e-9:
            return self._unreliable_observation(
                ObservationName.PUNCH_HEIGHT,
                "La hauteur du poing ne peut pas être évaluée avec fiabilité.",
            )

        relative_position = (
            wrist.y - shoulder_center.y
        ) / torso_height

        confidence = min(
            wrist.visibility,
            shoulder_center.visibility,
            hip_center.visibility,
        )

        tolerance = self._thresholds.punch_height_tolerance_shoulders

        if -tolerance <= relative_position <= 0.65:
            message = (
                "La hauteur du poing paraît cohérente avec une cible située "
                "au niveau du buste."
            )
            severity = FeedbackSeverity.INFO
        elif relative_position > 0.65:
            message = (
                "La trajectoire du poing semble légèrement trop basse."
            )
            severity = FeedbackSeverity.MEDIUM
        else:
            message = (
                "La trajectoire du poing paraît légèrement trop haute."
            )
            severity = FeedbackSeverity.MEDIUM

        return TechnicalObservation(
            name=ObservationName.PUNCH_HEIGHT,
            reliable=True,
            confidence=confidence,
            measured_value=relative_position,
            unit="torso_ratio",
            severity=severity,
            message=message,
        )

    def _analyze_torso(
        self,
        frame: PoseFrame,
    ) -> TechnicalObservation:
        """Mesure l'écart du buste par rapport à l'axe vertical."""

        shoulder_center = calculate_shoulder_center(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )
        hip_center = calculate_hip_center(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )

        if shoulder_center is None or hip_center is None:
            return self._unreliable_observation(
                ObservationName.TORSO_INCLINATION,
                "L'inclinaison du buste ne peut pas être mesurée avec fiabilité.",
            )

        inclination = calculate_vertical_inclination(
            shoulder_center,
            hip_center,
            self._thresholds.minimum_visibility,
        )

        if inclination is None:
            return self._unreliable_observation(
                ObservationName.TORSO_INCLINATION,
                "L'inclinaison du buste ne peut pas être mesurée avec fiabilité.",
            )

        confidence = min(
            shoulder_center.visibility,
            hip_center.visibility,
        )

        if inclination < self._thresholds.torso_stable_degrees:
            message = "Le buste semble rester stable pendant l'impact."
            severity = FeedbackSeverity.INFO
        elif inclination <= self._thresholds.torso_warning_degrees:
            message = (
                "Le buste paraît légèrement incliné au moment de l'impact."
            )
            severity = FeedbackSeverity.MEDIUM
        else:
            message = (
                "Le buste semble fortement incliné au moment de l'impact."
            )
            severity = FeedbackSeverity.HIGH

        return TechnicalObservation(
            name=ObservationName.TORSO_INCLINATION,
            reliable=True,
            confidence=confidence,
            measured_value=inclination,
            unit="degrees",
            severity=severity,
            message=message,
        )

    def _analyze_opposite_guard(
        self,
        frame: PoseFrame,
        striking_side: BodySide,
    ) -> TechnicalObservation:
        """Mesure la proximité de la main opposée avec le haut du corps."""

        opposite_wrist = frame.get_landmark(
            self._wrist_name(striking_side.opposite)
        )
        shoulder_center = calculate_shoulder_center(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )
        shoulder_width = calculate_shoulder_width(
            frame.landmarks,
            self._thresholds.minimum_visibility,
        )

        if (
            opposite_wrist is None
            or shoulder_center is None
            or shoulder_width is None
        ):
            return self._unreliable_observation(
                ObservationName.OPPOSITE_GUARD,
                "La position de la main opposée ne peut pas être vérifiée.",
            )

        distance = normalized_distance(
            opposite_wrist,
            shoulder_center,
            shoulder_width,
            self._thresholds.minimum_visibility,
        )

        if distance is None:
            return self._unreliable_observation(
                ObservationName.OPPOSITE_GUARD,
                "La position de la main opposée ne peut pas être vérifiée.",
            )

        confidence = min(
            opposite_wrist.visibility,
            shoulder_center.visibility,
        )

        if distance <= self._thresholds.guard_good_distance_shoulders:
            message = (
                "La main opposée semble rester proche du haut du corps."
            )
            severity = FeedbackSeverity.INFO
        elif distance <= self._thresholds.guard_warning_distance_shoulders:
            message = (
                "La main opposée paraît légèrement éloignée de la garde."
            )
            severity = FeedbackSeverity.MEDIUM
        else:
            message = (
                "La main opposée semble trop éloignée du corps au moment "
                "de l'impact."
            )
            severity = FeedbackSeverity.HIGH

        return TechnicalObservation(
            name=ObservationName.OPPOSITE_GUARD,
            reliable=True,
            confidence=confidence,
            measured_value=distance,
            unit="shoulder_widths",
            severity=severity,
            message=message,
        )

    def _analyze_legs(
        self,
        frame: PoseFrame,
        striking_side: BodySide,
    ) -> TechnicalObservation:
        """Analyse approximativement la flexion du genou avant.

        Pour le MVP, la jambe opposée au bras qui frappe est considérée comme
        la jambe avant. Cette hypothèse peut être incorrecte selon la garde ou
        un effet miroir de la caméra.
        """

        front_side = striking_side.opposite

        hip = frame.get_landmark(self._hip_name(front_side))
        knee = frame.get_landmark(self._knee_name(front_side))
        ankle = frame.get_landmark(self._ankle_name(front_side))

        if hip is None or knee is None or ankle is None:
            return self._unreliable_observation(
                ObservationName.LEGS,
                "La position des jambes ne peut pas être évaluée avec fiabilité.",
            )

        knee_angle = calculate_angle(
            hip,
            knee,
            ankle,
            self._thresholds.minimum_visibility,
        )

        if knee_angle is None:
            return self._unreliable_observation(
                ObservationName.LEGS,
                "La position des jambes ne peut pas être évaluée avec fiabilité.",
            )

        confidence = min(
            hip.visibility,
            knee.visibility,
            ankle.visibility,
        )

        if (
            self._thresholds.knee_good_min_degrees
            <= knee_angle
            <= self._thresholds.knee_good_max_degrees
        ):
            message = (
                "La flexion visible du genou avant paraît cohérente."
            )
            severity = FeedbackSeverity.INFO
        elif knee_angle > self._thresholds.knee_good_max_degrees:
            message = (
                "La jambe avant semble presque tendue ; la base pourrait "
                "manquer légèrement de flexion."
            )
            severity = FeedbackSeverity.MEDIUM
        else:
            message = (
                "Le genou avant paraît très fléchi sur cette image."
            )
            severity = FeedbackSeverity.MEDIUM

        return TechnicalObservation(
            name=ObservationName.LEGS,
            reliable=True,
            confidence=confidence,
            measured_value=knee_angle,
            unit="degrees",
            severity=severity,
            message=message,
        )

    def _analyze_return_to_guard(
        self,
        frames: tuple[PoseFrame, ...],
        impact_frame_index: int,
        side: BodySide,
        impact_extension: float,
    ) -> TechnicalObservation:
        """Vérifie si le poignet revient vers l'épaule après l'impact."""

        post_impact_frames = [
            frame
            for frame in frames
            if frame.frame_index > impact_frame_index
        ]

        if len(post_impact_frames) < 2 or impact_extension <= 1e-9:
            return self._unreliable_observation(
                ObservationName.RETURN_TO_GUARD,
                "Le retour en garde ne peut pas être évalué sur cette séquence.",
            )

        extensions: list[tuple[float, float]] = []

        for frame in post_impact_frames:
            extension = arm_extension_distance(
                frame,
                side,
                self._thresholds.minimum_visibility,
            )

            arm_points = self._arm_landmarks(frame, side)

            if extension is None or arm_points is None:
                continue

            visibility = min(
                point.visibility
                for point in arm_points
            )
            extensions.append((extension, visibility))

        if len(extensions) < 2:
            return self._unreliable_observation(
                ObservationName.RETURN_TO_GUARD,
                "Le retour en garde ne peut pas être évalué avec fiabilité.",
            )

        minimum_post_extension = min(
            extension
            for extension, _ in extensions
        )

        return_ratio = max(
            0.0,
            (impact_extension - minimum_post_extension)
            / impact_extension,
        )

        confidence = sum(
            visibility
            for _, visibility in extensions
        ) / len(extensions)

        if return_ratio >= self._thresholds.good_return_ratio:
            message = (
                "Le poignet semble revenir correctement vers la garde "
                "après l'impact."
            )
            severity = FeedbackSeverity.INFO
        elif return_ratio >= self._thresholds.minimum_return_ratio:
            message = (
                "Le retour du poignet est visible, mais paraît assez limité."
            )
            severity = FeedbackSeverity.MEDIUM
        else:
            message = (
                "Le poignet semble rester trop longtemps en extension "
                "après l'impact."
            )
            severity = FeedbackSeverity.HIGH

        return TechnicalObservation(
            name=ObservationName.RETURN_TO_GUARD,
            reliable=True,
            confidence=confidence,
            measured_value=return_ratio,
            unit="ratio",
            severity=severity,
            message=message,
        )

    def _select_context_frames(
        self,
        frames: tuple[PoseFrame, ...],
        impact_frame_index: int,
    ) -> tuple[PoseFrame, ...]:
        """Conserve quelques images avant et après l'impact."""

        context = self._thresholds.impact_context_frames

        return tuple(
            frame
            for frame in frames
            if impact_frame_index - context
            <= frame.frame_index
            <= impact_frame_index + context
        )

    def _sequence_visibility(
        self,
        frames: tuple[PoseFrame, ...],
    ) -> float:
        """Calcule une confiance moyenne simple sur la séquence."""

        valid_visibilities = [
            frame.mean_visibility()
            for frame in frames
            if frame.pose_detected and frame.landmarks
        ]

        if not valid_visibilities:
            return 0.0

        return sum(valid_visibilities) / len(valid_visibilities)

    def _arm_landmarks(
        self,
        frame: PoseFrame,
        side: BodySide,
    ) -> tuple[Landmark, Landmark, Landmark] | None:
        """Retourne épaule, coude et poignet d'un côté."""

        shoulder = frame.get_landmark(self._shoulder_name(side))
        elbow = frame.get_landmark(self._elbow_name(side))
        wrist = frame.get_landmark(self._wrist_name(side))

        if shoulder is None or elbow is None or wrist is None:
            return None

        if not all(
            point.is_reliable(self._thresholds.minimum_visibility)
            for point in (shoulder, elbow, wrist)
        ):
            return None

        return shoulder, elbow, wrist

    @staticmethod
    def _unreliable_observation(
        name: ObservationName,
        message: str,
    ) -> TechnicalObservation:
        """Construit une observation explicitement non fiable."""

        return TechnicalObservation(
            name=name,
            reliable=False,
            confidence=0.0,
            measured_value=None,
            unit=None,
            severity=FeedbackSeverity.MEDIUM,
            message=message,
        )

    @staticmethod
    def _shoulder_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_SHOULDER
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_SHOULDER
        )

    @staticmethod
    def _elbow_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_ELBOW
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_ELBOW
        )

    @staticmethod
    def _wrist_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_WRIST
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_WRIST
        )

    @staticmethod
    def _hip_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_HIP
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_HIP
        )

    @staticmethod
    def _knee_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_KNEE
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_KNEE
        )

    @staticmethod
    def _ankle_name(side: BodySide) -> LandmarkName:
        return (
            LandmarkName.LEFT_ANKLE
            if side is BodySide.LEFT
            else LandmarkName.RIGHT_ANKLE
        )