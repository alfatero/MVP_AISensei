"""Calcul du score final de Karate Coach.

Ce module transforme les observations géométriques produites par
``GyakuZukiAnalyzer`` en :

- sous-scores par critère ;
- score total sur 100 ;
- points positifs ;
- corrections prioritaires ;
- avertissements de fiabilité.

Il ne réalise aucun calcul géométrique et ne détecte aucune articulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.gyaku_zuki_analyzer import (
    GyakuZukiAnalysis,
    ObservationName,
    TechnicalObservation,
)
from src.models import (
    AnalysisCriterion,
    AnalysisResult,
    AnalysisStatus,
    CriterionResult,
    FeedbackItem,
    FeedbackSeverity,
)


CRITERION_MAX_SCORES: Final[dict[AnalysisCriterion, float]] = {
    AnalysisCriterion.EXTENSION: 25.0,
    AnalysisCriterion.TRAJECTORY: 20.0,
    AnalysisCriterion.TORSO: 20.0,
    AnalysisCriterion.LEGS: 20.0,
    AnalysisCriterion.GUARD: 15.0,
}

TOTAL_MAX_SCORE: Final[float] = sum(CRITERION_MAX_SCORES.values())

SEVERITY_SCORE_RATIOS: Final[dict[FeedbackSeverity, float]] = {
    FeedbackSeverity.INFO: 1.00,
    FeedbackSeverity.LOW: 0.85,
    FeedbackSeverity.MEDIUM: 0.60,
    FeedbackSeverity.HIGH: 0.25,
}

SEVERITY_PRIORITY: Final[dict[FeedbackSeverity, int]] = {
    FeedbackSeverity.HIGH: 3,
    FeedbackSeverity.MEDIUM: 2,
    FeedbackSeverity.LOW: 1,
    FeedbackSeverity.INFO: 0,
}


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Configuration du moteur de score.

    Les valeurs sont des paramètres initiaux du MVP. Elles devront être
    ajustées après validation sur des vidéos réelles avec un professeur
    de karaté.

    Attributes:
        minimum_global_confidence: Confiance minimale de l'analyse initiale.
        minimum_observation_confidence: Confiance minimale d'une observation.
        minimum_reliable_criteria: Nombre minimal de critères calculables.
        require_extension: Exige que l'extension soit mesurable.
        opposite_guard_weight: Poids de la position de la main opposée.
        return_to_guard_weight: Poids du retour en garde.
    """

    minimum_global_confidence: float = 0.60
    minimum_observation_confidence: float = 0.60
    minimum_reliable_criteria: int = 4
    require_extension: bool = True
    opposite_guard_weight: float = 0.40
    return_to_guard_weight: float = 0.60

    def __post_init__(self) -> None:
        for name, value in (
            (
                "minimum_global_confidence",
                self.minimum_global_confidence,
            ),
            (
                "minimum_observation_confidence",
                self.minimum_observation_confidence,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} doit être compris entre 0 et 1."
                )

        if not 1 <= self.minimum_reliable_criteria <= len(
            CRITERION_MAX_SCORES
        ):
            raise ValueError(
                "Le nombre minimal de critères fiables doit être compris "
                f"entre 1 et {len(CRITERION_MAX_SCORES)}."
            )

        if self.opposite_guard_weight < 0.0:
            raise ValueError(
                "Le poids de la main opposée ne peut pas être négatif."
            )

        if self.return_to_guard_weight < 0.0:
            raise ValueError(
                "Le poids du retour en garde ne peut pas être négatif."
            )

        total_guard_weight = (
            self.opposite_guard_weight
            + self.return_to_guard_weight
        )

        if abs(total_guard_weight - 1.0) > 1e-9:
            raise ValueError(
                "Les poids de la garde doivent avoir une somme égale à 1."
            )


@dataclass(frozen=True, slots=True)
class _CriterionScore:
    """Sous-score interne avant construction du résultat final."""

    criterion: AnalysisCriterion
    score: float
    max_score: float
    reliable: bool
    confidence: float
    message: str
    severity: FeedbackSeverity
    measured_value: float | None = None
    unit: str | None = None


class ScoringEngine:
    """Convertit une analyse géométrique en score final sur 100."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
    ) -> None:
        self._config = config or ScoringConfig()

    @property
    def config(self) -> ScoringConfig:
        """Retourne la configuration active."""

        return self._config

    def calculate(
        self,
        analysis: GyakuZukiAnalysis,
    ) -> AnalysisResult:
        """Calcule le résultat final d'une analyse de Gyaku-zuki.

        Aucun score n'est produit lorsque :

        - l'analyse géométrique a échoué ;
        - la confiance globale est insuffisante ;
        - trop peu de critères sont fiables ;
        - l'extension est indisponible alors qu'elle est obligatoire.
        """

        if not analysis.success or analysis.impact is None:
            return AnalysisResult.failure(
                status=AnalysisStatus.PROCESSING_ERROR,
                warning=self._failure_warning(analysis),
                confidence=analysis.confidence,
            )

        if analysis.confidence < self._config.minimum_global_confidence:
            return AnalysisResult.failure(
                status=AnalysisStatus.INSUFFICIENT_VISIBILITY,
                warning=(
                    "L'analyse n'est pas suffisamment fiable. "
                    "Refilme le mouvement en gardant tout le corps visible."
                ),
                confidence=analysis.confidence,
            )

        observations = self._index_observations(
            analysis.observations
        )

        criterion_scores = (
            self._score_single_observation(
                criterion=AnalysisCriterion.EXTENSION,
                observation=observations.get(
                    ObservationName.ARM_EXTENSION
                ),
            ),
            self._score_single_observation(
                criterion=AnalysisCriterion.TRAJECTORY,
                observation=observations.get(
                    ObservationName.PUNCH_HEIGHT
                ),
            ),
            self._score_single_observation(
                criterion=AnalysisCriterion.TORSO,
                observation=observations.get(
                    ObservationName.TORSO_INCLINATION
                ),
            ),
            self._score_single_observation(
                criterion=AnalysisCriterion.LEGS,
                observation=observations.get(
                    ObservationName.LEGS
                ),
            ),
            self._score_guard(
                opposite_guard=observations.get(
                    ObservationName.OPPOSITE_GUARD
                ),
                return_to_guard=observations.get(
                    ObservationName.RETURN_TO_GUARD
                ),
            ),
        )

        reliable_scores = tuple(
            criterion_score
            for criterion_score in criterion_scores
            if criterion_score.reliable
        )

        extension_score = next(
            score
            for score in criterion_scores
            if score.criterion is AnalysisCriterion.EXTENSION
        )

        if (
            self._config.require_extension
            and not extension_score.reliable
        ):
            return AnalysisResult.failure(
                status=AnalysisStatus.INSUFFICIENT_VISIBILITY,
                warning=(
                    "L'extension du bras ne peut pas être mesurée avec "
                    "suffisamment de fiabilité. Aucun score n'a été généré."
                ),
                confidence=analysis.confidence,
            )

        if len(reliable_scores) < self._config.minimum_reliable_criteria:
            return AnalysisResult.failure(
                status=AnalysisStatus.INSUFFICIENT_VISIBILITY,
                warning=(
                    "Trop peu de critères sont suffisamment fiables pour "
                    "calculer un score. Refilme le mouvement en gardant les "
                    "poignets, épaules, hanches, genoux et chevilles visibles."
                ),
                confidence=analysis.confidence,
            )

        score_total = self._calculate_total_score(
            reliable_scores
        )

        criterion_results = tuple(
            self._to_criterion_result(score)
            for score in criterion_scores
        )

        strengths = self._select_strengths(reliable_scores)
        corrections = self._select_corrections(reliable_scores)

        warnings = list(analysis.warnings)

        unreliable_criteria = tuple(
            score.criterion.value
            for score in criterion_scores
            if not score.reliable
        )

        if unreliable_criteria:
            warnings.append(
                "Critères non comptabilisés faute de fiabilité : "
                + ", ".join(unreliable_criteria)
                + "."
            )

        subscores = {
            score.criterion: round(score.score, 1)
            for score in reliable_scores
        }

        return AnalysisResult(
            success=True,
            status=AnalysisStatus.SUCCESS,
            confidence=round(analysis.confidence, 3),
            score_total=round(score_total, 1),
            subscores=subscores,
            strengths=strengths,
            corrections=corrections,
            warnings=tuple(warnings),
            impact_frame=analysis.impact.frame_index,
            criterion_results=criterion_results,
        )

    def _score_single_observation(
        self,
        criterion: AnalysisCriterion,
        observation: TechnicalObservation | None,
    ) -> _CriterionScore:
        """Calcule un critère associé à une observation unique."""

        max_score = CRITERION_MAX_SCORES[criterion]

        if not self._is_observation_reliable(observation):
            return _CriterionScore(
                criterion=criterion,
                score=0.0,
                max_score=max_score,
                reliable=False,
                confidence=(
                    observation.confidence
                    if observation is not None
                    else 0.0
                ),
                message=(
                    observation.message
                    if observation is not None
                    else (
                        "Ce critère ne peut pas être évalué car la mesure "
                        "nécessaire est absente."
                    )
                ),
                severity=(
                    observation.severity
                    if observation is not None
                    else FeedbackSeverity.MEDIUM
                ),
                measured_value=(
                    observation.measured_value
                    if observation is not None
                    else None
                ),
                unit=(
                    observation.unit
                    if observation is not None
                    else None
                ),
            )

        assert observation is not None

        score_ratio = SEVERITY_SCORE_RATIOS[
            observation.severity
        ]

        return _CriterionScore(
            criterion=criterion,
            score=max_score * score_ratio,
            max_score=max_score,
            reliable=True,
            confidence=observation.confidence,
            message=observation.message,
            severity=observation.severity,
            measured_value=observation.measured_value,
            unit=observation.unit,
        )

    def _score_guard(
        self,
        opposite_guard: TechnicalObservation | None,
        return_to_guard: TechnicalObservation | None,
    ) -> _CriterionScore:
        """Combine la garde opposée et le retour du poignet.

        Si une seule mesure est fiable, elle est utilisée seule au lieu de
        fabriquer une valeur pour la mesure absente.
        """

        criterion = AnalysisCriterion.GUARD
        max_score = CRITERION_MAX_SCORES[criterion]

        reliable_observations: list[
            tuple[TechnicalObservation, float]
        ] = []

        if self._is_observation_reliable(opposite_guard):
            assert opposite_guard is not None
            reliable_observations.append(
                (
                    opposite_guard,
                    self._config.opposite_guard_weight,
                )
            )

        if self._is_observation_reliable(return_to_guard):
            assert return_to_guard is not None
            reliable_observations.append(
                (
                    return_to_guard,
                    self._config.return_to_guard_weight,
                )
            )

        if not reliable_observations:
            messages = [
                observation.message
                for observation in (
                    opposite_guard,
                    return_to_guard,
                )
                if observation is not None
            ]

            return _CriterionScore(
                criterion=criterion,
                score=0.0,
                max_score=max_score,
                reliable=False,
                confidence=0.0,
                message=(
                    " ".join(messages)
                    if messages
                    else (
                        "La garde et le retour du poignet ne peuvent pas "
                        "être évalués."
                    )
                ),
                severity=FeedbackSeverity.MEDIUM,
            )

        available_weight = sum(
            weight
            for _, weight in reliable_observations
        )

        weighted_ratio = sum(
            SEVERITY_SCORE_RATIOS[observation.severity]
            * weight
            for observation, weight in reliable_observations
        ) / available_weight

        confidence = sum(
            observation.confidence * weight
            for observation, weight in reliable_observations
        ) / available_weight

        severity = max(
            (
                observation.severity
                for observation, _ in reliable_observations
            ),
            key=lambda value: SEVERITY_PRIORITY[value],
        )

        message = " ".join(
            observation.message
            for observation, _ in reliable_observations
        )

        return _CriterionScore(
            criterion=criterion,
            score=max_score * weighted_ratio,
            max_score=max_score,
            reliable=True,
            confidence=confidence,
            message=message,
            severity=severity,
        )

    @staticmethod
    def _calculate_total_score(
        reliable_scores: tuple[_CriterionScore, ...],
    ) -> float:
        """Normalise le total lorsque certains critères sont indisponibles.

        Exemple : si 80 points sont mesurables et que 64 sont obtenus, le
        résultat global vaut ``64 / 80 * 100 = 80``.

        Le sous-score d'un critère indisponible reste absent du résultat.
        """

        available_maximum = sum(
            score.max_score
            for score in reliable_scores
        )

        if available_maximum <= 0.0:
            raise ValueError(
                "Le score maximal disponible doit être positif."
            )

        obtained_score = sum(
            score.score
            for score in reliable_scores
        )

        return obtained_score / available_maximum * TOTAL_MAX_SCORE

    def _is_observation_reliable(
        self,
        observation: TechnicalObservation | None,
    ) -> bool:
        """Vérifie la fiabilité minimale d'une observation."""

        return (
            observation is not None
            and observation.reliable
            and observation.confidence
            >= self._config.minimum_observation_confidence
        )

    @staticmethod
    def _index_observations(
        observations: tuple[TechnicalObservation, ...],
    ) -> dict[ObservationName, TechnicalObservation]:
        """Indexe les observations et refuse les doublons."""

        indexed: dict[
            ObservationName,
            TechnicalObservation,
        ] = {}

        for observation in observations:
            if observation.name in indexed:
                raise ValueError(
                    "Plusieurs observations ont été reçues pour "
                    f"'{observation.name.value}'."
                )

            indexed[observation.name] = observation

        return indexed

    @staticmethod
    def _to_criterion_result(
        criterion_score: _CriterionScore,
    ) -> CriterionResult:
        """Convertit un sous-score interne en modèle public."""

        return CriterionResult(
            criterion=criterion_score.criterion,
            score=round(criterion_score.score, 1),
            max_score=criterion_score.max_score,
            reliable=criterion_score.reliable,
            confidence=criterion_score.confidence,
            message=criterion_score.message,
            severity=criterion_score.severity,
            measured_value=criterion_score.measured_value,
            unit=criterion_score.unit,
        )

    @staticmethod
    def _select_strengths(
        scores: tuple[_CriterionScore, ...],
    ) -> tuple[FeedbackItem, ...]:
        """Sélectionne au maximum deux critères positifs."""

        candidates = sorted(
            (
                score
                for score in scores
                if score.severity in {
                    FeedbackSeverity.INFO,
                    FeedbackSeverity.LOW,
                }
            ),
            key=lambda score: (
                score.score / score.max_score,
                score.confidence,
            ),
            reverse=True,
        )

        return tuple(
            FeedbackItem(
                criterion=score.criterion,
                message=score.message,
                severity=score.severity,
                confidence=score.confidence,
            )
            for score in candidates[:2]
        )

    @staticmethod
    def _select_corrections(
        scores: tuple[_CriterionScore, ...],
    ) -> tuple[FeedbackItem, ...]:
        """Sélectionne au maximum trois corrections prioritaires."""

        candidates = sorted(
            (
                score
                for score in scores
                if score.severity in {
                    FeedbackSeverity.MEDIUM,
                    FeedbackSeverity.HIGH,
                }
            ),
            key=lambda score: (
                SEVERITY_PRIORITY[score.severity],
                1.0 - score.score / score.max_score,
                score.confidence,
            ),
            reverse=True,
        )

        return tuple(
            FeedbackItem(
                criterion=score.criterion,
                message=score.message,
                severity=score.severity,
                confidence=score.confidence,
            )
            for score in candidates[:3]
        )

    @staticmethod
    def _failure_warning(
        analysis: GyakuZukiAnalysis,
    ) -> str:
        """Produit le message d'échec le plus utile disponible."""

        if analysis.warnings:
            return analysis.warnings[0]

        return (
            "Le mouvement n'a pas pu être analysé avec suffisamment "
            "de fiabilité."
        )


def calculate_score(
    analysis: GyakuZukiAnalysis,
    config: ScoringConfig | None = None,
) -> AnalysisResult:
    """Fonction pratique pour calculer un score sans conserver le moteur."""

    return ScoringEngine(config).calculate(analysis)