"""Tests unitaires du moteur de score de Karate Coach."""

from __future__ import annotations

import pytest

from src.gyaku_zuki_analyzer import (
    GyakuZukiAnalysis,
    ObservationName,
    TechnicalObservation,
)
from src.models import (
    AnalysisCriterion,
    AnalysisStatus,
    BodySide,
    FeedbackSeverity,
    ImpactFrame,
)
from src.scoring import (
    ScoringConfig,
    ScoringEngine,
    calculate_score,
)


def make_observation(
    name: ObservationName,
    severity: FeedbackSeverity,
    *,
    reliable: bool = True,
    confidence: float = 0.90,
    message: str | None = None,
) -> TechnicalObservation:
    """Construit une observation synthétique."""

    return TechnicalObservation(
        name=name,
        reliable=reliable,
        confidence=confidence,
        message=message or f"Observation {name.value}.",
        severity=severity,
        measured_value=1.0 if reliable else None,
        unit="test_unit" if reliable else None,
    )


def make_impact() -> ImpactFrame:
    """Construit une image d'impact synthétique valide."""

    return ImpactFrame(
        frame_index=42,
        timestamp_seconds=1.4,
        striking_side=BodySide.RIGHT,
        confidence=0.90,
        wrist_shoulder_distance=2.0,
        elbow_angle_degrees=170.0,
        wrist_speed=1.5,
    )


def make_analysis(
    *,
    global_confidence: float = 0.90,
    extension_severity: FeedbackSeverity = FeedbackSeverity.INFO,
    trajectory_severity: FeedbackSeverity = FeedbackSeverity.INFO,
    torso_severity: FeedbackSeverity = FeedbackSeverity.INFO,
    legs_severity: FeedbackSeverity = FeedbackSeverity.INFO,
    opposite_guard_severity: FeedbackSeverity = FeedbackSeverity.INFO,
    return_severity: FeedbackSeverity = FeedbackSeverity.INFO,
) -> GyakuZukiAnalysis:
    """Construit une analyse complète et fiable."""

    observations = (
        make_observation(
            ObservationName.ARM_EXTENSION,
            extension_severity,
            message="Le bras semble correctement étendu.",
        ),
        make_observation(
            ObservationName.PUNCH_HEIGHT,
            trajectory_severity,
            message="La hauteur du poing paraît cohérente.",
        ),
        make_observation(
            ObservationName.TORSO_INCLINATION,
            torso_severity,
            message="Le buste semble stable.",
        ),
        make_observation(
            ObservationName.LEGS,
            legs_severity,
            message="La position visible des jambes paraît cohérente.",
        ),
        make_observation(
            ObservationName.OPPOSITE_GUARD,
            opposite_guard_severity,
            message="La main opposée semble proche du corps.",
        ),
        make_observation(
            ObservationName.RETURN_TO_GUARD,
            return_severity,
            message="Le poignet semble revenir vers la garde.",
        ),
    )

    return GyakuZukiAnalysis(
        success=True,
        confidence=global_confidence,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=observations,
    )


def test_perfect_analysis_scores_100() -> None:
    """Une analyse entièrement positive doit obtenir 100 points."""

    result = calculate_score(make_analysis())

    assert result.success is True
    assert result.status is AnalysisStatus.SUCCESS
    assert result.score_total == pytest.approx(100.0)
    assert result.impact_frame == 42


def test_perfect_analysis_has_expected_subscores() -> None:
    """Les pondérations maximales doivent respecter le cahier des charges."""

    result = calculate_score(make_analysis())

    assert result.subscores == {
        AnalysisCriterion.EXTENSION: 25.0,
        AnalysisCriterion.TRAJECTORY: 20.0,
        AnalysisCriterion.TORSO: 20.0,
        AnalysisCriterion.LEGS: 20.0,
        AnalysisCriterion.GUARD: 15.0,
    }


def test_medium_extension_scores_60_percent() -> None:
    """Une sévérité moyenne obtient 60 % des points du critère."""

    analysis = make_analysis(
        extension_severity=FeedbackSeverity.MEDIUM,
    )

    result = calculate_score(analysis)

    assert result.success is True
    assert result.subscores[
        AnalysisCriterion.EXTENSION
    ] == pytest.approx(15.0)

    assert result.score_total == pytest.approx(90.0)


def test_high_extension_problem_scores_25_percent() -> None:
    """Une correction prioritaire conserve seulement 25 % des points."""

    analysis = make_analysis(
        extension_severity=FeedbackSeverity.HIGH,
    )

    result = calculate_score(analysis)

    assert result.success is True
    assert result.subscores[
        AnalysisCriterion.EXTENSION
    ] == pytest.approx(6.2, abs=0.1)


def test_guard_combines_opposite_hand_and_return() -> None:
    """La garde doit combiner ses deux observations avec les bons poids."""

    analysis = make_analysis(
        opposite_guard_severity=FeedbackSeverity.INFO,
        return_severity=FeedbackSeverity.HIGH,
    )

    result = calculate_score(analysis)

    expected_ratio = (
        1.00 * 0.40
        + 0.25 * 0.60
    )

    assert result.subscores[
        AnalysisCriterion.GUARD
    ] == pytest.approx(
        15.0 * expected_ratio,
        abs=0.1,
    )


def test_strengths_are_limited_to_two() -> None:
    """Le résultat ne doit contenir que deux points positifs maximum."""

    result = calculate_score(make_analysis())

    assert len(result.strengths) == 2
    assert len(result.corrections) == 0


def test_corrections_are_limited_to_three() -> None:
    """Les corrections doivent être triées et limitées à trois."""

    analysis = make_analysis(
        extension_severity=FeedbackSeverity.HIGH,
        trajectory_severity=FeedbackSeverity.MEDIUM,
        torso_severity=FeedbackSeverity.HIGH,
        legs_severity=FeedbackSeverity.MEDIUM,
        opposite_guard_severity=FeedbackSeverity.HIGH,
        return_severity=FeedbackSeverity.HIGH,
    )

    result = calculate_score(analysis)

    assert result.success is True
    assert len(result.corrections) == 3
    assert all(
        correction.severity in {
            FeedbackSeverity.HIGH,
            FeedbackSeverity.MEDIUM,
        }
        for correction in result.corrections
    )


def test_high_severity_corrections_are_prioritized() -> None:
    """Les problèmes importants doivent passer avant les problèmes moyens."""

    analysis = make_analysis(
        extension_severity=FeedbackSeverity.MEDIUM,
        trajectory_severity=FeedbackSeverity.HIGH,
        torso_severity=FeedbackSeverity.MEDIUM,
    )

    result = calculate_score(analysis)

    assert result.corrections
    assert (
        result.corrections[0].criterion
        is AnalysisCriterion.TRAJECTORY
    )
    assert (
        result.corrections[0].severity
        is FeedbackSeverity.HIGH
    )


def test_failed_analysis_does_not_generate_score() -> None:
    """Une analyse géométrique échouée ne doit produire aucun score."""

    analysis = GyakuZukiAnalysis(
        success=False,
        confidence=0.20,
        warnings=(
            "Le bras qui frappe n'a pas pu être identifié.",
        ),
    )

    result = calculate_score(analysis)

    assert result.success is False
    assert result.score_total is None
    assert result.impact_frame is None
    assert result.warnings


def test_low_global_confidence_does_not_generate_score() -> None:
    """Une confiance globale faible doit bloquer le score."""

    analysis = make_analysis(
        global_confidence=0.40,
    )

    result = calculate_score(analysis)

    assert result.success is False
    assert (
        result.status
        is AnalysisStatus.INSUFFICIENT_VISIBILITY
    )
    assert result.score_total is None


def test_unreliable_extension_blocks_score() -> None:
    """L'extension est un critère obligatoire par défaut."""

    analysis = make_analysis()

    observations = tuple(
        make_observation(
            observation.name,
            observation.severity,
            reliable=(
                observation.name
                is not ObservationName.ARM_EXTENSION
            ),
            confidence=(
                0.0
                if observation.name
                is ObservationName.ARM_EXTENSION
                else 0.90
            ),
        )
        for observation in analysis.observations
    )

    unreliable_analysis = GyakuZukiAnalysis(
        success=True,
        confidence=0.90,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=observations,
    )

    result = calculate_score(unreliable_analysis)

    assert result.success is False
    assert result.score_total is None
    assert (
        result.status
        is AnalysisStatus.INSUFFICIENT_VISIBILITY
    )


def test_too_few_reliable_criteria_blocks_score() -> None:
    """Moins de quatre critères fiables ne doivent pas donner de score."""

    reliable_names = {
        ObservationName.ARM_EXTENSION,
        ObservationName.PUNCH_HEIGHT,
        ObservationName.TORSO_INCLINATION,
    }

    observations = tuple(
        make_observation(
            name,
            FeedbackSeverity.INFO,
            reliable=name in reliable_names,
            confidence=0.90 if name in reliable_names else 0.10,
        )
        for name in ObservationName
    )

    analysis = GyakuZukiAnalysis(
        success=True,
        confidence=0.90,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=observations,
    )

    result = calculate_score(analysis)

    assert result.success is False
    assert result.score_total is None


def test_one_missing_non_required_criterion_is_normalized() -> None:
    """Un critère non obligatoire absent peut être exclu du total."""

    analysis = make_analysis()

    observations = tuple(
        make_observation(
            observation.name,
            observation.severity,
            reliable=(
                observation.name
                is not ObservationName.LEGS
            ),
            confidence=(
                0.0
                if observation.name
                is ObservationName.LEGS
                else 0.90
            ),
            message=observation.message,
        )
        for observation in analysis.observations
    )

    partial_analysis = GyakuZukiAnalysis(
        success=True,
        confidence=0.90,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=observations,
    )

    result = calculate_score(partial_analysis)

    assert result.success is True
    assert result.score_total == pytest.approx(100.0)
    assert AnalysisCriterion.LEGS not in result.subscores
    assert result.warnings


def test_low_observation_confidence_makes_criterion_unreliable() -> None:
    """Une observation sous le seuil ne doit pas entrer dans le calcul."""

    analysis = make_analysis()

    observations = tuple(
        make_observation(
            observation.name,
            observation.severity,
            confidence=(
                0.40
                if observation.name is ObservationName.LEGS
                else 0.90
            ),
            message=observation.message,
        )
        for observation in analysis.observations
    )

    partial_analysis = GyakuZukiAnalysis(
        success=True,
        confidence=0.90,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=observations,
    )

    result = calculate_score(partial_analysis)

    assert result.success is True
    assert AnalysisCriterion.LEGS not in result.subscores


def test_duplicate_observation_is_rejected() -> None:
    """Deux observations du même type signalent une erreur d'architecture."""

    duplicate = make_observation(
        ObservationName.ARM_EXTENSION,
        FeedbackSeverity.INFO,
    )

    analysis = GyakuZukiAnalysis(
        success=True,
        confidence=0.90,
        striking_side=BodySide.RIGHT,
        impact=make_impact(),
        observations=(
            duplicate,
            duplicate,
            make_observation(
                ObservationName.PUNCH_HEIGHT,
                FeedbackSeverity.INFO,
            ),
            make_observation(
                ObservationName.TORSO_INCLINATION,
                FeedbackSeverity.INFO,
            ),
            make_observation(
                ObservationName.LEGS,
                FeedbackSeverity.INFO,
            ),
            make_observation(
                ObservationName.OPPOSITE_GUARD,
                FeedbackSeverity.INFO,
            ),
            make_observation(
                ObservationName.RETURN_TO_GUARD,
                FeedbackSeverity.INFO,
            ),
        ),
    )

    with pytest.raises(ValueError):
        calculate_score(analysis)


def test_invalid_guard_weights_are_rejected() -> None:
    """Les deux poids de garde doivent avoir une somme égale à un."""

    with pytest.raises(ValueError):
        ScoringConfig(
            opposite_guard_weight=0.50,
            return_to_guard_weight=0.60,
        )


def test_custom_minimum_confidence_is_applied() -> None:
    """La configuration doit permettre d'ajuster la confiance minimale."""

    engine = ScoringEngine(
        ScoringConfig(
            minimum_global_confidence=0.80,
        )
    )

    result = engine.calculate(
        make_analysis(global_confidence=0.70)
    )

    assert result.success is False
    assert result.score_total is None