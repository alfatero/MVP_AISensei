"""Tests unitaires du pipeline complet de Karate Coach."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisPipelineConfig,
)
from src.gyaku_zuki_analyzer import GyakuZukiAnalysis
from src.models import (
    AnalysisResult,
    AnalysisStatus,
    BodySide,
    ImpactFrame,
    PoseFrame,
    VideoMetadata,
)
from src.video_processor import (
    InvalidVideoError,
    VideoProcessingResult,
)


def make_pose_frame(
    frame_index: int,
    *,
    pose_detected: bool = True,
    person_count: int = 1,
) -> PoseFrame:
    """Construit une image de pose minimale pour les tests du pipeline."""

    return PoseFrame(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10.0,
        landmarks={},
        pose_detected=pose_detected,
        person_count=person_count if pose_detected else 0,
    )


def make_processing_result(
    tmp_path: Path,
    *,
    frame_count: int = 8,
    detected_pose_frames: int | None = None,
    multiple_people_frames: int = 0,
) -> VideoProcessingResult:
    """Construit un résultat technique synthétique."""

    if detected_pose_frames is None:
        detected_pose_frames = frame_count

    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "annotated.mp4"

    source_path.write_bytes(b"source")
    output_path.write_bytes(b"annotated")

    pose_frames = tuple(
        make_pose_frame(
            frame_index=index,
            pose_detected=index < detected_pose_frames,
            person_count=(
                2
                if index < multiple_people_frames
                else 1
            ),
        )
        for index in range(frame_count)
    )

    metadata = VideoMetadata(
        source_path=source_path,
        fps=10.0,
        width=320,
        height=240,
        frame_count=frame_count,
        duration_seconds=frame_count / 10.0,
    )

    return VideoProcessingResult(
        metadata=metadata,
        output_path=output_path,
        pose_frames=pose_frames,
        processed_width=320,
        processed_height=240,
        detected_pose_frames=detected_pose_frames,
        multiple_people_frames=multiple_people_frames,
    )


def make_technical_analysis(
    *,
    success: bool = True,
    confidence: float = 0.90,
) -> GyakuZukiAnalysis:
    """Construit une analyse métier synthétique."""

    if not success:
        return GyakuZukiAnalysis(
            success=False,
            confidence=confidence,
            warnings=(
                "Le bras qui frappe n'a pas pu être identifié.",
            ),
        )

    impact = ImpactFrame(
        frame_index=4,
        timestamp_seconds=0.4,
        striking_side=BodySide.RIGHT,
        confidence=0.90,
        wrist_shoulder_distance=2.0,
        elbow_angle_degrees=170.0,
        wrist_speed=1.5,
    )

    return GyakuZukiAnalysis(
        success=True,
        confidence=confidence,
        striking_side=BodySide.RIGHT,
        impact=impact,
        observations=(),
    )


def make_scoring_result() -> AnalysisResult:
    """Construit un résultat final synthétique."""

    return AnalysisResult(
        success=True,
        status=AnalysisStatus.SUCCESS,
        confidence=0.90,
        score_total=82.0,
        subscores={},
        strengths=(),
        corrections=(),
        warnings=(),
        impact_frame=4,
    )


class FakeVideoProcessor:
    """Faux processeur vidéo injectable."""

    def __init__(
        self,
        result: VideoProcessingResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[
            tuple[Path, Path, Any]
        ] = []

    def process(
        self,
        source_path: Path | str,
        output_path: Path | str,
        progress_callback: Any = None,
    ) -> VideoProcessingResult:
        self.calls.append(
            (
                Path(source_path),
                Path(output_path),
                progress_callback,
            )
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise RuntimeError(
                "Le faux processeur ne contient aucun résultat."
            )

        return self.result


class FakeAnalyzer:
    """Faux analyseur Gyaku-zuki."""

    def __init__(
        self,
        result: GyakuZukiAnalysis,
    ) -> None:
        self.result = result
        self.received_frames: tuple[PoseFrame, ...] | None = None

    def analyze(
        self,
        frames: tuple[PoseFrame, ...],
    ) -> GyakuZukiAnalysis:
        self.received_frames = tuple(frames)
        return self.result


class FakeScoringEngine:
    """Faux moteur de score."""

    def __init__(
        self,
        result: AnalysisResult,
    ) -> None:
        self.result = result
        self.received_analysis: GyakuZukiAnalysis | None = None

    def calculate(
        self,
        analysis: GyakuZukiAnalysis,
    ) -> AnalysisResult:
        self.received_analysis = analysis
        return self.result


def test_pipeline_executes_all_components(
    tmp_path: Path,
) -> None:
    """Le pipeline doit chaîner traitement, analyse et score."""

    processing_result = make_processing_result(tmp_path)
    technical_analysis = make_technical_analysis()
    scoring_result = make_scoring_result()

    video_processor = FakeVideoProcessor(processing_result)
    analyzer = FakeAnalyzer(technical_analysis)
    scoring_engine = FakeScoringEngine(scoring_result)

    pipeline = AnalysisPipeline(
        video_processor=video_processor,
        analyzer=analyzer,
        scoring_engine=scoring_engine,
    )

    result = pipeline.analyze_video(
        source_path=processing_result.metadata.source_path,
        output_path=processing_result.output_path,
    )

    assert result.success is True
    assert result.score_total == pytest.approx(82.0)
    assert result.output_video_path == processing_result.output_path

    assert analyzer.received_frames == processing_result.pose_frames
    assert scoring_engine.received_analysis is technical_analysis
    assert len(video_processor.calls) == 1


def test_pipeline_forwards_progress_callback(
    tmp_path: Path,
) -> None:
    """La fonction de progression doit être transmise au processeur."""

    processing_result = make_processing_result(tmp_path)
    video_processor = FakeVideoProcessor(processing_result)

    callback_calls: list[tuple[int, int]] = []

    def progress_callback(
        current: int,
        total: int,
    ) -> None:
        callback_calls.append((current, total))

    pipeline = AnalysisPipeline(
        video_processor=video_processor,
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    pipeline.analyze_video(
        source_path=processing_result.metadata.source_path,
        output_path=processing_result.output_path,
        progress_callback=progress_callback,
    )

    assert video_processor.calls[0][2] is progress_callback


def test_pipeline_rejects_video_without_pose(
    tmp_path: Path,
) -> None:
    """Aucune pose détectée ne doit produire aucun score."""

    processing_result = make_processing_result(
        tmp_path,
        detected_pose_frames=0,
    )

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is False
    assert result.status is AnalysisStatus.NO_POSE_DETECTED
    assert result.score_total is None
    assert result.output_video_path == processing_result.output_path


def test_pipeline_rejects_multiple_people(
    tmp_path: Path,
) -> None:
    """Une seconde personne visible doit bloquer l'analyse."""

    processing_result = make_processing_result(
        tmp_path,
        multiple_people_frames=2,
    )

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is False
    assert (
        result.status
        is AnalysisStatus.MULTIPLE_PEOPLE_DETECTED
    )
    assert result.score_total is None


def test_pipeline_rejects_low_pose_detection_ratio(
    tmp_path: Path,
) -> None:
    """Une pose trop rarement détectée doit bloquer le score."""

    processing_result = make_processing_result(
        tmp_path,
        frame_count=10,
        detected_pose_frames=4,
    )

    pipeline = AnalysisPipeline(
        config=AnalysisPipelineConfig(
            minimum_pose_detection_ratio=0.60,
        ),
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is False
    assert (
        result.status
        is AnalysisStatus.INSUFFICIENT_VISIBILITY
    )
    assert result.confidence == pytest.approx(0.40)


def test_pipeline_rejects_too_short_sequence(
    tmp_path: Path,
) -> None:
    """Une séquence de moins de cinq images doit être refusée."""

    processing_result = make_processing_result(
        tmp_path,
        frame_count=4,
    )

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is False
    assert result.status is AnalysisStatus.MOVEMENT_TOO_SHORT
    assert result.score_total is None


def test_pipeline_converts_analyzer_failure(
    tmp_path: Path,
) -> None:
    """Un échec métier ne doit pas appeler le moteur de score."""

    processing_result = make_processing_result(tmp_path)
    analyzer = FakeAnalyzer(
        make_technical_analysis(
            success=False,
            confidence=0.50,
        )
    )
    scoring_engine = FakeScoringEngine(make_scoring_result())

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=analyzer,
        scoring_engine=scoring_engine,
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is False
    assert (
        result.status
        is AnalysisStatus.INSUFFICIENT_VISIBILITY
    )
    assert result.score_total is None
    assert scoring_engine.received_analysis is None


def test_pipeline_converts_invalid_video_error(
    tmp_path: Path,
) -> None:
    """Une vidéo invalide doit devenir un résultat d'échec propre."""

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(
            error=InvalidVideoError(
                "La vidéo ne peut pas être ouverte."
            )
        ),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        tmp_path / "invalid.mp4"
    )

    assert result.success is False
    assert result.status is AnalysisStatus.INVALID_VIDEO
    assert result.score_total is None
    assert result.warnings


def test_pipeline_can_allow_multiple_people(
    tmp_path: Path,
) -> None:
    """La règle multi-personnes doit rester configurable."""

    processing_result = make_processing_result(
        tmp_path,
        multiple_people_frames=2,
    )

    pipeline = AnalysisPipeline(
        config=AnalysisPipelineConfig(
            reject_multiple_people=False,
        ),
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    result = pipeline.analyze_video(
        processing_result.metadata.source_path
    )

    assert result.success is True
    assert result.score_total == pytest.approx(82.0)


def test_pipeline_config_rejects_invalid_ratio() -> None:
    """Un taux de détection hors de la plage 0-1 doit être refusé."""

    with pytest.raises(ValueError):
        AnalysisPipelineConfig(
            minimum_pose_detection_ratio=1.10,
        )


def test_pipeline_cannot_be_used_after_close(
    tmp_path: Path,
) -> None:
    """Un pipeline fermé ne doit plus accepter de traitement."""

    processing_result = make_processing_result(tmp_path)

    pipeline = AnalysisPipeline(
        video_processor=FakeVideoProcessor(processing_result),
        analyzer=FakeAnalyzer(make_technical_analysis()),
        scoring_engine=FakeScoringEngine(make_scoring_result()),
    )

    pipeline.close()

    with pytest.raises(RuntimeError):
        pipeline.analyze_video(
            processing_result.metadata.source_path
        )