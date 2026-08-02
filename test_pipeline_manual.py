"""Test manuel du pipeline complet de Karate Coach."""

from __future__ import annotations

import logging
from pathlib import Path

from src.analysis_pipeline import AnalysisPipeline


SOURCE_PATH = Path("data/uploads/test.mp4")
OUTPUT_PATH = Path("data/outputs/test_pipeline_annotated.mp4")


def display_progress(
    processed_frames: int,
    total_frames: int,
) -> None:
    """Affiche une progression simple dans le terminal."""

    if total_frames <= 0:
        return

    percentage = min(
        100.0,
        processed_frames / total_frames * 100.0,
    )

    print(
        f"\rTraitement : {percentage:5.1f}% "
        f"({processed_frames}/{total_frames})",
        end="",
        flush=True,
    )


def main() -> None:
    """Exécute le pipeline sur la vidéo de test."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Vidéo de test introuvable : {SOURCE_PATH.resolve()}"
        )

    with AnalysisPipeline() as pipeline:
        result = pipeline.analyze_video(
            source_path=SOURCE_PATH,
            output_path=OUTPUT_PATH,
            progress_callback=display_progress,
        )

    print()
    print("-" * 60)
    print(f"Analyse réussie : {result.success}")
    print(f"Statut : {result.status.value}")
    print(f"Confiance : {result.confidence:.1%}")

    if result.output_video_path is not None:
        print(
            "Vidéo annotée : "
            f"{result.output_video_path.resolve()}"
        )

    if not result.success:
        print("Aucun score généré.")

        for warning in result.warnings:
            print(f"Avertissement : {warning}")

        return

    print(f"Score global : {result.score_total:.1f}/100")
    print(f"Image d'impact : {result.impact_frame}")

    print("\nSous-scores :")
    for criterion, score in result.subscores.items():
        print(f"- {criterion.value}: {score:.1f}")

    if result.strengths:
        print("\nPoints positifs :")
        for strength in result.strengths:
            print(f"- {strength.message}")

    if result.corrections:
        print("\nCorrections prioritaires :")
        for correction in result.corrections:
            print(f"- {correction.message}")

    if result.warnings:
        print("\nAvertissements :")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()