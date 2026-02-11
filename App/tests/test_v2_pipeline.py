#!/usr/bin/env python3
"""
Tests pour le pipeline v2 de detection de moments viraux.

Valide les imports, les dataclasses, les algorithmes d'agregation,
la fusion des scores et l'extraction JSON robuste.

Usage:
    python tests/test_v2_pipeline.py              # Tous les tests
    python tests/test_v2_pipeline.py --quick       # Imports seulement
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Ajouter le dossier racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Supprimer les warnings MediaPipe/TF
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------
# Infrastructure de test (meme pattern que test_pipeline.py)
# ------------------------------------------------------------------

@dataclass
class TestResult:
    """Resultat d'un test."""
    name: str
    success: bool
    message: str
    details: Optional[str] = None


def _run_test(name: str, fn) -> TestResult:
    """Execute une fonction de test et capture les exceptions."""
    try:
        fn()
        return TestResult(name=name, success=True, message='OK')
    except AssertionError as e:
        return TestResult(name=name, success=False, message=f'ASSERT: {e}')
    except Exception as e:
        return TestResult(name=name, success=False, message=f'ERROR: {e}')


# ------------------------------------------------------------------
# Tests d'imports
# ------------------------------------------------------------------

def test_imports() -> List[TestResult]:
    """Teste les imports de tous les modules v2."""
    results = []

    v2_modules = [
        ('src.v2', 'ViralDetectorV2, detect_viral_moments'),
        ('src.v2.models', 'ContentType, WordTimestamp, TranscriptSegment, '
         'AudioFeatures, AudioEngagementScore, LLMChapter, LLMScoredMoment, '
         'ViralCandidate, PipelineConfig'),
        ('src.v2.audio_features', 'AudioFeatureExtractor, DiarizationExtractor, '
         'EmotionExtractor'),
        ('src.v2.score_aggregator', 'AudioScoreAggregator'),
        ('src.v2.fusion', 'ScoreFusion'),
        ('src.v2.pipeline', 'ViralDetectorV2, detect_viral_moments'),
    ]

    for module_name, classes in v2_modules:
        try:
            class_list = [c.strip() for c in classes.split(',')]
            module = __import__(module_name, fromlist=class_list)
            missing = [c for c in class_list if not hasattr(module, c)]
            if missing:
                results.append(TestResult(
                    name=f'Import {module_name}',
                    success=False,
                    message=f'Classes manquantes : {", ".join(missing)}'
                ))
            else:
                results.append(TestResult(
                    name=f'Import {module_name}',
                    success=True,
                    message=f'OK ({len(class_list)} classes)'
                ))
        except Exception as e:
            results.append(TestResult(
                name=f'Import {module_name}',
                success=False,
                message=str(e)
            ))

    # Test import du llm_analyzer (peut echouer si llama-cpp-python absent)
    try:
        from src.v2.llm_analyzer import LLMAnalyzer
        results.append(TestResult(
            name='Import src.v2.llm_analyzer',
            success=True,
            message='OK (LLMAnalyzer)'
        ))
    except ImportError as e:
        results.append(TestResult(
            name='Import src.v2.llm_analyzer',
            success=True,  # Import echoue est acceptable (dep optionnelle)
            message=f'OK (skip attendu : {e})'
        ))
    except Exception as e:
        results.append(TestResult(
            name='Import src.v2.llm_analyzer',
            success=False,
            message=str(e)
        ))

    return results


# ------------------------------------------------------------------
# Tests des modeles (dataclasses)
# ------------------------------------------------------------------

def test_models() -> List[TestResult]:
    """Teste les dataclasses du module models."""
    results = []

    from src.v2.models import (
        ContentType, WordTimestamp, TranscriptSegment,
        AudioFeatures, AudioEngagementScore, LLMChapter,
        LLMScoredMoment, ViralCandidate, PipelineConfig,
    )

    # Test ContentType enum
    def _test_content_type_enum():
        assert ContentType.PODCAST.value == 'podcast'
        assert ContentType.UNKNOWN.value == 'unknown'
        assert ContentType('comedy') == ContentType.COMEDY

    results.append(_run_test('ContentType enum', _test_content_type_enum))

    # Test ViralCandidate.to_viral_moment()
    def _test_to_viral_moment():
        candidate = ViralCandidate(
            start=10.0, end=70.0,
            final_score=0.85,
            emotion='surprise',
            reason='Moment intense',
        )
        vm = candidate.to_viral_moment()
        assert vm.start_time == 10.0, f'start_time={vm.start_time}'
        assert vm.end_time == 70.0, f'end_time={vm.end_time}'
        assert vm.score == 0.85, f'score={vm.score}'
        assert '[surprise]' in vm.reason, f'reason={vm.reason}'

    results.append(_run_test('ViralCandidate.to_viral_moment()', _test_to_viral_moment))

    # Test ViralCandidate.to_viral_moment() sans emotion
    def _test_to_viral_moment_no_emotion():
        candidate = ViralCandidate(
            start=5.0, end=35.0,
            final_score=0.70,
            emotion='',
            reason='Audio peak only',
        )
        vm = candidate.to_viral_moment()
        assert vm.reason == 'Audio peak only', f'reason={vm.reason}'
        assert '[' not in vm.reason, f'brackets in reason: {vm.reason}'

    results.append(_run_test(
        'ViralCandidate.to_viral_moment() no emotion',
        _test_to_viral_moment_no_emotion
    ))

    # Test ViralCandidate.duration property
    def _test_duration_property():
        candidate = ViralCandidate(start=10.0, end=70.0)
        assert candidate.duration == 60.0, f'duration={candidate.duration}'

    results.append(_run_test('ViralCandidate.duration', _test_duration_property))

    # Test LLMScoredMoment.compute_composite()
    def _test_compute_composite():
        moment = LLMScoredMoment(
            start=0.0, end=60.0,
            hook_strength=0.8,
            emotional_intensity=0.7,
            standalone_clarity=0.9,
            quotability=0.6,
            tension_arc=0.5,
            controversy=0.4,
        )
        score = moment.compute_composite()
        # Verification manuelle :
        # 0.8*0.25 + 0.7*0.20 + 0.9*0.20 + 0.6*0.15 + 0.5*0.10 + 0.4*0.10
        # = 0.20 + 0.14 + 0.18 + 0.09 + 0.05 + 0.04 = 0.70
        assert abs(score - 0.70) < 0.001, f'score={score} expected 0.70'
        assert moment.composite_score == score

    results.append(_run_test('LLMScoredMoment.compute_composite()', _test_compute_composite))

    # Test PipelineConfig defaults
    def _test_pipeline_config_defaults():
        config = PipelineConfig()
        assert config.min_clip_duration == 30.0
        assert config.max_clip_duration == 90.0
        assert config.max_clips == 5
        total = config.weight_llm + config.weight_audio + config.weight_speech + config.weight_structural
        assert abs(total - 1.0) < 0.001, f'Weights sum to {total}, expected 1.0'

    results.append(_run_test('PipelineConfig defaults', _test_pipeline_config_defaults))

    return results


# ------------------------------------------------------------------
# Tests du score aggregator
# ------------------------------------------------------------------

def test_score_aggregator() -> List[TestResult]:
    """Teste l'agregation des scores audio."""
    results = []

    from src.v2.models import AudioFeatures, AudioEngagementScore, ContentType
    from src.v2.score_aggregator import AudioScoreAggregator, _zscore_normalize

    # Test zscore_normalize avec valeurs identiques (std=0)
    def _test_zscore_normalize_constant():
        import numpy as np
        values = np.array([5.0, 5.0, 5.0])
        normalized = _zscore_normalize(values)
        # Tous a 0.5 quand std=0
        for v in normalized:
            assert abs(v - 0.5) < 0.001, f'Expected 0.5, got {v}'

    results.append(_run_test('zscore_normalize constant', _test_zscore_normalize_constant))

    # Test zscore_normalize avec valeurs variees
    def _test_zscore_normalize_varied():
        import numpy as np
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = _zscore_normalize(values)
        # Le plus bas doit etre < 0.5, le plus haut > 0.5
        assert normalized[0] < 0.5, f'Min should be < 0.5, got {normalized[0]}'
        assert normalized[-1] > 0.5, f'Max should be > 0.5, got {normalized[-1]}'
        # Tout doit etre dans [0, 1]
        for v in normalized:
            assert 0.0 <= v <= 1.0, f'Out of range: {v}'

    results.append(_run_test('zscore_normalize varied', _test_zscore_normalize_varied))

    # Test zscore_normalize vide
    def _test_zscore_normalize_empty():
        import numpy as np
        values = np.array([])
        normalized = _zscore_normalize(values)
        assert len(normalized) == 0

    results.append(_run_test('zscore_normalize empty', _test_zscore_normalize_empty))

    # Test aggregate() avec des features synthetiques
    def _test_aggregate_basic():
        features = [
            AudioFeatures(start=0.0, end=10.0, rms_energy=0.1, pitch_std=5.0, speech_rate_delta=-0.5),
            AudioFeatures(start=10.0, end=20.0, rms_energy=0.5, pitch_std=20.0, speech_rate_delta=1.5),
            AudioFeatures(start=20.0, end=30.0, rms_energy=0.3, pitch_std=10.0, speech_rate_delta=0.0),
        ]
        aggregator = AudioScoreAggregator(content_type=ContentType.UNKNOWN)
        scores = aggregator.aggregate(features)

        assert len(scores) == 3, f'Expected 3, got {len(scores)}'
        for s in scores:
            assert 0.0 <= s.score <= 1.5, f'Score out of range: {s.score}'
            assert s.start >= 0.0

    results.append(_run_test('aggregate basic', _test_aggregate_basic))

    # Test aggregate() vide
    def _test_aggregate_empty():
        aggregator = AudioScoreAggregator()
        scores = aggregator.aggregate([])
        assert scores == []

    results.append(_run_test('aggregate empty', _test_aggregate_empty))

    # Test find_peak_regions()
    def _test_find_peak_regions():
        aggregator = AudioScoreAggregator()
        # Creer des scores synthetiques avec un pic clair
        scores = [
            AudioEngagementScore(start=0.0, end=10.0, score=0.3),
            AudioEngagementScore(start=10.0, end=20.0, score=0.3),
            AudioEngagementScore(start=20.0, end=30.0, score=0.9),
            AudioEngagementScore(start=30.0, end=40.0, score=0.85),
            AudioEngagementScore(start=40.0, end=50.0, score=0.3),
            AudioEngagementScore(start=50.0, end=60.0, score=0.3),
        ]
        peaks = aggregator.find_peak_regions(scores, min_duration=15.0, max_duration=90.0)
        # Le pic 20-40s devrait etre detecte et ajuste a min 15s
        assert len(peaks) >= 1, f'Expected at least 1 peak, got {len(peaks)}'
        # Le meilleur pic devrait couvrir la zone 20-40s
        best_start, best_end, best_score = peaks[0]
        assert best_start <= 25.0, f'Peak start too late: {best_start}'
        assert best_end >= 35.0, f'Peak end too early: {best_end}'

    results.append(_run_test('find_peak_regions', _test_find_peak_regions))

    # Test find_peak_regions() vide
    def _test_find_peak_regions_empty():
        aggregator = AudioScoreAggregator()
        peaks = aggregator.find_peak_regions([])
        assert peaks == []

    results.append(_run_test('find_peak_regions empty', _test_find_peak_regions_empty))

    # Test content-type weights
    def _test_content_type_weights():
        podcast_agg = AudioScoreAggregator(content_type=ContentType.PODCAST)
        comedy_agg = AudioScoreAggregator(content_type=ContentType.COMEDY)
        assert podcast_agg.weights['speech_dynamics'] > comedy_agg.weights['speech_dynamics']
        assert comedy_agg.weights['energy'] > podcast_agg.weights['energy']

    results.append(_run_test('content-type weights', _test_content_type_weights))

    return results


# ------------------------------------------------------------------
# Tests de la fusion
# ------------------------------------------------------------------

def test_fusion() -> List[TestResult]:
    """Teste le systeme de fusion des scores."""
    results = []

    from src.v2.models import (
        AudioEngagementScore, LLMScoredMoment, ViralCandidate, PipelineConfig,
    )
    from src.v2.fusion import ScoreFusion

    # Test _remove_overlaps()
    def _test_remove_overlaps():
        candidates = [
            ViralCandidate(start=0.0, end=60.0, final_score=0.9),
            ViralCandidate(start=10.0, end=70.0, final_score=0.8),  # chevauche fortement le 1er
            ViralCandidate(start=120.0, end=180.0, final_score=0.7),  # pas de chevauchement
        ]
        kept = ScoreFusion._remove_overlaps(candidates)
        assert len(kept) == 2, f'Expected 2, got {len(kept)}'
        assert kept[0].final_score == 0.9
        assert kept[1].final_score == 0.7

    results.append(_run_test('remove_overlaps', _test_remove_overlaps))

    # Test _remove_overlaps() avec un seul candidat
    def _test_remove_overlaps_single():
        candidates = [ViralCandidate(start=0.0, end=60.0, final_score=0.9)]
        kept = ScoreFusion._remove_overlaps(candidates)
        assert len(kept) == 1

    results.append(_run_test('remove_overlaps single', _test_remove_overlaps_single))

    # Test _remove_overlaps() vide
    def _test_remove_overlaps_empty():
        kept = ScoreFusion._remove_overlaps([])
        assert kept == []

    results.append(_run_test('remove_overlaps empty', _test_remove_overlaps_empty))

    # Test _is_covered_by_llm()
    def _test_is_covered_by_llm():
        llm_moments = [
            LLMScoredMoment(start=10.0, end=70.0),
        ]
        # Pic entierement couvert
        assert ScoreFusion._is_covered_by_llm(20.0, 50.0, llm_moments) is True
        # Pic partiellement couvert (mais > 50%)
        assert ScoreFusion._is_covered_by_llm(50.0, 80.0, llm_moments) is True
        # Pic non couvert
        assert ScoreFusion._is_covered_by_llm(100.0, 160.0, llm_moments) is False

    results.append(_run_test('is_covered_by_llm', _test_is_covered_by_llm))

    # Test _compute_overlapping_audio_score()
    def _test_overlapping_audio_score():
        audio_scores = [
            AudioEngagementScore(start=0.0, end=10.0, score=0.3),
            AudioEngagementScore(start=10.0, end=20.0, score=0.8),
            AudioEngagementScore(start=20.0, end=30.0, score=0.6),
        ]
        # Fenetre qui chevauche 10-20 et 20-30
        avg = ScoreFusion._compute_overlapping_audio_score(15.0, 25.0, audio_scores)
        assert abs(avg - 0.7) < 0.001, f'Expected 0.7, got {avg}'

        # Fenetre sans chevauchement
        avg_empty = ScoreFusion._compute_overlapping_audio_score(50.0, 60.0, audio_scores)
        assert avg_empty == 0.0

    results.append(_run_test('overlapping audio score', _test_overlapping_audio_score))

    # Test fuse() avec des donnees synthetiques
    def _test_fuse_basic():
        config = PipelineConfig(
            min_viral_score=0.3,
            max_clips=3,
        )
        fusion = ScoreFusion(config)

        llm_moments = [
            LLMScoredMoment(
                start=10.0, end=70.0,
                hook_strength=0.8, emotional_intensity=0.7,
                standalone_clarity=0.9, quotability=0.6,
                tension_arc=0.5, controversy=0.4,
                hook_text='Question forte', emotion='surprise',
                reason='Moment intense', composite_score=0.70,
            ),
        ]
        audio_scores = [
            AudioEngagementScore(start=0.0, end=10.0, score=0.3, speech_dynamics_score=0.2),
            AudioEngagementScore(start=10.0, end=20.0, score=0.7, speech_dynamics_score=0.6),
            AudioEngagementScore(start=20.0, end=30.0, score=0.8, speech_dynamics_score=0.7),
            AudioEngagementScore(start=30.0, end=40.0, score=0.6, speech_dynamics_score=0.5),
            AudioEngagementScore(start=40.0, end=50.0, score=0.5, speech_dynamics_score=0.4),
            AudioEngagementScore(start=50.0, end=60.0, score=0.4, speech_dynamics_score=0.3),
            AudioEngagementScore(start=60.0, end=70.0, score=0.3, speech_dynamics_score=0.2),
        ]
        audio_peaks = []

        candidates = fusion.fuse(
            llm_moments=llm_moments,
            audio_scores=audio_scores,
            audio_peaks=audio_peaks,
        )
        assert len(candidates) >= 1, f'Expected >= 1 candidate, got {len(candidates)}'
        assert candidates[0].final_score > 0.0

    results.append(_run_test('fuse basic', _test_fuse_basic))

    # Test fuse() sans LLM mais avec pics audio
    def _test_fuse_audio_only():
        config = PipelineConfig(
            min_viral_score=0.1,
            max_clips=2,
            weight_llm=0.0,
            weight_audio=0.60,
            weight_speech=0.30,
            weight_structural=0.10,
        )
        fusion = ScoreFusion(config)

        audio_scores = [
            AudioEngagementScore(start=0.0, end=10.0, score=0.3, speech_dynamics_score=0.2),
            AudioEngagementScore(start=10.0, end=20.0, score=0.9, speech_dynamics_score=0.8),
            AudioEngagementScore(start=20.0, end=30.0, score=0.85, speech_dynamics_score=0.75),
        ]
        audio_peaks = [(10.0, 60.0, 0.85)]

        candidates = fusion.fuse(
            llm_moments=[],
            audio_scores=audio_scores,
            audio_peaks=audio_peaks,
        )
        assert len(candidates) >= 1, f'Expected >= 1 candidate, got {len(candidates)}'
        assert candidates[0].llm_score == 0.0, 'LLM score should be 0'

    results.append(_run_test('fuse audio-only', _test_fuse_audio_only))

    return results


# ------------------------------------------------------------------
# Tests de l'extraction JSON (llm_analyzer)
# ------------------------------------------------------------------

def test_json_extraction() -> List[TestResult]:
    """Teste l'extraction JSON robuste du LLMAnalyzer."""
    results = []

    try:
        from src.v2.llm_analyzer import LLMAnalyzer
    except Exception:
        results.append(TestResult(
            name='JSON extraction (skip)',
            success=True,
            message='LLMAnalyzer non importable — tests skip'
        ))
        return results

    # On instancie sans se connecter a un modele (on teste juste _extract_json)
    try:
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._content_type = 'unknown'
        analyzer._language = 'auto'
        analyzer._temperature = 0.3
        analyzer._max_retries = 1
        analyzer._model_format = 'phi'
        analyzer._stop_tokens = ['<|end|>']
    except Exception as e:
        results.append(TestResult(
            name='LLMAnalyzer instance',
            success=False,
            message=f'Cannot create instance: {e}'
        ))
        return results

    # Test 1 : JSON valide direct
    def _test_json_direct():
        data = analyzer._extract_json('[{"start": 10, "end": 60}]')
        assert isinstance(data, list), f'Expected list, got {type(data)}'
        assert data[0]['start'] == 10

    results.append(_run_test('JSON direct', _test_json_direct))

    # Test 2 : JSON dans un bloc code
    def _test_json_code_block():
        text = 'Voici le resultat:\n```json\n[{"start": 5, "end": 30}]\n```\n'
        data = analyzer._extract_json(text)
        assert isinstance(data, list)
        assert data[0]['start'] == 5

    results.append(_run_test('JSON code block', _test_json_code_block))

    # Test 3 : JSON avec texte avant/apres
    def _test_json_with_noise():
        text = 'Bien sur, voici les moments:\n[{"start": 100, "end": 200}]\nFin.'
        data = analyzer._extract_json(text)
        assert isinstance(data, list)
        assert data[0]['start'] == 100

    results.append(_run_test('JSON with noise', _test_json_with_noise))

    # Test 4 : JSON avec virgule pendante
    def _test_json_trailing_comma():
        text = '[{"start": 10, "end": 60,}]'
        data = analyzer._extract_json(text)
        assert data is not None

    results.append(_run_test('JSON trailing comma', _test_json_trailing_comma))

    # Test 5 : JSON objet unique (doit etre encapsule dans une liste)
    def _test_json_single_object():
        text = '{"start": 10, "end": 60, "title": "Test"}'
        data = analyzer._extract_json(text)
        assert isinstance(data, list), f'Expected list, got {type(data)}'
        assert len(data) == 1
        assert data[0]['start'] == 10

    results.append(_run_test('JSON single object', _test_json_single_object))

    # Test 6 : Texte vide
    def _test_json_empty():
        data = analyzer._extract_json('')
        assert data is None

    results.append(_run_test('JSON empty', _test_json_empty))

    # Test 7 : Texte sans JSON
    def _test_json_no_json():
        data = analyzer._extract_json('Ceci est un texte sans JSON')
        assert data is None

    results.append(_run_test('JSON no JSON', _test_json_no_json))

    # Test 8 : JSON avec guillemets simples
    def _test_json_single_quotes():
        text = "[{'start': 10, 'end': 60}]"
        data = analyzer._extract_json(text)
        # Peut echouer selon la robustesse du repair, c'est acceptable
        if data is not None:
            assert isinstance(data, list)

    results.append(_run_test('JSON single quotes', _test_json_single_quotes))

    # Test 9 : Objets JSON multiples concatenes
    def _test_json_multiple_objects():
        text = '{"start": 10, "end": 60}{"start": 100, "end": 160}'
        data = analyzer._extract_json(text)
        assert data is not None

    results.append(_run_test('JSON multiple objects', _test_json_multiple_objects))

    return results


# ------------------------------------------------------------------
# Test d'integration (pipeline v2 avec mock LLM)
# ------------------------------------------------------------------

def test_pipeline_integration() -> List[TestResult]:
    """Teste l'integration du pipeline v2 sans LLM reel."""
    results = []

    from src.v2.models import PipelineConfig, TranscriptSegment, WordTimestamp
    from src.v2.pipeline import ViralDetectorV2

    # Test : le pipeline ne crash pas avec des segments vides
    def _test_empty_segments():
        config = PipelineConfig(min_viral_score=0.3, max_clips=2)
        detector = ViralDetectorV2(config=config)
        # Pas de video reelle, pas de segments -> doit retourner une liste vide
        moments = detector.detect(
            video_path='/tmp/nonexistent_test_video.mp4',
            transcript_segments=[],
            words=[],
            content_type='unknown',
        )
        assert isinstance(moments, list), f'Expected list, got {type(moments)}'
        # Sans audio ni LLM, on attend une liste vide
        assert len(moments) == 0, f'Expected 0 moments, got {len(moments)}'

    results.append(_run_test('pipeline empty segments', _test_empty_segments))

    # Test : _estimate_duration_from_segments()
    def _test_estimate_duration():
        segments = [
            TranscriptSegment(start=0.0, end=30.0, text='Hello'),
            TranscriptSegment(start=30.0, end=60.0, text='World'),
            TranscriptSegment(start=60.0, end=120.0, text='End'),
        ]
        duration = ViralDetectorV2._estimate_duration_from_segments(segments)
        assert duration == 120.0, f'Expected 120.0, got {duration}'

    results.append(_run_test('estimate_duration', _test_estimate_duration))

    # Test : _estimate_duration_from_segments() avec liste vide
    def _test_estimate_duration_empty():
        duration = ViralDetectorV2._estimate_duration_from_segments([])
        assert duration == 0.0, f'Expected 0.0, got {duration}'

    results.append(_run_test('estimate_duration empty', _test_estimate_duration_empty))

    # Test : _apply_content_type()
    def _test_apply_content_type():
        config = PipelineConfig()
        detector = ViralDetectorV2(config=config)
        detector._apply_content_type('podcast')
        assert detector.config.content_type.value == 'podcast'

        detector._apply_content_type('INVALID_TYPE')
        assert detector.config.content_type.value == 'unknown'

    results.append(_run_test('apply_content_type', _test_apply_content_type))

    # Test : _adjust_weights_for_fallback()
    def _test_adjust_weights():
        config = PipelineConfig()
        detector = ViralDetectorV2(config=config)

        # Tout OK -> poids inchanges
        normal_config = detector._adjust_weights_for_fallback(audio_ok=True, llm_ok=True)
        assert normal_config.weight_llm == config.weight_llm
        assert normal_config.weight_audio == config.weight_audio

        # LLM seul -> audio a 0
        llm_only = detector._adjust_weights_for_fallback(audio_ok=False, llm_ok=True)
        assert llm_only.weight_audio == 0.0
        assert llm_only.weight_speech == 0.0
        assert llm_only.weight_llm > config.weight_llm  # augmente

        # Audio seul -> LLM a 0
        audio_only = detector._adjust_weights_for_fallback(audio_ok=True, llm_ok=False)
        assert audio_only.weight_llm == 0.0
        assert audio_only.weight_audio > config.weight_audio  # augmente

    results.append(_run_test('adjust_weights fallback', _test_adjust_weights))

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tests pipeline v2')
    parser.add_argument('--quick', action='store_true', help='Imports seulement')
    args = parser.parse_args()

    print('=' * 60)
    print('  ClipGenius — Tests du pipeline v2')
    print('=' * 60)
    print()

    all_results: List[TestResult] = []

    # Toujours tester les imports
    print('[1/5] Tests d\'imports...')
    all_results.extend(test_imports())

    if not args.quick:
        print('[2/5] Tests des modeles...')
        all_results.extend(test_models())

        print('[3/5] Tests du score aggregator...')
        all_results.extend(test_score_aggregator())

        print('[4/5] Tests de la fusion...')
        all_results.extend(test_fusion())

        print('[5/5] Tests d\'extraction JSON...')
        all_results.extend(test_json_extraction())

        print('[bonus] Tests d\'integration pipeline...')
        all_results.extend(test_pipeline_integration())

    # Affichage des resultats
    print()
    print('-' * 60)
    passed = sum(1 for r in all_results if r.success)
    failed = sum(1 for r in all_results if not r.success)

    for r in all_results:
        status = 'PASS' if r.success else 'FAIL'
        icon = '+' if r.success else 'X'
        print(f'  [{icon}] {status}  {r.name}: {r.message}')
        if r.details:
            print(f'         {r.details}')

    print()
    print('-' * 60)
    print(f'  Total : {passed}/{len(all_results)} passes, {failed} echecs')
    print('-' * 60)

    if failed > 0:
        print('\nCERTAINS TESTS ONT ECHOUE')
        sys.exit(1)
    else:
        print('\nTOUS LES TESTS SONT PASSES')
        sys.exit(0)


if __name__ == '__main__':
    main()
