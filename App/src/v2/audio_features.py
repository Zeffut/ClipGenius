"""
Extraction de features audio pour la detection de moments viraux.

Analyse un fichier video/audio et retourne des AudioFeatures par segment :
- Energie RMS et pics d'energie
- Pitch moyen, ecart-type et etendue (F0 via pyin)
- Debit de parole et variation relative
- Diarisation des locuteurs (pyannote, optionnel)
- Emotion vocale (speechbrain, optionnel)

Fonctionne 100% hors-ligne sur macOS Apple Silicon.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
from rich.console import Console

from .models import AudioFeatures, WordTimestamp

console = Console()

# Frequence d'echantillonnage cible pour le chargement audio
_SAMPLE_RATE: int = 16000

# Parametres pyin pour l'estimation de F0 (voix humaine)
_PYIN_FMIN: float = 50.0
_PYIN_FMAX: float = 500.0

# Seuil pour la detection de pic d'energie : mean + k * std
_ENERGY_PEAK_K: float = 1.5

# Duree minimale d'un segment exploitable (en secondes)
_MIN_SEGMENT_DURATION: float = 0.1


# ------------------------------------------------------------------
# Extracteurs optionnels
# ------------------------------------------------------------------

class DiarizationExtractor:
    """Encapsule pyannote-audio pour la diarisation des locuteurs.

    Si pyannote n'est pas installe ou si le modele n'est pas disponible,
    l'extracteur se desactive proprement (self.available = False).
    """

    def __init__(self) -> None:
        self.available: bool = False
        self._pipeline = None
        try:
            from pyannote.audio import Pipeline as _Pipeline  # noqa: F811
            self._pipeline_cls = _Pipeline
            self.available = True
            console.print('[dim]pyannote-audio disponible pour la diarisation[/dim]')
        except ImportError:
            console.print(
                '[yellow]pyannote-audio non installe — '
                'diarisation desactivee[/yellow]'
            )
        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Erreur lors de l\'import pyannote : {exc} — '
                'diarisation desactivee[/yellow]'
            )

    # ------------------------------------------------------------------

    def _load_pipeline(self) -> bool:
        """Charge le pipeline de diarisation si necessaire."""
        if self._pipeline is not None:
            return True
        if not self.available:
            return False
        try:
            self._pipeline = self._pipeline_cls.from_pretrained(
                'pyannote/speaker-diarization-3.1'
            )
            return True
        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Impossible de charger le modele pyannote : {exc}[/yellow]'
            )
            self.available = False
            return False

    # ------------------------------------------------------------------

    def extract(
        self,
        audio_path: str,
        segments: List[AudioFeatures],
    ) -> None:
        """Remplit num_speakers, speaker_changes et overlap_ratio.

        Args:
            audio_path: Chemin vers le fichier audio/video.
            segments: Liste de AudioFeatures a enrichir.
        """
        if not segments:
            return
        if not self._load_pipeline():
            return

        try:
            diarization = self._pipeline(audio_path)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Echec de la diarisation : {exc}[/yellow]'
            )
            return

        # Convertir la diarisation en liste d'intervalles (start, end, speaker)
        turns: List[tuple] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        for seg in segments:
            seg_start = seg.start
            seg_end = seg.end

            # Filtrer les tours qui chevauchent ce segment
            overlapping = [
                (max(t[0], seg_start), min(t[1], seg_end), t[2])
                for t in turns
                if t[0] < seg_end and t[1] > seg_start
            ]

            if not overlapping:
                seg.num_speakers = 0
                seg.speaker_changes = 0
                seg.overlap_ratio = 0.0
                continue

            # Nombre de locuteurs uniques
            speakers_in_seg = {t[2] for t in overlapping}
            seg.num_speakers = len(speakers_in_seg)

            # Changements de locuteur
            changes = 0
            prev_speaker: Optional[str] = None
            for _, _, spk in sorted(overlapping, key=lambda x: x[0]):
                if prev_speaker is not None and spk != prev_speaker:
                    changes += 1
                prev_speaker = spk
            seg.speaker_changes = changes

            # Ratio de chevauchement (overlap)
            # Construire un tableau pour compter les locuteurs actifs par echantillon
            seg_duration = seg_end - seg_start
            if seg_duration <= 0:
                seg.overlap_ratio = 0.0
                continue

            resolution = 0.01  # 10 ms
            n_bins = max(1, int(seg_duration / resolution))
            counts = np.zeros(n_bins, dtype=np.int32)
            for t_start, t_end, _ in overlapping:
                i_start = int((t_start - seg_start) / resolution)
                i_end = int((t_end - seg_start) / resolution)
                i_start = max(0, min(i_start, n_bins - 1))
                i_end = max(0, min(i_end, n_bins))
                counts[i_start:i_end] += 1

            overlap_bins = np.sum(counts > 1)
            seg.overlap_ratio = float(overlap_bins) / float(n_bins)


class EmotionExtractor:
    """Encapsule speechbrain pour la reconnaissance d'emotions vocales.

    Si speechbrain ou torchaudio ne sont pas installes, l'extracteur
    se desactive proprement (self.available = False).
    """

    def __init__(self) -> None:
        self.available: bool = False
        self._classifier = None
        try:
            import speechbrain  # noqa: F401
            import torchaudio  # noqa: F401
            self.available = True
            console.print(
                '[dim]speechbrain disponible pour la reconnaissance '
                'd\'emotions[/dim]'
            )
        except ImportError:
            console.print(
                '[yellow]speechbrain/torchaudio non installe — '
                'reconnaissance d\'emotions desactivee[/yellow]'
            )
        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Erreur lors de l\'import speechbrain : {exc} — '
                'reconnaissance d\'emotions desactivee[/yellow]'
            )

    # ------------------------------------------------------------------

    def _load_classifier(self) -> bool:
        """Charge le classifieur d'emotions si necessaire."""
        if self._classifier is not None:
            return True
        if not self.available:
            return False
        try:
            from speechbrain.inference.classifiers import (
                EncoderClassifier,
            )
            self._classifier = EncoderClassifier.from_hparams(
                source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
                savedir='/tmp/speechbrain_emotion',
            )
            return True
        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Impossible de charger le modele speechbrain : '
                f'{exc}[/yellow]'
            )
            self.available = False
            return False

    # ------------------------------------------------------------------

    def extract(
        self,
        audio_path: str,
        segments: List[AudioFeatures],
    ) -> None:
        """Remplit emotion et emotion_confidence pour chaque segment.

        Args:
            audio_path: Chemin vers le fichier audio/video.
            segments: Liste de AudioFeatures a enrichir.
        """
        if not segments:
            return
        if not self._load_classifier():
            return

        try:
            import torch
            import torchaudio

            # Charger l'audio complet une seule fois
            waveform, sr = torchaudio.load(audio_path)

            # Convertir en mono si necessaire
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resampler a 16 kHz si necessaire (speechbrain attend 16 kHz)
            if sr != _SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, _SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = _SAMPLE_RATE

            total_samples = waveform.shape[1]

            for seg in segments:
                try:
                    start_sample = int(seg.start * sr)
                    end_sample = int(seg.end * sr)

                    # Bornes de securite
                    start_sample = max(0, min(start_sample, total_samples))
                    end_sample = max(start_sample, min(end_sample, total_samples))

                    if end_sample - start_sample < int(0.5 * sr):
                        # Segment trop court pour une analyse fiable
                        continue

                    chunk = waveform[:, start_sample:end_sample]

                    with torch.no_grad():
                        out_prob, score, index, label = (
                            self._classifier.classify_batch(chunk)
                        )

                    seg.emotion = label[0]
                    seg.emotion_confidence = float(score.squeeze())
                except Exception:  # noqa: BLE001
                    # On ne fait pas echouer tout le pipeline pour un segment
                    continue

        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Erreur lors de l\'analyse d\'emotions : {exc}[/yellow]'
            )


# ------------------------------------------------------------------
# Extracteur principal
# ------------------------------------------------------------------

class AudioFeatureExtractor:
    """Orchestre l'extraction de toutes les features audio.

    Charge l'audio via librosa, decoupe en segments temporels et
    appelle chaque sous-extracteur pour remplir les champs de
    AudioFeatures.
    """

    def __init__(self) -> None:
        self._diarization = DiarizationExtractor()
        self._emotion = EmotionExtractor()

    # ------------------------------------------------------------------
    # Methode publique
    # ------------------------------------------------------------------

    def extract_all(
        self,
        audio_path: str,
        segment_duration: float = 10.0,
        words: Optional[List[WordTimestamp]] = None,
    ) -> List[AudioFeatures]:
        """Extrait les features audio par segment.

        Args:
            audio_path: Chemin vers le fichier audio ou video.
            segment_duration: Duree de chaque segment en secondes.
            words: Liste optionnelle de WordTimestamp pour le calcul
                du debit de parole. Si None, speech_rate reste a 0.

        Returns:
            Liste de AudioFeatures, un par segment temporel.
        """
        path = Path(audio_path)
        if not path.exists():
            console.print(f"[red]Fichier introuvable : {audio_path}[/red]")
            return []

        # Charger l'audio
        console.print(f'[dim]Chargement audio : {path.name}[/dim]')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y, sr = librosa.load(
                    str(path), sr=_SAMPLE_RATE, mono=True
                )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Impossible de charger l'audio : {exc}[/red]")
            return []

        if y is None or len(y) == 0:
            console.print("[yellow]Audio vide — aucune feature extraite[/yellow]")
            return []

        total_duration: float = float(len(y)) / float(sr)
        if total_duration < _MIN_SEGMENT_DURATION:
            console.print(
                f"[yellow]Audio trop court ({total_duration:.2f}s) — "
                "aucune feature extraite[/yellow]"
            )
            return []

        # Creer les segments
        segments = self._build_segments(total_duration, segment_duration)
        if not segments:
            return []

        console.print(
            f'[dim]Extraction de features sur {len(segments)} segments '
            f'({segment_duration:.0f}s chacun)[/dim]'
        )

        # Sous-extracteurs (chacun remplit les champs en place)
        self._extract_energy(y, sr, segments)
        self._extract_pitch(y, sr, segments)

        if words is not None:
            self._extract_speech_rate(words, segments)

        # Extracteurs optionnels (pyannote / speechbrain)
        if self._diarization.available:
            try:
                self._diarization.extract(audio_path, segments)
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f'[yellow]Diarisation echouee : {exc}[/yellow]'
                )

        if self._emotion.available:
            try:
                self._emotion.extract(audio_path, segments)
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f'[yellow]Analyse d\'emotions echouee : {exc}[/yellow]'
                )

        console.print('[green]Extraction de features audio terminee[/green]')
        return segments

    # ------------------------------------------------------------------
    # Construction des segments
    # ------------------------------------------------------------------

    @staticmethod
    def _build_segments(
        total_duration: float,
        segment_duration: float,
    ) -> List[AudioFeatures]:
        """Decoupe la duree totale en segments de taille fixe.

        Le dernier segment est raccourci si necessaire ; il est ignore
        s'il est trop court.
        """
        if segment_duration <= 0:
            segment_duration = 10.0

        segments: List[AudioFeatures] = []
        t = 0.0
        while t < total_duration:
            end = min(t + segment_duration, total_duration)
            if end - t >= _MIN_SEGMENT_DURATION:
                segments.append(AudioFeatures(start=t, end=end))
            t = end
        return segments

    # ------------------------------------------------------------------
    # Sous-extracteurs prives
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_energy(
        y: np.ndarray,
        sr: int,
        segments: List[AudioFeatures],
    ) -> None:
        """Calcule l'energie RMS et detecte les pics d'energie.

        Un segment est marque comme pic si son RMS depasse
        mean + _ENERGY_PEAK_K * std sur l'ensemble des segments.
        """
        try:
            # Calculer le RMS sur tout le signal (frame-level)
            rms_all = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

            # Assigner l'energie moyenne par segment
            hop_duration = 512.0 / float(sr)

            for seg in segments:
                frame_start = int(seg.start / hop_duration)
                frame_end = int(seg.end / hop_duration)
                frame_start = max(0, min(frame_start, len(rms_all) - 1))
                frame_end = max(frame_start + 1, min(frame_end, len(rms_all)))

                seg_rms = rms_all[frame_start:frame_end]
                if len(seg_rms) > 0:
                    seg.rms_energy = float(np.mean(seg_rms))
                else:
                    seg.rms_energy = 0.0

            # Detection des pics : mean + k * std
            energies = np.array([s.rms_energy for s in segments])
            if len(energies) > 1:
                mean_e = float(np.mean(energies))
                std_e = float(np.std(energies))
                threshold = mean_e + _ENERGY_PEAK_K * std_e
                for seg in segments:
                    seg.energy_peak = seg.rms_energy > threshold
            elif len(energies) == 1:
                # Un seul segment : pas de pic relatif possible
                segments[0].energy_peak = False

        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Extraction d\'energie echouee : {exc}[/yellow]'
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pitch(
        y: np.ndarray,
        sr: int,
        segments: List[AudioFeatures],
    ) -> None:
        """Estime le pitch (F0) par segment via librosa.pyin.

        Les frames non-voisees (NaN) sont ignorees dans les calculs
        de moyenne, ecart-type et etendue.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    y,
                    fmin=_PYIN_FMIN,
                    fmax=_PYIN_FMAX,
                    sr=sr,
                    frame_length=2048,
                    hop_length=512,
                )

            if f0 is None:
                return

            hop_duration = 512.0 / float(sr)

            for seg in segments:
                frame_start = int(seg.start / hop_duration)
                frame_end = int(seg.end / hop_duration)
                frame_start = max(0, min(frame_start, len(f0) - 1))
                frame_end = max(frame_start + 1, min(frame_end, len(f0)))

                seg_f0 = f0[frame_start:frame_end]
                # Filtrer les NaN (frames non-voisees)
                valid = seg_f0[~np.isnan(seg_f0)]

                if len(valid) == 0:
                    seg.pitch_mean = 0.0
                    seg.pitch_std = 0.0
                    seg.pitch_range = 0.0
                    continue

                seg.pitch_mean = float(np.mean(valid))
                seg.pitch_std = float(np.std(valid))
                seg.pitch_range = float(np.ptp(valid))  # max - min

        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Extraction du pitch echouee : {exc}[/yellow]'
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_speech_rate(
        words: List[WordTimestamp],
        segments: List[AudioFeatures],
    ) -> None:
        """Calcule le debit de parole (mots/seconde) par segment.

        speech_rate_delta = (segment_rate - mean_rate) / std_rate,
        soit un z-score par rapport a la moyenne globale.
        Si std_rate == 0 (debit constant), delta vaut 0.
        """
        if not words or not segments:
            return

        try:
            for seg in segments:
                # Compter les mots dont le centre tombe dans le segment
                count = 0
                for w in words:
                    word_center = (w.start + w.end) / 2.0
                    if seg.start <= word_center < seg.end:
                        count += 1

                duration = seg.end - seg.start
                if duration > 0:
                    seg.speech_rate = float(count) / duration
                else:
                    seg.speech_rate = 0.0

            # Calculer le delta (z-score) par rapport a la moyenne globale
            rates = np.array([s.speech_rate for s in segments])
            mean_rate = float(np.mean(rates))
            std_rate = float(np.std(rates))

            for seg in segments:
                if std_rate > 0:
                    seg.speech_rate_delta = (seg.speech_rate - mean_rate) / std_rate
                else:
                    seg.speech_rate_delta = 0.0

        except Exception as exc:  # noqa: BLE001
            console.print(
                f'[yellow]Extraction du debit de parole echouee : {exc}[/yellow]'
            )
