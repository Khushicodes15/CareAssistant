import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_COPY_ON_SYMLINK_ERROR"] = "1"
import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

# ── Load Model ───────────────────────────────────────────
class SpeakerEmbeddingModel:
    def __init__(self):
        print("Loading speaker embedding model...")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa_tdnn",
            run_opts={"device": "cpu"}
        )
        print("Model loaded.")

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """
        Takes audio file path
        Returns speaker embedding as numpy array
        """
        signal, fs = torchaudio.load(audio_path)

        # resample to 16khz if needed
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)

        # get embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(signal)
            embedding = embedding.squeeze().numpy()

        return embedding

    def get_embedding_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Takes raw audio numpy array
        Returns speaker embedding
        """
        signal = torch.tensor(audio_array, dtype=torch.float32)

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            signal = resampler(signal)

        with torch.no_grad():
            embedding = self.model.encode_batch(signal)
            embedding = embedding.squeeze().numpy()

        return embedding