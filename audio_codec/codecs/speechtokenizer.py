import os
import torch
import torchaudio
from torchaudio.functional import resample
from speechtokenizer import SpeechTokenizer

class SpeechTokenizerDecoder:
    def __init__(
        self,
        hub_name: str = None,
        sample_rate: int = None,
        device: str = "cpu",
        config_path: str = None,
        ckpt_path: str = None,
    ):
        # figure out project root (two levels up from this file)
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # paths (override defaults if provided)
        self.config_path = (
            config_path
            or os.path.join(repo_root, "config", "config.json")
        )
        self.ckpt_path = (
            ckpt_path
            or os.path.join(repo_root, "checkpoints", "SpeechTokenizer.pt")
        )

        self.device = torch.device(device)
        print(f"  → Loading SpeechTokenizer from {self.ckpt_path} onto {self.device}")

        self.model = (
            SpeechTokenizer.load_from_checkpoint(
                config_path=self.config_path,
                ckpt_path=self.ckpt_path
            )
            .to(self.device)
            .eval()
        )

        # determine sample rate from model if not passed
        self.sample_rate = getattr(self.model, "sample_rate", sample_rate)
        if self.sample_rate is None:
            raise ValueError("Sample rate not set and not found on model")

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)              # [channels, T]
        if wav.size(0) > 1:
            wav = wav[:1]                            # to mono
        if sr != self.sample_rate:
            wav = resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        return wav.unsqueeze(0).to(self.device)      # [1,1,T]

    def _save(self, wav: torch.Tensor, out_path: str):
        wav = wav.detach().cpu()[0]                  # [channels, T]
        torchaudio.save(out_path, wav, self.sample_rate)

    def decode_file(self, src_path: str, out_dir: str) -> str:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_speechtokenizer.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        wav = self._load(src_path)
        with torch.no_grad():
            codes = self.model.encode(wav)
            recon = self.model.decode(codes)

        self._save(recon, out_path)
        del wav, codes, recon
        return out_name
