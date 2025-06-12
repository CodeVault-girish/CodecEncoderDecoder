import os
import torch
import torchaudio
from torchaudio.functional import resample
from speechtokenizer import SpeechTokenizer

class SpeechTokenizerDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        # Default paths (you can change these to the actual paths)
        self.config_path = "../configs/speech_tokenizer.yaml"
        self.ckpt_path = "..//checkpoints/ST.ckpt"

        print(f"  → Loading SpeechTokenizer from {self.ckpt_path} onto {self.device}")
        self.model = SpeechTokenizer.load_from_checkpoint(
            config_path=self.config_path,
            ckpt_path=self.ckpt_path
        ).to(self.device).eval()

    def _load(self, path: str):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav[:1]
        if sr != self.sample_rate:
            wav = resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        return wav.unsqueeze(0).to(self.device)  # [1, 1, T]

    def _save(self, wav, out_path: str):
        wav = wav.detach().cpu()[0]  # [channels, T]
        torchaudio.save(out_path, wav, self.sample_rate)

    def decode_file(self, src_path: str, out_dir: str):
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_speechtokenizer.wav"
        out_path = os.path.join(out_dir, out_name)

        wav = self._load(src_path)
        with torch.no_grad():
            codes = self.model.encode(wav)
            recon = self.model.decode(codes)
        self._save(recon, out_path)

        del wav, codes, recon
        return out_name
