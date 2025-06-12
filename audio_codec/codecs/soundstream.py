import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from soundstream import from_pretrained, load as load_audio

class SoundStreamDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        # SoundStream doesn't take hub_name; ignore it
        self.device       = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        # model always runs at 16 kHz
        self.model_sr     = 16000
        # desired output rate
        self.sample_rate  = sample_rate
        self.name         = f"soundstream_{self.model_sr//1000}khz"
        print(f"  → Loading SoundStream model onto {self.device}")
        # load & move to device
        self.codec = from_pretrained().eval().to(self.device)

    def decode_file(self, src_path: str, out_dir: str):
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) load @16 kHz
        waveform = load_audio(src_path)                   # [1, T] @16 kHz
        waveform = waveform.to(self.device)

        # 2) encode & decode
        with torch.inference_mode():
            quantized = self.codec(waveform, mode="encode")
            recovered = self.codec(quantized, mode="decode")  # [1, C, T']

        # 3) detach, cpu & squeeze batch dim → [C, T']
        recovered = recovered.detach().cpu()[0]

        # 4) optionally resample to target rate
        if self.sample_rate != self.model_sr:
            resampler = Resample(orig_freq=self.model_sr, new_freq=self.sample_rate)
            recovered = resampler(recovered)

        # 5) save (expects [channels, time])
        torchaudio.save(out_path, recovered, self.sample_rate)

        # 6) cleanup
        del waveform, quantized, recovered
        return out_name
