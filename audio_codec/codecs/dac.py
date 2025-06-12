import os
import torch
import torchaudio
import torch.nn.functional as F
from transformers import DacModel, AutoProcessor

class DACDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"  → Loading DAC model {hub_name} onto {self.device}")
        # exactly as in your snippet:
        self.model     = DacModel.from_pretrained(hub_name).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(hub_name)
        # use the processor’s sampling_rate (should match hub_name)
        self.sample_rate = self.processor.sampling_rate
        self.name        = hub_name.replace("/", "_")

    def decode_file(self, src_path: str, out_dir: str):
        """
        Encode + decode one WAV via DAC, saving to:
          out_dir/<basename>_<hub_name>.wav
        """
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        # --- your snippet begins here ---
        wav_path = src_path
        audio_sample, sr = torchaudio.load(wav_path)

        if sr != self.processor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.processor.sampling_rate
            )
            audio_sample = resampler(audio_sample)

        inputs = self.processor(
            raw_audio=audio_sample.squeeze().numpy(),
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt"
        )

        encoder_outputs = self.model.encode(inputs["input_values"])
        audio_codes = encoder_outputs.audio_codes
        print(f"Original shape of audio codes: {audio_codes.shape}")

        audio_codes = F.interpolate(
            audio_codes.permute(0, 2, 1),
            size=1024,
            mode="linear"
        ).permute(0, 2, 1)
        print(f"Interpolated shape of audio codes: {audio_codes.shape}")

        audio_codes = audio_codes.float()
        decoded_audio = self.model.decode(audio_codes)[0]

        if decoded_audio.ndim == 1:  # If it's 1D, assume mono and add a channel
            decoded_audio = decoded_audio.unsqueeze(0)
        decoded_audio = decoded_audio.to(torch.float32)

        torchaudio.save(out_path, decoded_audio.cpu(), self.processor.sampling_rate)
        # --- snippet ends ---

        return out_name
