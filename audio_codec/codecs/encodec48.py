import os
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

class Encodec48Decoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device      = torch.device(device)
        print(f"  → Loading Encodec model {hub_name} onto {self.device}")
        self.model       = EncodecModel.from_pretrained(hub_name).eval().to(self.device)
        self.processor   = AutoProcessor.from_pretrained(hub_name)
        self.sample_rate = self.processor.sampling_rate
        self.name        = hub_name.replace("/", "_")

    def decode_file(self, src_path: str, out_dir: str):
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) load + stereo + resample
        wav, sr = torchaudio.load(src_path)                       # (channels, T)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)                                # duplicate to stereo (2, T)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(wav)

        # 2) prepare inputs
        audio_np   = wav.cpu().numpy()                            # (C, T)
        inputs     = self.processor(
                        raw_audio=audio_np,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                     )
        input_vals = inputs["input_values"].to(self.device)
        padding    = inputs.get("padding_mask", None)
        if padding is not None:
            padding = padding.to(self.device)

        # 3) encode + decode
        with torch.inference_mode():
            enc     = self.model.encode(input_vals, padding_mask=padding)
            codes   = enc.audio_codes
            scales  = enc.audio_scales
            decoded = self.model.decode(codes, scales, padding_mask=padding)[0]

        # 4) squeeze to 2D (channels, samples)
        if decoded.ndim == 3:
            decoded = decoded.squeeze(0)
        elif decoded.ndim == 1:
            decoded = decoded.unsqueeze(0)

        decoded = decoded.to(torch.float32).cpu()

        # 5) save
        torchaudio.save(out_path, decoded, self.sample_rate)

        # 6) cleanup
        del wav, inputs, input_vals, padding, enc, codes, scales, decoded
        return out_name
