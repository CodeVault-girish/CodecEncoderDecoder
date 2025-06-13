# audio_codec/codecs/dac.py

import os
import torch
import dac
from audiotools import AudioSignal

class DACDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        """
        hub_name: will be something like "dac_16khz" or "dac_24khz" (matches registry name)
        sample_rate: unused here (dac knows its own rate), but kept for API consistency
        device: "cpu" or "cuda"
        """
        # pick device
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        # derive model_type from hub_name
        # e.g. hub_name="descript/dac_16khz" or simply "dac_16khz"
        model_type = hub_name.split("/")[-1].replace("dac_", "")
        print(f"  → Downloading + loading DAC model (→ {model_type}) onto {self.device}")
        # download returns a local path to the weights
        model_path = dac.utils.download(model_type=model_type)
        # load the DAC object
        self.model = dac.DAC.load(model_path).to(self.device).eval()
        # name for output files
        self.name = f"dac_{model_type}"

    def decode_file(self, src_path: str, out_dir: str) -> str:
        """
        Round-trip one WAV through DAC, writing:
            out_dir/<basename>_dac_<rate>.wav
        Uses model.compress + model.decompress under the hood (with chunking).
        """
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Load as an AudioSignal (handles sampling + mono/stereo)
        signal = AudioSignal(src_path)
        signal = signal.to(self.model.device)

        # 2) Compress → a DACFile (handles chunking for long files)
        compressed = self.model.compress(signal)

        # 3) Decompress → back to an AudioSignal
        recon = self.model.decompress(compressed)

        # 4) Move back to CPU and write WAV
        recon = recon.to("cpu")
        recon.write(out_path)

        # 5) Cleanup
        del signal, compressed, recon
        return out_name
