# audio_codec/registry.py

CODEC_REGISTRY = {
    "1": {
        "name":        "snac_24khz",
        "module":      "audio_codec.codecs.snac",
        "class":       "SNACDecoder",
        "hub_name":    "hubertsiuzdak/snac_24khz",
        "sample_rate": 24000,
    },
    "2": {
        "name":        "snac_32khz",
        "module":      "audio_codec.codecs.snac",
        "class":       "SNACDecoder",
        "hub_name":    "hubertsiuzdak/snac_32khz",
        "sample_rate": 32000,
    },
    "3": {
        "name":        "snac_44khz",
        "module":      "audio_codec.codecs.snac",
        "class":       "SNACDecoder",
        "hub_name":    "hubertsiuzdak/snac_44khz",
        "sample_rate": 44100,
    },
    # ─── DAC variants ───
    "4": {
        "name":        "dac_16khz",
        "module":      "audio_codec.codecs.dac",
        "class":       "DACDecoder",
        "hub_name":    "descript/dac_16khz",
        "sample_rate": 16000,
    },
    "5": {
        "name":        "dac_24khz",
        "module":      "audio_codec.codecs.dac",
        "class":       "DACDecoder",
        "hub_name":    "descript/dac_24khz",
        "sample_rate": 24000,
    },
    "6": {
        "name":        "dac_44khz",
        "module":      "audio_codec.codecs.dac",
        "class":       "DACDecoder",
        "hub_name":    "descript/dac_44khz",
        "sample_rate": 44100,
    },

    # ─── Encodec variants ───
    "7": {
        "name":        "encodec_24khz",
        "module":      "audio_codec.codecs.encodec24",
        "class":       "Encodec24Decoder",
        "hub_name":    "facebook/encodec_24khz",
        "sample_rate": 24000,
    },
    "8": {
        "name":        "encodec_48khz",
        "module":      "audio_codec.codecs.encodec48",
        "class":       "Encodec48Decoder",
        "hub_name":    "facebook/encodec_48khz",
        "sample_rate": 48000,
    },
    "9": {
        "name":        "soundstream_16khz",
        "module":      "audio_codec.codecs.soundstream",
        "class":       "SoundStreamDecoder",
        "hub_name":    "SoundStream/soundstream_16khz",
        "sample_rate": 16000,
    },
        # ─── SpeechTokenizer variant ───
    "10": {
        "name":        "speechtokenizer",
        "module":      "audio_codec.codecs.speechtokenizer",
        "class":       "SpeechTokenizerDecoder",
        "config_path": "config/config.json",
        "ckpt_path":   "checkpoints/SpeechTokenizer.pt",
        "sample_rate": None,
    },

}
