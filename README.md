# NeuralCodecDecoder

```
from audio_codec.registry import CODEC_REGISTRY
from audio_codec.cli import decoder_list, decode_folder
```

# list available
```
decoder_list()
```

# decode folder
```
decode_folder('2', 'raw_wavs', 'decoded', 'cpu')
decode_folder('10', '/home/girish/Girish/Reseach/Health-care/Audio_Data/Audio_Data/HC', 'output/', 'cuda')
```


# For fnlp/SpeechTokenizer use this link to get it's model 

[fnlp/SpeechTokenizer model](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)

Add this model in this 
<repo-root>/
  audio_codec/
    codecs/
  config/
    config.json
  checkpoints/
    SpeechTokenizer.pt