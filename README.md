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

Add this model file as follows:

```
NeuralCodecDecoder/
  audio_codec/
    codecs/
  config/
    config.json
  checkpoints/
    SpeechTokenizer.pt
```

Place the downloaded `SpeechTokenizer.pt` file into the `checkpoints/` directory as shown above.


```
pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: tokenizers

pip install git+https://github.com/ga642381/AudioCodec-Hub.git soundfile
```
additional
```
pip install --no-deps --force-reinstall git+https://github.com/ga642381/AudioCodec-Hub.git soundfile
```
