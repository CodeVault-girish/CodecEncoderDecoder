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



## For Funcoded
```
pyhton3 -m venv funcodec
source funcodec/bin/activate
```
```
git clone https://github.com/alibaba-damo-academy/FunCodec.git
cd FunCodec
pip install -e .
```
```
pip install torch torchaudio numpy soundfile
```
```
cd egs/LibriTTS/codec
mkdir -p exp
model_name="audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch"
# Download the model
git lfs install
git clone https://huggingface.co/alibaba-damo/${model_name}
```

# Encoding
```
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh \
  --stage 1 \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_devices "0" \
  --model_dir exp/${model_name} \
  --bit_width 16000 \
  --file_sampling_rate 16000 \
  --wav_scp input.scp \
  --out_dir outputs/codecs
```
# Decoding
```
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh \
  --stage 2 \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_devices "0" \
  --model_dir exp/${model_name} \
  --bit_width 16000 \
  --file_sampling_rate 16000 \
  --wav_scp outputs/codecs/codecs.txt \
  --out_dir outputs/recon_wavs
```