# NeuralCodecDecoder


from audio_codec.registry import CODEC_REGISTRY
from audio_codec.cli import decoder_list, decode_folder

# list available
decoder_list()

# decode folder
decode_folder('2', 'raw_wavs', 'decoded', 'cpu')
decode_folder('9', '/home/girish/Girish/Reseach/Health-care/Audio_Data/Audio_Data/HC', 'output/', 'cuda')