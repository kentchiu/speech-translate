import warnings
from transformers import AutoProcessor, SeamlessM4Tv2Model
import librosa
from datasets import load_dataset
import numpy as np

# warnings.filterwarnings(
#     "ignore", category=FutureWarning, module="transformers.deepspeed"
# )
#

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")


audio_file = "test-data/sample-en-01.mp3"
# Load the audio file using librosa
audio_data, sampling_rate = librosa.load(audio_file, sr=16000)
# now, process it
audio_inputs = processor(
    audios=np.array([audio_data]), sampling_rate=sampling_rate, return_tensors="pt"
)

# now, process some English text as well
text_inputs = processor(
    text="Hello, my dog is cute", src_lang="eng", return_tensors="pt"
)


audio_array_from_text = (
    model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
)
print(f"🟥[5]: m4t.py:40: audio_array_from_text={audio_array_from_text}")
audio_array_from_audio = (
    model.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
)


# from audio
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_audio = processor.decode(
    output_tokens[0].tolist()[0], skip_special_tokens=True
)
print(f"🟥[2]: m4t.py:46: translated_text_from_audio={translated_text_from_audio}")

# from text
output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_text = processor.decode(
    output_tokens[0].tolist()[0], skip_special_tokens=True
)
print(f"🟥[3]: m4t.py:53: translated_text_from_text={translated_text_from_text}")
