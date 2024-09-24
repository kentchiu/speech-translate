from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

# 加載 Whisper 模型，支持語音轉錄任務
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# 轉錄語音
# Load the audio file
waveform, original_sampling_rate = torchaudio.load("test-data/serenity-zh.mp3")
# Resample the audio to 16000 Hz
resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
waveform = resampler(waveform)
# Pass the resampled audio data and the new sampling rate to the processor
input_features = processor(audio=waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
transcription_ids = model.generate(input_features)
transcription = processor.decode(transcription_ids[0], skip_special_tokens=True)
print(f"Transcription: {transcription}")

# 加載 Whisper 模型，支持語音翻譯任務
translation_ids = model.generate(input_features, task="translate")
translation = processor.decode(translation_ids[0], skip_special_tokens=True)
print(f"Translation: {translation}")
