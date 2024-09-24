from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

# 加載 Whisper 模型，支持語音轉錄任務
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# 轉錄語音
# Load the audio file
waveform, sampling_rate = torchaudio.load("test-data/serenity-zh.mp3")
# Pass the loaded audio data and the sampling rate to the processor
input_features = processor(audio=waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_features
transcription_ids = model.generate(input_features)
transcription = processor.decode(transcription_ids[0], skip_special_tokens=True)
print(f"Transcription: {transcription}")

# 加載 Whisper 模型，支持語音翻譯任務
translation_ids = model.generate(input_features, task="translate")
translation = processor.decode(translation_ids[0], skip_special_tokens=True)
print(f"Translation: {translation}")
