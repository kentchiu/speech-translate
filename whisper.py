from transformers import pipeline

transcibe_pipe = pipeline(model="openai/whisper-large-v3")
transcription = transcibe_pipe("test-data/serenity-zh.mp3")
print(transcription)


# 加載 Whisper 模型，支持語音翻譯任務
translation_pipe = pipeline(task="translation", model="openai/whisper-large-v3")
translation = translation_pipe("test-data/serenity-zh.mp3", task="translation")
print(f"🟥[1]: whisper.py:8: translation={translation}")
