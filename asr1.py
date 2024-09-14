import os
import time
import whisper
import torch

def transcribe_and_translate(file_path, model="tiny", device="cpu"):
    """
    Transcribes the given audio file using the specified Whisper model and translates to multiple languages.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (str): The Whisper model to use for transcription. Default is "tiny".
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".

    Returns:
        dict: A dictionary containing the original transcription and translations in multiple languages.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    start_time = time.time()

    torch.set_num_threads(8)
    model = whisper.load_model(model).to(device)

    # Transcribe in the original language
    result = model.transcribe(file_path)
    original_lang = result["language"]
    original_text = result["text"]

    # Translate to other languages
    languages = ["en", "zh", "ja", "ko", "th"]
    translations = {}

    for lang in languages:
        if lang != original_lang:
            translation = model.transcribe(file_path, task="translate", language=lang)
            translations[lang] = translation["text"]

    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "original": original_text,
        "original_lang": original_lang,
        "translations": translations,
        "execution_time": execution_time
    }

if __name__ == "__main__":
    folder = "/home/kent/dev/playgroud/speech-translate/test-data"
    audio_files = ["sample-zh-01.mp3", "sample-zh-02.mp3"]

    for i, audio_file in enumerate(audio_files, 1):
        file_path = os.path.join(folder, audio_file)
        result = transcribe_and_translate(file_path, model="large-v3")

        print(f"Audio {i}:")
        print(f"Original Language: {result['original_lang']}")
        print(f"Original Transcription: {result['original']}")
        print("Translations:")
        for lang, text in result['translations'].items():
            print(f"  {lang}: {text}")
        print(f"Execution time: {result['execution_time']:.2f} seconds\n")
