import os
import time
import warnings

import whisper
import torch
import langcodes

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")


def print_result(audio_name, lang, text, time, expect, note):
    print(f"\n{audio_name}")
    print(f"Language: {lang}")
    print(f"Transcription: {text}")
    print(f"Expected: {expect}")
    print(f"Note (Chinese translation): {note}")
    print(f"Execution time: {time:.2f} seconds")


def transcribe_audio(file_path, model="tiny", device="cpu", expect="", note=""):
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (str): The Whisper model to use for transcription. Default is "tiny". can be tiny, small, medium, large, large-v2, large-v3
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".
        expect (str): The expected correct transcription of the original audio. Default is an empty string.
        note (str): The correct Chinese translation of the audio content. Default is an empty string.

    Returns:
        tuple: A tuple containing the transcribed text, the detected language, the execution time, the expected text, and the note.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    start_time = time.time()

    torch.set_num_threads(8)
    model = whisper.load_model(model).to(device)
    result = model.transcribe(file_path)

    end_time = time.time()
    execution_time = end_time - start_time

    detected_language = result["language"]
    transcribed_text = result["text"]

    # 如果檢測到的語言是中文，確保使用繁體中文
    if detected_language in ["zh", "yue"]:
        lang = langcodes.Language.get(detected_language)
        lang = lang.maximize()
        if lang.script == "Hant":
            # 已經是繁體中文，不需要額外處理
            pass
        else:
            # 這裡可以添加簡體到繁體的轉換邏輯
            # 例如使用 OpenCC 庫進行轉換
            # 注意：這需要額外安裝 OpenCC 庫
            # from opencc import OpenCC
            # cc = OpenCC('s2t')  # 簡體到繁體
            # transcribed_text = cc.convert(transcribed_text)
            pass

    return transcribed_text, detected_language, execution_time, expect, note


if __name__ == "__main__":
    folder = "/home/kent/dev/playgroud/speech-translate/test-data"
    audio_file_1 = os.path.join(folder, "sample-zh-01.mp3")
    text_1, lang_1, time_1, expect_1, note_1 = transcribe_audio(audio_file_1, model="tiny", expect="中文語音辨識測試", note="Chinese speech recognition test")
    print_result("Audio 1 (Chinese 1)", lang_1, text_1, time_1, expect_1, note_1)

    audio_file_2 = os.path.join(folder, "sample-zh-02.mp3")
    text_2, lang_2, time_2 = transcribe_audio(audio_file_2, model="tiny")
    print_result("Audio 2 (Chinese 2)", lang_2, text_2, time_2)

    audio_file_3 = os.path.join(folder, "sample-en-01.mp3")
    text_3, lang_3, time_3 = transcribe_audio(audio_file_3, model="tiny")
    print_result("Audio 3 (English)", lang_3, text_3, time_3)

    audio_file_4 = os.path.join(folder, "sample-jp-01.mp3")
    text_4, lang_4, time_4 = transcribe_audio(audio_file_4, model="tiny")
    print_result("Audio 4 (Japanese)", lang_4, text_4, time_4)

    audio_file_5 = os.path.join(folder, "sample-kr-01.mp3")
    text_5, lang_5, time_5 = transcribe_audio(audio_file_5, model="tiny")
    print_result("Audio 5 (Korean)", lang_5, text_5, time_5)

    audio_file_6 = os.path.join(folder, "sample-th-01.mp3")
    text_6, lang_6, time_6 = transcribe_audio(audio_file_6, model="tiny")
    print_result("Audio 6 (Thai)", lang_6, text_6, time_6)
