import os
import time

import whisper
import torch
import langcodes


def transcribe_audio(file_path, model="tiny", device="cpu", expect="", note=""):
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (str): The Whisper model to use for transcription. Default is "tiny". can be tiny, small, medium, large, large-v2, large-v3
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".

    Returns:
        tuple: A tuple containing the transcribed text, the detected language, and the execution time.

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

    return transcribed_text, detected_language, execution_time


if __name__ == "__main__":
    folder = "/home/kent/dev/playgroud/speech-translate/test-data"
    audio_file_1 = os.path.join(folder, "sample-zh-01.mp3")
    text_1, lang_1, time_1 = transcribe_audio(
        audio_file_1, model="tiny", expect="中文語音辨識測試", note="中文語音辨識測試"
    )
    print(f"Audio 1 - Language: {lang_1}")
    print(f"Transcription: {text_1}")
    print(f"Execution time: {time_1:.2f} seconds\n")

    audio_file_2 = os.path.join(folder, "sample-zh-02.mp3")
    text_2, lang_2, time_2 = transcribe_audio(
        audio_file_2, model="tiny", expect="吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮", note="吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮"
    )
    print(f"Audio 2 - Language: {lang_2}")
    print(f"Transcription: {text_2}")
    print(f"Execution time: {time_2:.2f} seconds")

    # Hello! This is a English audio speech recongnition testing.

    audio_file_3 = os.path.join(folder, "sample-en-01.mp3")
    text_3, lang_3, time_3 = transcribe_audio(
        audio_file_3, model="tiny", expect="Hello! This is a English audio speech recongnition testing", note="哈囉! 這是一個英文語音辨識測試"
    )
    print(f"Audio 2 - Language: {lang_3}")
    print(f"Transcription: {text_3}")
    print(f"Execution time: {time_3:.2f} seconds")

    # 感受痛苦吧:  痛みを感じる
    audio_file_4 = os.path.join(folder, "sample-jp-01.mp3")
    text_4, lang_4, time_4 = transcribe_audio(
        audio_file_4, model="tiny", expect="痛みを感じる", note="感受痛苦吧"
    )
    print(f"Audio 2 - Language: {lang_4}")
    print(f"Transcription: {text_4}")
    print(f"Execution time: {time_4:.2f} seconds")

    # 生日快樂我的朋友：내 친구 생일 축하해.
    audio_file_5 = os.path.join(folder, "sample-kr-01.mp3")
    text_5, lang_5, time_5 = transcribe_audio(
        audio_file_5, model="tiny", expect="내 친구 생일 축하해", note="生日快樂我的朋友"
    )
    print(f"Audio 2 - Language: {lang_5}")
    print(f"Transcription: {text_5}")
    print(f"Execution time: {time_5:.2f} seconds")

    # 今天天氣如何？วันนี้อากาศเป็นอย่างไร?
    audio_file_6 = os.path.join(folder, "sample-th-01.mp3")
    text_6, lang_6, time_6 = transcribe_audio(
        audio_file_6, model="tiny", expect="วันนี้อากาศเป็นอย่างไร?", note="今天天氣如何？)"
    )
    print(f"Audio 2 - Language: {lang_6}")
    print(f"Transcription: {text_6}")
    print(f"Execution time: {time_6:.2f} seconds")
