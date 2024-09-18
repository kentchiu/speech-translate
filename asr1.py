import os
import time
import difflib

import whisper
import torch
import langcodes


def print_result(model, audio_number, filename, lang, text, time, expect, note):
    print(f"Model: whisper-{model}")
    print(f"Audio: {filename}")
    print(f"Language: {lang}")

    # 使用 difflib 比较 text 和 expect
    d = difflib.SequenceMatcher(None, expect.split(), text.split())
    highlighted_text = []
    for op, i1, i2, j1, j2 in d.get_opcodes():
        if op == "equal":
            highlighted_text.extend(text.split()[j1:j2])
        else:
            highlighted_text.extend(f"[{word}]" for word in text.split()[j1:j2])

    print(f"Expected: {expect}")
    print(f"Transcription: {' '.join(highlighted_text)}")
    print(f"Note: {note}")
    print(f"Execution time: {time:.2f} seconds\n")
    print("=============================")
    print("\n\n\n")


def transcribe_audio(file_path, model="tiny", device="cpu", expect="", note=""):
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (str): The Whisper model to use for transcription. Default is "tiny". can be tiny, base, small, medium, large, large-v2, large-v3
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".

    Returns:
        tuple: A tuple containing the transcribed text, the detected language, and the execution time.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    start_time = time.time()

    torch.set_num_threads(4)
    time1 = time.time()
    model = whisper.load_model(model).to(device)
    time2 = time.time()
    result = model.transcribe(file_path)
    time3 = time.time()

    end_time = time.time()
    # print all time diff
    #
    print(f"load model time: {time2 - time1}")
    print(f"transcribe time: {time3 - time2}")

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


def evaluate(model="tiny"):
    folder = "/home/kent/dev/playgroud/speech-translate/test-data"
    audio_files = [
        ("sample-zh-01.mp3", "中文語音辨識測試", "中文語音辨識測試"),
        (
            "sample-zh-02.mp3",
            "吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮",
            "吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮",
        ),
        (
            "sample-en-01.mp3",
            "Hello! This is a English audio speech recognition testing",
            "哈囉! 這是一個英文語音辨識測試",
        ),
        ("sample-jp-01.mp3", "痛みを感じる", "感受痛苦吧"),
        ("sample-kr-01.mp3", "내 친구 생일 축하해", "生日快樂我的朋友"),
        ("sample-th-01.mp3", "วันนี้อากาศเป็นอย่างไร?", "今天天氣如何？"),
    ]

    for i, (file_name, expect, note) in enumerate(audio_files, 1):
        audio_file = os.path.join(folder, file_name)
        try:
            text, lang, time_taken = transcribe_audio(
                audio_file, model=model, expect=expect, note=note
            )
            print_result(model, i, file_name, lang, text, time_taken, expect, note)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}\n")


if __name__ == "__main__":
    # evaluate("tiny")
    # evaluate("small")
    # evaluate("medium")
    evaluate("large-v3")
