import csv
import os
import time
import warnings

import langcodes
import torch
import whisper

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")


class Record:
    def __init__(
        self,
        model: str = "",
        filename: str = "",
        lang: str = "",
        load_time: float = 0,
        transcribe_time: float = 0,
        expect: str = "",
        note: str = "",
        transcribe: str = "",
    ):
        self.model = model
        self.filename = filename
        self.lang = lang
        self.transcribe = transcribe
        self.load_time = load_time  # 使用 float 來儲存時間
        self.transcribe_time = transcribe_time  # 使用 float 來儲存時間
        self.expect = expect
        self.note = note

    def display_info(self):
        print(f"Model: {self.model}")
        print(f"Filename: {self.filename}")
        print(f"Lang: {self.lang}")
        print(f"Load Time: {self.load_time}")
        print(f"Transcribe Time: {self.transcribe_time}")
        print(f"Expect: {self.expect}")
        print(f"Transcribe: {self.transcribe}")
        print(f"Note: {self.note}")


def transcribe_audio(file_path, model, device="cpu", expect="", note=""):
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (WhisperModel): The loaded Whisper model to use for transcription.
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".

    Returns:
        tuple: A tuple containing the transcribed text, the detected language, and the execution time.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    torch.set_num_threads(4)

    start_time = time.time()
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

    # 確保檢測到的語言是正確的
    if detected_language in ["zh", "yue"]:
        detected_language = "zh"
    elif detected_language in ["ja"]:
        detected_language = "ja"
    elif detected_language in ["ko"]:
        detected_language = "ko"
    elif detected_language in ["th"]:
        detected_language = "th"
    elif detected_language in ["en"]:
        detected_language = "en"

    return transcribed_text, detected_language, execution_time


def evaluate(model="tiny"):
    folder = "test-data"
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

    print(f"============= {model} ============")
    # Load the model once before the loop
    device = "cpu"  # or "cuda" if you have a GPU
    time1 = time.time()
    loaded_model = whisper.load_model(model).to(device)
    time2 = time.time()

    records = []

    for i, (file_name, expect, note) in enumerate(audio_files, 1):
        record = Record()
        audio_file = os.path.join(folder, file_name)
        record.model = model
        try:
            transcribe, lang, time_taken = transcribe_audio(
                audio_file, model=loaded_model, device=device, expect=expect, note=note
            )
            record.filename = file_name
            record.lang = lang
            record.load_time = time2 - time1
            record.transcribe_time = time_taken
            record.expect = expect
            record.note = note
            record.transcribe = transcribe
            record.display_info()
            records.append(record)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}\n")
    return records


def write_records_to_csv(records, filename):
    """
    Writes the records to a CSV file with headers.

    Args:
        records (list): List of Record objects.
        filename (str): Name of the CSV file to write to.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
                "Filename",
                "Lang",
                "Load Time",
                "Transcribe Time",
                "Expect",
                "Note",
                "Transcribe",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.model,
                    record.filename,
                    record.lang,
                    record.load_time,
                    record.transcribe_time,
                    record.expect,
                    record.note,
                    record.transcribe,
                ]
            )


if __name__ == "__main__":
    records = []
    records.extend(evaluate("tiny"))
    records.extend(evaluate("small"))
    records.extend(evaluate("medium"))
    records.extend(evaluate("large-v3"))

    write_records_to_csv(records, "dist/cpu-kent.csv")
