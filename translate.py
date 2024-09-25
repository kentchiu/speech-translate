import whisper
import time
import csv
import warnings


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
        transcription_time: float = 0,
        transcription: str = "",
        translation: str = "",
    ):
        self.model = model
        self.filename = filename
        self.lang = lang
        self.load_time = load_time
        self.transcription_time = transcription_time
        self.transcription = transcription
        self.translation = translation

    def display_info(self):
        print(f"Model: {self.model}")
        print(f"Filename: {self.filename}")
        print(f"Lang: {self.lang}")
        print(f"Load Time: {self.load_time}")
        print(f"Transcription Time: {self.transcription_time}")
        print(f"Transcripition: {self.transcription}")
        print(f"translation: {self.translation}")


def transcribe_and_translate(audio_file, model):
    """使用指定的模型轉錄音頻文件並翻譯成英文"""
    start_time = time.time()
    # 轉錄原始語音
    result = model.transcribe(audio_file)

    # # 翻譯成英文
    translation = model.transcribe(audio_file, task="translate")
    end_time = time.time()

    return {
        "transcription": result["text"],
        "language": result["language"],
        "translation": translation["text"],
        "time": end_time - start_time,
    }


def write_records_to_csv(records, filename):
    """將記錄寫入CSV文件"""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
                "Filename",
                "Lang",
                "Load Time",
                "Transcribe Time",
                "Transcription",
                "Translation",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.model,
                    record.filename,
                    record.lang,
                    record.load_time,
                    record.transcription_time,
                    record.transcription,
                    record.translation,
                ]
            )


def transscribe_all(
    filename: str, langs=["en", "zh", "ja", "ko", "th"], model_size="tiny"
):
    test_data_dir = "test-data"
    time_1 = time.time()
    model = whisper.load_model(model_size)
    time_2 = time.time()
    records = []

    audio_files = [f"{test_data_dir}/{filename}-{lang}.mp3" for lang in langs]

    for audio_file in audio_files:
        result = transcribe_and_translate(audio_file, model)
        record = Record(
            model=model_size,
            filename=audio_file,
            lang=result["language"],
            transcription=result["transcription"],
            translation=result["translation"],
            transcription_time=result["time"],
            load_time=time_2 - time_1,
        )
        records.append(record)
        print(record.display_info())
        print("\n" + "=" * 40 + "\n")

    return records


def test_results(filename, model_sizes=["small", "medium", "large-v3"]):
    records = []
    langs = ["en", "zh", "ja", "ko", "th"]

    for model_size in model_sizes:
        records += transscribe_all(filename, langs=langs, model_size=model_size)

    return records


def main():
    records = test_results("serenity", model_sizes=["large-v3"])
    records += test_results("spiderman", model_sizes=["large-v3"])
    records += test_results("thinking", model_sizes=["large-v3"])
    filename = f"dist/translation-{time.strftime('%Y%m%d-%H%M')}.csv"
    write_records_to_csv(records, filename)


if __name__ == "__main__":
    main()
