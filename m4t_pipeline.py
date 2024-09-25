from transformers import pipeline
import sys
import time
import csv

from transformers.configuration_utils import re


class Record:
    def __init__(
        self,
        model: str = "",
        filename: str = "",
        lang: str = "",
        translation_time: float = 0,
        transcription_time: float = 0,
        transcription: str = "",
        translation: str = "",
    ):
        self.model = model
        self.filename = filename
        self.lang = lang
        self.translation_time = translation_time
        self.transcription_time = transcription_time
        self.translation = translation
        self.transcription = transcription

    def display_info(self):
        print(f"Model: {self.model}")
        print(f"Filename: {self.filename}")
        print(f"Lang: {self.lang}")
        print(f"Translation Time: {self.translation_time}")
        print(f"Transcriiption Time: {self.transcription_time}")
        print(f"Transcripition: {self.transcription}")
        print(f"translation: {self.translation}")


# "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

model_name = "facebook/seamless-m4t-v2-large"
# device = -1  # -1 for CPU
device = 0  # 0 for CUDA
transcriber = pipeline(
    task="automatic-speech-recognition", model=model_name, device=device
)
translator = pipeline(task="translation", model=model_name, device=-1)


def transcribe(audio_file: str, target_lang: str = "eng"):
    try:
        transcription = transcriber(
            audio_file, generate_kwargs={"tgt_lang": target_lang}
        )
        return transcription
    except Exception as e:
        print(f"Error transcribing {audio_file}: {str(e)}", file=sys.stderr)
        return None


def translate(text: str, src_lang: str) -> str:
    """Translate text to English"""
    try:
        translation = translator(
            text, src_lang=src_lang, tgt_lang="eng", max_length=400
        )

        return translation[0].get("translation_text", "")
    except Exception as e:
        print(f"Error translating text: {str(e)}", file=sys.stderr)
        return text


def print_transcription(file: str, lang: str):
    result = transcribe(file, target_lang=lang)
    if result:
        transcription = result["text"]
        translation = translate(transcription, lang)
        print(f"🟥 lang={lang},trans={translation} ,text={transcription}")


def process_files():
    print_transcription("test-data/sample-en-01.mp3", "eng")
    print_transcription("test-data/sample-zh-01.mp3", "cmn_Hant")
    print_transcription("test-data/sample-ja-01.mp3", "jpn")
    print_transcription("test-data/sample-ko-01.mp3", "kor")
    print_transcription("test-data/sample-th-01.mp3", "tha")

    print_transcription("test-data/serenity-en.mp3", "eng")
    print_transcription("test-data/serenity-zh.mp3", "cmn_Hant")
    print_transcription("test-data/serenity-ja.mp3", "jpn")
    print_transcription("test-data/serenity-ko.mp3", "kor")
    print_transcription("test-data/serenity-th.mp3", "tha")

    print_transcription("test-data/spiderman-en.mp3", "eng")
    print_transcription("test-data/spiderman-zh.mp3", "cmn_Hant")
    print_transcription("test-data/spiderman-ja.mp3", "jpn")
    print_transcription("test-data/spiderman-ko.mp3", "kor")
    print_transcription("test-data/spiderman-th.mp3", "tha")

    print_transcription("test-data/thinking-en.mp3", "eng")
    print_transcription("test-data/thinking-zh.mp3", "cmn_Hant")
    print_transcription("test-data/thinking-ja.mp3", "jpn")
    print_transcription("test-data/thinking-ko.mp3", "kor")
    print_transcription("test-data/thinking-th.mp3", "tha")


def write_records_to_csv(records, filename):
    """將記錄寫入CSV文件"""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
                "Filename",
                "Lang",
                "Translation Time",
                "Transcription Time",
                "Translation",
                "Transcription",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.model,
                    record.filename,
                    record.lang,
                    record.translation_time,
                    record.transcription_time,
                    record.transcription,
                    record.translation,
                ]
            )


def test_results(filename):
    records = []
    langs = ["en", "zh", "ja", "ko", "th"]
    langs2 = ["eng", "cmn_Hant", "jpn", "kor", "tha"]
    for lang, target_lang in zip(langs, langs2):
        file = f"test-data/{filename}-{lang}.mp3"
        time1 = time.time()
        result = transcribe(file, target_lang=target_lang)
        time2 = time.time()
        transcription = result["text"]
        translation = translate(transcription, target_lang)
        time3 = time.time()
        print(f"🟥 lang={lang},trans={translation} ,text={transcription}")
        record = Record(
            model=model_name,
            filename=file,
            lang=lang,
            transcription=transcription,
            translation=translation,
            translation_time=round(time3 - time2, 1),
            transcription_time=round(time2 - time1, 1),
        )
        records.append(record)
        record.display_info()

    return records


def main():
    records = test_results("serenity")
    records += test_results("spiderman")
    records += test_results("thinking")
    filename = f"dist/translation-{time.strftime('%Y%m%d-%H%M')}.csv"
    write_records_to_csv(records, filename)


if __name__ == "__main__":
    # process_files()
    main()
