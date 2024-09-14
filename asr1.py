import whisper
import torch
import os


def transcribe_audio(file_path, model="tiny", device="cpu"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")
    torch.set_num_threads(8)
    model = whisper.load_model(model).to(device)
    result = model.transcribe(file_path)
    return result["text"]


if __name__ == "__main__":
    folder = "/home/kent/dev/playgroud/speech-translate/test-data"
    audio_file_1 = os.path.join(folder, "sample-zh-01.mp3")
    print(transcribe_audio(audio_file_1))

    audio_file_2 = os.path.join(folder, "sample-zh-02.mp3")
    print(transcribe_audio(audio_file_2))
