import whisper
import torch
import os


def transcribe_audio(file_path, model="tiny", device="cpu"):
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file to be transcribed.
        model (str): The Whisper model to use for transcription. Default is "tiny". can be tiny, smale, medium, large, large-v2, large-v3
        device (str): The device to run the model on. Default is "cpu". Can be "cpu" or "cuda".

    Returns:
        str: The transcribed text from the audio file.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
    """
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
