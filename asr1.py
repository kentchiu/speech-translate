import os
import time

try:
    import whisper
    import torch
    import langcodes
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required dependencies by running:")
    print("pip install openai-whisper torch langcodes")
    exit(1)


def transcribe_audio(file_path, model="tiny", device="cpu"):
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
    text_1, lang_1, time_1 = transcribe_audio(audio_file_1, model="large-v3")
    print(f"Audio 1 - Language: {lang_1}")
    print(f"Transcription: {text_1}")
    print(f"Execution time: {time_1:.2f} seconds\n")

    audio_file_2 = os.path.join(folder, "sample-zh-02.mp3")
    text_2, lang_2, time_2 = transcribe_audio(audio_file_2, model="large-v3")
    print(f"Audio 2 - Language: {lang_2}")
    print(f"Transcription: {text_2}")
    print(f"Execution time: {time_2:.2f} seconds")
