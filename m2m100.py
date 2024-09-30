from transformers import pipeline
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer

# 初始化翻譯pipeline
MODEL_NAME = "facebook/m2m100_418M"
translator = pipeline("translation", model=MODEL_NAME)

# 定義語言代碼
LANG_CODES = {"中文": "zh", "英文": "en", "日文": "ja", "韓文": "ko", "泰文": "th"}


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    使用 M2M100 模型將文本從源語言翻譯為目標語言。

    Args:
        text (str): 要翻譯的文本
        source_lang (str): 源語言
        target_lang (str): 目標語言

    Returns:
        str: 翻譯後的文本
    """
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = LANG_CODES[source_lang]

    translated = translator(
        text,
        src_lang=LANG_CODES[source_lang],
        tgt_lang=LANG_CODES[target_lang],
    )
    return translated[0]["translation_text"]


def main():
    # 測試翻譯
    test_sentences = {
        "中文": "今天天气真好。",
        "英文": "The weather is nice today.",
        "日文": "今日の天気はとても良いです。",
        "韓文": "오늘 날씨가 정말 좋습니다.",
        "泰文": "วันนี้อากาศดีจริงๆ",
    }

    # 進行所有語言對的翻譯
    for source_lang, source_text in test_sentences.items():
        print(f"\n原文 ({source_lang}): {source_text}")
        for target_lang in LANG_CODES.keys():
            if source_lang != target_lang:
                translated = translate(source_text, source_lang, target_lang)
                print(f"翻譯成 {target_lang}: {translated}")


if __name__ == "__main__":
    main()
