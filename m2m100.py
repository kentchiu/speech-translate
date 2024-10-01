from transformers import pipeline
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer


class M2M100Translator:
    """
    M2M100 是一個能夠在 100 種語言之間直接翻譯的模型。

    主要功能：
    1. 初始化 M2M100 翻譯模型
    2. 提供翻譯功能，支持中文、英文、日文、韓文和泰文之間的互譯

    使用方法：
    1. 創建 M2M100Translator 類的實例
    2. 使用 translate 方法進行翻譯
    3. 使用 test_translations 方法測試不同語言間的翻譯
    """

    # MODEL_NAME = "facebook/m2m100_418M"
    MODEL_NAME = "facebook/m2m100_1.2B"
    # 支持的語言及其對應的語言代碼
    LANG_CODES = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "th": "th"}

    def __init__(self):
        self.translator = pipeline("translation", model=self.MODEL_NAME)
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.MODEL_NAME)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        使用 M2M100 模型將文本從源語言翻譯為目標語言。

        Args:
            text (str): 要翻譯的文本
            source_lang (str): 來源語言
            target_lang (str): 目標語言

        Returns:
            str: 翻譯後的文本
        """
        self.tokenizer.src_lang = self.to_language_code(source_lang)

        translated = self.translator(
            text,
            src_lang=self.to_language_code(source_lang),
            tgt_lang=self.to_language_code(target_lang),
        )
        return translated[0]["translation_text"]

    def to_language_code(self, language: str) -> str:
        """
        將語言名稱轉換為語言代碼。
        Args:
            language (str): 語言名稱
        Returns:
            str: 語言代碼
        """
        return self.LANG_CODES[language]

    def test_translations(self):
        test_sentences = {
            "zh": "今天天气真好。",
            "en": "The weather is nice today.",
            "ja": "今日の天気はとても良いです。",
            "ko": "오늘 날씨가 정말 좋습니다.",
            "th": "วันนี้อากาศดีจริงๆ",
        }

        for source_lang, source_text in test_sentences.items():
            print(f"\n原文 ({source_lang}): {source_text}")
            for target_lang in self.LANG_CODES.keys():
                if source_lang != target_lang:
                    translated = self.translate(source_text, source_lang, target_lang)
                    print(f"翻譯成 {target_lang}: {translated}")


def main():
    translator = M2M100Translator()
    translator.test_translations()


if __name__ == "__main__":
    main()
