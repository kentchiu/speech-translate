"""
m2m100.py - M2M100 多語言翻譯模型實現

這個模塊實現了使用 Facebook 的 M2M100 模型進行多語言翻譯的功能。
M2M100 是一個能夠在 100 種語言之間直接翻譯的模型。

主要功能：
1. 初始化 M2M100 翻譯模型
2. 提供翻譯功能，支持中文、英文、日文、韓文和泰文之間的互譯
3. 包含一個測試函數，用於展示不同語言間的翻譯效果

使用方法：
1. 創建 M2M100Translator 類的實例
2. 使用 translate 方法進行翻譯
3. 使用 test_translations 方法測試不同語言間的翻譯

範例：
    translator = M2M100Translator()
    translated_text = translator.translate("你好", "中文", "英文")
    print(translated_text)

注意：
- 首次運行時，模型會自動下載，可能需要一些時間
- 翻譯質量可能因語言對和具體內容而異
"""

from transformers import pipeline
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer

class M2M100Translator:
    # 使用 facebook/m2m100_418M 模型，這是一個相對較小但效果不錯的多語言翻譯模型
    MODEL_NAME = "facebook/m2m100_418M"
    # 支持的語言及其對應的語言代碼
    LANG_CODES = {"中文": "zh", "英文": "en", "日文": "ja", "韓文": "ko", "泰文": "th"}

    def __init__(self):
        # 初始化翻譯管道和分詞器
        self.translator = pipeline("translation", model=self.MODEL_NAME)
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.MODEL_NAME)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        使用 M2M100 模型將文本從源語言翻譯為目標語言。

        Args:
            text (str): 要翻譯的文本
            source_lang (str): 源語言
            target_lang (str): 目標語言

        Returns:
            str: 翻譯後的文本
        """
        self.tokenizer.src_lang = self.LANG_CODES[source_lang]

        translated = self.translator(
            text,
            src_lang=self.LANG_CODES[source_lang],
            tgt_lang=self.LANG_CODES[target_lang],
        )
        return translated[0]["translation_text"]

    def test_translations(self):
        test_sentences = {
            "中文": "今天天气真好。",
            "英文": "The weather is nice today.",
            "日文": "今日の天気はとても良いです。",
            "韓文": "오늘 날씨가 정말 좋습니다.",
            "泰文": "วันนี้อากาศดีจริงๆ",
        }

        for source_lang, source_text in test_sentences.items():
            print(f"\n原文 ({source_lang}): {source_text}")
            for target_lang in self.LANG_CODES.keys():
                if source_lang != target_lang:
                    translated = self.translate(source_text, source_lang, target_lang)
                    print(f"翻譯成 {target_lang}: {translated}")

def main():
    # 創建翻譯器實例
    translator = M2M100Translator()
    # 運行測試翻譯，展示不同語言間的翻譯效果
    translator.test_translations()

if __name__ == "__main__":
    main()

# 要運行這個腳本，請使用以下命令：
# uv run m2m100.py
