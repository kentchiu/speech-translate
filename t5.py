from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Translator:
    """
    T5 (Text-to-Text Transfer Transformer) 是一個多功能的序列到序列模型，能夠執行多種自然語言處理任務，包括翻譯。

    主要功能：
    1. 初始化 T5 翻譯模型
    2. 提供翻譯功能，支持中文、英文、日文、韓文和泰文之間的互譯

    使用方法：
    1. 創建 T5Translator 類的實例
    2. 使用 translate 方法進行翻譯
    3. 使用 test_translations 方法測試不同語言間的翻譯
    """

    MODEL_NAME = "t5-base"
    # 支持的語言及其對應的語言代碼
    LANG_CODES = {
        "中文": "zh",
        "英文": "en",
        "日文": "ja",
        "韓文": "ko",
        "泰文": "th"
    }

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        使用 T5 模型將文本從源語言翻譯為目標語言。

        Args:
            text (str): 要翻譯的文本
            source_lang (str): 源語言
            target_lang (str): 目標語言

        Returns:
            str: 翻譯後的文本
        """
        task_prefix = f"translate {self.LANG_CODES[source_lang]} to {self.LANG_CODES[target_lang]}: "
        input_text = task_prefix + text
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.model.generate(input_ids, max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    translator = T5Translator()
    translator.test_translations()


if __name__ == "__main__":
    main()
