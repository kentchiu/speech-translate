from transformers import pipeline
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer

# 初始化翻譯pipeline
model_name = "facebook/m2m100_418M"
translator = pipeline("translation", model=model_name)

# 定義語言代碼
lang_codes = {"中文": "zh", "英文": "en", "日文": "ja", "韓文": "ko", "泰文": "th"}


# 翻譯函數
def translate(text, source_lang, target_lang):
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    tokenizer.src_lang = lang_codes[source_lang]
    inputs = tokenizer(text, return_tensors="pt")

    translated = translator(
        **inputs,
        src_lang=lang_codes[source_lang],
        tgt_lang=lang_codes[target_lang],
    )
    return translated[0]["translation_text"]


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
    for target_lang in lang_codes.keys():
        if source_lang != target_lang:
            translated = translate(source_text, source_lang, target_lang)
            print(f"翻譯成 {target_lang}: {translated}")
