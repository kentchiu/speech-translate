from transformers import pipeline

# Example 1: Using T5 for English to German translation
t5_translator = pipeline("translation_en_to_de", model="t5-small")

english_text = "Hello, how are you?"
german_translation = t5_translator(english_text)
print(f"T5 Translation (EN to DE): {german_translation[0]['translation_text']}")

# Example 2: Using Helsinki-NLP for Chinese to English translation
helsinki_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

chinese_text = "你好，最近如何？"
english_translation = helsinki_translator(chinese_text)
print(
    f"Helsinki-NLP Translation (ZH to EN): {english_translation[0]['translation_text']}"
)

# Example 3: Using mT5 for multi-language translation
mt5_translator = pipeline("translation", model="google/mt5-small")

french_text = "Bonjour, comment allez-vous?"
english_translation = mt5_translator(french_text, max_length=40)
print(f"mT5 Translation (FR to EN): {english_translation[0]['translation_text']}")

# Note: For mT5, you might need to specify the source and target languages
# mt5_translator = pipeline("translation", model="google/mt5-small", src_lang="fr", tgt_lang="en")
