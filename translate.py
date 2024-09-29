from transformers import pipeline


model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("How are you?")
# [{'translation_text': 'Comment allez-vous ?'}]
