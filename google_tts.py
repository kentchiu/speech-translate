import requests


def download_google_translate_speech(text, filename="tts", lang="zh"):
    base_url = "https://translate.google.com/translate_tts"
    params = {"ie": "UTF-8", "q": text, "tl": lang, "client": "tw-ob"}
    response = requests.get(base_url, params=params, stream=True)
    if response.status_code == 200:
        save_to = f"dist/{filename}-{lang}.mp3"
        with open(save_to, "wb") as f:
            f.write(response.content)
        print(f"下載完成 - {save_to}")
    else:
        print("下載失敗")


prayer = {
    "en": "God, Grant me the serenity to accept the thing I cannot change, the courage to change the thing I can change, and wisdom to separate the difference.",
    "zh": "神啊!請賜給我雅量從容的接受不可改變的事，賜給我勇氣去改變應該改變的事，並賜給我智慧去分辨什麼是可以改變的，什麼是不可以改變的。",
    "ja": "神様、私に変えられないことを受け入れる落ち着きを与えてください、変えられることを変える勇気を与えてください、そして変えられるものと変えられないものを区別する知恵を与えてください。",
    "ko": "하나님, 나에게 변할 수 없는 것을 받아들이는 온화함을 주시고, 변할 수 있는 것을 변화시키는 용기를 주시며, 변할 수 있는 것과 변할 수 없는 것을 구별하는 지혜를 주시옵소서。",
    "th": "พระเจ้า โปรดประทานความสงบสุขให้ข้าพเจ้ายอมรับสิ่งที่เปลี่ยนแปลงไม่ได้ มีความกล้าที่จะเปลี่ยนแปลงสิ่งที่เปลี่ยนแปลงได้ และมีปัญญาที่จะแยกแยะความแตกต่าง",
}

# for lang, text in prayer.items():
#     download_google_translate_speech(text, lang=lang)

spiderman = {
    "en": "With great power comes great responsibility.",
    "zh": "能力越大，責任越大。",
    "ja": "大いなる力には大いなる責任が伴う",
    "ko": "큰 힘에는 큰 책임이 따른다",
    "th": "ความสามารถที่ยิ่งใหญ่ ความรับผิดชอบก็ยิ่งมาก",
}

# for lang, text in spiderman.items():
#     download_google_translate_speech(text, filename="spiderman", lang=lang)
#

thinking = {
    "en": "I’m not daydreaming; I’m deeply contemplating the mysteries of the universe.",
    "zh": "我不是在發呆，我是在深度思考宇宙的奧秘。",
    "ja": "私はぼーっとしているのではなく、宇宙の神秘について深く考えているのです。",
    "ko": "나는 멍하니 있는 게 아니라 우주의 신비에 대해 깊이 생각하고 있어.",
    "th": "ฉันไม่ได้คิดน้อยใจนะ ฉันกำลังคิดอย่างลึกซึ้งเกี่ยวกับความลึกลับของจักรวาล",
}

for lang, text in thinking.items():
    download_google_translate_speech(text, filename="thinking", lang=lang)
