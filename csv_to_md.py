import pandas as pd


df = pd.read_csv("test-data/compare-20240925.csv").sort_values(by="Filename")

# convert facebook/seamless-m4t-v2-large to m4t in model column
df["Model"] = df["Model"].apply(
    lambda x: "m4t" if x == "facebook/seamless-m4t-v2-large" else x
)
df["Model"] = df["Model"].apply(lambda x: "whisper" if x == "large-v3" else x)


grouped = df.groupby("Filename")


prayer = {
    "en": "God, Grant me the serenity to accept the thing I cannot change, the courage to change the thing I can change, and wisdom to separate the difference.",
    "zh": "神啊!請賜給我雅量從容的接受不可改變的事，賜給我勇氣去改變應該改變的事，並賜給我智慧去分辨什麼是可以改變的，什麼是不可以改變的。",
    "ja": "神様、私に変えられないことを受け入れる落ち着きを与えてください、変えられることを変える勇気を与えてください、そして変えられるものと変えられないものを区別する知恵を与えてください。",
    "ko": "하나님, 나에게 변할 수 없는 것을 받아들이는 온화함을 주시고, 변할 수 있는 것을 변화시키는 용기를 주시며, 변할 수 있는 것과 변할 수 없는 것을 구별하는 지혜를 주시옵소서。",
    "th": "พระเจ้า โปรดประทานความสงบสุขให้ข้าพเจ้ายอมรับสิ่งที่เปลี่ยนแปลงไม่ได้ มีความกล้าที่จะเปลี่ยนแปลงสิ่งที่เปลี่ยนแปลงได้ และมีปัญญาที่จะแยกแยะความแตกต่าง",
}


spiderman = {
    "en": "With great power comes great responsibility.",
    "zh": "能力越大，責任越大。",
    "ja": "大いなる力には大いなる責任が伴う",
    "ko": "큰 힘에는 큰 책임이 따른다",
    "th": "ความสามารถที่ยิ่งใหญ่ ความรับผิดชอบก็ยิ่งมาก",
}


thinking = {
    "en": "I’m not daydreaming; I’m deeply contemplating the mysteries of the universe.",
    "zh": "我不是在發呆，我是在深度思考宇宙的奧秘。",
    "ja": "私はぼーっとしているのではなく、宇宙の神秘について深く考えているのです。",
    "ko": "나는 멍하니 있는 게 아니라 우주의 신비에 대해 깊이 생각하고 있어.",
    "th": "ฉันไม่ได้คิดน้อยใจนะ ฉันกำลังคิดอย่างลึกซึ้งเกี่ยวกับความลึกลับของจักรวาล",
}

google = {
    "serenity": prayer,
    "spiderman": spiderman,
    "thinking": thinking,
}

markdown_output = ""

for filename, group in grouped:
    m4t_data = group[group["Model"] == "m4t"].iloc[0]
    whisper_data = group[group["Model"] == "whisper"].iloc[0]
    lang = m4t_data["Lang"]
    # filename substring betweend "test-data/" and "-xx.mp3"
    name = filename[10:-7]

    tts = google[name][lang]
    google_translation = google[name]["en"]

    markdown_output += f"#### {filename}\n"
    markdown_output += "\n"
    markdown_output += f"* Translation time (m4t, whisper)  : {m4t_data['Translation Time']}, {whisper_data['Translation Time']}\n"
    markdown_output += f"* Transcription time (m4t, whisper): {m4t_data['Transcription Time']}, {whisper_data['Transcription Time']}\n"
    markdown_output += "\n"

    markdown_output += "##### Transcription\n"
    markdown_output += "\n"
    markdown_output += "```text\n"
    markdown_output += f"tts    : {tts.strip()}\n"
    markdown_output += f"m4t    : {m4t_data['Transcription'].strip()}.\n"
    markdown_output += f"whisper: {whisper_data['Transcription'].strip()}\n"
    markdown_output += "```\n"
    markdown_output += "##### Translation\n"
    markdown_output += "\n"
    markdown_output += "```text\n"
    markdown_output += f"google : {google_translation.strip()}\n"
    markdown_output += f"m4t    : {m4t_data['Translation'].strip()}\n"
    markdown_output += f"whisper: {whisper_data['Translation'].strip()}\n"
    markdown_output += "```\n"
    markdown_output += "\n"


print(markdown_output)
