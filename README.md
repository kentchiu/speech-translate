# 語音辨識

## Run

```bash
uv run asr_v3.py /home/kent/dev/playgroud/speech-translate/test-data/sample4.mp3 --refernect_test  "吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮" --model "openai/whisper-small" --device cpu
```

Write A CLI script which called "fy.py". fy will take a language and a text argument and will output a set of translations of input text.

ex:

`fy --lang zh --model m2m100  "今天天气真好。"`

and will output

```
zh: 今天天气真好。,
en: The weather is nice today.,
ja: 今日の天気はとても良いです。,
ko: 오늘 날씨가 정말 좋습니다.,
th: วันนี้อากาศดีจริงๆ,
```

under the hook, the translate is deon by m2m100.py and mbard
