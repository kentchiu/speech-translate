import argparse
from m2m100 import M2M100Translator
from mbart import MBARTTranslator


def main():
    parser = argparse.ArgumentParser(
        description="Translate text using m2m100 or mbart models."
    )
    parser.add_argument("--lang", required=True, help="Source language of the text.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["m2m100", "mbart"],
        help="Translation model to use.",
    )
    parser.add_argument("text", help="Text to translate.")

    args = parser.parse_args()

    if args.model == "m2m100":
        translator = M2M100Translator()
    elif args.model == "mbart":
        translator = MBARTTranslator()

    translations = {}
    for target_lang in translator.LANG_CODES.keys():
        if args.lang != target_lang:
            translated_text = translator.translate(args.text, args.lang, target_lang)
            print(f"🟥[1]: fy.py:21: lang={args.lang} text={translated_text}")
            translations[target_lang] = translated_text

    print(f"{args.lang}: {args.text},")
    for lang, translation in translations.items():
        print(f"{lang}: {translation},")


if __name__ == "__main__":
    main()
