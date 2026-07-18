import json
import os
import sys

# ضبط الترميز للطباعة في بيئات Windows Terminal و PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(
    SCRIPT_DIR,
    "hadith-json-main",
    "db",
    "by_book",
    "the_9_books",
    "bukhari.json",
)


def load_json_safe(path):
    if not os.path.exists(path):
        print(f"❌ الملف غير موجود في: {os.path.abspath(path)}")
        sys.exit(1)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"📁 حجم الملف: {size_mb:.2f} MB")
    print("⏳ جاري التحميل...")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ تم تحميل الملف بنجاح")
        return data

    except json.JSONDecodeError as e:
        print(f"❌ خطأ في JSON: {e}")
        sys.exit(1)
    except MemoryError:
        print("❌ MemoryError: الملف كبير جداً على الذاكرة الحالية")
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"❌ خطأ في encoding: {e}")
        sys.exit(1)


def find_hadiths_list(data):
    """
    يكتشف تلقائياً إذا البيانات list أو dict،
    ويرجع (list_of_hadiths, key_used)
    """
    if isinstance(data, list):
        print("🔍 الهيكل: قائمة (list) مباشرة")
        return data, None

    if isinstance(data, dict):
        print(f"🔍 الهيكل: قاموس (dict) يحتوي على المفاتيح: {list(data.keys())}")

        common_keys = [
            "hadiths",
            "data",
            "items",
            "records",
            "hadith",
            "chapters",
            "bukhari",
            "ahadith",
            "narrations",
        ]
        for key in common_keys:
            if key in data and isinstance(data[key], list):
                print(f"✅ وجدت القائمة في المفتاح: {key!r}")
                return data[key], key

        for key, value in data.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"✅ اكتشفت القائمة تلقائياً في المفتاح: {key!r}")
                return value, key

        print("⚠️ لم أجد قائمة واضحة، سأعامل الـ dict كعنصر واحد")
        return [data], None

    print(f"❌ نوع غير متوقع: {type(data)}")
    sys.exit(1)


def clean_text(text):
    """تنظيف النص من علامات RTL وغيرها للعرض"""
    if not isinstance(text, str):
        text = str(text)

    rtl_chars = ["\u200f", "\u200e", "\u202b", "\u202c", "\u202a"]
    for ch in rtl_chars:
        text = text.replace(ch, "")
    return text.strip()


def print_hadith(index, hadith):
    """طباعة حديث بشكل واضح"""
    print("\n" + "═" * 60)
    print(f"  الحديث رقم [{index + 1}]")
    print("═" * 60)

    if isinstance(hadith, dict):
        important_fields = [
            "id",
            "hadithNumber",
            "number",
            "arab",
            "arabic",
            "text",
            "hadith",
            "narrator",
            "book",
            "chapter",
            "chapterName",
            "grade",
            "reference",
        ]

        for field in important_fields:
            if field in hadith:
                val = clean_text(hadith[field])
                print(f"  {field}: {val}")

        for field in hadith:
            if field not in important_fields:
                val = clean_text(hadith[field])
                print(f"  {field}: {val}")
    else:
        print(f"  {clean_text(str(hadith))}")


def analyze_structure(hadiths):
    """تحليل هيكل الحديث لفهم الحقول المتاحة"""
    if hadiths and isinstance(hadiths[0], dict):
        print(f"\n📊 حقول كل حديث: {list(hadiths[0].keys())}")


def main():
    print("=" * 60)
    print("   فحص ملف صحيح البخاري")
    print("=" * 60)

    data = load_json_safe(FILE_PATH)
    hadiths, key_used = find_hadiths_list(data)

    total = len(hadiths)
    print(f"\n📿 إجمالي الأحاديث: {total:,}")

    if total < 1000:
        print("⚠️  تحذير: العدد أقل من المتوقع (7000+)، قد تكون البيانات ناقصة")
    elif total >= 7000:
        print("✅ العدد يطابق المتوقع لصحيح البخاري")

    analyze_structure(hadiths)

    print("\n" + "─" * 60)
    print("  عرض أول حديثين:")
    for i in range(min(2, total)):
        print_hadith(i, hadiths[i])

    print("\n" + "═" * 60)
    print("✅ انتهى الفحص بنجاح")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
