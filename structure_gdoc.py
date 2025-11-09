# Парсинг документа по абзацам и заголовкам

# Импорт библиотек
from docx import Document
import json
import re

# === НАСТРОЙКИ ===
DOCX_FILE = 'tinkoff_api_full.docx'  
OUTPUT_JSON = 'structured_document.json'

# Размер фрагмента (символов)
FRAGMENT_SIZE = 1000000

def clean_text(text):
    """Очищает текст от лишних пробелов и переносов."""
    return re.sub(r'\s+', ' ', text).strip()

def structure_docx():
    print(f"Открываем документ: {DOCX_FILE}")
    doc = Document(DOCX_FILE)
    
    structured = []
    current_h1 = "Общие сведения"
    current_h2 = None
    buffer = []

    print("Парсинг документа по абзацам и заголовкам...")

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name

        # === H1 ===
        if style.startswith('Heading 1'):
            if buffer:
                structured.append(make_fragment(current_h1, current_h2, buffer))
            current_h1 = clean_text(text)
            current_h2 = None
            buffer = []
            print(f"  [H1] {current_h1}")

        # === H2 ===
        elif style.startswith('Heading 2'):
            if buffer:
                structured.append(make_fragment(current_h1, current_h2, buffer))
            current_h2 = clean_text(text)
            buffer = []
            print(f"    [H2] {current_h2}")

        # === Обычный текст ===
        else:
            buffer.append(clean_text(text))

    # Последний фрагмент
    if buffer:
        structured.append(make_fragment(current_h1, current_h2, buffer))

    # Сохранение
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"\nГОТОВО!")
    print(f"   • Файл: {OUTPUT_JSON}")
    print(f"   • Фрагментов: {len(structured)}")
    print(f"   • Пример:")
    print(json.dumps(structured[0] if structured else {}, ensure_ascii=False, indent=2))

def make_fragment(h1, h2, texts):
    full_text = ' '.join(texts)
    truncated = full_text[:FRAGMENT_SIZE]
    if len(full_text) > FRAGMENT_SIZE:
        truncated += '...'
    return {
        'h1': h1,
        'h2': h2 or "Без подзаголовка",
        'fragment': truncated
    }

if __name__ == '__main__':
    structure_docx()