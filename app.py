# Импорт библиотек

import streamlit as st
import requests
import os
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from litellm import completion
import tiktoken  
from dotenv import load_dotenv

# === Загрузка .env ===
load_dotenv()

# === Конфигурация ===
DOC_ID = "1T_X6a4uRjPLvsHnYKsCwBda77RfHxJmgFxtQsbtUFq8"
INDEX_PATH = "tinkoff_faiss.index"
CHUNKS_PATH = "chunks.npy"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 12000  # Оставляем запас для ответа
ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

# === Промпт ===
PROMPT_TEMPLATE = """Ты — Tinkoff API Helper, эксперт по Tinkoff Invest API.
Ты помогаешь разработчикам с интеграцией через библиотеку `tinkoff-invest-api`.

Правила:
- Отвечай  кодом на Python, советом и рекомендацией.
- Используй `with Client(...) as client:`.
- Указывай: sandbox — для тестов, production — реальные деньги.
- Если не уверен: "См. официальную документацию: https://tinkoff.github.io/investAPI/".
- Используй ТОЛЬКО контекст ниже.

Контекст:
{context}

Вопрос: {question}

Ответь как Tinkoff API Helper, с полным рабочим примером кода:"""

# === Подсчёт токенов ===
def count_tokens(text):
    return len(ENCODING.encode(text))

# === Кэширование эмбеддера ===
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDER_MODEL)

# === Загрузка Google Дока ===
@st.cache_data(show_spinner=False)
def fetch_google_doc(doc_id):
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Ошибка загрузки документа: {e}")
        return None

# === Создание/загрузка FAISS ===
@st.cache_resource(show_spinner="Создаём базу знаний...")
def build_or_load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        st.info("Загружаем сохранённую базу знаний...")
        index = faiss.read_index(INDEX_PATH)
        chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
        return index, chunks

    raw_text = fetch_google_doc(DOC_ID)
    if not raw_text:
        st.stop()

    st.info("Создаём базу знаний из документа...")
    chunks = [raw_text[i:i+1200] for i in range(0, len(raw_text), 1000)]
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=16)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks))
    st.success("База знаний готова!")
    return index, chunks

# === Поиск с обрезкой по токенам ===
def search_and_trim(query, index, chunks, max_tokens=MAX_TOKENS):
    embedder = get_embedder()
    q_vec = embedder.encode([query]).astype('float32')
    D, I = index.search(q_vec, 10)  # Берём больше, потом обрезаем

    selected_chunks = []
    total_tokens = 0
    prompt_overhead = count_tokens(PROMPT_TEMPLATE.format(context="", question=query)) + 200  # запас

    for idx in I[0]:
        chunk = chunks[idx]
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens + prompt_overhead <= max_tokens:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break

    return selected_chunks, total_tokens

# === Генерация ответа ===
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка LLM: {e}"

# === Streamlit UI ===
st.set_page_config(page_title="Tinkoff API Helper", page_icon="Chart", layout="centered")
st.title("Tinkoff API Helper")
st.caption("RAG + gpt-4o-mini + токен-контроль")

# Инициализация
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
**Привет!** Я — **Tinkoff API Helper**.  
Задай вопрос по API — дам **рабочий код, совет или рекомендацию**.

**Например:**
- _"Как получить портфель в sandbox?"_
- _"Как купить 1 лот Сбера по рынку?"_
"""}
    ]

# Отображение чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод
if prompt := st.chat_input("Например: Как получить список инструментов?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("RAG-поиск + генерация кода..."):
            try:
                index, chunks = build_or_load_index()
                context_chunks, context_tokens = search_and_trim(prompt, index, chunks)

                # Показ токенов
                st.caption(f"Контекст: **{context_tokens} токенов** (из {MAX_TOKENS})")

                answer = generate_answer(prompt, context_chunks)
                st.markdown(answer)

                # Источники
                with st.expander(f"Источники ({len(context_chunks)} фрагмента, {context_tokens} токенов)"):
                    for i, doc in enumerate(context_chunks):
                        st.caption(f"Фрагмент {i+1} ({count_tokens(doc)} токенов)")
                        st.code(doc[:800] + ("..." if len(doc) > 800 else ""), language="text")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Ошибка: {e}")

# === Сайдбар ===
with st.sidebar:
    st.header("Настройки")

    if st.button("Обновить базу знаний"):
        with st.spinner("Удаляем кэш..."):
            for path in [INDEX_PATH, CHUNKS_PATH]:
                if os.path.exists(path):
                    os.remove(path) if not os.path.isdir(path) else shutil.rmtree(path)
            st.success("База удалена. Перезапустите.")
            st.rerun()

    if st.button("Очистить чат"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    st.markdown("---")
    st.markdown("**Документация:**")
    st.markdown(f"[Google Doc](https://docs.google.com/document/d/{DOC_ID}/edit)")
    st.markdown("[Официально](https://tinkoff.github.io/investAPI/)")
    st.caption("Модель: gpt-4o-mini | Токены: контролируются")