# Импорт библиотек

import streamlit as st
import requests
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from litellm import completion
import tiktoken
from dotenv import load_dotenv
import multiprocessing as mp

# === ОТКЛЮЧАЕМ МУЛЬТИПРОЦЕССИНГ ДЛЯ FAISS И TRANSFORMERS ===
# Предотвращаем segmentation fault и утечки семафоров
mp.set_start_method('spawn', force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Отключаем параллелизм HuggingFace
os.environ["OMP_NUM_THREADS"] = "1"  # Отключаем OpenMP в NumPy/FAISS

# === Загрузка .env ===
load_dotenv()

# === Конфигурация ===
DOC_ID = "1T_X6a4uRjPLvsHnYKsCwBda77RfHxJmgFxtQsbtUFq8"
INDEX_PATH = "tinkoff_faiss.index"
CHUNKS_PATH = "chunks.npy"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 12000
ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

# === Промпт ===
PROMPT_TEMPLATE = """Ты — Tinkoff API Helper, эксперт по Tinkoff Invest API.
Ты помогаешь разработчикам с интеграцией через библиотеку `tinkoff-invest-api`.

Правила:
- Отвечай кодом на Python, советом и рекомендацией.
- Используй `with Client(...) as client:`.
- Указывай: sandbox — для тестов, production — реальные деньги.
- Если не уверен: "См. официальную документацию: https://tinkoff.github.io/investAPI/".
- Используй ТОЛЬКО контекст ниже.

Контекст:
{context}

Вопрос: {question}

Ответь как Tinkoff API Helper, с полным рабочим примером кода:"""

# === Подсчёт токенов ===
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

# === Кэшированный эмбеддер (ленивый) ===
@st.cache_resource(show_spinner="Загружаем модель эмбеддингов...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDER_MODEL, device="cpu")  # CPU безопаснее

# === Загрузка Google Дока ===
@st.cache_data(show_spinner=False, ttl=3600)  # Документ обновляется раз в час


def fetch_google_doc(_doc_id: str) -> str:
    url = f"https://docs.google.com/document/d/{_doc_id}/export?format=txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Ошибка загрузки документа: {e}")
        return ""

# === Создание/загрузка FAISS индекса ===
@st.cache_resource(show_spinner="Создаём или загружаем базу знаний...")
def build_or_load_index() -> tuple[faiss.IndexFlatL2, list]:
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
            st.info("База знаний загружена из кэша.")
            return index, chunks
        except Exception as e:
            st.warning(f"Ошибка загрузки индекса: {e}. Пересоздаём...")

    # Загружаем документ
    raw_text = fetch_google_doc(DOC_ID)
    if not raw_text.strip():
        st.error("Не удалось загрузить документ. Проверьте DOC_ID.")
        st.stop()

    # Чанки с перекрытием
    chunks = []
    step = 1000
    overlap = 200
    for i in range(0, len(raw_text), step - overlap):
        chunk = raw_text[i:i + 1200]
        if len(chunk.strip()) > 50:  # Игнорируем слишком короткие
            chunks.append(chunk)

    # Эмбеддинги
    embedder = get_embedder()
    embeddings = embedder.encode(
        chunks,
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True
    ).astype('float32')

    # FAISS индекс
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Сохранение
    faiss.write_index(index, INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks))
    st.success("База знаний создана и сохранена!")

    return index, chunks

# === Поиск с контролем токенов ===
def search_relevant_chunks(query: str, index, chunks, max_tokens: int = MAX_TOKENS):
    embedder = get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True).astype('float32')
    D, I = index.search(q_vec, k=15)

    selected = []
    used_tokens = 0
    overhead = count_tokens(PROMPT_TEMPLATE.format(context="", question=query)) + 300

    for idx in I[0]:
        if idx >= len(chunks):
            continue
        chunk = chunks[idx]
        chunk_tokens = count_tokens(chunk)
        if used_tokens + chunk_tokens + overhead <= max_tokens:
            selected.append(chunk)
            used_tokens += chunk_tokens
        else:
            break

    return selected, used_tokens

# === Генерация ответа ===
def generate_answer(question: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка генерации ответа: {str(e)}\n\nСм. документацию: https://tinkoff.github.io/investAPI/"

# === Streamlit UI ===
st.set_page_config(page_title="Tinkoff API Helper", page_icon="Chart", layout="centered")
st.title("Tinkoff API Helper")
st.caption("RAG + gpt-4o-mini + токен-контроль + стабильность")

# Инициализация чата
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """
**Привет!** Я — **Tinkoff API Helper**.  
Задавай вопросы по Tinkoff Invest API — получишь **рабочий код + советы**.

**Примеры:**
- _"Как получить портфель в sandbox?"_
- _"Как купить 1 лот Сбера по рынку?"_
- _"Как получить список свечей?"_
"""}]

# Отображение истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод вопроса
if prompt := st.chat_input("Например: Как получить список инструментов?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Поиск в документации + генерация кода..."):
            try:
                index, chunks = build_or_load_index()
                context_chunks, token_count = search_relevant_chunks(prompt, index, chunks)

                st.caption(f"Контекст: **{token_count} токенов** (макс. {MAX_TOKENS})")

                answer = generate_answer(prompt, context_chunks)
                st.markdown(answer)

                # Источники
                with st.expander(f"Источники ({len(context_chunks)} фрагмента)"):
                    for i, chunk in enumerate(context_chunks):
                        preview = chunk[:700] + ("..." if len(chunk) > 700 else "")
                        st.caption(f"Фрагмент {i+1} — {count_tokens(chunk)} токенов")
                        st.code(preview, language="text")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Критическая ошибка: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# === Сайдбар ===
with st.sidebar:
    st.header("Управление")

    if st.button("🔄 Обновить базу знаний"):
        for path in [INDEX_PATH, CHUNKS_PATH]:
            if os.path.exists(path):
                try:
                    os.remove(path) if os.path.isfile(path) else None
                except:
                    pass
        st.success("Кэш удалён. Перезапустите приложение.")
        st.rerun()

    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    st.markdown("---")
    st.markdown("**Документация:**")
    st.markdown(f"[Google Doc](https://docs.google.com/document/d/{DOC_ID}/edit)")
    st.markdown("[Официальная](https://tinkoff.github.io/investAPI/)")