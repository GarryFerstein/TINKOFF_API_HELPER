# Tinkoff API Helper

Помощник по **Tinkoff Invest API** с RAG и документацией в Google Доке.

---

## Особенности

- Поиск по твоей документации (FAQ, примеры, ошибки)
- Ответы с кодом на Python (`tinkoff-invest-api`)
- Чат-интерфейс на Streamlit
- Автообновление базы знаний
- Работает в **Streamlit Cloud**

---

## Как запустить локально

```bash
git clone https://github.com/твой-username/tinkoff-api-helper.git
cd tinkoff-api-helper
cp .env.example .env  # Вставь свой OPENAI_API_KEY
pip install -r requirements.txt
streamlit run app.py
