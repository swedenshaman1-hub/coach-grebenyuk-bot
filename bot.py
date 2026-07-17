import os, sys, asyncio, logging, json, sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Инициализируем auth.json из env-переменной (для Railway/облака)
_nlm_auth_env = os.environ.get("NOTEBOOKLM_AUTH", "")
if _nlm_auth_env:
    from platformdirs import user_data_dir
    _auth_dir = Path(user_data_dir("notebooklm-mcp-2026"))
    _auth_dir.mkdir(parents=True, exist_ok=True)
    (_auth_dir / "auth.json").write_text(_nlm_auth_env, encoding="utf-8")

from notebooklm_mcp_2026.tools.query import query_notebook as nlm_query

from groq import Groq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# --- Конфиг ---
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
OWNER_ID        = int(os.getenv("OWNER_CHAT_ID", "0"))
NOTEBOOK_ID     = "85da7d6e-6980-4da0-89a9-4efabc9542bc"
DB_PATH         = Path(__file__).parent / "data" / "coach.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

groq = Groq(api_key=GROQ_API_KEY)
DB_PATH.parent.mkdir(exist_ok=True)


# --- База данных ---
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                name TEXT,
                questions INTEGER DEFAULT 0,
                last_seen TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                topic TEXT,
                done_at TEXT
            )
        """)


def upsert_user(user_id: int, name: str):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            INSERT INTO users(user_id, name, last_seen)
            VALUES(?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                name=excluded.name, last_seen=excluded.last_seen
        """, (user_id, name, datetime.now().isoformat()))


def inc_questions(user_id: int):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("UPDATE users SET questions=questions+1 WHERE user_id=?", (user_id,))


def save_lesson(user_id: int, topic: str):
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO lessons(user_id, topic, done_at) VALUES(?,?,?)",
            (user_id, topic[:80], datetime.now().isoformat())
        )


def get_progress(user_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT name, questions FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        topics = c.execute(
            "SELECT topic FROM lessons WHERE user_id=? ORDER BY id DESC LIMIT 10",
            (user_id,)
        ).fetchall()
    return {
        "name": row[0] if row else "Студент",
        "questions": row[1] if row else 0,
        "topics": [t[0] for t in topics],
    }


# --- NotebookLM + Groq ---
def ask_notebooklm(question: str, conv_id: str | None = None) -> tuple[str, str]:
    result = nlm_query(NOTEBOOK_ID, question, conversation_id=conv_id)
    if result.get("status") == "success":
        return result["answer"], result.get("conversation_id", "")
    log.error("NLM error: %s", result.get("error"))
    return "", ""


def format_teaching(raw: str, question: str, name: str) -> str:
    """Groq форматирует сырой ответ NotebookLM в обучающий стиль."""
    if not raw:
        return "Не удалось получить ответ из базы знаний. Попробуй позже."
    prompt = (
        f"Ты — коуч по методологии Михаила Гребенюка. "
        f"Студента зовут {name}. "
        f"Вот материал из базы знаний:\n\n{raw}\n\n"
        f"Вопрос студента: {question}\n\n"
        f"Дай чёткий, практичный ответ в стиле Гребенюка: конкретно, без воды, "
        f"с примером или заданием если уместно. Максимум 400 слов."
    )
    resp = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# --- Handlers ---
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    upsert_user(user.id, user.first_name)
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("📚 Урок дня", callback_data="lesson"),
        InlineKeyboardButton("📊 Прогресс", callback_data="progress"),
    ]])
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я — Coach Grebenyuk, твой наставник по методологии "
        "Михаила Гребенюка «Ноль справа».\n\n"
        "Задавай любые вопросы по продажам, управлению командой "
        "и масштабированию бизнеса — отвечу из базы знаний.\n\n"
        "Или выбери действие:",
        reply_markup=kb
    )


async def cmd_lesson(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    progress = get_progress(user.id)
    covered = ", ".join(progress["topics"][:5]) or "нет"
    msg = await update.message.reply_text("⏳ Готовлю урок дня...")

    raw, _ = ask_notebooklm(
        f"Дай один конкретный урок или инструмент из методологии Гребенюка, "
        f"который ещё не изучался. Уже пройдено: {covered}. "
        f"Формат: название темы, суть метода, конкретное задание на сегодня."
    )
    answer = format_teaching(raw, "урок дня", user.first_name)
    topic = answer.split("\n")[0][:80]
    save_lesson(user.id, topic)

    await msg.edit_text(f"📚 *Урок дня*\n\n{answer}", parse_mode="Markdown")


async def cmd_progress(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    p = get_progress(user.id)
    topics_str = "\n".join(f"  • {t}" for t in p["topics"]) or "  Уроков пока нет"
    await update.message.reply_text(
        f"📊 *Прогресс {p['name']}*\n\n"
        f"Вопросов задано: {p['questions']}\n"
        f"Последние уроки:\n{topics_str}\n\n"
        f"Продолжай в том же духе! 💪",
        parse_mode="Markdown"
    )


async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "lesson":
        # Имитируем вызов cmd_lesson через message
        user = q.from_user
        progress = get_progress(user.id)
        covered = ", ".join(progress["topics"][:5]) or "нет"
        msg = await q.message.reply_text("⏳ Готовлю урок дня...")
        raw, _ = ask_notebooklm(
            f"Дай один конкретный урок из методологии Гребенюка. "
            f"Уже пройдено: {covered}. "
            f"Формат: название, суть, задание на сегодня."
        )
        answer = format_teaching(raw, "урок дня", user.first_name)
        save_lesson(user.id, answer.split("\n")[0][:80])
        await msg.edit_text(f"📚 *Урок дня*\n\n{answer}", parse_mode="Markdown")
    elif q.data == "progress":
        p = get_progress(q.from_user.id)
        topics_str = "\n".join(f"  • {t}" for t in p["topics"]) or "  Уроков пока нет"
        await q.message.reply_text(
            f"📊 *Прогресс*\n\nВопросов: {p['questions']}\n\nУроки:\n{topics_str}",
            parse_mode="Markdown"
        )


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    upsert_user(user.id, user.first_name)
    question = update.message.text.strip()
    msg = await update.message.reply_text("🔍 Ищу в базе знаний Гребенюка...")

    raw, _ = ask_notebooklm(question)
    answer = format_teaching(raw, question, user.first_name)
    inc_questions(user.id)

    await msg.edit_text(answer)


async def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Нет TELEGRAM_TOKEN в .env")
    if not GROQ_API_KEY:
        raise ValueError("Нет GROQ_API_KEY в .env")

    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("lesson", cmd_lesson))
    app.add_handler(CommandHandler("progress", cmd_progress))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Coach Grebenyuk bot started")
    async with app:
        await app.start()
        await app.updater.start_polling()
        log.info("Polling started. Press Ctrl+C to stop.")
        await asyncio.Event().wait()
        await app.updater.stop()
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
