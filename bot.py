"""
Telegram-бот: Coach Grebenyuk — обучение по методологии Михаила Гребенюка «Ноль справа»
- Принимает вопросы текстом и голосом
- NotebookLM (ноутбук Гребенюка, 254 источника) — база знаний
- Gemini — постобработка в коучинговый стиль
"""

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import defaultdict
from functools import partial

from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as genai_types
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

load_dotenv()

# На Railway: восстанавливаем auth и создаём клиент из переменной окружения
_nb_auth_json = os.getenv("NOTEBOOKLM_AUTH_JSON", "").strip()
_nb_auth_json_b64 = os.getenv("NOTEBOOKLM_AUTH_JSON_B64", "").strip()
_nb_data_dir = os.getenv("NOTEBOOKLM_MCP_DATA_DIR", "").strip()
_NB_AUTH_DATA: dict = {}  # хранится в памяти для переподключения при 401

if (_nb_auth_json or _nb_auth_json_b64) and _nb_data_dir:
    import httpx as _httpx
    os.makedirs(_nb_data_dir, exist_ok=True)
    _auth_path = os.path.join(_nb_data_dir, "auth.json")
    if _nb_auth_json_b64:
        _nb_auth_json = base64.b64decode(_nb_auth_json_b64).decode("utf-8")
    _NB_AUTH_DATA = json.loads(_nb_auth_json)
    # Получаем свежий CSRF с текущего IP (Railway), т.к. сохранённый CSRF с другого IP не работает
    try:
        _jar = _httpx.Cookies()
        for _k, _v in _NB_AUTH_DATA.get("cookies", {}).items():
            _jar.set(_k, _v, domain=".google.com")
        _hdrs = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        with _httpx.Client(cookies=_jar, headers=_hdrs, follow_redirects=True, timeout=20.0) as _hc:
            _pg = _hc.get("https://notebooklm.google.com/")
        if _pg.status_code == 200 and "accounts.google.com" not in str(_pg.url):
            _m = re.search(r'"SNlM0e":"([^"]+)"', _pg.text)
            if _m:
                _NB_AUTH_DATA["csrf_token"] = _m.group(1)
                _m2 = re.search(r'"FdrFJe":"(\d+)"', _pg.text)
                if _m2:
                    _NB_AUTH_DATA["session_id"] = _m2.group(1)
                print(f"Startup CSRF OK: {_NB_AUTH_DATA['csrf_token'][:35]}...", flush=True)
            else:
                print("Startup CSRF: SNlM0e not in page, using stored token", flush=True)
            # Авто-определяем build label — Google меняет его раз в несколько недель.
            # Устанавливаем env var ДО первого импорта notebooklm пакета.
            _bl = re.search(r'boq_labs-tailwind-frontend_[\w.]+', _pg.text)
            if _bl:
                _detected_bl = _bl.group(0).rstrip('.')
                os.environ["NOTEBOOKLM_BL"] = _detected_bl
                print(f"Build label: {_detected_bl}", flush=True)
        else:
            print(f"Startup CSRF: page {_pg.status_code}, using stored token", flush=True)
    except Exception as _e:
        print(f"Startup CSRF refresh failed, using stored token: {_e}", flush=True)

    with open(_auth_path, "w", encoding="utf-8") as _f:
        json.dump(_NB_AUTH_DATA, _f)

    # Пре-создаём синглтон клиент напрямую — обходим load_tokens() полностью
    try:
        from notebooklm_mcp_2026 import server as _nb_server_startup
        from notebooklm_mcp_2026.client import NotebookLMClient as _NbClient
        _nb_server_startup._client = _NbClient(
            cookies=_NB_AUTH_DATA.get("cookies", {}),
            csrf_token=_NB_AUTH_DATA.get("csrf_token", ""),
            session_id=_NB_AUTH_DATA.get("session_id", ""),
        )
        print("NotebookLMClient singleton создан", flush=True)
    except Exception as _ce:
        print(f"Ошибка создания клиента: {_ce}", flush=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

NOTEBOOK_ID = "85da7d6e-6980-4da0-89a9-4efabc9542bc"

# История диалога: chat_id -> список {"role": "user"|"assistant", "text": str}
_history: dict[int, list[dict]] = defaultdict(list)
HISTORY_LIMIT = 6

# conversation_id для продолжения диалога в NotebookLM
_nb_conversations: dict[int, str] = {}

# Локальный прокси (опционально): Railway-бот → локальная машина → NotebookLM
_NB_LOCAL_URL = os.getenv("NOTEBOOKLM_LOCAL_URL", "").strip().rstrip("/")
_NB_LOCAL_SECRET = os.getenv("NOTEBOOKLM_LOCAL_SECRET", "").strip()
_NB_REFRESH_MAX_AGE = 25 * 60
_nb_last_refresh_at = time.time() if _NB_AUTH_DATA else 0.0
_nb_query_lock = threading.Lock()

# ─── Промпты ──────────────────────────────────────────────────────────────────

TRANSCRIBE_PROMPT = """Расшифруй это голосовое сообщение на русском языке.

Контекст: пользователь задаёт вопросы о методологии Михаила Гребенюка «Ноль справа» — системе построения отделов продаж и масштабирования бизнеса.
Термины: Ноль справа, KPI, конверсия, воронка продаж, РОП, мотивация, скрипты, декомпозиция, маржа, оборот.

Правила:
- Пиши точно как сказано, без пересказа
- Только текст расшифровки, без комментариев"""


COACH_SYSTEM_PROMPT = """Ты — коуч и наставник, глубоко знающий методологию Михаила Гребенюка «Ноль справа».

Твоя роль: обучать системному построению бизнеса, отделов продаж и масштабированию прибыли на основе авторских материалов Гребенюка. Отвечать чётко, конкретно и по делу — как опытный бизнес-наставник, без воды. Использовать термины методологии естественно. Давать практические инструменты и конкретные шаги. При необходимости задавать уточняющие вопросы.

Формат ответа — ОБЯЗАТЕЛЬНО:
Пиши сплошным живым текстом, как говоришь вслух. Никаких звёздочек, никаких дефисов в начале строк, никаких тире как маркеров списка, никакого markdown вообще. Только обычные слова и предложения. Абзацы разделяй пустой строкой. Длина ответа — строго не более 200 слов. Завершай ответ коротким вопросом или конкретным заданием на сегодня."""


def _build_notebooklm_query(question: str, history: list[dict]) -> str:
    context = ""
    if history:
        lines = []
        for msg in history[-4:]:
            role = "Ученик" if msg["role"] == "user" else "Коуч"
            lines.append(f"{role}: {msg['text']}")
        context = "Контекст предыдущего диалога:\n" + "\n".join(lines) + "\n\n"
    return (
        f"{context}"
        f"Вопрос по методологии Гребенюка «Ноль справа»:\n{question}\n\n"
        "Дай развёрнутый ответ, опираясь на материалы методологии."
    )


def _strip_markdown(text: str) -> str:
    text = re.sub(r'\s*\[\d+(?:[,\-\s]\s*\d+)*\]', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _persist_notebooklm_auth() -> None:
    if not _nb_data_dir or not _NB_AUTH_DATA:
        return
    os.makedirs(_nb_data_dir, exist_ok=True)
    with open(os.path.join(_nb_data_dir, "auth.json"), "w", encoding="utf-8") as f:
        json.dump(_NB_AUTH_DATA, f)


def _refresh_notebooklm_auth_sync() -> bool:
    if not _NB_AUTH_DATA:
        return False

    import httpx as _h

    jar = _h.Cookies()
    for key, value in _NB_AUTH_DATA.get("cookies", {}).items():
        jar.set(key, value, domain=".google.com")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with _h.Client(cookies=jar, headers=headers, follow_redirects=True, timeout=25.0) as client:
            page = client.get("https://notebooklm.google.com/")
    except Exception as exc:
        logger.warning(f"NotebookLM auth refresh failed: {exc}")
        return False

    if page.status_code != 200 or "accounts.google.com" in str(page.url):
        logger.warning(f"NotebookLM auth refresh unexpected page: {page.status_code} {page.url}")
        return False

    csrf_match = re.search(r'"SNlM0e":"([^"]+)"', page.text)
    if csrf_match:
        _NB_AUTH_DATA["csrf_token"] = csrf_match.group(1)
        session_match = re.search(r'"FdrFJe":"(\d+)"', page.text)
        if session_match:
            _NB_AUTH_DATA["session_id"] = session_match.group(1)

    build_match = re.search(r'boq_labs-tailwind-frontend_[\w.]+', page.text)
    build_label = build_match.group(0).rstrip(".") if build_match else None
    if build_label:
        os.environ["NOTEBOOKLM_BL"] = build_label

    try:
        _persist_notebooklm_auth()
    except Exception as exc:
        logger.warning(f"NotebookLM auth persist failed: {exc}")

    try:
        from notebooklm_mcp_2026 import server as nb_server
        from notebooklm_mcp_2026.client import NotebookLMClient
        config_module = sys.modules.get("notebooklm_mcp_2026.config")
        if config_module and build_label:
            config_module.BUILD_LABEL = build_label
        nb_server._client = NotebookLMClient(
            cookies=_NB_AUTH_DATA.get("cookies", {}),
            csrf_token=_NB_AUTH_DATA.get("csrf_token", ""),
            session_id=_NB_AUTH_DATA.get("session_id", ""),
        )
    except Exception as exc:
        logger.warning(f"NotebookLM client refresh failed: {exc}")
        return False

    logger.info(f"NotebookLM auth refresh OK: BL={build_label or 'N/A'} CSRF={'OK' if csrf_match else 'N/A'}")
    return True


def _query_notebooklm_once(query: str, conversation_id: str | None) -> dict:
    script = r"""
import json
import sys
from notebooklm_mcp_2026.tools.query import query_notebook

payload = json.load(sys.stdin)
result = query_notebook(
    notebook_id=payload["notebook_id"],
    query=payload["query"],
    conversation_id=payload.get("conversation_id") or None,
)
print(json.dumps(result, ensure_ascii=False))
"""
    payload = {
        "notebook_id": NOTEBOOK_ID,
        "query": query,
        "conversation_id": conversation_id,
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=85,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "NotebookLM timeout after 85s"}

    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or proc.stdout)[-2000:]}

    stdout = proc.stdout.strip()
    if not stdout:
        return {"status": "error", "error": "NotebookLM subprocess returned empty output"}

    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"NotebookLM subprocess JSON error: {exc}; output={stdout[-1000:]}"}


def _coach_reformat(raw_answer: str, question: str, history: list[dict]) -> str:
    client = google_genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(timeout=60_000),
    )
    history_text = ""
    if history:
        lines = [
            f"{'Ученик' if m['role'] == 'user' else 'Коуч'}: {m['text']}"
            for m in history[-4:]
        ]
        history_text = "\n\nКонтекст диалога:\n" + "\n".join(lines)

    prompt = (
        f"{COACH_SYSTEM_PROMPT}\n\n"
        f"Вопрос ученика: {question}{history_text}\n\n"
        f"Информация из материалов методологии (используй как источник, перепиши своими словами):\n{raw_answer}\n\n"
        "Дай ответ в роли коуча. Только ответ, без вводных фраз типа 'Конечно!' или 'Отличный вопрос!'."
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()


# ─── NotebookLM ──────────────────────────────────────────────────────────────

def _ask_notebooklm(query: str, chat_id: int = 0) -> str | None:
    global _nb_last_refresh_at
    logger.info(f"NotebookLM query: {query[:80]}")

    if _NB_LOCAL_URL:
        # Прокси-режим: запрос уходит на локальный сервер пользователя
        try:
            import urllib.request
            payload = json.dumps({"query": query, "chat_id": chat_id}).encode("utf-8")
            req = urllib.request.Request(
                f"{_NB_LOCAL_URL}/ask",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Secret": _NB_LOCAL_SECRET,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                answer = data.get("answer", "").strip()
                logger.info(f"NotebookLM proxy: {len(answer)} символов")
                return answer or None
            else:
                logger.error(f"NotebookLM proxy error: {data.get('error')}")
                return None
        except Exception as e:
            logger.exception(f"NotebookLM proxy exception: {e}")
            return None

    # Прямой режим: импорт notebooklm_mcp_2026 с 401-retry
    conv_id = _nb_conversations.get(chat_id)

    if time.time() - _nb_last_refresh_at > _NB_REFRESH_MAX_AGE:
        with _nb_query_lock:
            if time.time() - _nb_last_refresh_at > _NB_REFRESH_MAX_AGE:
                if _refresh_notebooklm_auth_sync():
                    _nb_last_refresh_at = time.time()

    for _attempt in range(3):
        try:
            result = _query_notebooklm_once(query, conv_id or None)
            logger.info(f"NotebookLM status: {result.get('status')} | attempt={_attempt}")
            if result.get("status") == "success":
                new_conv = result.get("conversation_id")
                if new_conv:
                    _nb_conversations[chat_id] = new_conv
                return result.get("answer", "").strip() or None

            error = result.get("error", "")
            if "401" in str(error) and _attempt < 2:
                logger.info("NotebookLM 401, refreshing auth and retrying...")
                with _nb_query_lock:
                    if _refresh_notebooklm_auth_sync():
                        _nb_last_refresh_at = time.time()
                continue
                try:
                    _nb_server.reset_client()
                    from notebooklm_mcp_2026.client import NotebookLMClient as _NbClient
                    # csrf_token="" → конструктор сам получит свежий CSRF через _refresh_auth_tokens()
                    _nb_server._client = _NbClient(
                        cookies=_NB_AUTH_DATA.get("cookies", {}),
                        csrf_token="",
                        session_id=_NB_AUTH_DATA.get("session_id", ""),
                    )
                except Exception as _re:
                    logger.warning(f"Клиент не пересоздан: {_re}")
                continue

            logger.error(f"NotebookLM error: {error} | hint: {result.get('hint', '')}")
            return None
        except Exception as e:
            logger.exception(f"NotebookLM exception: {e}")
            return None
    return None


# ─── Транскрипция голоса ──────────────────────────────────────────────────────

def _transcribe(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    client = google_genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(timeout=120_000),
    )
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                    TRANSCRIBE_PROMPT,
                ],
            )
            return response.text.strip()
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise


# ─── TTS через Gemini ─────────────────────────────────────────────────────────

_TTS_CHUNK_LIMIT = 800


def _tts_chunk(text: str) -> str:
    client = google_genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(timeout=120_000),
    )
    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name="Sadaltager"
                            )
                        )
                    ),
                ),
            )
            break
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ("deadline", "504", "timeout", "timed", "503", "unavailable")) and attempt < 2:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    if response is None:
        raise RuntimeError("TTS: все попытки исчерпаны")

    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)
    return path


def _split_for_tts(text: str) -> list[str]:
    if len(text) <= _TTS_CHUNK_LIMIT:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > _TTS_CHUNK_LIMIT:
        cut = remaining[:_TTS_CHUNK_LIMIT]
        last_dot = cut.rfind(".")
        if last_dot > _TTS_CHUNK_LIMIT // 2:
            cut = cut[:last_dot + 1]
        chunks.append(cut.strip())
        remaining = remaining[len(cut):].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _text_to_speech(text: str) -> list[str]:
    parts = _split_for_tts(text)
    return [_tts_chunk(p) for p in parts]


# ─── Вспомогательные ─────────────────────────────────────────────────────────

async def _run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args))


async def _periodic_notebooklm_refresh():
    global _nb_last_refresh_at
    while True:
        await asyncio.sleep(1800)
        try:
            def _locked_refresh():
                with _nb_query_lock:
                    ok = _refresh_notebooklm_auth_sync()
                    if ok:
                        return time.time()
                    return 0.0

            refreshed_at = await _run_blocking(_locked_refresh)
            if refreshed_at:
                _nb_last_refresh_at = refreshed_at
        except Exception:
            logger.exception("NotebookLM periodic refresh failed")


async def _post_init(app: Application):
    asyncio.create_task(_periodic_notebooklm_refresh())
    print("Periodic NotebookLM auth refresh scheduled (every 30m)", flush=True)


async def _send_long(update: Update, text: str):
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


async def _answer(update: Update, question: str):
    chat_id = update.effective_chat.id
    history = _history[chat_id]

    await update.message.reply_text("Ищу в материалах Гребенюка... ⏳")
    query = _build_notebooklm_query(question, history)
    raw = await _run_blocking(_ask_notebooklm, query, chat_id)

    if not raw:
        await update.message.reply_text(
            "Не удалось получить ответ из базы знаний. "
            "Попробуй переформулировать вопрос или повторить чуть позже."
        )
        return

    await update.message.reply_text("Формулирую ответ... 💭")
    try:
        answer = await _run_blocking(_coach_reformat, raw, question, history)
    except Exception:
        logger.exception("Gemini reformat error")
        answer = raw
    answer = _strip_markdown(answer)

    history.append({"role": "user", "text": question})
    history.append({"role": "assistant", "text": answer[:500]})
    if len(history) > HISTORY_LIMIT:
        _history[chat_id] = history[-HISTORY_LIMIT:]

    await _send_long(update, answer)

    # Для голоса берём первые 600 символов — быстрая генерация, текст уже доставлен
    voice_text = answer[:600]
    dot = voice_text.rfind(".")
    if dot > 200:
        voice_text = voice_text[:dot + 1].strip()

    audio_paths: list[str] = []
    try:
        audio_paths = await asyncio.wait_for(
            _run_blocking(_text_to_speech, voice_text),
            timeout=50.0,
        )
        for path in audio_paths:
            with open(path, "rb") as f:
                await update.message.reply_voice(f)
    except asyncio.TimeoutError:
        logger.warning("TTS timeout, skipping voice")
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
    finally:
        for path in audio_paths:
            try:
                os.unlink(path)
            except Exception:
                pass


# ─── Handlers ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    _history[chat_id].clear()
    await update.message.reply_text(
        "Привет! Я Coach Grebenyuk — наставник по методологии Михаила Гребенюка «Ноль справа».\n\n"
        "Задавай вопросы текстом или голосом — отвечу по авторским материалам.\n\n"
        "С чего начнём?\n"
        "— Что такое «Ноль справа» и как работает методология\n"
        "— Как построить отдел продаж с нуля\n"
        "— KPI, скрипты и мотивация команды\n"
        "— Декомпозиция целей и масштабирование прибыли\n\n"
        "/reset — начать диалог заново\n"
        "/id — узнать свой Telegram ID\n"
        "/debug — диагностика подключения"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    _history[chat_id].clear()
    _nb_conversations.pop(chat_id, None)
    await update.message.reply_text("Диалог сброшен. Начинаем с чистого листа. О чём поговорим?")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Твой Telegram chat_id: `{update.effective_chat.id}`", parse_mode="Markdown"
    )


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    auth_json_set = bool(os.getenv("NOTEBOOKLM_AUTH_JSON", "").strip())
    auth_json_b64_set = bool(os.getenv("NOTEBOOKLM_AUTH_JSON_B64", "").strip())
    data_dir = os.getenv("NOTEBOOKLM_MCP_DATA_DIR", "").strip()
    lines.append(f"NOTEBOOKLM_AUTH_JSON_B64 set: {auth_json_b64_set}")
    lines.append(f"NOTEBOOKLM_AUTH_JSON задан: {auth_json_set}")
    lines.append(f"NOTEBOOKLM_MCP_DATA_DIR: {data_dir or '(не задан)'}")
    lines.append(f"NOTEBOOKLM_LOCAL_URL: {_NB_LOCAL_URL or '(не задан, прямой режим)'}")

    if data_dir:
        auth_path = os.path.join(data_dir, "auth.json")
        exists = os.path.exists(auth_path)
        lines.append(f"auth.json существует: {exists}")
        if exists:
            try:
                with open(auth_path) as f:
                    data = json.load(f)
                cookies = data.get("cookies", {})
                csrf = data.get("csrf_token", "")
                lines.append(f"Кук: {list(cookies.keys())[:4]}...")
                lines.append(f"CSRF: {csrf[:40]}..." if csrf else "CSRF: (пусто)")
            except Exception as e:
                lines.append(f"Ошибка чтения auth.json: {e}")

    # Статус singleton клиента
    try:
        from notebooklm_mcp_2026 import server as _nb_srv
        from notebooklm_mcp_2026.auth import load_tokens as _lt
        lines.append(f"nb_server._client: {'создан' if _nb_srv._client else 'None'}")
        tok = _lt()
        lines.append(f"load_tokens(): {'OK' if tok else 'NONE!'}")
    except Exception as _de:
        lines.append(f"diagnostics error: {_de}")

    lines.append(f"_NB_AUTH_DATA cookies: {list(_NB_AUTH_DATA.get('cookies', {}).keys())[:3]}")

    lines.append("\nЗапрашиваю NotebookLM (тест)...")
    await update.message.reply_text("\n".join(lines))
    lines = []

    try:
        answer = await _run_blocking(
            _ask_notebooklm,
            "Что такое «Ноль справа»?",
            update.effective_chat.id,
        )
        lines.append(f"Status: {'success' if answer else 'error'}")
        if answer:
            lines.append(f"Answer preview:\n{answer[:400]}")
        await update.message.reply_text("\n".join(lines))
        return
        from notebooklm_mcp_2026.tools.query import query_notebook
        result = query_notebook(notebook_id=NOTEBOOK_ID, query="Что такое «Ноль справа»?")
        status = result.get("status")
        error = result.get("error", "")
        hint = result.get("hint", "")
        answer = result.get("answer", "")
        lines.append(f"Статус: {status}")
        if error:
            lines.append(f"Ошибка: {error}")
        if hint:
            lines.append(f"Подсказка: {hint}")
        if answer:
            lines.append(f"Ответ (200 симв.):\n{answer[:200]}")
    except Exception as e:
        import traceback
        lines.append(f"Исключение: {e}")
        lines.append(traceback.format_exc()[-800:])

    await update.message.reply_text("\n".join(lines))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if question:
        await _answer(update, question)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Расшифровываю... 🎤")
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    try:
        question = await _run_blocking(_transcribe, tmp_path)
        await update.message.reply_text(f"_{question}_", parse_mode="Markdown")
        await _answer(update, question)
    except Exception as e:
        logger.exception("Transcription error")
        await update.message.reply_text(f"Не удалось расшифровать: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─── Запуск ──────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN не задан в .env")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY не задан в .env")
        sys.exit(1)

    mode = f"прокси → {_NB_LOCAL_URL}" if _NB_LOCAL_URL else "прямой импорт"
    print(f"Coach Grebenyuk запускается... NotebookLM: {mode}")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("debug", cmd_debug))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Бот запущен. Ожидаю сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
