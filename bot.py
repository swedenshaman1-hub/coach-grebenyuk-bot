"""Telegram-бот «Архитектор роста» — персональный AI-бизнес-коуч."""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
import zipfile
from functools import partial

import access_control as access_db
import edge_tts
import speech_recognition as sr
import vosk
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as genai_types
from telegram import (
    BotCommand,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters,
)

from notebook_registry import load_registry
from notebooklm_gateway import NotebookLMGateway
from strict_contract import ErrorType, ResultStatus
from strict_service import StrictKnowledgeService
from verified_repository import VerifiedRepository

load_dotenv()

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
_admin_ids_env = os.getenv("ADMIN_CHAT_IDS", "1288155468")
ADMIN_CHAT_IDS: set[int] = {
    int(value.strip()) for value in _admin_ids_env.split(",") if value.strip()
}
BOT_USERNAME = os.getenv("BOT_USERNAME", "growth_architect_ai_bot").lstrip("@")

_GEMINI_QUOTA_RECHECK_SECONDS = 15 * 60
_gemini_quota_blocked_until = 0.0
_VOSK_MODEL_NAME = "vosk-model-small-ru-0.22"
_VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{_VOSK_MODEL_NAME}.zip"
_VOSK_MODEL_DIR = os.getenv(
    "VOSK_MODEL_DIR",
    os.path.join(tempfile.gettempdir(), _VOSK_MODEL_NAME),
)
_vosk_model = None
_vosk_model_lock = threading.Lock()

HISTORY_LIMIT = 6
_chat_answer_locks: dict[int, asyncio.Lock] = {}

_nb_health_ok: bool | None = None
_nb_last_admin_alert_at = 0.0
_nb_last_error = ""
_nb_last_error_type = ErrorType.NONE
_NB_ADMIN_ALERT_INTERVAL = 6 * 60 * 60

NOTEBOOK_REGISTRY = load_registry()
VERIFIED_REPOSITORY = VerifiedRepository()
NOTEBOOK_GATEWAY = NotebookLMGateway()
STRICT_KNOWLEDGE = StrictKnowledgeService(
    NOTEBOOK_REGISTRY,
    NOTEBOOK_GATEWAY,
    VERIFIED_REPOSITORY,
    collection_id=os.getenv("NOTEBOOK_COLLECTION", "grebenyuk"),
)

# ─── Промпты ──────────────────────────────────────────────────────────────────

TRANSCRIBE_PROMPT = """Расшифруй это голосовое сообщение на русском языке.

Контекст: пользователь обсуждает развитие бизнеса, продажи, управление и масштабирование.
Термины: KPI, конверсия, воронка продаж, РОП, мотивация, скрипты, декомпозиция, маржа, оборот, прибыль.

Правила:
- Пиши точно как сказано, без пересказа
- Только текст расшифровки, без комментариев"""


def _needs_missing_context_clarification(question: str, history: list[dict]) -> bool:
    if history:
        return False
    normalized = " ".join(re.sub(r"[^а-яёa-z0-9]+", " ", question.lower()).split())
    return normalized in {
        "давай дальше",
        "продолжай",
        "продолжим",
        "что дальше",
        "поехали дальше",
        "да",
        "ок",
        "хорошо",
    }




def _prewarm_primary_knowledge_sync() -> bool:
    """Health is successful only when the registered NotebookLM sources load."""
    global _nb_last_error, _nb_last_error_type
    ok, detail = STRICT_KNOWLEDGE.health()
    _nb_last_error = "" if ok else detail
    _nb_last_error_type = ErrorType.NONE if ok else ErrorType.UNKNOWN
    if ok:
        logger.info("Strict NotebookLM health OK: %s", detail)
    else:
        logger.warning("Strict NotebookLM health failed: %s", detail)
    return ok


# ─── Транскрипция голоса ──────────────────────────────────────────────────────

def _is_gemini_quota_error(exc: Exception) -> bool:
    error = str(exc).lower()
    return "resource_exhausted" in error or "prepayment credits are depleted" in error


def _get_vosk_model():
    global _vosk_model
    with _vosk_model_lock:
        if _vosk_model is not None:
            return _vosk_model

        if not os.path.isdir(_VOSK_MODEL_DIR):
            model_parent = os.path.dirname(_VOSK_MODEL_DIR)
            os.makedirs(model_parent, exist_ok=True)
            zip_path = os.path.join(model_parent, f"{_VOSK_MODEL_NAME}.zip")
            logger.info("Downloading offline Russian speech model...")
            with urllib.request.urlopen(_VOSK_MODEL_URL, timeout=90) as response:
                with open(zip_path, "wb") as output:
                    shutil.copyfileobj(response, output)
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    parent_real = os.path.realpath(model_parent)
                    for member in archive.infolist():
                        target = os.path.realpath(os.path.join(model_parent, member.filename))
                        if target != parent_real and not target.startswith(parent_real + os.sep):
                            raise RuntimeError("Unsafe path in Vosk model archive")
                    archive.extractall(model_parent)
            finally:
                try:
                    os.unlink(zip_path)
                except OSError:
                    pass

        vosk.SetLogLevel(-1)
        _vosk_model = vosk.Model(_VOSK_MODEL_DIR)
        logger.info("Offline Russian speech model is ready")
        return _vosk_model


def _convert_to_wav(file_path: str) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    converted = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", file_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if converted.returncode != 0:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg conversion failed: {converted.stderr[-300:]}")
    return wav_path


def _transcribe_vosk(file_path: str) -> str:
    """Полностью автономная русская расшифровка внутри Railway."""
    wav_path = _convert_to_wav(file_path)
    try:
        model = _get_vosk_model()
        pieces: list[str] = []
        with wave.open(wav_path, "rb") as audio:
            recognizer = vosk.KaldiRecognizer(model, audio.getframerate())
            while True:
                data = audio.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    if text:
                        pieces.append(text)
            final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
            if final_text:
                pieces.append(final_text)

        result = " ".join(pieces).strip()
        if not result:
            raise RuntimeError("offline transcription returned empty text")
        return result
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _transcribe_google_web(file_path: str) -> str:
    """Резервная расшифровка без Gemini API и платных кредитов."""
    wav_path = _convert_to_wav(file_path)
    try:
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 30
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        result = recognizer.recognize_google(audio, language="ru-RU").strip()
        if not result:
            raise RuntimeError("fallback transcription returned empty text")
        return result
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _transcribe(file_path: str) -> str:
    global _gemini_quota_blocked_until
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    if time.time() >= _gemini_quota_blocked_until:
        client = google_genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=genai_types.HttpOptions(timeout=25_000),
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                    TRANSCRIBE_PROMPT,
                ],
            )
            return response.text.strip()
        except Exception as exc:
            if _is_gemini_quota_error(exc):
                _gemini_quota_blocked_until = time.time() + _GEMINI_QUOTA_RECHECK_SECONDS
                logger.warning("Gemini transcription quota exhausted; switching to fallback")
            else:
                logger.warning(f"Gemini transcription failed; switching to fallback: {type(exc).__name__}")

    try:
        return _transcribe_vosk(file_path)
    except Exception as exc:
        logger.exception(f"Offline transcription failed; trying Google Speech: {exc}")
        return _transcribe_google_web(file_path)


# ─── TTS через Gemini ─────────────────────────────────────────────────────────

_TTS_CHUNK_LIMIT = 2000
_TTS_CACHE_TTL = 24 * 60 * 60
_TTS_CACHE_MAX = 200
_tts_answers: dict[str, tuple[int, str, float]] = {}
_tts_in_progress: set[str] = set()


def _edge_tts_chunk(text: str) -> str:
    """Резервная озвучка без Gemini API и платных кредитов."""
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    async def _save() -> None:
        communicate = edge_tts.Communicate(
            text,
            voice="ru-RU-DmitryNeural",
            connect_timeout=10,
            receive_timeout=30,
        )
        await communicate.save(path)

    try:
        asyncio.run(_save())
        if os.path.getsize(path) == 0:
            raise RuntimeError("Edge TTS returned an empty audio file")
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _tts_chunk(text: str) -> str:
    global _gemini_quota_blocked_until
    if time.time() < _gemini_quota_blocked_until:
        return _edge_tts_chunk(text)

    client = google_genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(timeout=30_000),
    )
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
    except Exception as exc:
        if _is_gemini_quota_error(exc):
            _gemini_quota_blocked_until = time.time() + _GEMINI_QUOTA_RECHECK_SECONDS
            logger.warning("Gemini TTS quota exhausted; switching to Edge TTS")
            return _edge_tts_chunk(text)
        logger.warning(f"Gemini TTS failed; switching to Edge TTS: {type(exc).__name__}")
        return _edge_tts_chunk(text)

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


def _store_tts_answer(chat_id: int, text: str) -> str:
    """Store the complete answer behind a short Telegram callback token."""
    now = time.time()
    expired = [
        token
        for token, (_, _, created_at) in _tts_answers.items()
        if now - created_at > _TTS_CACHE_TTL
    ]
    for token in expired:
        _tts_answers.pop(token, None)

    while len(_tts_answers) >= _TTS_CACHE_MAX:
        oldest = next(iter(_tts_answers))
        _tts_answers.pop(oldest, None)

    token = secrets.token_urlsafe(6)
    _tts_answers[token] = (chat_id, text, now)
    return token


# ─── Вспомогательные ─────────────────────────────────────────────────────────

async def _run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args))


async def _notify_admin_notebooklm(bot, error: str, force: bool = False):
    global _nb_last_admin_alert_at
    now = time.time()
    if not force and now - _nb_last_admin_alert_at < _NB_ADMIN_ALERT_INTERVAL:
        return
    _nb_last_admin_alert_at = now
    error_value = str(error).lower()
    if _nb_last_error_type is ErrorType.AUTH or any(
        marker in error_value for marker in ("401", "403", "authentication", "auth_required")
    ):
        reason = (
            "истекла авторизация Google. Нужно один раз повторно войти в "
            "NotebookLM на компьютере, после чего обновить сессию Railway."
        )
    elif _nb_last_error_type is ErrorType.RATE_LIMIT or "429" in error_value:
        reason = "NotebookLM временно ограничил частоту запросов."
    elif _nb_last_error_type in {
        ErrorType.TIMEOUT, ErrorType.SERVER, ErrorType.NETWORK,
    }:
        reason = "временная ошибка соединения; следующая проверка будет выполнена автоматически."
    else:
        reason = "основной источник не ответил; следующая проверка будет выполнена автоматически."
    text = (
        "⚠️ Строгая база NotebookLM временно недоступна. Неподтверждённые "
        "ответы пользователям не выдаются.\n\n"
        f"Причина: {reason}"
    )
    for admin_id in ADMIN_CHAT_IDS:
        try:
            await bot.send_message(chat_id=admin_id, text=text)
        except Exception:
            logger.exception("Could not send NotebookLM alert to admin %s", admin_id)


async def _periodic_notebooklm_health(app: Application):
    global _nb_health_ok
    await asyncio.sleep(300)
    while True:
        try:
            was_ok = _nb_health_ok
            is_ok = await _run_blocking(_prewarm_primary_knowledge_sync)
            _nb_health_ok = is_ok
            if not is_ok:
                await _notify_admin_notebooklm(app.bot, _nb_last_error or "health check failed")
            elif was_ok is False:
                for admin_id in ADMIN_CHAT_IDS:
                    await app.bot.send_message(
                        chat_id=admin_id,
                        text="✅ Связь с основной базой знаний восстановлена.",
                    )
        except Exception:
            logger.exception("NotebookLM periodic health check failed")
        await asyncio.sleep(1800)


async def _prewarm_vosk_model():
    try:
        await _run_blocking(_get_vosk_model)
    except Exception:
        logger.exception("Offline speech model prewarm failed")


async def _prewarm_notebooklm_sources(app: Application):
    global _nb_health_ok
    try:
        _nb_health_ok = await _run_blocking(_prewarm_primary_knowledge_sync)
        if not _nb_health_ok:
            await _notify_admin_notebooklm(app.bot, _nb_last_error or "startup health check failed")
    except Exception:
        _nb_health_ok = False
        logger.exception("NotebookLM source prewarm failed")


def _is_admin(chat_id: int) -> bool:
    return chat_id in ADMIN_CHAT_IDS


def _is_allowed(chat_id: int) -> bool:
    return _is_admin(chat_id) or access_db.has_active_access(chat_id)


async def _send_access_denied(message):
    await message.reply_text(
        "Доступ к ассистенту не активирован или уже истёк. "
        "Попроси у администратора персональную ссылку-приглашение."
    )


async def _post_init(app: Application):
    # Telegram rate-limits profile mutations very aggressively. The profile is
    # already configured in BotFather, so production restarts must not rewrite
    # it. Enable this flag only for an intentional one-time profile update.
    if os.getenv("SYNC_TELEGRAM_PROFILE", "false").lower() in {"1", "true", "yes", "on"}:
        try:
            await app.bot.set_my_name("Архитектор роста")
            await app.bot.set_my_short_description(
                "AI-бизнес-коуч по продажам, управлению, прибыли и системному росту."
            )
            await app.bot.set_my_description(
                "Персональный AI-бизнес-коуч. Помогает диагностировать бизнес, "
                "находить точки роста, усиливать продажи, выстраивать команду, "
                "декомпозировать цели и превращать идеи в конкретный план действий."
            )
            await app.bot.set_my_commands([
                BotCommand("start", "Начать работу"),
                BotCommand("help", "Проверить доступ"),
                BotCommand("reset", "Начать новый диалог"),
            ])
            for admin_id in ADMIN_CHAT_IDS:
                await app.bot.set_my_commands(
                    [
                        BotCommand("admin", "Панель администратора"),
                        BotCommand("invite7", "Создать доступ на 7 дней"),
                        BotCommand("users", "Активные пользователи"),
                        BotCommand("start", "Открыть ассистента"),
                        BotCommand("reset", "Начать новый диалог"),
                        BotCommand("help", "Команды администратора"),
                        BotCommand("id", "Показать Telegram ID"),
                        BotCommand("health", "Проверить все компоненты"),
                        BotCommand("sources", "Активные блокноты"),
                        BotCommand("verify", "Свежая проверка вопроса"),
                        BotCommand("cache", "Проверить карточку вопроса"),
                        BotCommand("debug", "Безопасная диагностика"),
                    ],
                    scope=BotCommandScopeChat(chat_id=admin_id),
                )
            logger.info("Telegram profile configured: Архитектор роста")
        except Exception:
            logger.exception("Telegram profile configuration failed")
    else:
        logger.info("Telegram profile sync skipped on startup")

    asyncio.create_task(_periodic_notebooklm_health(app))
    asyncio.create_task(_prewarm_vosk_model())
    asyncio.create_task(_prewarm_notebooklm_sources(app))
    print("Periodic knowledge health check scheduled (every 30m)", flush=True)


async def _send_long(update: Update, text: str, reply_markup=None):
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)] or [text]
    for index, chunk in enumerate(chunks):
        await update.message.reply_text(
            chunk,
            reply_markup=reply_markup if index == len(chunks) - 1 else None,
        )


async def _answer_unlocked(update: Update, question: str, force_fresh: bool = False):
    global _nb_health_ok, _nb_last_error, _nb_last_error_type
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        await _send_access_denied(update.message)
        return
    history = access_db.get_chat_history(chat_id, HISTORY_LIMIT)
    if _needs_missing_context_clarification(question, history):
        await update.message.reply_text(
            "Мне не хватает предыдущего контекста, чтобы продолжить точно. "
            "Напиши одним предложением, какую задачу мы разбирали, — и я сразу подхвачу."
        )
        return
    started_at = time.monotonic()
    await update.message.reply_text("Анализирую вопрос... ⏳")
    result = await _run_blocking(
        STRICT_KNOWLEDGE.answer,
        question,
        history,
        chat_id,
        force_fresh,
    )
    _nb_last_error_type = result.error_type

    if result.status is ResultStatus.VERIFIED:
        _nb_health_ok = True
        _nb_last_error = ""
    elif result.source_kind == "verified_cache" and result.text:
        _nb_health_ok = False
        _nb_last_error = result.error_type.value
        await _notify_admin_notebooklm(update.get_bot(), _nb_last_error)
    elif result.status is ResultStatus.INSUFFICIENT:
        _nb_health_ok = True
        _nb_last_error = ""
        clarification = result.text.strip() or (
            "Чтобы продолжить предметно, напиши, чем занимается твой бизнес, "
            "какой сейчас результат и какую цель ты хочешь получить."
        )
        access_db.append_chat_exchange(
            chat_id,
            question,
            clarification,
            HISTORY_LIMIT,
        )
        logger.info(
            "Strict clarification ready in %.2fs | request=%s",
            time.monotonic() - started_at,
            result.request_id,
        )
        await update.message.reply_text(clarification)
        return
    elif result.status is ResultStatus.AUTH_REQUIRED:
        _nb_health_ok = False
        _nb_last_error = "auth_required"
        await _notify_admin_notebooklm(update.get_bot(), _nb_last_error, force=True)
        await update.message.reply_text(
            "Основная база знаний временно недоступна. Содержательный ответ "
            "не сформирован, чтобы не подменять подтверждённые сведения догадками."
        )
        return
    elif result.status in {ResultStatus.PARTIAL, ResultStatus.UNAVAILABLE}:
        _nb_health_ok = False
        _nb_last_error = result.error_type.value
        await _notify_admin_notebooklm(update.get_bot(), _nb_last_error)
        await update.message.reply_text(
            "Не удалось получить и проверить ответ по основной базе знаний. "
            "Неподтверждённый ответ не будет показан. Попробуй повторить чуть позже."
        )
        return

    answer = result.text.strip()
    if not answer:
        logger.error("Strict service returned no text: request=%s", result.request_id)
        await update.message.reply_text(
            "Проверенный ответ получился пустым. Неподтверждённый вариант не показан."
        )
        return

    # Access may be revoked while a long knowledge query is running.
    if not _is_allowed(chat_id):
        await _send_access_denied(update.message)
        return

    logger.info(
        "Strict answer ready in %.2fs | source=%s | chars=%s | request=%s",
        time.monotonic() - started_at,
        result.source_kind,
        len(answer),
        result.request_id,
    )

    access_db.append_chat_exchange(
        chat_id,
        question,
        answer,
        HISTORY_LIMIT,
    )

    tts_token = _store_tts_answer(chat_id, answer)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "🔊 Озвучить полностью",
            callback_data=f"tts:{tts_token}",
        )
    ]])
    await _send_long(update, answer, reply_markup=keyboard)


async def _answer(update: Update, question: str, force_fresh: bool = False):
    """Keep each chat ordered while allowing different chats to run concurrently."""
    chat_id = update.effective_chat.id
    lock = _chat_answer_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_answer_locks[chat_id] = lock
    async with lock:
        await _answer_unlocked(update, question, force_fresh=force_fresh)


async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not _is_allowed(update.effective_chat.id):
        await query.answer("Доступ не активирован или уже истёк.", show_alert=True)
        return
    token = (query.data or "").partition(":")[2]
    entry = _tts_answers.get(token)

    if not entry or time.time() - entry[2] > _TTS_CACHE_TTL:
        _tts_answers.pop(token, None)
        await query.answer(
            "Эта кнопка устарела. Задай вопрос ещё раз.",
            show_alert=True,
        )
        return

    chat_id, text, _ = entry
    if chat_id != update.effective_chat.id:
        await query.answer("Эта кнопка относится к другому чату.", show_alert=True)
        return

    if token in _tts_in_progress:
        await query.answer("Этот ответ уже озвучивается.")
        return

    await query.answer("Начинаю полную озвучку")
    _tts_in_progress.add(token)
    status_message = await query.message.reply_text("🔊 Озвучиваю весь ответ...")
    parts = _split_for_tts(text)
    sent_parts = 0

    try:
        for index, part in enumerate(parts, start=1):
            audio_path = None
            try:
                audio_path = await asyncio.wait_for(
                    _run_blocking(_tts_chunk, part),
                    timeout=75.0,
                )
                caption = (
                    f"🔊 Часть {index}/{len(parts)}"
                    if len(parts) > 1
                    else "🔊 Полная озвучка"
                )
                with open(audio_path, "rb") as audio:
                    await query.message.reply_voice(audio, caption=caption)
                sent_parts += 1
            finally:
                if audio_path:
                    try:
                        os.unlink(audio_path)
                    except OSError:
                        pass

        await status_message.edit_text("✅ Полная озвучка готова.")
    except asyncio.TimeoutError:
        logger.warning("Full-answer TTS timed out after part %s", sent_parts)
        await status_message.edit_text(
            f"Не удалось озвучить часть {sent_parts + 1}. Нажми кнопку ещё раз."
        )
    except Exception as exc:
        logger.exception("Full-answer TTS failed: %s", exc)
        await status_message.edit_text(
            f"Не удалось завершить озвучку после части {sent_parts}. Попробуй ещё раз."
        )
    finally:
        _tts_in_progress.discard(token)


# ─── Handlers ────────────────────────────────────────────────────────────────


def _admin_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔗 Создать доступ на 7 дней", callback_data="admin:invite7")],
        [InlineKeyboardButton("👥 Активные пользователи", callback_data="admin:users")],
        [InlineKeyboardButton("ℹ️ Инструкция", callback_data="admin:help")],
    ])


async def _send_invitation(message, token: str):
    invite_url = f"https://t.me/{BOT_USERNAME}?start={token}"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🚀 Принять приглашение", url=invite_url)
    ]])
    await message.reply_text(
        "🚀 <b>Приглашение в «Архитектор роста»</b>\n\n"
        "Персональный AI-бизнес-коуч поможет вам:\n\n"
        "• найти главные точки роста бизнеса;\n"
        "• усилить продукт, продажи и управление;\n"
        "• превратить финансовую цель в понятный план;\n"
        "• получить конкретные решения и следующие шаги.\n\n"
        "🎁 Доступ предоставляется на 7 дней с момента активации.\n"
        "Приглашение персональное и действует для одного Telegram-аккаунта.\n\n"
        f"👉 <a href=\"{invite_url}\"><b>Принять приглашение</b></a>\n\n"
        "Нажмите на надпись выше, чтобы начать.",
        parse_mode="HTML",
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


async def _send_admin_panel(message, context: ContextTypes.DEFAULT_TYPE, pin: bool = False):
    panel = await message.reply_text(
        "Панель администратора\n\n"
        "Здесь можно создать одноразовую ссылку на 7 дней, посмотреть "
        "активных пользователей, продлить или отключить доступ.",
        reply_markup=_admin_keyboard(),
    )
    if pin:
        try:
            await context.bot.pin_chat_message(
                chat_id=message.chat_id,
                message_id=panel.message_id,
                disable_notification=True,
            )
        except Exception as error:
            logger.warning("Could not pin admin panel: %s", error)
    return panel


async def _send_active_users(message):
    users = access_db.list_active_users()
    if not users:
        await message.reply_text(
            "Сейчас нет активных тестировщиков.",
            reply_markup=_admin_keyboard(),
        )
        return

    lines = ["Активные пользователи:"]
    buttons = []
    for user in users:
        chat_id = int(user["chat_id"])
        name = user["display_name"] or "Без имени"
        username = f' @{user["username"]}' if user["username"] else ""
        expiry = access_db.format_expiry(int(user["expires_at"]))
        lines.append(f"{name}{username}\nID: {chat_id}\nДо: {expiry}")
        buttons.append([
            InlineKeyboardButton(
                f"➕ 7 дней: {name[:18]}",
                callback_data=f"admin:extend:{chat_id}",
            ),
            InlineKeyboardButton(
                "⛔ Отключить",
                callback_data=f"admin:revoke:{chat_id}",
            ),
        ])
    buttons.append([InlineKeyboardButton("⬅️ Панель", callback_data="admin:panel")])
    await message.reply_text(
        "\n\n".join(lines),
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    await _send_admin_panel(update.message, context, pin=True)


async def cmd_invite7(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    token = access_db.create_invite(update.effective_chat.id, 7)
    await _send_invitation(update.message, token)


async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    await _send_active_users(update.message)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if _is_admin(chat_id):
        await update.message.reply_text(
            "Команды администратора:\n"
            "/admin — открыть и закрепить панель\n"
            "/invite7 — создать ссылку на 7 дней\n"
            "/users — активные пользователи\n"
            "/health — состояние Telegram, базы и NotebookLM\n"
            "/sources — активная коллекция и UUID\n"
            "/verify <вопрос> — запрос без резервной карточки\n"
            "/cache <вопрос> — статус проверенной карточки\n"
            "/debug — безопасная диагностика\n"
            "/id — показать Telegram ID"
        )
        return

    access = access_db.get_access(chat_id)
    if access and int(access["expires_at"]) > int(time.time()):
        await update.message.reply_text(
            "Доступ активен до "
            f"{access_db.format_expiry(int(access['expires_at']))}.\n\n"
            "Можно задавать вопросы о развитии бизнеса текстом и голосом, "
            "получать практические ответы и запускать полную озвучку."
        )
    else:
        await _send_access_denied(update.message)


async def handle_admin_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not _is_admin(update.effective_chat.id):
        await query.answer("Недостаточно прав", show_alert=True)
        return
    await query.answer()
    data = query.data or ""

    if data == "admin:panel":
        await _send_admin_panel(query.message, context)
    elif data == "admin:invite7":
        token = access_db.create_invite(update.effective_chat.id, 7)
        await _send_invitation(query.message, token)
    elif data == "admin:users":
        await _send_active_users(query.message)
    elif data == "admin:help":
        await query.message.reply_text(
            "Нажми «Создать доступ на 7 дней» и перешли полученную ссылку. "
            "После активации человек появится в списке пользователей. "
            "Там же можно продлить или отключить его доступ."
        )
    elif data.startswith("admin:extend:"):
        chat_id = int(data.rsplit(":", 1)[1])
        expires_at = access_db.extend_access(chat_id, 7)
        if expires_at:
            await query.message.reply_text(
                "Доступ продлён до " + access_db.format_expiry(expires_at)
            )
        await _send_active_users(query.message)
    elif data.startswith("admin:revoke:"):
        chat_id = int(data.rsplit(":", 1)[1])
        access_db.revoke_access(chat_id)
        await query.message.reply_text(f"Доступ пользователя {chat_id} отключён.")
        await _send_active_users(query.message)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    access_db.clear_chat_history(chat_id)
    STRICT_KNOWLEDGE.reset_session(chat_id)

    if context.args and not _is_admin(chat_id):
        token = context.args[0].strip()
        user = update.effective_user
        status, expires_at = access_db.activate_invite(
            token,
            chat_id,
            user.full_name or "Без имени",
            user.username,
        )
        if status == "activated":
            await update.message.reply_text(
                "Доступ активирован на 7 дней.\n"
                f"Он действует до {access_db.format_expiry(expires_at)}."
            )
        elif status == "already":
            await update.message.reply_text(
                "Эта ссылка уже активирована тобой. Доступ действует до "
                f"{access_db.format_expiry(expires_at)}."
            )
        elif status == "used":
            await update.message.reply_text("Эта ссылка уже использована другим человеком.")
            return
        else:
            await update.message.reply_text("Ссылка недействительна.")
            return

    if not _is_allowed(chat_id):
        await _send_access_denied(update.message)
        return

    await update.message.reply_text(
        "Привет! Я «Архитектор роста» — твой персональный AI-бизнес-коуч.\n\n"
        "Я помогаю предпринимателям находить главные ограничения бизнеса и превращать их в понятный план роста. Со мной можно:\n\n"
        "— провести диагностику бизнеса и найти точку максимального влияния;\n"
        "— усилить продукт, предложение и позиционирование;\n"
        "— построить воронку, отдел продаж, KPI и мотивацию команды;\n"
        "— декомпозировать финансовую цель и увеличить прибыль;\n"
        "— разобрать сложную ситуацию и получить конкретные следующие шаги.\n\n"
        "Отправь вопрос текстом или голосовым сообщением. Я дам практическую трактовку, предложу решение и задам следующий вопрос. Под ответом будет кнопка полной озвучки.\n\n"
        "Чтобы начать, напиши: «Помоги провести диагностику моего бизнеса».\n\n"
        "/reset — начать новый диалог\n"
        "/help — проверить срок доступа"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        await _send_access_denied(update.message)
        return
    access_db.clear_chat_history(chat_id)
    STRICT_KNOWLEDGE.reset_session(chat_id)
    await update.message.reply_text("Диалог сброшен. Начинаем с чистого листа. О чём поговорим?")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    await update.message.reply_text(
        f"Твой Telegram chat_id: `{update.effective_chat.id}`", parse_mode="Markdown"
    )


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _nb_health_ok, _nb_last_error, _nb_last_error_type
    if not _is_admin(update.effective_chat.id):
        return
    await update.message.reply_text("Проверяю строгую связь с NotebookLM…")
    started_at = time.monotonic()
    _nb_health_ok, detail = await _run_blocking(STRICT_KNOWLEDGE.health)
    elapsed = time.monotonic() - started_at
    if _nb_health_ok:
        _nb_last_error = ""
        _nb_last_error_type = ErrorType.NONE
        await update.message.reply_text(
            "✅ Статус: работает\n"
            "Политика: strict_notebooklm\n"
            f"Хранилище: {VERIFIED_REPOSITORY.backend_name}\n"
            f"Проверка: {detail}\n"
            f"Время: {elapsed:.1f} с"
        )
    else:
        _nb_last_error = detail
        logger.error("Admin knowledge diagnostic failed: %s", detail)
        await update.message.reply_text(
            "❌ Статус: NotebookLM временно недоступен. "
            f"Класс ошибки: {_nb_last_error_type.value}. "
            "Cookies, CSRF и токены не выводятся."
        )


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    await cmd_debug(update, context)


async def cmd_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    await update.message.reply_text(STRICT_KNOWLEDGE.source_summary())


async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text("Использование: /verify <вопрос>")
        return
    await _answer(update, question, force_fresh=True)


async def cmd_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_chat.id):
        return
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text("Использование: /cache <вопрос>")
        return
    history = access_db.get_chat_history(update.effective_chat.id, HISTORY_LIMIT)
    card = STRICT_KNOWLEDGE.cache_info(question, history)
    if card is None:
        await update.message.reply_text("Проверенной карточки для этого вопроса нет.")
        return
    await update.message.reply_text(
        "Проверенная карточка найдена.\n"
        f"Статус: verified\n"
        f"Проверена: {access_db.format_expiry(card.verified_at)}\n"
        f"Действительна до: {access_db.format_expiry(card.expires_at)}\n"
        f"Утверждений с доказательствами: {len(card.claims)}"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update.effective_chat.id):
        await _send_access_denied(update.message)
        return
    question = (update.message.text or "").strip()
    if question:
        await _answer(update, question)


async def _download_voice_with_retry(context, voice, destination: str) -> None:
    """Telegram occasionally times out while serving voice files; retry safely."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            telegram_file = await asyncio.wait_for(
                context.bot.get_file(voice.file_id), timeout=30.0
            )
            await asyncio.wait_for(
                telegram_file.download_to_drive(destination), timeout=45.0
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= 2:
                break
            logger.warning(
                "Telegram voice download retry %s after %s",
                attempt + 1,
                type(exc).__name__,
            )
            await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(
        f"Telegram voice download failed: {type(last_error).__name__}"
    ) from last_error


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update.effective_chat.id):
        await _send_access_denied(update.message)
        return
    await update.message.reply_text("Расшифровываю... 🎤")
    voice = update.message.voice
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await _download_voice_with_retry(context, voice, tmp_path)
        question = await _run_blocking(_transcribe, tmp_path)
        await update.message.reply_text(f"_{question}_", parse_mode="Markdown")
        await _answer(update, question)
    except Exception as e:
        logger.exception("Transcription error")
        await update.message.reply_text(
            "Не удалось разобрать голосовое сообщение. Отправь его ещё раз "
            "или напиши вопрос текстом."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def handle_application_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Keep transient Telegram errors observable without leaking user content."""
    logger.error(
        "Unhandled Telegram update error: %s",
        type(context.error).__name__,
        exc_info=context.error,
    )


# ─── Запуск ──────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN не задан в .env")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY не задан в .env")
        sys.exit(1)

    access_db.init_db()
    STRICT_KNOWLEDGE.init()

    mode = "strict NotebookLM proxy" if NOTEBOOK_GATEWAY.local_url else "strict NotebookLM direct"
    print(f"Архитектор роста запускается... knowledge mode: {mode}")
    print(f"Администраторы: {sorted(ADMIN_CHAT_IDS)}")
    print(f"База доступа: {access_db.DB_PATH}")
    print(f"Проверенные карточки: {VERIFIED_REPOSITORY.backend_name}")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("sources", cmd_sources))
    app.add_handler(CommandHandler("verify", cmd_verify))
    app.add_handler(CommandHandler("cache", cmd_cache))
    app.add_handler(CommandHandler("debug", cmd_debug))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("invite7", cmd_invite7))
    app.add_handler(CommandHandler("users", cmd_users))
    app.add_handler(CallbackQueryHandler(handle_admin_button, pattern=r"^admin:"))
    app.add_handler(CallbackQueryHandler(handle_tts, pattern=r"^tts:"))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(handle_application_error)

    print("Бот запущен. Ожидаю сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
