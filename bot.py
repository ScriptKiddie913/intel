import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.sessions import StringSession


load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_intel_bot")


TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", "").strip()
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "").strip()
TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
DEST_CHANNEL = os.getenv("DEST_CHANNEL", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openchat/openchat-7b").strip()


ALLOWED_CATEGORIES = {
    "terrorism",
    "cybersecurity",
    "stock_market",
    "geopolitics",
    "crime",
    "protest",
    "military",
    "economic",
    "other",
}

DEFAULT_TAGS = {
    "location": "unknown",
    "date": "unknown",
    "category": "other",
    "entities": [],
    "keywords": [],
}

TAGGING_URL = "https://openrouter.ai/api/v1/chat/completions"
TAGGING_SYSTEM_PROMPT = (
    "You are an intelligence analyst AI that extracts structured metadata from news/intelligence text. "
    "Return STRICT JSON only with keys: location, date, category, entities, keywords. "
    "Categories must be one of: terrorism, cybersecurity, stock_market, geopolitics, crime, protest, military, economic, other. "
    "If unknown, use unknown for location/date and other for category. "
    "Do not add markdown, code fences, explanations, or extra keys."
)

MAX_CAPTION_LENGTH = 1024
MAX_TEXT_LENGTH = 4096
API_RETRY_ATTEMPTS = 3
API_RETRY_BASE_DELAY = 1.5
DEDUP_TTL_SECONDS = int(os.getenv("DEDUP_TTL_SECONDS", "3600"))
HEALTHCHECK_PORT = int(os.getenv("PORT", os.getenv("HEALTHCHECK_PORT", "10000")))
HEALTHCHECK_HOST = os.getenv("HEALTHCHECK_HOST", "0.0.0.0")


if not TELEGRAM_API_ID or not TELEGRAM_API_HASH or (not TELEGRAM_SESSION and not BOT_TOKEN) or not SOURCE_CHANNEL or not DEST_CHANNEL:
    raise RuntimeError(
        "Missing required Telegram configuration. Set TELEGRAM_API_ID, TELEGRAM_API_HASH, SOURCE_CHANNEL, DEST_CHANNEL, and either TELEGRAM_SESSION or BOT_TOKEN."
    )

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY.")

try:
    TELEGRAM_API_ID_INT = int(TELEGRAM_API_ID)
except ValueError as exc:
    raise RuntimeError("TELEGRAM_API_ID must be an integer.") from exc


def parse_source_channel_value(raw_value: str) -> tuple[set[int], set[str]]:
    numeric_ids: set[int] = set()
    usernames: set[str] = set()

    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        cleaned = part.lstrip("@").strip()
        try:
            numeric_ids.add(int(cleaned))
        except ValueError:
            usernames.add(cleaned.lower())

    return numeric_ids, usernames


def parse_destination_channel_value(raw_value: str) -> int | str:
    cleaned = raw_value.strip().lstrip("@").strip()
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


SOURCE_CHANNEL_IDS, SOURCE_CHANNEL_USERNAMES = parse_source_channel_value(SOURCE_CHANNEL)
DEST_CHANNEL_ENTITY = parse_destination_channel_value(DEST_CHANNEL)


@dataclass
class TagPayload:
    location: str
    date: str
    category: str
    entities: List[str]
    keywords: List[str]


class RecentMessageCache:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        async with self._lock:
            self._purge()
            if key in self._items:
                return True
            self._items[key] = time.time()
            return False

    async def release(self, key: str) -> None:
        async with self._lock:
            self._items.pop(key, None)

    def _purge(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        stale_keys = [key for key, value in self._items.items() if value < cutoff]
        for key in stale_keys:
            self._items.pop(key, None)


processed_messages = RecentMessageCache(ttl_seconds=DEDUP_TTL_SECONDS)
tagging_semaphore = asyncio.Semaphore(int(os.getenv("TAGGING_CONCURRENCY", "2")))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def build_message_key(event: events.NewMessage.Event) -> str:
    message = event.message
    parts = [
        str(event.chat_id),
        str(message.id),
        str(message.grouped_id or ""),
        hashlib.sha256((message.message or "").encode("utf-8", errors="ignore")).hexdigest(),
    ]
    return ":".join(parts)


def is_image_message(message: Any) -> bool:
    if getattr(message, "photo", None):
        return True
    file = getattr(message, "file", None)
    mime_type = getattr(file, "mime_type", None) if file else None
    return bool(mime_type and mime_type.startswith("image/"))


def extract_source_text(message: Any) -> str:
    raw_text = message.message or message.text or ""
    return normalize_whitespace(raw_text)


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        cleaned: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_tag_payload(payload: Dict[str, Any], message_date: datetime) -> TagPayload:
    location = str(payload.get("location") or "unknown").strip() or "unknown"
    date_value = str(payload.get("date") or "unknown").strip() or "unknown"
    if date_value.lower() == "unknown":
        date_value = message_date.date().isoformat()

    category = str(payload.get("category") or "other").strip().lower() or "other"
    if category not in ALLOWED_CATEGORIES:
        category = "other"

    entities = safe_list(payload.get("entities"))
    keywords = safe_list(payload.get("keywords"))

    return TagPayload(
        location=location,
        date=date_value,
        category=category,
        entities=entities,
        keywords=keywords,
    )


def format_tags(tags: TagPayload) -> str:
    entities = ", ".join(tags.entities) if tags.entities else "unknown"
    keywords = ", ".join(tags.keywords) if tags.keywords else "unknown"
    return (
        "\n\n---\n\n"
        f"📍 Location: {tags.location}\n"
        f"📅 Date: {tags.date}\n"
        f"⚠️ Category: {tags.category}\n"
        f"🏷 Entities: {entities}\n"
        f"🔎 Keywords: {keywords}\n"
        "\n---"
    )


def assemble_post_text(original_text: str, tags: TagPayload) -> str:
    base = original_text.strip()
    tagged = f"{base}{format_tags(tags)}" if base else format_tags(tags).lstrip("\n")
    return tagged.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("OpenRouter response did not contain valid JSON.")


async def call_openrouter(text: str, message_date: datetime) -> TagPayload:
    user_prompt = (
        f"Message date: {message_date.date().isoformat()}\n"
        "Extract the intelligence metadata from the following text and return STRICT JSON only.\n\n"
        f"{text}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": TAGGING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                response = await client.post(TAGGING_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                return normalize_tag_payload(parsed, message_date)
            except Exception as exc:
                last_error = exc
                logger.warning("OpenRouter tagging failed on attempt %s/%s: %s", attempt, API_RETRY_ATTEMPTS, exc)
                if attempt < API_RETRY_ATTEMPTS:
                    await asyncio.sleep(API_RETRY_BASE_DELAY * attempt)

    logger.error("Falling back to default tags after OpenRouter failures: %s", last_error)
    return normalize_tag_payload(DEFAULT_TAGS, message_date)


async def download_image_to_tempfile(message: Any) -> Optional[str]:
    suffix = ".jpg"
    file = getattr(message, "file", None)
    if file and getattr(file, "ext", None):
        suffix = file.ext if str(file.ext).startswith(".") else f".{file.ext}"

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.close()
    try:
        result = await message.download_media(file=handle.name)
        if not result:
            os.unlink(handle.name)
            return None
        return handle.name
    except Exception:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise


def split_text_for_telegram(text: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    current = []
    current_length = 0
    for paragraph in text.split("\n\n"):
        paragraph_piece = paragraph if not current else f"\n\n{paragraph}"
        if current_length + len(paragraph_piece) <= max_length:
            current.append(paragraph)
            current_length += len(paragraph_piece)
            continue

        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0

        if len(paragraph) <= max_length:
            current.append(paragraph)
            current_length = len(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start : start + max_length])
            start += max_length

    if current:
        chunks.append("\n\n".join(current))

    return chunks


async def send_text_chunks(client: TelegramClient, text: str) -> None:
    chunks = split_text_for_telegram(text, MAX_TEXT_LENGTH)
    for chunk in chunks:
        await client.send_message(DEST_CHANNEL_ENTITY, chunk)


async def post_to_destination(client: TelegramClient, source_message: Any, caption: str) -> None:
    file_path: Optional[str] = None
    try:
        if is_image_message(source_message):
            file_path = await download_image_to_tempfile(source_message)
            if not file_path:
                logger.warning("Image download returned nothing for message %s", source_message.id)
                await send_text_chunks(client, caption)
                return

            if len(caption) <= MAX_CAPTION_LENGTH:
                await client.send_file(
                    DEST_CHANNEL_ENTITY,
                    file_path,
                    caption=caption,
                    force_document=True,
                )
            else:
                prefix = caption[:MAX_CAPTION_LENGTH]
                await client.send_file(
                    DEST_CHANNEL_ENTITY,
                    file_path,
                    caption=prefix,
                    force_document=True,
                )
                remainder = caption[MAX_CAPTION_LENGTH:].strip()
                if remainder:
                    await send_text_chunks(client, remainder)
        else:
            await send_text_chunks(client, caption)
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                logger.debug("Failed to remove temp file: %s", file_path)


async def process_message(client: TelegramClient, event: events.NewMessage.Event) -> None:
    message = event.message
    raw_text = extract_source_text(message)
    has_media = bool(message.media)

    if not raw_text and not has_media:
        logger.info("Skipping empty message %s", message.id)
        return

    cache_key = build_message_key(event)
    if await processed_messages.acquire(cache_key):
        logger.info("Skipping duplicate message %s", message.id)
        return

    try:
        message_date = message.date
        if message_date.tzinfo is None:
            message_date = message_date.replace(tzinfo=timezone.utc)

        async with tagging_semaphore:
            tags = await call_openrouter(raw_text or "", message_date)

        outgoing_text = assemble_post_text(raw_text, tags)
        await post_to_destination(client, message, outgoing_text)
        logger.info("Reposted message %s with tags", message.id)
    except Exception:
        await processed_messages.release(cache_key)
        raise


async def healthcheck_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        data = await reader.read(1024)
        request_line = data.splitlines()[0].decode("utf-8", errors="ignore") if data else ""
        path = "/"
        parts = request_line.split(" ")
        if len(parts) >= 2:
            path = parts[1]

        if path == "/health":
            body = "ok"
            status = "HTTP/1.1 200 OK"
        else:
            body = "telegram-intel-bot"
            status = "HTTP/1.1 200 OK"

        response = (
            f"{status}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            "Connection: close\r\n"
            "\r\n"
            f"{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
    except Exception:
        logger.exception("Healthcheck handler error")
    finally:
        writer.close()
        await writer.wait_closed()


async def run_healthcheck_server() -> None:
    server = await asyncio.start_server(healthcheck_handler, HEALTHCHECK_HOST, HEALTHCHECK_PORT)
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    logger.info("Healthcheck server listening on %s", sockets)
    async with server:
        await server.serve_forever()


async def main() -> None:
    if BOT_TOKEN:
        client = TelegramClient(StringSession(), TELEGRAM_API_ID_INT, TELEGRAM_API_HASH)
    else:
        client = TelegramClient(StringSession(TELEGRAM_SESSION), TELEGRAM_API_ID_INT, TELEGRAM_API_HASH)

    async def log_raw_update(update: Any) -> None:
        logger.info("RAW UPDATE: %s", update)

    client.add_event_handler(log_raw_update, events.Raw())

    @client.on(events.NewMessage(incoming=True))
    async def handler(event: events.NewMessage.Event) -> None:
        try:
            if event.chat_id is None:
                return

            if SOURCE_CHANNEL_IDS or SOURCE_CHANNEL_USERNAMES:
                if event.chat_id in SOURCE_CHANNEL_IDS:
                    await process_message(client, event)
                    return

                chat = await event.get_chat()
                username = (getattr(chat, "username", None) or "").lower()
                if username not in SOURCE_CHANNEL_USERNAMES:
                    return

            await process_message(client, event)
        except Exception:
            logger.exception("Failed to process message %s", event.message.id)

    logger.info("Starting Telegram client")
    if BOT_TOKEN:
        await client.start(bot_token=BOT_TOKEN)
    else:
        await client.start()
    logger.info(
        "Bot is running. Source=%s Destination=%s | source_ids=%s | source_usernames=%s",
        SOURCE_CHANNEL,
        DEST_CHANNEL,
        sorted(SOURCE_CHANNEL_IDS),
        sorted(SOURCE_CHANNEL_USERNAMES),
    )
    health_task = asyncio.create_task(run_healthcheck_server())
    try:
        await client.run_until_disconnected()
    finally:
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
