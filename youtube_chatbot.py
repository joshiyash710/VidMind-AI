# ================== IMPORTS ==================
import os
import re
import json
import random
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ================== ENV ==================
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "vidmind-ai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"

# ================== LLM ==================
llm         = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7,  api_key=OPENAI_API_KEY)
llm_precise = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2,  api_key=OPENAI_API_KEY)

# ================== UTILS ==================
def get_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    try:
        parsed = urlparse(url)
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            vid = parsed.path.lstrip('/').split('/')[0].split('?')[0]
            if re.match(r'^[A-Za-z0-9_-]{11}$', vid):
                return vid
        if parsed.hostname and 'youtube.com' in parsed.hostname:
            qs = parse_qs(parsed.query)
            if 'v' in qs and qs['v'][0]:
                vid = qs['v'][0]
                if re.match(r'^[A-Za-z0-9_-]{11}$', vid):
                    return vid
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) >= 2:
                prefix = path_parts[0]
                vid    = path_parts[1]
                if prefix in ('embed', 'v', 'shorts', 'live', 'e'):
                    if re.match(r'^[A-Za-z0-9_-]{11}$', vid):
                        return vid
    except Exception:
        pass
    match = re.search(
        r'(?:v=|vi=|v\/|youtu\.be\/|embed\/|shorts\/|live\/)([A-Za-z0-9_-]{11})', url
    )
    if match:
        return match.group(1)
    return None


def _format_time(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ================== TRANSCRIPT ITEM NORMALIZER ==================
def _item_to_dict(item) -> Optional[dict]:
    """
    Convert ANY transcript item type to {"text": str, "start": float}.
    Handles every known format from all three fetcher methods.
    """
    text  = None
    start = None

    # ── dict (yt-dlp / innertube / old youtube-transcript-api) ──
    if isinstance(item, dict):
        text  = item.get("text", "")
        start = item.get("start", 0.0)

    else:
        # ── Try attribute access first (FetchedTranscriptSnippet v0.7+) ──
        try:
            text  = item.text
            start = item.start
        except AttributeError:
            pass

        # ── Try subscript access (legacy namedtuple) ──
        if text is None:
            try:
                text  = item["text"]
                start = item["start"]
            except (KeyError, TypeError, IndexError):
                pass

        # ── Last resort: repr parsing ──
        if text is None:
            try:
                s = str(item)
                m = re.search(r"text['\"]?\s*[:=]\s*['\"]([^'\"]+)", s)
                if m:
                    text = m.group(1)
                m2 = re.search(r"start['\"]?\s*[:=]\s*([\d.]+)", s)
                if m2:
                    start = float(m2.group(1))
            except Exception:
                pass

    if text is None:
        return None

    text  = str(text).strip()
    start = float(start) if start is not None else 0.0

    if not text:
        return None

    return {"text": text, "start": start}


def _build_chunks(items: list, video_id: str) -> list:
    """
    Group normalized transcript items into ~30-second chunks.
    Each item must be a plain dict with 'text' and 'start' keys.
    """
    chunks     = []
    buf        = []
    chunk_start = 0.0

    for item in items:
        if not buf:
            chunk_start = item["start"]
        buf.append(item["text"])

        if item["start"] - chunk_start >= 30:
            chunks.append({
                "start_timestamp": int(chunk_start),
                "timestamp":       _format_time(int(chunk_start)),
                "text":            " ".join(buf),
                "video_id":        video_id,
            })
            buf = []

    if buf:
        chunks.append({
            "start_timestamp": int(chunk_start),
            "timestamp":       _format_time(int(chunk_start)),
            "text":            " ".join(buf),
            "video_id":        video_id,
        })

    return chunks


# ================== SUBTITLE PARSERS ==================
def _parse_subtitle_content(content: str, ext: str) -> list:
    stripped = content.strip()
    if not stripped:
        return []

    # Auto-detect format
    if not ext or ext not in ("json3", "srv1", "srv2", "srv3", "vtt", "ttml", "srt"):
        if stripped.startswith("{") or stripped.startswith("["):
            ext = "json3"
        elif "WEBVTT" in stripped[:50]:
            ext = "vtt"
        elif re.search(r'<transcript|<text ', stripped[:500]):
            ext = "srv1"
        elif re.search(r'<tt |xmlns', stripped[:300]):
            ext = "ttml"
        elif re.match(r'^\d+\s*\n', stripped):
            ext = "srt"
        else:
            ext = "vtt"

    parsers = {
        "json3": [_parse_json3, _parse_vtt, _parse_srv],
        "srv1":  [_parse_srv, _parse_json3, _parse_vtt],
        "srv2":  [_parse_srv, _parse_json3, _parse_vtt],
        "srv3":  [_parse_srv, _parse_json3, _parse_vtt],
        "vtt":   [_parse_vtt, _parse_json3, _parse_srv],
        "ttml":  [_parse_ttml, _parse_vtt],
        "srt":   [_parse_srt, _parse_vtt],
    }

    for parser in parsers.get(ext, [_parse_json3, _parse_vtt, _parse_srv]):
        try:
            items = parser(content)
            if items:
                return items
        except Exception:
            continue
    return []


def _parse_json3(content: str) -> list:
    items = []
    try:
        data   = json.loads(content)
        events = data.get("events", [])
        for event in events:
            start_ms = event.get("tStartMs", 0)
            segs     = event.get("segs", [])
            if not segs:
                continue
            text = "".join(s.get("utf8", "") for s in segs if isinstance(s, dict))
            text = text.replace("\n", " ").strip()
            if text:
                items.append({"text": text, "start": start_ms / 1000.0})
    except Exception:
        pass
    return items


def _parse_srv(content: str) -> list:
    import xml.etree.ElementTree as ET
    items = []

    def _try_parse(xml_str):
        root   = ET.fromstring(xml_str)
        result = []
        for elem in root.iter("text"):
            raw = (elem.text or "")
            raw = (raw.replace("&amp;", "&").replace("&lt;",  "<")
                      .replace("&gt;",  ">").replace("&quot;", '"')
                      .replace("&#39;", "'").strip())
            text  = raw.replace("\n", " ").strip()
            start = float(elem.get("start", 0))
            if text:
                result.append({"text": text, "start": start})
        return result

    try:
        items = _try_parse(content)
    except ET.ParseError:
        try:
            clean = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', content)
            items = _try_parse(clean)
        except Exception:
            pass
    return items


def _parse_vtt(content: str) -> list:
    items  = []
    lines  = content.splitlines()
    i      = 0

    # Skip to WEBVTT header
    while i < len(lines) and "WEBVTT" not in lines[i]:
        i += 1
    i += 1

    TS_LONG  = re.compile(r'(\d+):(\d{2}):(\d{2})[.,](\d+)\s*-->')
    TS_SHORT = re.compile(r'^(\d+):(\d{2})[.,](\d+)\s*-->')
    TAG_RE   = re.compile(r'<[^>]+>')

    def _parse_long(line):
        m = TS_LONG.match(line)
        if m:
            h, mn, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            return h * 3600 + mn * 60 + s + int(ms) / (10 ** len(ms))
        return None

    def _parse_short(line):
        m = TS_SHORT.match(line)
        if m:
            mn, s, ms = int(m.group(1)), int(m.group(2)), m.group(3)
            return mn * 60 + s + int(ms) / (10 ** len(ms))
        return None

    seen_texts = set()  # deduplicate rolling captions

    while i < len(lines):
        line  = lines[i].strip()
        start = _parse_long(line) or _parse_short(line)
        if start is not None:
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                raw = TAG_RE.sub("", lines[i]).strip()
                if raw:
                    text_lines.append(raw)
                i += 1
            text = " ".join(text_lines).strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                items.append({"text": text, "start": start})
        else:
            i += 1

    return items


def _parse_srt(content: str) -> list:
    items  = []
    blocks = re.split(r'\n\s*\n', content.strip())
    TS_RE  = re.compile(r'(\d+):(\d{2}):(\d{2})[,.](\d+)\s*-->')
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        ts_line    = None
        text_lines = []
        for line in lines:
            m = TS_RE.match(line.strip())
            if m and ts_line is None:
                h, mn, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
                ts_line = h * 3600 + mn * 60 + s + int(ms) / (10 ** len(ms))
            elif ts_line is not None and line.strip() and not line.strip().isdigit():
                text_lines.append(re.sub(r'<[^>]+>', '', line).strip())
        if ts_line is not None and text_lines:
            text = " ".join(text_lines).strip()
            if text:
                items.append({"text": text, "start": ts_line})
    return items


def _parse_ttml(content: str) -> list:
    import xml.etree.ElementTree as ET
    items = []
    content_clean = re.sub(r'\sxmlns(?::\w+)?="[^"]*"', '', content)
    content_clean = re.sub(r'<\?[^?]+\?>', '', content_clean)
    try:
        root = ET.fromstring(content_clean)
    except ET.ParseError:
        try:
            content_clean = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', content_clean)
            root = ET.fromstring(content_clean)
        except Exception:
            return items

    def _t(t: str) -> float:
        if not t:
            return 0.0
        parts = t.strip().replace(',', '.').split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            else:
                return float(t.rstrip('s'))
        except Exception:
            return 0.0

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'p':
            start = _t(elem.get('begin', ''))
            parts = []
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            for child in elem:
                if child.text and child.text.strip():
                    parts.append(child.text.strip())
                if child.tail and child.tail.strip():
                    parts.append(child.tail.strip())
            text = " ".join(parts).strip()
            if text:
                items.append({"text": text, "start": start})
    return items


# ================== METHOD 1: youtube-transcript-api ==================
def _fetch_via_youtube_transcript_api(video_id: str) -> tuple:
    """
    Exhaustively tries every known API variant across all versions.
    Returns (raw_items, lang) or (None, None).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("[yta] Not installed")
        return None, None

    # ── Detect what's available on this class ──
    has_get_transcript   = hasattr(YouTubeTranscriptApi, 'get_transcript')
    has_list_transcripts = hasattr(YouTubeTranscriptApi, 'list_transcripts')
    can_instantiate      = False
    instance             = None

    try:
        instance        = YouTubeTranscriptApi()
        can_instantiate = True
        print(f"[yta] v0.7+ instantiated. Methods: "
              f"{[m for m in dir(instance) if not m.startswith('_')]}")
    except Exception as e:
        print(f"[yta] Cannot instantiate: {e}")

    # ────────────────────────────────────────────
    # ATTEMPT 1: Instance-based fetch (v0.7+)
    # ────────────────────────────────────────────
    if can_instantiate and instance is not None:
        has_fetch = hasattr(instance, 'fetch')
        has_list  = hasattr(instance, 'list')

        if has_fetch:
            for lang_codes in [["en", "en-US", "en-GB", "en-IN"], None]:
                try:
                    if lang_codes:
                        try:
                            raw = instance.fetch(video_id, languages=lang_codes)
                        except TypeError:
                            # API doesn't accept languages kwarg
                            raw = instance.fetch(video_id)
                        except Exception as inner_e:
                            # Any other fetch error — try without lang filter
                            print(f"[yta] inner fetch error ({lang_codes}): "
                                  f"{type(inner_e).__name__}: {inner_e}")
                            try:
                                raw = instance.fetch(video_id)
                            except Exception as fallback_e:
                                print(f"[yta] fallback fetch error: "
                                      f"{type(fallback_e).__name__}: {fallback_e}")
                                continue
                    else:
                        raw = instance.fetch(video_id)

                    if raw is None:
                        print(f"[yta] instance.fetch() returned None for {lang_codes}")
                        continue

                    raw_list = list(raw)
                    if raw_list:
                        lang = lang_codes[0] if lang_codes else "en"
                        print(f"[yta] ✅ instance.fetch() → {len(raw_list)} items "
                              f"| type={type(raw_list[0])}")
                        return raw_list, lang

                except Exception as e:
                    name = type(e).__name__
                    print(f"[yta] instance.fetch({lang_codes}): {name}: {str(e)[:120]}")
                    if any(x in name for x in ["TranscriptsDisabled", "VideoUnavailable"]):
                        return None, None

        if has_list:
            try:
                transcript_list = instance.list(video_id)
                all_t = list(transcript_list)
                print(f"[yta] instance.list() → {len(all_t)} transcript(s)")

                if all_t:
                    def _rank(t):
                        lc = getattr(t, 'language_code', '') or ''
                        ig = getattr(t, 'is_generated', True)
                        return (0 if lc.startswith('en') else 1, 0 if not ig else 1)

                    all_t.sort(key=_rank)
                    chosen = all_t[0]
                    lc     = getattr(chosen, 'language_code', 'en') or 'en'

                    for fetch_method in [
                        lambda t: list(t.fetch()),
                        lambda t: list(t),
                    ]:
                        try:
                            raw_list = fetch_method(chosen)
                            if raw_list:
                                print(f"[yta] ✅ instance.list().fetch() → "
                                      f"{len(raw_list)} | lang={lc}")
                                return raw_list, lc
                        except Exception as e2:
                            print(f"[yta] fetch method failed: {e2}")

            except Exception as e:
                print(f"[yta] instance.list(): {type(e).__name__}: {str(e)[:150]}")

    # ────────────────────────────────────────────
    # ATTEMPT 2: Class-level list_transcripts
    # ────────────────────────────────────────────
    if has_list_transcripts:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            all_t = list(transcript_list)
            print(f"[yta] list_transcripts() → {len(all_t)} transcript(s)")

            if all_t:
                def _rank(t):
                    lc = getattr(t, 'language_code', '') or ''
                    ig = getattr(t, 'is_generated', True)
                    return (0 if lc.startswith('en') else 1, 0 if not ig else 1)

                all_t.sort(key=_rank)
                chosen = all_t[0]
                lc     = getattr(chosen, 'language_code', 'en') or 'en'

                for fetch_method in [
                    lambda t: list(t.fetch()),
                    lambda t: list(t),
                ]:
                    try:
                        raw_list = fetch_method(chosen)
                        if raw_list:
                            print(f"[yta] ✅ list_transcripts().fetch() → "
                                  f"{len(raw_list)} | lang={lc}")
                            return raw_list, lc
                    except Exception as e2:
                        print(f"[yta] fetch method: {e2}")

        except AttributeError:
            print("[yta] list_transcripts not available")
        except Exception as e:
            name = type(e).__name__
            print(f"[yta] list_transcripts(): {name}: {str(e)[:150]}")
            if any(x in name for x in ["TranscriptsDisabled", "VideoUnavailable"]):
                return None, None

    # ────────────────────────────────────────────
    # ATTEMPT 3: Class-level get_transcript
    # ────────────────────────────────────────────
    if has_get_transcript:
        for lang_codes in [["en", "en-US", "en-GB", "en-IN"], None]:
            try:
                if lang_codes:
                    raw = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_codes)
                else:
                    raw = YouTubeTranscriptApi.get_transcript(video_id)

                raw_list = list(raw)
                if raw_list:
                    lang = lang_codes[0] if lang_codes else "en"
                    print(f"[yta] ✅ get_transcript() → {len(raw_list)} | lang={lang}")
                    return raw_list, lang
            except Exception as e:
                name = type(e).__name__
                print(f"[yta] get_transcript({lang_codes}): {name}: {str(e)[:120]}")
                if any(x in name for x in ["TranscriptsDisabled", "VideoUnavailable"]):
                    return None, None

    print("[yta] All attempts exhausted")
    return None, None


# ================== METHOD 2: yt-dlp ==================
def _fetch_via_ytdlp(video_id: str) -> tuple:
    """Download subtitle URL and parse it."""
    try:
        import yt_dlp
    except ImportError:
        print("[ytdlp] Not installed")
        return None, None

    import urllib.request

    url = f"https://www.youtube.com/watch?v={video_id}"
    ua  = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    ydl_opts = {
        "skip_download":     True,
        "writesubtitles":    True,
        "writeautomaticsub": True,
        "quiet":             True,
        "no_warnings":       True,
        "socket_timeout":    30,
        "extractor_args":    {"youtube": {"skip": ["hls", "dash"]}},
        "http_headers": {
            "User-Agent":      ua,
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if not info:
                print(f"[ytdlp] No info returned (attempt {attempt + 1})")
                time.sleep(3 * (2 ** attempt))
                continue

            manual_subs   = info.get("subtitles",          {}) or {}
            auto_captions = info.get("automatic_captions", {}) or {}

            print(f"[ytdlp] Manual : {list(manual_subs.keys())[:8]}")
            print(f"[ytdlp] Auto   : {list(auto_captions.keys())[:12]}")

            # Build priority candidate list
            candidates = []

            for lang in ["en", "en-US", "en-GB", "en-IN"]:
                if manual_subs.get(lang):
                    candidates.append((lang, manual_subs[lang], False))

            for lang, subs in manual_subs.items():
                if subs and not lang.startswith("zxx"):
                    if not any(c[0] == lang for c in candidates):
                        candidates.append((lang, subs, False))

            for lang in ["en", "en-US", "en-GB", "en-orig"]:
                if auto_captions.get(lang):
                    candidates.append((lang, auto_captions[lang], True))

            for lang, subs in auto_captions.items():
                if subs and not lang.startswith("zxx") and "-orig" not in lang:
                    if not any(c[0] == lang for c in candidates):
                        candidates.append((lang, subs, True))

            if not candidates:
                print("[ytdlp] No subtitle candidates")
                time.sleep(3 * (2 ** attempt))
                continue

            print(f"[ytdlp] {len(candidates)} candidate(s)")

            candidate_succeeded = False

            for chosen_lang, chosen_subs, is_auto in candidates:
                # Pick best format
                chosen_format = None
                for fmt in ["json3", "srv3", "srv2", "srv1", "vtt", "ttml", "srt"]:
                    for entry in chosen_subs:
                        if isinstance(entry, dict) and entry.get("ext") == fmt:
                            chosen_format = entry
                            break
                    if chosen_format:
                        break

                if not chosen_format and chosen_subs:
                    first = chosen_subs[0]
                    chosen_format = first if isinstance(first, dict) else None

                if not chosen_format or not chosen_format.get("url"):
                    continue

                sub_url = chosen_format["url"]
                ext     = chosen_format.get("ext", "json3")

                try:
                    req      = urllib.request.Request(
                        sub_url,
                        headers={
                            "User-Agent":      ua,
                            "Accept-Language": "en-US,en;q=0.9",
                        }
                    )
                    response = urllib.request.urlopen(req, timeout=20)
                    content  = response.read().decode("utf-8", errors="replace")

                    if not content.strip():
                        continue

                    items = _parse_subtitle_content(content, ext)
                    if items:
                        print(f"[ytdlp] ✅ {len(items)} items | "
                              f"lang={chosen_lang} | ext={ext}")
                        return items, chosen_lang

                    candidate_succeeded = True  # downloaded but parse returned empty

                except Exception as e:
                    print(f"[ytdlp] Download error ({chosen_lang}): "
                          f"{type(e).__name__}: {e}")
                    continue

            # All candidates tried — retry the whole attempt if worth it
            if not candidate_succeeded:
                print(f"[ytdlp] All candidates failed (attempt {attempt + 1}), retrying...")
                time.sleep(3 * (2 ** attempt))
                continue

        except Exception as e:
            err = str(e)
            print(f"[ytdlp] Attempt {attempt + 1}: {type(e).__name__}: {err[:150]}")
            if "429" in err or "rate" in err.lower():
                wait = 10 * (2 ** attempt)
                print(f"[ytdlp] Rate limit — waiting {wait}s")
                time.sleep(wait)
            elif attempt < 2:
                time.sleep(3 * (2 ** attempt))

    print("[ytdlp] All attempts exhausted")
    return None, None


# ================== METHOD 3: InnerTube / Direct Scrape ==================
def _fetch_via_innertube(video_id: str) -> tuple:
    """Scrape YouTube page for captionTracks and download directly."""
    import urllib.request

    print(f"[innertube] Starting for {video_id}")

    page_url = f"https://www.youtube.com/watch?v={video_id}"
    headers  = {
        "User-Agent":      (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }

    for attempt in range(3):
        try:
            req      = urllib.request.Request(page_url, headers=headers)
            response = urllib.request.urlopen(req, timeout=25)

            # Handle gzip
            raw_bytes = response.read()
            try:
                import gzip
                html = gzip.decompress(raw_bytes).decode("utf-8", errors="replace")
            except Exception:
                html = raw_bytes.decode("utf-8", errors="replace")

            if len(html) < 5000:
                print(f"[innertube] HTML too short ({len(html)}) — likely bot detection")
                time.sleep(3 * (attempt + 1))
                continue

            caption_tracks = []

            # ── Pattern 1: captionTracks JSON array ──
            for pattern in [
                r'"captionTracks":(\[.*?\])',
                r'"captionTracks"\s*:\s*(\[.*?\])',
            ]:
                m = re.search(pattern, html)
                if m:
                    try:
                        tracks = json.loads(m.group(1))
                        for track in tracks:
                            bu = track.get("baseUrl", "")
                            if bu:
                                caption_tracks.append({
                                    "url":    bu,
                                    "lang":   track.get("languageCode", "en"),
                                    "is_asr": track.get("kind", "") == "asr",
                                    "name":   track.get("name", {}).get("simpleText", ""),
                                })
                        if caption_tracks:
                            break
                    except Exception as e:
                        print(f"[innertube] Pattern parse error: {e}")

            # ── Pattern 2: ytInitialPlayerResponse ──
            if not caption_tracks:
                m2 = re.search(r'ytInitialPlayerResponse\s*=\s*(\{)', html)
                if m2:
                    start_idx         = m2.start(1)
                    depth, end_idx    = 0, start_idx
                    for i, ch in enumerate(
                        html[start_idx:start_idx + 500000], start_idx
                    ):
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                    try:
                        player_data = json.loads(html[start_idx:end_idx])
                        tracks = (
                            player_data
                            .get("captions", {})
                            .get("playerCaptionsTracklistRenderer", {})
                            .get("captionTracks", [])
                        )
                        for track in tracks:
                            bu = track.get("baseUrl", "")
                            if bu:
                                caption_tracks.append({
                                    "url":    bu,
                                    "lang":   track.get("languageCode", "en"),
                                    "is_asr": track.get("kind", "") == "asr",
                                    "name":   track.get("name", {}).get("simpleText", ""),
                                })
                    except Exception as e:
                        print(f"[innertube] Player response parse: {e}")

            # ── Pattern 3: Direct baseUrl in HTML ──
            if not caption_tracks:
                for m in re.finditer(
                    r'"baseUrl"\s*:\s*"(https://www\.youtube\.com/api/timedtext[^"]+)"',
                    html
                ):
                    url_raw = m.group(1).replace("\\u0026", "&").replace("\\/", "/")
                    lang_m  = re.search(r'[?&]lang=([^&]+)', url_raw)
                    lang    = lang_m.group(1) if lang_m else "en"
                    caption_tracks.append({
                        "url":    url_raw,
                        "lang":   lang,
                        "is_asr": "kind=asr" in url_raw,
                        "name":   "",
                    })
                # Deduplicate
                seen_urls      = set()
                caption_tracks = [
                    t for t in caption_tracks
                    if t["url"] not in seen_urls and not seen_urls.add(t["url"])
                ]

            if not caption_tracks:
                print(f"[innertube] No tracks found (attempt {attempt + 1})")
                time.sleep(2)
                continue

            print(f"[innertube] {len(caption_tracks)} track(s): "
                  f"{[(t['lang'], t['is_asr']) for t in caption_tracks[:5]]}")

            # Sort: prefer non-ASR English first
            caption_tracks.sort(
                key=lambda t: (t["is_asr"], 0 if t["lang"].startswith("en") else 1)
            )

            for track in caption_tracks:
                for fmt in ["json3", "xml", "vtt"]:
                    try:
                        track_url = track["url"]

                        # Inject/replace format param
                        if "fmt=" in track_url:
                            track_url = re.sub(r'fmt=[^&]*', f'fmt={fmt}', track_url)
                        else:
                            sep       = "&" if "?" in track_url else "?"
                            track_url = track_url + sep + f"fmt={fmt}"

                        print(f"[innertube] Fetching lang={track['lang']} fmt={fmt}")
                        req_obj  = urllib.request.Request(
                            track_url,
                            headers={
                                "User-Agent":      headers["User-Agent"],
                                "Accept-Language": "en-US,en;q=0.9",
                            }
                        )
                        resp  = urllib.request.urlopen(req_obj, timeout=20)
                        raw_b = resp.read()
                        try:
                            import gzip
                            content = gzip.decompress(raw_b).decode(
                                "utf-8", errors="replace"
                            )
                        except Exception:
                            content = raw_b.decode("utf-8", errors="replace")

                        if not content.strip() or len(content) < 30:
                            continue

                        ext_map = {"json3": "json3", "xml": "srv1", "vtt": "vtt"}
                        items   = _parse_subtitle_content(
                            content, ext_map.get(fmt, fmt)
                        )

                        if items:
                            print(f"[innertube] ✅ {len(items)} items | "
                                  f"lang={track['lang']} | fmt={fmt}")
                            return items, track["lang"]

                    except Exception as e:
                        print(f"[innertube] Track {track['lang']}/{fmt}: "
                              f"{type(e).__name__}: {e}")
                        continue

            time.sleep(2)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (2 ** attempt) + random.uniform(0, 2)
                print(f"[innertube] 429 — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"[innertube] HTTP {e.code}: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"[innertube] Attempt {attempt + 1}: {type(e).__name__}: {e}")
            time.sleep(2)

    print("[innertube] All attempts failed")
    return None, None


# ================== MAIN TRANSCRIPT FETCHER ==================
def get_transcript_with_timestamps(video_id: str):
    """
    Try 3 methods in sequence. Each returns (raw_items, lang) or (None, None).
    After success, normalize all items to {"text": str, "start": float}.
    Returns (plain_text, chunks, raw_items, detected_lang) or (None, [], [], None).
    """
    print(f"\n{'=' * 60}")
    print(f"[Transcript] Fetching: {video_id}")
    print(f"{'=' * 60}")

    raw_items:     Optional[list] = None
    detected_lang: str            = "en"

    methods = [
        ("youtube-transcript-api", _fetch_via_youtube_transcript_api),
        ("yt-dlp",                 _fetch_via_ytdlp),
        ("innertube",              _fetch_via_innertube),
    ]

    for method_name, method_fn in methods:
        print(f"\n[Transcript] ── Trying: {method_name}")
        try:
            result_items, result_lang = method_fn(video_id)
            if result_items:
                raw_items     = result_items
                detected_lang = result_lang or "en"
                print(f"[Transcript] ✅ {method_name} succeeded → "
                      f"{len(raw_items)} raw items")
                break
        except Exception as e:
            print(f"[Transcript] {method_name} CRASHED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not raw_items:
        print(f"\n[Transcript] ❌ All 3 methods failed for {video_id}")
        return None, [], [], None

    # ── Normalize all items to plain dicts ──
    normalized = []
    for i, item in enumerate(raw_items):
        try:
            d = _item_to_dict(item)
            if d:
                normalized.append(d)
        except Exception as e:
            print(f"[Transcript] Normalization error item {i}: "
                  f"{type(e).__name__}: {e}")
            continue

    if not normalized:
        print(f"[Transcript] ❌ 0 items after normalization "
              f"(raw had {len(raw_items)})")
        if raw_items:
            print(f"[Transcript] First raw item type: {type(raw_items[0])}")
            print(f"[Transcript] First raw item repr: {repr(raw_items[0])[:200]}")
        return None, [], [], None

    plain_text = " ".join(i["text"] for i in normalized)
    chunks     = _build_chunks(normalized, video_id)

    print(
        f"\n[Transcript] ✅ Done: {len(plain_text):,} chars | "
        f"{len(normalized)} items | {len(chunks)} chunks | lang={detected_lang}"
    )
    # Return normalized (plain dicts) as raw_items so callers never see raw objects
    return plain_text, chunks, normalized, detected_lang


# ================== MEMORY ==================
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages = []

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()


store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# ================== SUMMARY ==================
def generate_summary(transcript: str) -> dict:
    # Sample from beginning, middle, and end for better coverage
    t_len  = len(transcript)
    sample = (
        transcript[:3000]
        + "\n...\n"
        + transcript[t_len // 2: t_len // 2 + 2000]
        + "\n...\n"
        + transcript[-1000:]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content analyst and learning strategist.

Analyze the transcript and return ONLY valid JSON with no markdown fences:
{{
  "title": "A concise, descriptive title for the video",
  "summary": "A 4-sentence educational summary with main ideas and examples",
  "why_it_matters": "A 1-2 sentence explanation of real-world relevance",
  "key_concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],
  "difficulty": "Beginner or Intermediate or Advanced",
  "study_time_minutes": 15
}}

Return ONLY the JSON object. No extra text. No markdown fences."""),
        ("human", "Transcript:\n{transcript}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke({"transcript": sample})
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        parsed.setdefault("why_it_matters",
                          "Understanding this content builds essential knowledge.")
        parsed.setdefault(
            "study_time_minutes",
            max(10, min(60, len(transcript.split()) // 150))
        )
        return parsed
    except Exception:
        return {
            "title":              "Video Summary",
            "summary":            raw[:300] if raw else "Summary unavailable.",
            "why_it_matters":     "This content is relevant for exam preparation.",
            "key_concepts":       [],
            "difficulty":         "Intermediate",
            "study_time_minutes": 15,
        }


# ================== MULTI-VIDEO MERGER ==================
def merge_multi_video_summary(summaries: List[dict]) -> dict:
    if not summaries:
        return {
            "title":              "Multi-Video Session",
            "summary":            "",
            "why_it_matters":     "",
            "key_concepts":       [],
            "difficulty":         "Intermediate",
            "study_time_minutes": 0,
        }
    if len(summaries) == 1:
        return summaries[0]

    titles   = [s.get("title", "") for s in summaries]
    concepts = list(
        dict.fromkeys(c for s in summaries for c in s.get("key_concepts", []))
    )[:8]
    diffs    = [s.get("difficulty", "Intermediate") for s in summaries]
    overall  = (
        "Advanced" if "Advanced" in diffs
        else ("Beginner" if all(d == "Beginner" for d in diffs) else "Intermediate")
    )
    total_t  = sum(s.get("study_time_minutes", 15) for s in summaries)

    return {
        "title": (
            f"Combined: {' + '.join(titles[:2])}"
            + (" + more" if len(titles) > 2 else "")
        ),
        "summary":            " ".join(
            s.get("summary", "") for s in summaries[:3]
        )[:450],
        "why_it_matters":     "Multi-video learning provides comprehensive understanding.",
        "key_concepts":       concepts,
        "difficulty":         overall,
        "study_time_minutes": total_t,
    }


# ================== QUIZ ==================
def generate_quiz(
    transcript: str,
    section:    Optional[str] = None,
    topic:      Optional[str] = None,
) -> list:
    topic_line   = f"Exam Topic: {topic.strip()}\n"   if topic   else ""
    section_line = f"Focus Area: {section.strip()}\n" if section else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert exam paper setter.
Generate 5 high-quality MCQs based on the transcript content.
Mix recall, understanding, and application questions. Make wrong options plausible.

Return ONLY a valid JSON array — no markdown fences:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "Why this is correct."
  }}
]"""),
        ("human", f"{topic_line}{section_line}Transcript:\n{{transcript}}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke(
        {"transcript": transcript[:10000]}
    )
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        if isinstance(result, dict):
            return result.get("questions", [])
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ================== NOTES ==================
def generate_notes(transcript: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic tutor. Generate structured exam-ready study notes.

Structure:
# [Topic Title]
## 1. Introduction & Context
## 2. Key Concepts
- **Definition**: ...
- **How it works**: ...
- **Example**: ...
## Summary
## Key Takeaways

Be thorough and educational."""),
        ("human", "Transcript:\n{transcript}")
    ])

    result = (prompt | llm | StrOutputParser()).invoke(
        {"transcript": transcript[:10000]}
    )
    return result if result and result.strip() else "Notes could not be generated."


# ================== FLASHCARDS ==================
def generate_flashcards(transcript: str) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Create exactly 8 flashcards from the transcript.
Return ONLY valid JSON array — no markdown fences:
[{{"front": "Concept or question", "back": "Clear explanation or answer"}}]"""),
        ("human", "Transcript:\n{transcript}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke(
        {"transcript": transcript[:6000]}
    )
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ================== EXAM PLAN ==================
def generate_exam_plan(transcript: str, topic: str, hours: float) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert exam strategist.
Create an optimized study plan.

Return ONLY valid JSON — no markdown fences:
{{
  "total_minutes": 120,
  "sections": [
    {{
      "title": "Topic Name",
      "duration_mins": 20,
      "concepts": ["concept1", "concept2"],
      "checkpoint_question": "Quick verification question"
    }}
  ],
  "quick_tips": ["tip1", "tip2", "tip3"]
}}"""),
        ("human", "Topic: {topic}\nHours Available: {hours}\n\nTranscript:\n{transcript}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke({
        "transcript": transcript[:8000],
        "topic":      topic,
        "hours":      str(hours),
    })
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        data.setdefault("total_minutes", int(hours * 60))
        return data
    except Exception:
        return {"total_minutes": int(hours * 60), "sections": [], "quick_tips": []}


# ================== IMPORTANT QUESTIONS ==================
def generate_important_questions(transcript: str, topic: str) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate EXACTLY 8 important exam questions with detailed answers.
Return ONLY valid JSON array — no markdown fences:
[
  {{
    "question": "Exam-style question",
    "answer": "Detailed answer (3-5 sentences with examples)",
    "marks": 5,
    "type": "Conceptual or Application or Problem-Solving"
  }}
]"""),
        ("human", "Topic: {topic}\n\nTranscript:\n{transcript}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke({
        "transcript": transcript[:10000],
        "topic":      topic,
    })
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ================== REVISION NOTES ==================
def generate_revision_notes(transcript: str, topic: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Create LAST-MINUTE REVISION NOTES — concise and high-yield.
Return ONLY valid JSON — no markdown fences:
{{
  "must_remember": ["critical point 1", "critical point 2", "critical point 3",
                    "critical point 4", "critical point 5"],
  "key_formulas": ["formula or rule 1", "formula or rule 2", "formula or rule 3"],
  "common_mistakes": ["mistake 1", "mistake 2", "mistake 3"],
  "quick_tips": ["tip 1", "tip 2", "tip 3"]
}}"""),
        ("human", "Topic: {topic}\n\nTranscript:\n{transcript}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke({
        "transcript": transcript[:10000],
        "topic":      topic,
    })
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result
    except Exception:
        return {
            "must_remember":  [],
            "key_formulas":   [],
            "common_mistakes": [],
            "quick_tips":     [],
        }


# ================== CONFUSION DETECTOR ==================
def detect_confusion(question: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rate student confusion 1-5.
1=clear, 3=somewhat confused, 5=very confused.

Return ONLY valid JSON — no markdown fences:
{{
  "score": 3,
  "reason": "brief explanation",
  "confused_topic": "specific topic name or empty string"
}}"""),
        ("human", "Student question: {question}")
    ])

    raw = (prompt | llm_precise | StrOutputParser()).invoke({"question": question})
    try:
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        result.setdefault("confused_topic", "")
        return result
    except Exception:
        return {"score": 1, "reason": "could not assess", "confused_topic": ""}


# ================== HYBRID RAG ==================
class _RetrieverWithStore:
    def __init__(self, retriever, vectorstore):
        self._retriever  = retriever
        self.vectorstore = vectorstore

    def invoke(self, query):
        return self._retriever.invoke(query)

    def __getattr__(self, name):
        return getattr(self._retriever, name)


def answer_with_hybrid_rag(
    question:   str,
    retriever,
    session_id: str,
    confused:   bool = False,
) -> dict:
    use_world_knowledge = True
    context_docs        = []

    try:
        vs              = retriever.vectorstore
        docs_and_scores = vs.similarity_search_with_score(question, k=4)
        if docs_and_scores:
            best_l2 = min(score for _, score in docs_and_scores)
            if best_l2 < 1.8:
                use_world_knowledge = False
                context_docs = [doc for doc, _ in docs_and_scores]
    except Exception:
        try:
            context_docs        = retriever.invoke(question)
            use_world_knowledge = len(context_docs) == 0
        except Exception:
            use_world_knowledge = True

    if use_world_knowledge:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful tutor. Answer using general knowledge."),
            ("human", "{question}")
        ])
        answer = (prompt | llm | StrOutputParser()).invoke({"question": question})
        return {
            "answer":       answer,
            "mode":         "world_knowledge",
            "source_label": "⚠️ General knowledge answer",
        }

    context        = "\n\n".join(doc.page_content for doc in context_docs)
    confusion_note = (
        "\n\nIMPORTANT: Student seems confused. "
        "Start with a simple analogy then give the technical explanation."
        if confused else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are VidMind AI. Answer using ONLY the transcript context. "
         f"Be educational and clear.{confusion_note}"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer":       answer,
        "mode":         "video_grounded",
        "source_label": "📹 Answered from video",
    }


# ================== RAG CHAIN BUILDER ==================
def build_rag_chain(transcript: str, chunks: list, video_label: str = "Video 1"):
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    docs        = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever      = _RetrieverWithStore(base_retriever, vectorstore)

    ts_retriever = None
    if chunks:
        ts_docs = splitter.create_documents(
            [c["text"] for c in chunks],
            metadatas=[{**c, "source_label": video_label} for c in chunks],
        )
        ts_store     = FAISS.from_documents(ts_docs, embeddings)
        ts_retriever = ts_store.as_retriever(search_kwargs={"k": 3})

    def get_timestamps(question: str) -> list:
        if not ts_retriever:
            return []
        try:
            results   = ts_retriever.invoke(question)
            seen, out = set(), []
            for doc in results:
                ts    = doc.metadata.get("timestamp", "")
                start = doc.metadata.get("start_timestamp", 0)
                vid   = doc.metadata.get("video_id", "")
                label = doc.metadata.get("source_label", video_label)
                if ts and ts not in seen:
                    seen.add(ts)
                    out.append({
                        "timestamp": ts,
                        "time":      start,
                        "label":     f"{label} @ {ts}",
                        "video_id":  vid,
                    })
            return out
        except Exception:
            return []

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are VidMind AI. Answer based on the transcript context. "
         "Be educational, clear, and helpful.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnableParallel({
            "context":  lambda x: "\n\n".join(
                d.page_content for d in retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, get_timestamps, retriever


# ================== CROSS-VIDEO STORE ==================
cross_video_store: Dict[str, Dict] = {}


def add_video_to_session(session_id, video_id, transcript, chunks, label):
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    docs        = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)

    ts_vectorstore = None
    if chunks:
        ts_docs = splitter.create_documents(
            [c["text"] for c in chunks],
            metadatas=[{**c, "source_label": label} for c in chunks],
        )
        ts_vectorstore = FAISS.from_documents(ts_docs, embeddings)

    if session_id not in cross_video_store:
        cross_video_store[session_id] = {}

    cross_video_store[session_id][video_id] = {
        "vectorstore":    vectorstore,
        "ts_vectorstore": ts_vectorstore,
        "label":          label,
        "video_id":       video_id,
    }


def answer_cross_video(session_id: str, question: str) -> dict:
    if session_id not in cross_video_store or not cross_video_store[session_id]:
        return {"answer": "No videos loaded for cross-video Q&A.", "sources": []}

    all_docs = []
    for vid_id, entry in cross_video_store[session_id].items():
        try:
            results = entry["vectorstore"].similarity_search(question, k=2)
            for doc in results:
                doc.metadata["source_label"] = entry["label"]
                doc.metadata["video_id"]     = vid_id
            all_docs.extend(results)
        except Exception:
            continue

    if not all_docs:
        return {"answer": "Could not retrieve relevant content.", "sources": []}

    context = "\n\n".join(
        f"[{doc.metadata.get('source_label', 'Video')}]: {doc.page_content}"
        for doc in all_docs
    )

    sources, seen_ts = [], set()
    for vid_id, entry in cross_video_store[session_id].items():
        if entry.get("ts_vectorstore"):
            try:
                for doc in entry["ts_vectorstore"].similarity_search(question, k=2):
                    ts = doc.metadata.get("timestamp", "")
                    if ts and (vid_id, ts) not in seen_ts:
                        seen_ts.add((vid_id, ts))
                        sources.append({
                            "label":     entry["label"],
                            "timestamp": ts,
                            "time":      doc.metadata.get("start_timestamp", 0),
                            "video_id":  vid_id,
                        })
            except Exception:
                continue

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert tutor synthesizing information from multiple video "
         "transcripts. Cite sources using 'According to [Video Label]...' format. "
         "Be educational."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"answer": answer, "sources": sources}


def remove_video_session(session_id: str) -> None:
    """Clean up cross_video_store for a session (call on session delete)."""
    cross_video_store.pop(session_id, None)


# ================== PDF EXPORT ==================
def generate_pdf_from_notes(notes_text: str, title: str = "VidMind AI Notes") -> bytes:
    try:
        from fpdf import FPDF

        class NotesPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(120, 120, 120)
                self.cell(0, 8, "VidMind AI - Study Notes", align="R")
                self.ln(2)
                self.set_draw_color(220, 220, 220)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(4)

            def footer(self):
                self.set_y(-14)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(160, 160, 160)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        def safe(t):
            t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
            t = re.sub(r'\*(.+?)\*',     r'\1', t)
            t = re.sub(r'`(.+?)`',       r'\1', t)
            for a, b in [
                ('\u2019', "'"), ('\u2018', "'"),
                ('\u201c', '"'), ('\u201d', '"'),
                ('\u2014', '-'), ('\u2013', '-'),
            ]:
                t = t.replace(a, b)
            return t

        pdf = NotesPDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()
        pdf.set_margins(15, 15, 15)

        for line in notes_text.split('\n'):
            s = line.strip()
            if not s:
                pdf.ln(2)
                continue
            if s.startswith('# ') and not s.startswith('## '):
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(30, 58, 95)
                pdf.multi_cell(0, 9, safe(s[2:]))
                pdf.ln(2)
            elif s.startswith('## '):
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(46, 117, 182)
                pdf.ln(4)
                pdf.multi_cell(0, 8, safe(s[3:]))
                pdf.ln(2)
            elif s.startswith('### '):
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(50, 50, 80)
                pdf.ln(2)
                pdf.multi_cell(0, 7, safe(s[4:]))
            elif s.startswith('- '):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 6, f"  * {safe(s[2:])}")
            else:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 6, safe(s))

        return bytes(pdf.output())

    except ImportError:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from io import BytesIO

        buf    = BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story  = []
        for line in notes_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), styles['Normal']))
                story.append(Spacer(1, 4))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()


# ================== CHAT HISTORY EXPORT ==================
def export_chat_history(session_id: str) -> str:
    history = get_session_history(session_id)
    if not history.messages:
        return "No conversation history found."
    lines = ["=== VidMind AI Chat Export ===", f"Session: {session_id}", ""]
    for msg in history.messages:
        role = "You" if msg.type == "human" else "VidMind AI"
        lines += [f"[{role}]", str(msg.content), ""]
    return "\n".join(lines)