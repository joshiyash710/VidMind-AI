"""
Run this BEFORE anything else to diagnose what's failing.
Usage: python debug_transcript.py <video_id_or_url>
"""
import sys
import re

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "dQw4w9WgXcQ"  # default test

# Strip URL to ID
m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', VIDEO_ID)
if m:
    VIDEO_ID = m.group(1)

print(f"\n{'='*60}")
print(f"  Diagnosing transcript fetch for: {VIDEO_ID}")
print(f"{'='*60}\n")

# ── Check 1: Package versions ──
print("── Package Versions ──")
try:
    import youtube_transcript_api as yta
    print(f"  youtube-transcript-api : {getattr(yta, '__version__', 'unknown')}")
except ImportError:
    print("  youtube-transcript-api : NOT INSTALLED ❌")

try:
    import yt_dlp
    print(f"  yt-dlp                 : {yt_dlp.version.__version__}")
except ImportError:
    print("  yt-dlp                 : NOT INSTALLED ❌")

# ── Check 2: youtube-transcript-api (new API) ──
print("\n── Method 1a: youtube-transcript-api NEW style (v0.7+) ──")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api    = YouTubeTranscriptApi()
    result = list(api.fetch(VIDEO_ID, languages=["en", "en-US"]))
    print(f"  ✅ Success: {len(result)} items")
    if result:
        first = result[0]
        print(f"  First item type : {type(first)}")
        print(f"  First item repr : {repr(first)[:120]}")
        # Check attribute access
        for attr in ["text", "start", "duration"]:
            val = getattr(first, attr, "MISSING")
            print(f"  .{attr} = {repr(val)[:60]}")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")

print("\n── Method 1b: youtube-transcript-api OLD style (v0.6) ──")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    result = list(YouTubeTranscriptApi.get_transcript(VIDEO_ID, languages=["en"]))
    print(f"  ✅ Success: {len(result)} items")
    if result:
        print(f"  First item: {result[0]}")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")

print("\n── Method 1c: list_transcripts ──")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        api = YouTubeTranscriptApi()
        tl  = list(api.list(VIDEO_ID))
    except Exception:
        tl  = list(YouTubeTranscriptApi.list_transcripts(VIDEO_ID))
    
    print(f"  Found {len(tl)} transcript(s):")
    for t in tl:
        lang = getattr(t, "language_code", "?")
        gen  = getattr(t, "is_generated", "?")
        name = getattr(t, "language", "?")
        print(f"    lang={lang}, generated={gen}, name={name}")
        try:
            raw = list(t.fetch())
            print(f"    → fetch() returned {len(raw)} items ✅")
            if raw:
                item = raw[0]
                print(f"      item type: {type(item)}")
                for attr in ["text", "start", "duration"]:
                    print(f"      .{attr} = {repr(getattr(item, attr, 'MISSING'))[:60]}")
        except Exception as fe:
            print(f"    → fetch() failed: {fe}")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")

# ── Check 3: yt-dlp ──
print("\n── Method 2: yt-dlp ──")
try:
    import yt_dlp
    url     = f"https://www.youtube.com/watch?v={VIDEO_ID}"
    ydl_opts = {
        "skip_download": True,
        "quiet":         True,
        "no_warnings":   True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    
    manual = info.get("subtitles", {}) or {}
    auto   = info.get("automatic_captions", {}) or {}
    print(f"  Manual subtitles  : {list(manual.keys())[:10]}")
    print(f"  Auto captions     : {list(auto.keys())[:10]}")
    
    # Try to get English
    for src_name, src_dict in [("manual", manual), ("auto", auto)]:
        for lang in ["en", "en-US", "en-GB"]:
            if src_dict.get(lang):
                formats = src_dict[lang]
                print(f"  ✅ Found {src_name}/{lang}: {[f.get('ext') for f in formats]}")
                break
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")

# ── Check 4: Network ──
print("\n── Network Check ──")
import urllib.request
try:
    r = urllib.request.urlopen("https://www.youtube.com", timeout=5)
    print(f"  YouTube reachable: ✅ (status {r.status})")
except Exception as e:
    print(f"  YouTube unreachable: ❌ {e}")

print(f"\n{'='*60}")
print("  Diagnosis complete")
print(f"{'='*60}\n")
# paste at bottom of debug_transcript.py and rerun
print("\n── Final normalization test ──")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api  = YouTubeTranscriptApi()
    tl   = list(api.list("3_TN1i3MTEU"))
    raw  = list(tl[0].fetch())          # Hindi transcript
    item = raw[0]

    # Simulate _item_to_dict
    result = {
        "text":  str(getattr(item, "text",  "")).strip(),
        "start": float(getattr(item, "start", 0.0)),
    }
    print(f"  ✅ Normalized: {result}")
except Exception as e:
    print(f"  ❌ {e}")