"""
build_haw_kb_from_pdf.py

Reads the HAW_Hamburg_Online_Services.xlsx, extracts all URLs from the Excel file,
fetches page content, detects language, and builds:
- haw_kb.json (original)
- haw_kb_en.json (English version)
- haw_kb_de.json (German version, all texts translated)

Run:
    python build_haw_kb_from_excel.py
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
import langid
from deep_translator import GoogleTranslator  # ✅ simpler and synchronous


EXCEL_PATH = "data/HAW_Hamburg_Online_Services.xlsx"
OUTPUT_JSON_ALL = "data/haw_kb.json"
OUTPUT_JSON_EN = "data/haw_kb_en.json"
OUTPUT_JSON_DE = "data/haw_kb_de.json"


# ------------------- URL Handling -------------------
def extract_urls_from_excel(excel_path):
    """Extract all http(s) URLs from the Excel file."""
    df = pd.read_excel(excel_path)
    urls = []
    for _, row in df.iterrows():
        label = str(row.get("Label", "")).strip()
        url = str(row.get("URL", "")).strip()
        if url:
            urls.append((label, url))
    return urls


def is_valid_url(url):
    """Check if URL has a valid scheme and netloc."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


# ------------------- Fetching -------------------
def fetch_page_text(url):
    """Fetch a web page and return cleaned visible text, including
    contact details extracted from mailto: and tel: links."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return None, f"Error fetching {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract contact details from links before stripping HTML
    contact_lines = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(strip=True)
        if href.startswith("mailto:"):
            email = href.replace("mailto:", "").split("?")[0].strip()
            contact_lines.append(f"Email: {email} ({label})" if label and label != email else f"Email: {email}")
        elif href.startswith("tel:"):
            phone = href.replace("tel:", "").strip()
            contact_lines.append(f"Phone: {phone} ({label})" if label and label != phone else f"Phone: {phone}")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    # Append extracted contact details at the end of the page text
    if contact_lines:
        text += " CONTACT DETAILS: " + " | ".join(contact_lines)

    return text, None


def split_text_into_chunks(text, chunk_size=500):
    """Split text into chunks of approximately `chunk_size` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def detect_language(text):
    """Detects whether a text is English or German."""
    if not text.strip():
        return "unknown"
    lang, confidence = langid.classify(text)
    return lang


def translate_text(text, target_lang):
    """Translate text to target language (en or de) using deep_translator."""
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated
    except Exception as e:
        return f"[Translation failed: {e}] {text}"


# ------------------- Build KB -------------------
def build_kb(excel_path, output_path_all, output_path_en, output_path_de):
    raw_data = extract_urls_from_excel(excel_path)
    urls = list(dict.fromkeys([u for _, u in raw_data]))  # de-duplicate while keeping order

    kb = []
    for label, url in raw_data:
        if not is_valid_url(url):
            print(f"Skipping invalid URL: {url}")
            kb.append({
                "url": url,
                "title": label,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": "invalid",
                "chunk_id": None,
                "text": "",
                "language": "unknown",
                "error": "Invalid URL"
            })
            continue

        print(f"Fetching {url} ...")
        text, error = fetch_page_text(url)

        if text:
            chunks = split_text_into_chunks(text, chunk_size=500)
            for idx, chunk in enumerate(chunks):
                # Prepend the page label so retrieval finds the right chunk
                # even when user words differ from page vocabulary (e.g. "programs" vs "degree courses")
                labeled_chunk = f"{label}: {chunk}"
                lang = detect_language(labeled_chunk)
                entry = {
                    "url": url,
                    "title": label,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                    "chunk_id": idx,
                    "text": labeled_chunk,
                    "language": lang,
                    "error": "",
                }
                kb.append(entry)
        else:
            entry = {
                "url": url,
                "title": label,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "chunk_id": None,
                "text": "",
                "language": "unknown",
                "error": error if error else "",
            }
            kb.append(entry)

    # Save combined version
    with open(output_path_all, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    # Build translated English and German versions
    kb_en = []
    kb_de = []

    print("\nTranslating all texts to English and German...")

    for entry in kb:
        text = entry["text"]

        # English version (translate everything to English)
        translated_en = translate_text(text, "en")

        # German version (translate everything to German)
        translated_de = translate_text(text, "de")

        en_entry = dict(entry)
        en_entry["text"] = translated_en
        en_entry["language"] = "en"
        kb_en.append(en_entry)

        de_entry = dict(entry)
        de_entry["text"] = translated_de
        de_entry["language"] = "de"
        kb_de.append(de_entry)

    with open(output_path_en, "w", encoding="utf-8") as f:
        json.dump(kb_en, f, ensure_ascii=False, indent=2)
    with open(output_path_de, "w", encoding="utf-8") as f:
        json.dump(kb_de, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Knowledge bases saved:")
    print(f"  - Original: {output_path_all} ({len(kb)} entries)")
    print(f"  - English:  {output_path_en} ({len(kb_en)} entries)")
    print(f"  - German:   {output_path_de} ({len(kb_de)} entries)")


if __name__ == "__main__":
    build_kb(EXCEL_PATH, OUTPUT_JSON_ALL, OUTPUT_JSON_EN, OUTPUT_JSON_DE)