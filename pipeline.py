#!/usr/bin/env python3
"""
Processing pipeline for parts manual digitization.

Steps:
1. PDF -> Images (PyMuPDF at 300 DPI)
2. Classify pages (diagram / table / other)
3. Pair sections (diagram + table pages)
4. YOLO detection on diagram pages
5. OCR callout numbers
6. Extract parts tables
7. Link callouts to parts
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Optional imports — gracefully degrade
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("ultralytics not available — YOLO detection disabled")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not available")


PART_NUMBER_RE = re.compile(r'\d{3}-\d{4}-\d{2}')
PART_NUMBER_SPACED_RE = re.compile(
    r'(\d)\s+(\d)\s+(\d)\s*-\s*(\d)\s+(\d)\s+(\d)\s+(\d)\s*-\s*(\d)\s+(\d)'
)


class ProcessingState:
    """Thread-safe processing state tracker."""

    def __init__(self):
        self.step = ""
        self.step_number = 0
        self.total_steps = 7
        self.progress = 0.0
        self.message = ""
        self.done = False
        self.error: Optional[str] = None

    def update(self, step_number: int, step: str, progress: float = 0.0,
               message: str = ""):
        self.step_number = step_number
        self.step = step
        self.progress = progress
        self.message = message

    def to_dict(self):
        return {
            "step": self.step,
            "step_number": self.step_number,
            "total_steps": self.total_steps,
            "progress": round(self.progress, 1),
            "message": self.message,
            "done": self.done,
            "error": self.error,
        }


# ── Text helpers for spaced-out PDF text ──────────────────────────────

def _compact_line(line: str) -> str:
    """Compact a spaced-out line: 'P A R T S  S E C T I O N' -> 'PARTS SECTION'.

    Splits on double-spaces (word boundaries), then collapses segments where
    most tokens are 1-2 chars (character-spaced text).
    """
    segments = re.split(r'  +', line)
    words = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        tokens = seg.split(' ')
        short_count = sum(1 for t in tokens if len(t) <= 2 and t)
        if len(tokens) > 1 and short_count / max(len(tokens), 1) > 0.5:
            words.append(''.join(tokens))
        else:
            words.append(seg)
    return ' '.join(words)


def _collapse_spaces(text: str) -> str:
    """Collapse spaced-out text line by line."""
    return '\n'.join(_compact_line(line) for line in text.split('\n'))


def _text_compacted_upper(text: str) -> str:
    """Remove ALL spaces and uppercase — for keyword detection."""
    return text.upper().replace(' ', '')


# ── Page classification ───────────────────────────────────────────────

def _is_table_text(text: str) -> bool:
    """Check if text contains parts-table markers (handles spaced-out text)."""
    compacted = _text_compacted_upper(text)
    has_item = 'ITEM' in compacted
    has_part = 'PART' in compacted
    has_list = 'PARTSLIST' in compacted
    has_part_nums = bool(PART_NUMBER_RE.search(text.replace(' ', '')))
    return (has_item and has_part) or has_list or (has_item and has_part_nums)


def _is_parts_section_header(text: str) -> bool:
    """Check if text contains 'PARTS SECTION' header."""
    return 'PARTSSECTION' in _text_compacted_upper(text)


def _is_cover_or_toc(page, page_num: int, total_pages: int) -> bool:
    """Detect cover, TOC, or back pages."""
    text = page.get_text()
    compacted = _text_compacted_upper(text)

    # First or second page with very little text = cover
    if page_num <= 2 and len(text.strip()) < 100:
        return True

    # Table of contents
    if 'TABLEOFCONTENTS' in compacted or 'CONTENTS' in compacted:
        return True

    # Last page(s) with very little text = back cover
    if page_num >= total_pages - 1 and len(text.strip()) < 100:
        return True

    return False


def is_diagram_page(page) -> bool:
    """Detect if a page contains an exploded diagram (not a table page)."""
    text = page.get_text()
    # Table pages have lots of text with ITEM/PART headers
    if _is_table_text(text) and len(text.strip()) > 500:
        return False
    # "PARTS SECTION" header with short text = diagram page
    if _is_parts_section_header(text):
        return True
    # Short text with images/drawings = diagram page
    image_list = page.get_images()
    drawings = page.get_drawings()
    if len(text.strip()) < 500 and (len(image_list) > 0 or len(drawings) > 50):
        return True
    return False


def is_table_page(page) -> bool:
    """Detect if a page contains a parts table."""
    text = page.get_text()
    return _is_table_text(text) and len(text.strip()) > 500


def extract_section_title(page) -> Optional[str]:
    """Extract section title from page text."""
    text = page.get_text()
    lines = text.split('\n')
    compacted_lines = [_compact_line(l.strip()) for l in lines if l.strip()]
    full_compacted = '\n'.join(compacted_lines)

    # Look for "PARTS SECTION: <title>"
    m = re.search(r'PARTS\s*SECTION[:\s]*(?:PAGE\s*\d+\s*)?(.+)',
                  full_compacted, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        title = re.sub(r'\s*PAGE\s*\d+\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^\d+\s*', '', title)
        title = re.sub(r'\s+\d+\s*$', '', title)
        if title and len(title) > 2:
            return title

    # Look for known section keywords
    keywords = ['DECK', 'ENGINE', 'FRAME', 'HYDRAULIC', 'ELECTRIC',
                'STEERING', 'SEAT', 'SUSPENSION', 'WHEEL', 'SPINDLE',
                'DRIVE', 'PODIUM', 'PUMP', 'FRONT END', 'HEIGHT',
                'BATTERY', 'MOTOR', 'SUB DECK', 'CHUTE', 'FENDER',
                'PLATFORM', 'TIRE', 'VANGUARD', 'KAWASAKI', 'DECAL',
                'INDUSTRIAL']
    for cl in compacted_lines:
        if any(kw in cl.upper() for kw in keywords):
            title = re.sub(r'^\d+\s*', '', cl)
            title = re.sub(r'\s+\d+\s*$', '', title)
            title = re.sub(r'\s*PAGE\s*\d+\s*', '', title, flags=re.IGNORECASE)
            if title and len(title) > 2:
                return title.strip()
    return None


# ── Step 1: PDF -> Images ────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300,
                  state: Optional[ProcessingState] = None) -> list[dict]:
    """Convert all PDF pages to PNG images. Returns page metadata list."""
    if state:
        state.update(1, "Converting PDF to images", 0, "Opening PDF...")

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    total = len(doc)

    for i in range(total):
        page = doc[i]
        filename = f"page_{i + 1:03d}.png"
        pix = page.get_pixmap(matrix=matrix)
        pix.save(str(pages_dir / filename))
        pages.append({
            "page_number": i + 1,
            "filename": filename,
            "width": pix.width,
            "height": pix.height,
        })
        if state:
            pct = ((i + 1) / total) * 100
            state.update(1, "Converting PDF to images", pct,
                         f"Page {i + 1}/{total}")

    doc.close()
    return pages


# ── Step 2: Classify Pages ───────────────────────────────────────────

def classify_pages(pdf_path: Path, pages: list[dict],
                   state: Optional[ProcessingState] = None) -> list[dict]:
    """Classify each page as diagram, table, or other."""
    if state:
        state.update(2, "Classifying pages", 0, "Analyzing page content...")

    doc = fitz.open(str(pdf_path))
    total = len(doc)

    for i in range(total):
        page = doc[i]
        page_num = i + 1

        if _is_cover_or_toc(page, page_num, total):
            pages[i]["type"] = "other"
        elif is_table_page(page):
            pages[i]["type"] = "table"
        elif is_diagram_page(page):
            pages[i]["type"] = "diagram"
        else:
            pages[i]["type"] = "other"

        title = extract_section_title(page)
        if title:
            pages[i]["section_title"] = title

        if state:
            pct = ((i + 1) / total) * 100
            state.update(2, "Classifying pages", pct,
                         f"Page {page_num}/{total}: {pages[i]['type']}")

    doc.close()
    return pages


# ── Step 3: Pair Sections ────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    return re.sub(r'[^a-z0-9]', '', title.lower())


def pair_sections(pages: list[dict],
                  state: Optional[ProcessingState] = None) -> list[dict]:
    """Group diagram + table pages into sections.

    Handles multiple consecutive diagram pages that belong to the same section
    (e.g., Podium pages 15-16-17 share table page 18).
    """
    if state:
        state.update(3, "Pairing sections", 0, "Grouping diagram and table pages...")

    sections = []
    i = 0
    total = len(pages)

    while i < total:
        p = pages[i]
        if p["type"] == "diagram":
            # Collect consecutive diagram pages with similar titles
            diagram_pages = [p]
            j = i + 1
            while j < total and pages[j]["type"] == "diagram":
                diagram_pages.append(pages[j])
                j += 1

            # Collect consecutive table pages after the diagram group
            table_page_nums = []
            while j < total and pages[j]["type"] == "table":
                table_page_nums.append(pages[j]["page_number"])
                j += 1

            # Determine section title from any page in the group
            title = None
            for dp in diagram_pages:
                if "section_title" in dp:
                    title = dp["section_title"]
                    break
            # Also check table pages for title
            if not title:
                for tp_num in table_page_nums:
                    tp = pages[tp_num - 1]
                    if "section_title" in tp:
                        title = tp["section_title"]
                        break

            if not title:
                title = f"Section {len(sections) + 1}"

            # Create one section per diagram page, sharing the same table
            for dp in diagram_pages:
                section = {
                    "diagram_page": dp["page_number"],
                    "diagram_image": dp["filename"],
                    "image_width": dp["width"],
                    "image_height": dp["height"],
                    "table_pages": table_page_nums,
                    "title": dp.get("section_title", title),
                }
                sections.append(section)

            i = j
        else:
            i += 1

    if state:
        state.update(3, "Pairing sections", 100,
                     f"Found {len(sections)} sections")

    return sections


# ── Step 4: YOLO Detection ───────────────────────────────────────────

def detect_callouts(sections: list[dict], pages_dir: Path,
                    model_path: str = "/models/best.pt",
                    conf: float = 0.3,
                    state: Optional[ProcessingState] = None) -> list[dict]:
    """Run YOLO detection on diagram pages."""
    if state:
        state.update(4, "Detecting callouts (YOLO)", 0, "Loading model...")

    if not HAS_YOLO:
        if state:
            state.update(4, "Detecting callouts (YOLO)", 100,
                         "YOLO not available — skipped")
        for sec in sections:
            sec["raw_detections"] = []
        return sections

    model_file = Path(model_path)
    if not model_file.exists():
        if state:
            state.update(4, "Detecting callouts (YOLO)", 100,
                         f"Model not found at {model_path} — skipped")
        for sec in sections:
            sec["raw_detections"] = []
        return sections

    model = YOLO(str(model_file))
    total = len(sections)

    for idx, sec in enumerate(sections):
        img_path = pages_dir / sec["diagram_image"]
        results = model(str(img_path), conf=conf, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                    "confidence": float(box.conf[0]),
                })

        sec["raw_detections"] = detections

        if state:
            pct = ((idx + 1) / total) * 100
            state.update(4, "Detecting callouts (YOLO)", pct,
                         f"Section {idx + 1}/{total}: {len(detections)} callouts")

    return sections


# ── Step 5: Read Callout Numbers ─────────────────────────────────────
#
# Primary method: extract digit text positions from the PDF text layer
# and match them spatially to YOLO-detected callout bounding boxes.
# This is ~100% accurate since we read source data instead of doing
# image OCR on small, low-res circle crops.
#
# Fallback: tesseract OCR on the rendered image crop (for PDFs without
# a text layer).

def _extract_text_numbers(pdf_path: Path, page_num: int,
                          dpi: int = 300) -> list[dict]:
    """Extract digit groups with pixel positions from a PDF page.

    PDF text layers often have characters individually positioned
    (e.g., "32" as "3" at x=100 and "2" at x=112).  We group adjacent
    single digits on the same line into multi-digit numbers.

    Returns list of {"value": int, "cx": float, "cy": float,
                      "x0","y0","x1","y1": float} in pixel coordinates.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    scale = dpi / 72  # PDF points -> pixels

    # Get every word with its bounding box
    words = page.get_text('words')
    # Each word: (x0, y0, x1, y1, text, block_no, line_no, word_no)

    # Collect digit-only words, convert to pixel coords
    digit_words = []
    for w in words:
        text = w[4].strip()
        if re.match(r'^\d{1,2}$', text):
            digit_words.append({
                'text': text,
                'x0': w[0] * scale,
                'y0': w[1] * scale,
                'x1': w[2] * scale,
                'y1': w[3] * scale,
            })

    # Sort by vertical then horizontal position
    digit_words.sort(key=lambda d: (round(d['y0'], -1), d['x0']))

    # Group adjacent digits that sit on the same line into numbers.
    # Two digits of the same callout number are typically <25px apart
    # horizontally and <10px apart vertically (at 300 DPI).
    groups = []
    used = set()
    for i, dw in enumerate(digit_words):
        if i in used:
            continue
        group = [dw]
        used.add(i)
        # Walk forward looking for the next digit on the same line
        for j in range(i + 1, len(digit_words)):
            if j in used:
                continue
            dw2 = digit_words[j]
            last = group[-1]
            cy_last = (last['y0'] + last['y1']) / 2
            cy2 = (dw2['y0'] + dw2['y1']) / 2
            x_gap = dw2['x0'] - last['x1']
            if abs(cy2 - cy_last) < 10 and 0 < x_gap < 25:
                group.append(dw2)
                used.add(j)
            elif dw2['y0'] > last['y1'] + 10:
                # Past this line, stop looking
                break

        number_str = ''.join(d['text'] for d in group)
        if re.match(r'^\d{1,2}$', number_str):
            val = int(number_str)
            if 1 <= val <= 99:
                gx0 = min(d['x0'] for d in group)
                gy0 = min(d['y0'] for d in group)
                gx1 = max(d['x1'] for d in group)
                gy1 = max(d['y1'] for d in group)
                groups.append({
                    'value': val,
                    'cx': (gx0 + gx1) / 2,
                    'cy': (gy0 + gy1) / 2,
                    'x0': gx0, 'y0': gy0,
                    'x1': gx1, 'y1': gy1,
                })

    doc.close()
    return groups


def _match_detections_to_text(detections: list[dict],
                              text_numbers: list[dict],
                              pad: float = 15) -> list[dict]:
    """Match YOLO bounding boxes to the nearest PDF text number.

    For each detection, finds the text number whose center falls inside
    (or within `pad` pixels of) the detection box and is closest to
    the box center.
    """
    results = []
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        bcx = x + w / 2
        bcy = y + h / 2

        best = None
        best_dist = float('inf')
        for tn in text_numbers:
            # Text center must be inside the expanded detection box
            if (x - pad <= tn['cx'] <= x + w + pad and
                    y - pad <= tn['cy'] <= y + h + pad):
                dist = ((tn['cx'] - bcx) ** 2 + (tn['cy'] - bcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = tn

        results.append({
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "confidence": det["confidence"],
            "ocr_confidence": 0.99 if best else 0.0,
            "item_number": best['value'] if best else None,
            "source": "pdf_text" if best else None,
        })

    return results


def _extract_digits(text: str) -> Optional[str]:
    """Extract a valid item number (1-99) from OCR text."""
    digits = re.sub(r'\D', '', text)
    if digits:
        try:
            val = int(digits)
            if 1 <= val <= 99:
                return str(val)
        except ValueError:
            pass
    return None


def _ocr_crop_tesseract(full_img: np.ndarray,
                        x: int, y: int, w: int, h: int) -> tuple:
    """Fallback: OCR a callout crop with tesseract.  Returns (digit_str, conf)."""
    if not HAS_TESSERACT:
        return None, 0.0

    # Inset to center 70%
    inset = 0.15
    ix1 = max(0, int(x + w * inset))
    iy1 = max(0, int(y + h * inset))
    ix2 = min(full_img.shape[1], int(x + w * (1 - inset)))
    iy2 = min(full_img.shape[0], int(y + h * (1 - inset)))

    crop = full_img[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None, 0.0

    ch, cw = crop.shape[:2]
    scale = max(1, 256 / min(ch, cw))
    big = cv2.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)

    if len(big.shape) == 3:
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    else:
        gray = big

    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fixed_hi = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, fixed_lo = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)

    variants = [otsu, cv2.bitwise_not(otsu), fixed_hi, fixed_lo,
                adaptive, cv2.bitwise_not(adaptive), gray]

    border = 30
    for img_var in variants:
        padded = cv2.copyMakeBorder(img_var, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=255)
        for psm in [7, 8, 10, 13]:
            try:
                text = pytesseract.image_to_string(
                    padded,
                    config=f'--psm {psm} -c tessedit_char_whitelist=0123456789'
                ).strip()
                digits = _extract_digits(text)
                if digits:
                    return digits, 0.80
            except Exception:
                continue

    return None, 0.0


def ocr_callouts(sections: list[dict], pages_dir: Path,
                 pdf_path: Optional[Path] = None,
                 state: Optional[ProcessingState] = None) -> tuple:
    """Read the callout number for each detected bounding box.

    Primary: match YOLO boxes to digit positions from the PDF text layer
    (fast, ~100% accurate).
    Fallback: tesseract OCR on the rendered image crop.
    """
    if state:
        state.update(5, "Reading callout numbers", 0,
                     "Extracting text positions from PDF...")

    total_callouts = sum(len(s.get("raw_detections", [])) for s in sections)
    processed = 0
    warnings = []

    for sec_idx, sec in enumerate(sections):
        detections = sec.get("raw_detections", [])
        if not detections:
            sec["callouts_raw"] = []
            continue

        page_num = sec["diagram_page"]

        # ── Primary: PDF text layer matching ──────────────────────────
        text_numbers = []
        if pdf_path and pdf_path.exists():
            try:
                text_numbers = _extract_text_numbers(pdf_path, page_num)
            except Exception as e:
                logger.debug(f"Text extraction failed for page {page_num}: {e}")

        if text_numbers:
            callouts = _match_detections_to_text(detections, text_numbers)
            # Count how many got a match
            text_matched = sum(1 for c in callouts if c["item_number"] is not None)
        else:
            callouts = None
            text_matched = 0

        # ── Fallback: tesseract for any unmatched boxes ───────────────
        if callouts is None or text_matched < len(detections):
            img_path = pages_dir / sec["diagram_image"]
            img = cv2.imread(str(img_path))

            if callouts is None:
                # No text layer at all — full tesseract
                callouts = []
                for det in detections:
                    x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                    digit_str, conf = _ocr_crop_tesseract(img, x, y, w, h) \
                        if img is not None else (None, 0.0)
                    callouts.append({
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "confidence": det["confidence"],
                        "ocr_confidence": conf,
                        "item_number": int(digit_str) if digit_str else None,
                        "source": "tesseract" if digit_str else None,
                    })
            else:
                # Fill gaps: tesseract only for boxes that text matching missed
                if img is not None:
                    for co in callouts:
                        if co["item_number"] is None:
                            b = co["bbox"]
                            digit_str, conf = _ocr_crop_tesseract(
                                img, b["x"], b["y"], b["w"], b["h"])
                            if digit_str:
                                co["item_number"] = int(digit_str)
                                co["ocr_confidence"] = conf
                                co["source"] = "tesseract"

        # Record warnings for any still-unmatched callouts
        for co in callouts:
            if co["item_number"] is None:
                b = co["bbox"]
                warnings.append(
                    f"Page {page_num}, callout at ({b['x']}, {b['y']}): "
                    f"could not read number")

        # Strip the internal 'source' key before storing
        for co in callouts:
            co.pop("source", None)

        sec["callouts_raw"] = callouts
        processed += len(detections)

        if state and total_callouts > 0:
            pct = (processed / total_callouts) * 100
            state.update(5, "Reading callout numbers", pct,
                         f"Section {sec_idx + 1}/{len(sections)}: "
                         f"{text_matched}/{len(detections)} from PDF text")

    if state:
        state.update(5, "Reading callout numbers", 100,
                     f"Done — {len(warnings)} warnings")
    return sections, warnings


# ── Step 6: Extract Parts Tables ─────────────────────────────────────

def _parse_table_rows(table) -> list[dict]:
    """Parse a PyMuPDF table into structured rows."""
    rows = []
    header_found = False
    for row in table.extract():
        cells = [c.strip() if c else "" for c in row]
        if not header_found:
            upper = [c.upper() for c in cells]
            if "ITEM" in upper and ("PART" in ' '.join(upper)):
                header_found = True
                continue
            continue
        if len(cells) < 3:
            continue

        item_val = None
        qty_val = None
        part_num = None
        desc = None

        for cell in cells:
            if item_val is None and re.match(r'^\d{1,2}$', cell):
                item_val = int(cell)
            elif item_val is not None and qty_val is None and re.match(r'^\d+$', cell):
                qty_val = int(cell)
            elif PART_NUMBER_RE.search(cell):
                m = PART_NUMBER_RE.search(cell)
                part_num = m.group(0)
            elif len(cell) > 3 and part_num is not None and desc is None:
                desc = cell

        if item_val is not None and part_num:
            rows.append({
                "item": item_val,
                "quantity": qty_val or 1,
                "part_number": part_num,
                "description": desc or "",
            })

    return rows


def _parse_text_table(text: str) -> list[dict]:
    """Parse parts table from raw text, handling spaced-out characters."""
    collapsed = _collapse_spaces(text)
    rows = []

    # Try structured regex on collapsed text
    pattern = re.compile(
        r'(\d{1,2})\s+(\d+)\s+(\d{3}-\d{4}-\d{2})\s+(.+?)(?:\n|$)'
    )
    for m in pattern.finditer(collapsed):
        rows.append({
            "item": int(m.group(1)),
            "quantity": int(m.group(2)),
            "part_number": m.group(3),
            "description": m.group(4).strip(),
        })

    if rows:
        return rows

    # Line-by-line parsing for PDFs where each field is on its own line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    i = 0
    while i < len(lines) - 2:
        cl = _compact_line(lines[i]).strip()

        if re.match(r'^\d{1,2}$', cl):
            item_num = int(cl)
            qty = None
            part_num = None
            desc = None

            for j in range(i + 1, min(i + 5, len(lines))):
                cl_j = _compact_line(lines[j]).strip()
                if qty is None and re.match(r'^\d{1,3}$', cl_j):
                    qty = int(cl_j)
                elif part_num is None and PART_NUMBER_RE.search(cl_j):
                    part_num = PART_NUMBER_RE.search(cl_j).group(0)
                elif part_num is None:
                    m = PART_NUMBER_SPACED_RE.search(lines[j])
                    if m:
                        groups = m.groups()
                        part_num = (''.join(groups[:3]) + '-' +
                                    ''.join(groups[3:7]) + '-' +
                                    ''.join(groups[7:9]))
                elif part_num is not None and desc is None and len(cl_j) > 3:
                    desc = cl_j

            if part_num:
                rows.append({
                    "item": item_num,
                    "quantity": qty or 1,
                    "part_number": part_num,
                    "description": desc or "",
                })
        i += 1

    return rows


def extract_tables(sections: list[dict], pdf_path: Path,
                   state: Optional[ProcessingState] = None) -> list[dict]:
    """Extract parts tables from table pages."""
    if state:
        state.update(6, "Extracting parts tables", 0, "Opening PDF...")

    doc = fitz.open(str(pdf_path))
    total = len(sections)

    # Cache parsed table pages (multiple sections may share the same tables)
    table_cache: dict[int, list[dict]] = {}

    for idx, sec in enumerate(sections):
        all_rows = []
        for tp in sec.get("table_pages", []):
            if tp in table_cache:
                all_rows.extend(table_cache[tp])
                continue

            page = doc[tp - 1]
            page_rows = []

            # Try PyMuPDF find_tables
            found_structured = False
            try:
                tabs = page.find_tables()
                if tabs and len(tabs.tables) > 0:
                    for t in tabs.tables:
                        rows = _parse_table_rows(t)
                        if rows:
                            page_rows.extend(rows)
                            found_structured = True
            except Exception:
                pass

            # Try splitting at midpoint
            if not found_structured:
                page_rect = page.rect
                mid_x = page_rect.width / 2
                for clip in [
                    fitz.Rect(0, 0, mid_x, page_rect.height),
                    fitz.Rect(mid_x, 0, page_rect.width, page_rect.height),
                ]:
                    try:
                        tabs = page.find_tables(clip=clip)
                        if tabs and len(tabs.tables) > 0:
                            for t in tabs.tables:
                                rows = _parse_table_rows(t)
                                if rows:
                                    page_rows.extend(rows)
                                    found_structured = True
                    except Exception:
                        pass

            # Text-based parsing (handles spaced-out text)
            if not found_structured:
                text = page.get_text()
                page_rows.extend(_parse_text_table(text))

            table_cache[tp] = page_rows
            all_rows.extend(page_rows)

        # Deduplicate by item number
        seen = set()
        unique_rows = []
        for r in all_rows:
            if r["item"] not in seen:
                seen.add(r["item"])
                unique_rows.append(r)

        sec["parts_table"] = sorted(unique_rows, key=lambda r: r["item"])

        if state:
            pct = ((idx + 1) / total) * 100
            state.update(6, "Extracting parts tables", pct,
                         f"Section {idx + 1}/{total}: {len(unique_rows)} parts")

    doc.close()
    return sections


# ── Step 7: Link Callouts -> Parts ───────────────────────────────────

def link_callouts(sections: list[dict],
                  state: Optional[ProcessingState] = None) -> tuple:
    """Match OCR'd callout numbers to parts table entries."""
    if state:
        state.update(7, "Linking callouts to parts", 0, "Building lookups...")

    total_detected = 0
    total_ocr_success = 0
    total_matched = 0
    all_warnings = []

    for idx, sec in enumerate(sections):
        lookup = {r["item"]: r for r in sec.get("parts_table", [])}
        matched_items = set()

        linked_callouts = []
        for co in sec.get("callouts_raw", []):
            total_detected += 1
            item_num = co.get("item_number")

            if item_num is not None:
                total_ocr_success += 1

                if item_num in lookup:
                    part = lookup[item_num]
                    co["part_number"] = part["part_number"]
                    co["description"] = part["description"]
                    co["quantity"] = part["quantity"]
                    matched_items.add(item_num)
                    total_matched += 1
                else:
                    co["part_number"] = None
                    co["description"] = None
                    co["quantity"] = None

            linked_callouts.append(co)

        # Keep all callouts that got an item number
        sec["callouts"] = [c for c in linked_callouts
                           if c.get("item_number") is not None]

        sec["unmatched_callouts"] = [
            c["item_number"] for c in linked_callouts
            if c.get("item_number") is not None and c.get("part_number") is None
        ]
        sec["unmatched_items"] = [
            item for item in lookup if item not in matched_items
        ]

        # Clean up intermediate data
        sec.pop("callouts_raw", None)
        sec.pop("raw_detections", None)

        if state:
            pct = ((idx + 1) / len(sections)) * 100
            state.update(7, "Linking callouts to parts", pct,
                         f"Section {idx + 1}/{len(sections)}")

    processing_log = {
        "total_callouts_detected": total_detected,
        "total_ocr_success": total_ocr_success,
        "total_matched": total_matched,
        "warnings": all_warnings,
    }

    if state:
        state.update(7, "Linking callouts to parts", 100,
                     f"Matched {total_matched}/{total_detected} callouts")

    return sections, processing_log


# ── Full Pipeline ─────────────────────────────────────────────────────

def process_manual(pdf_path: Path, manual_id: str, output_dir: Path,
                   model_path: str = "/models/best.pt",
                   state: Optional[ProcessingState] = None) -> dict:
    """Run the full processing pipeline on a PDF manual."""
    manual_dir = output_dir / manual_id
    manual_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: PDF -> Images
        pages = pdf_to_images(pdf_path, manual_dir, state=state)

        # Step 2: Classify pages
        pages = classify_pages(pdf_path, pages, state=state)

        # Step 3: Pair sections
        sections = pair_sections(pages, state=state)

        if not sections:
            raise ValueError("No diagram+table sections found in PDF")

        # Step 4: YOLO detection
        pages_dir = manual_dir / "pages"
        sections = detect_callouts(sections, pages_dir, model_path=model_path,
                                   state=state)

        # Step 5: Read callout numbers (PDF text layer + tesseract fallback)
        sections, ocr_warnings = ocr_callouts(sections, pages_dir,
                                              pdf_path=pdf_path, state=state)

        # Step 6: Extract parts tables
        sections = extract_tables(sections, pdf_path, state=state)

        # Step 7: Link callouts -> parts
        sections, processing_log = link_callouts(sections, state=state)
        processing_log["warnings"].extend(ocr_warnings)

        # Build manifest
        manifest = {
            "manual_id": manual_id,
            "filename": pdf_path.name,
            "total_pages": len(pages),
            "sections": sections,
            "processing_log": processing_log,
        }

        manifest_path = manual_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        if state:
            state.done = True
            state.progress = 100
            state.message = "Processing complete"

        return manifest

    except Exception as e:
        logger.exception(f"Pipeline failed for {manual_id}")
        if state:
            state.error = str(e)
            state.done = True
        raise
