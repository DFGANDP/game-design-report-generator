
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

# PDF extraction
import fitz  # PyMuPDF

import re
from pprint import pprint
from statistics import mean # do laczenia liter i ciagow znakow w tej samej lini

from rapidfuzz import process, fuzz
import re, unicodedata
import numpy as np 
import pandas as pd


import unicodedata

sections = {
    "Intro / Executive Summary": [],
    "Gameplay (overview + flow)": [],
    "Game World & Levels": [],
    "Graphics / Art Style": [],
    "Mechanics / Systems": [],
    "Characters": [],
    "AI / Enemies": [],
    "Controls / Interface": [],
    "Story / Narrative": [],
    "Sound / Music": [],
}

DOT_LEADERS = r".\u2026\u2024\u2027\u2219\u22C5"  # ., …, ․, ‧, ∙, ⋅
DOT_LEADERS_RE = re.compile(rf"[{DOT_LEADERS}]{{3,}}")  # co najmniej 3 z rzędu


class GddPdfInspector:
    """
    PDF layout analyzer tailored for Game Design Documents (GDD).

    Responsibilities:
        - Inspect pages and collect spans/images with geometry & typography.
        - Convert spans to a DataFrame with expanded bboxes.
        - Rebuild logical lines from spans in reading order.
        - Tag confident headers by fuzzy-matching Table of Contents (ToC).

    Parameters
    ----------
    y_tol : float
        Tolerance (pt) when clustering spans into the same text line.
    space_gap_pt : Optional[float]
        Fixed space insertion gap threshold (pt). If None, estimated per line.
    min_header_score : int
        Minimal fuzzy score for header tagging.
    """

    def __init__(
        self,
        y_tol: float = 2.0,
        space_gap_pt: Optional[float] = None,
        min_header_score: int = 78,
    ) -> None:
        self.y_tol = float(y_tol)
        self.space_gap_pt = space_gap_pt
        self.min_header_score = int(min_header_score)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def inspect_page(self, doc: fitz.Document, page_index: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Inspect a PDF page and collect layout-rich items.

        Returns two lists:
        - 'spans': text spans with typography & geometry (text, size, font, color, bbox).
        - 'images': image blocks with geometry and basic metadata (xref, bbox, width, height).
        """
        page = doc[page_index]

        # text spans (layout + typography)
        spans: List[Dict[str, Any]] = []
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox_line = tuple(line.get("bbox", (0, 0, 0, 0)))
                for s in line.get("spans", []):
                    spans.append(
                        {
                            "text": s.get("text", ""),
                            "size": round(s.get("size", 0.0), 2),
                            "font": s.get("font", ""),
                            "color": s.get("color", 0),  # int e.g. 0=black
                            "bbox": tuple(s.get("bbox", (0, 0, 0, 0))),
                            "line_bbox": bbox_line,
                        }
                    )

        # images (rawdict → reliable image blocks + xref)
        images: List[Dict[str, Any]] = []
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") == 1:
                images.append(
                    {
                        "xref": block.get("number"),
                        "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                        "width": block.get("width"),
                        "height": block.get("height"),
                    }
                )

        return {"spans": spans, "images": images}

    def inspect_document(
        self, filepath: str | bytes | fitz.Document, limit: Optional[int] = None
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Inspect multiple pages of a PDF, returning a list of per-page reports.
        """
        need_close = False
        if isinstance(filepath, fitz.Document):
            doc = filepath
        else:
            doc = fitz.open(filepath)
            need_close = True

        try:
            n_pages = len(doc) if limit is None else min(limit, len(doc))
            return [self.inspect_page(doc, i) for i in range(n_pages)]
        finally:
            if need_close:
                doc.close()

    def spans_to_df(self, spans_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Flatten spans (including bbox/line_bbox) into tabular columns.
        """
        df = pd.json_normalize(spans_list, sep="__")

        # Expand bbox / line_bbox if present
        for col, prefix in (("bbox", "bbox"), ("line_bbox", "line")):
            if col in df.columns:
                expanded = (
                    df[col]
                    .apply(lambda t: list(t) if isinstance(t, (list, tuple)) else [None] * 4)
                    .apply(pd.Series)
                    .rename(
                        columns={
                            0: f"{prefix}_x0",
                            1: f"{prefix}_y0",
                            2: f"{prefix}_x1",
                            3: f"{prefix}_y1",
                        }
                    )
                )
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

        # Dtypes
        for c in [c for c in df.columns if c.endswith(("_x0", "_y0", "_x1", "_y1"))] + ["size"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in [c for c in ["page", "color"] if c in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

        return df

    def assemble_lines(self, df_spans: pd.DataFrame, keep_meta: bool = True) -> pd.DataFrame:
        """
        Build logical lines from spans using geometry and reading order.
        """
        df = self._drop_empty_rows(df_spans)
        df = self._drop_page_number_rows(df)
        df = self._assign_line_ids(df, y_tol=self.y_tol)

        out_records: List[Dict[str, Any]] = []
        for (page, line_id), g in df.groupby(["page", "line_id"], sort=True):
            text, size, font, color = self._concat_line_spans(g)
            if not text:
                continue
            rec: Dict[str, Any] = {
                "page": int(page),
                "text": text,
                "size": float(size),
                "font": font,
                "color": int(color),
            }
            if keep_meta:
                rec["y_top"] = float(min(g["line_y0"] if "line_y0" in g else g["bbox_y0"]))
                rec["x_left"] = float(min(g["bbox_x0"]))
                rec["line_len"] = len(text)
                rec["word_count"] = len(text.split())
            out_records.append(rec)

        sort_cols = ["page", "y_top"] if keep_meta else ["page"]
        out = pd.DataFrame(out_records).sort_values(sort_cols, kind="mergesort")
        return out.reset_index(drop=True)

    def tag_confident_headers_simple(
        self,
        df: pd.DataFrame,
        toc: List[Dict[str, Any]],
        text_col: str = "text",
    ) -> pd.DataFrame:
        """
        Tag candidate header lines by fuzzy matching against ToC items.
        Adds: section_raw, section_title, section_numbering, section_level, section_confidence.
        """
        leading_num = re.compile(r"^\s*\d+(?:\.\d+)*\s*")

        def _norm(s: str) -> str:
            s = unicodedata.normalize("NFKD", s or "")
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return re.sub(r"[^a-z0-9]+", "", s.casefold())

        df = df.reset_index(drop=True).copy()

        if "_norm_no_num" not in df.columns:
            df["_norm_no_num"] = df[text_col].astype(str).map(
                lambda s: _norm(leading_num.sub("", s))
            )

        size_q80 = df["size"].quantile(0.80) if "size" in df else None
        is_big = (df["size"] >= size_q80) if size_q80 is not None else False
        is_short = df.get("word_count", pd.Series([999] * len(df))).le(6)
        candidates_mask = (is_big | is_short).astype(bool)
        candidates = df.loc[candidates_mask, "_norm_no_num"]

        for col in (
            "section_raw",
            "section_title",
            "section_numbering",
            "section_level",
            "section_confidence",
        ):
            df[col] = np.nan

        for t in toc:
            title = (t.get("subject") or "").strip()
            if not title:
                continue
            q = _norm(title)

            best = process.extractOne(q, candidates, scorer=fuzz.ratio)
            if not best or int(best[1]) < self.min_header_score:
                best = process.extractOne(q, candidates, scorer=fuzz.partial_ratio)

            if not best:
                continue

            score, idx = int(best[1]), int(best[2])
            if q == df.at[idx, "_norm_no_num"]:
                score = 100

            if score >= self.min_header_score:
                df.at[idx, "section_raw"] = title
                df.at[idx, "section_title"] = title
                df.at[idx, "section_numbering"] = (t.get("numbering") or "").strip()
                df.at[idx, "section_level"] = int(t.get("level") or 0)
                df.at[idx, "section_confidence"] = score

        return df.drop(columns=[c for c in ["_norm_no_num", "_norm"] if c in df.columns])

    # --------------------------------------------------------------------- #
    # Private helpers (implementation detail)
    # --------------------------------------------------------------------- #

    @staticmethod
    def _remove_dot_leaders(text: Optional[str]) -> str:
        if text is None:
            return ""
        cleaned = DOT_LEADERS_RE.sub(" ", text)
        return " ".join(cleaned.split())

    def _drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["text"].astype(str).fillna("").map(self._remove_dot_leaders)
        return df[df["text"].str.strip().str.len() > 0]

    @staticmethod
    def _drop_page_number_rows(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        mask = df[text_col].astype(str).str.fullmatch(r"\d+")
        return df[~mask].reset_index(drop=True)

    @staticmethod
    def _y_top(row: pd.Series) -> float:
        if "line_y0" in row and pd.notna(row["line_y0"]):
            return float(row["line_y0"])
        return float(row["bbox_y0"])

    def _assign_line_ids(self, df_spans: pd.DataFrame, y_tol: float) -> pd.DataFrame:
        df = df_spans.copy()
        for col in ("bbox_y0", "bbox_x0", "bbox_x1"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["y_top"] = df.apply(self._y_top, axis=1)
        df = df.sort_values(["page", "y_top", "bbox_x0"], kind="mergesort")

        line_ids: List[int] = []
        current_page: Optional[int] = None
        current_line_top: Optional[float] = None
        line_id = -1

        for _, row in df.iterrows():
            page = int(row["page"])
            y = float(row["y_top"])

            if (current_page is None) or (page != current_page):
                current_page = page
                current_line_top = None
                line_id += 1  # start sequence per page

            if current_line_top is None or abs(y - current_line_top) > y_tol:
                line_id += 1
                current_line_top = y

            line_ids.append(line_id)

        df["line_id"] = line_ids
        return df

    def _compute_gap_threshold(self, group: pd.DataFrame) -> float:
        if self.space_gap_pt is not None:
            return float(self.space_gap_pt)

        widths: List[float] = []
        for _, row in group.iterrows():
            txt = str(row["text"])
            if not txt:
                continue
            span_w = float(row["bbox_x1"]) - float(row["bbox_x0"])
            widths.append(span_w / max(len(txt), 1))

        if not widths:
            return 1.0
        return max(1.0, 0.5 * float(np.mean(widths)))

    def _concat_line_spans(self, group: pd.DataFrame) -> Tuple[str, float, str, int]:
        g = group.sort_values("bbox_x0", kind="mergesort")
        gap_thr = self._compute_gap_threshold(g)

        pieces: List[str] = []
        prev_x1: Optional[float] = None
        for _, row in g.iterrows():
            x0 = float(row["bbox_x0"])
            x1 = float(row["bbox_x1"])
            txt = str(row["text"])

            if prev_x1 is not None and (x0 - prev_x1) > gap_thr and (not pieces or not pieces[-1].endswith(" ")):
                pieces.append(" ")
            pieces.append(txt)
            prev_x1 = x1

        text = " ".join("".join(pieces).split())

        first = g.iloc[0]
        size = float(first.get("size", np.nan))
        font = str(first.get("font", ""))
        color = int(first.get("color", 0))
        return text, size, font, color



def extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Extract numbered TOC entries as {'subject': str, 'level': int, 'numbering': str, 'page': str|''}.

    Handles:
    - numeric hierarchies: 1, 1.2, 3.1.4
    - dot leaders (..... or ……) and page numbers (digits or roman numerals)
    - cases where numbering is on a separate line above the title

    Args:
        text: Raw table-of-contents text.

    Returns:
        A list of dicts with cleaned subject/title, inferred level, original numbering,
        and optional page string.
    """
    text = _prejoin_number_lines(text)

    # ^ num   title            (optional dot leaders + page)
    dot_cls = f"[{DOT_LEADERS}]"
    pattern = re.compile(
        rf"""^
            \s*(?P<num>\d+(?:\.\d+)*)
            \s+(?P<title>.*?)
            (?:\s*{dot_cls}{{3,}}\s*(?P<page>\d+|[ivxlcdmIVXLCDM]+))?
            \s*$
        """,
        re.MULTILINE | re.VERBOSE,
    )

    results: List[Dict[str, str]] = []
    for m in pattern.finditer(text):
        numbering = m.group("num")
        subject_raw = m.group("title").strip()

        # extra cleanup: sometimes PDFs insert stray leaders inside title
        subject = re.sub(rf"\s*{dot_cls}{{3,}}\s*$", "", subject_raw).strip()

        level = numbering.count(".") + 1
        page = (m.group("page") or "").strip()

        # skip junk lines that ended up with empty titles
        if not subject:
            continue

        results.append(
            {
                "subject": subject,
                "level": level,
                "numbering": numbering,
                "page": page,
                "raw": (f'{numbering} {subject}')
            }
        )

    return results

def _prejoin_number_lines(text: str) -> str:
    """
    Join lines where a numbering like '1', '1.2', '3.1.4' appears alone
    by merging it with the next non-empty line.

    This fixes ToCs where the number is on its own line and the title
    starts on the next line.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    i = 0
    num_only = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*$")
    while i < len(lines):
        m = num_only.match(lines[i])
        if m and i + 1 < len(lines) and lines[i + 1].strip():
            # merge: "1.2" + "Title .... 4" -> "1.2 Title .... 4"
            merged = f"{m.group(1)} {lines[i + 1].lstrip()}"
            out.append(merged)
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

class PDFTableofContents:
    """Utility class for extracting text from PDF files."""

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {self.filepath}")

    def extract_pages(self, pages: List[int]) -> str:
        """
        Extract text from specific pages of a PDF.

        Args:
            pages (List[int]): Page numbers (1-based indexing).

        Returns:
            str: Concatenated text from the specified pages.
        """
        if not pages:
            raise ValueError("Pages list cannot be empty.")

        try:
            doc = fitz.open(self.filepath)
            extracted_texts = []
            for page_num in pages:
                if page_num < 1 or page_num > len(doc):
                    raise IndexError(f"Page {page_num} out of range.")
                text = doc[page_num - 1].get_text("text")
                extracted_texts.append(text)
            doc.close()
            return "\n".join(extracted_texts)
        except Exception as e:
            raise RuntimeError(f"Failed to extract pages: {e}") from e

'''
Potrzebuje 2 klasy 

Pierwsza sobie robi preprocessing i wyciaganie spisu tresci 
Drugi robi wycaiganie danych z gotowego pdfa

Do tego interfacy potem dodac
'''

