#!/usr/bin/env python3
"""
Solid, testowalny moduł do preprocesingu FinalData (bez argparse).

Główne kroki:
1) odfiltrowanie wierszy „tylko punktor/bullet”
2) zostawienie tylko wierszy z section_title
3) stabilne sortowanie po kolejności czytania (section -> page -> y_top -> x_left)
4) agregacja po section_title (sklejanie tekstu z „smart spacing”)
5) opcjonalne sortowanie po section_numbering interpretowanym numerycznie

Użycie (programistyczne):

    from pathlib import Path
    import pandas as pd

    df = pd.read_csv("FinalData.csv", dtype={"page": "Int64"}, keep_default_na=True)

    # (opcjonalnie) normalizacja typów tekstowych:
    NormaliseSchema(["text", "section_raw", "section_title", "section_numbering"]).transform(df)

    service = CsvPreprocessingService.default()
    out = service.process_dataframe(df, sort_by_numbering=True)
    out.to_csv("FinalData_preprocessed.csv", index=False)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Protocol

import numpy as np
import pandas as pd


# =========================
# Konfiguracja i interfejsy
# =========================

@dataclass(frozen=True)
class PreprocessConfig:
    """Parametry domenowe i nazwy kolumn."""
    text_col: str = "text"
    title_col: str = "section_title"
    numbering_col: str = "section_numbering"
    page_col: str = "page"
    y_top_col: str = "y_top"
    x_left_col: str = "x_left"

    bullet_tokens: frozenset[str] = frozenset({"", "•", "-", "–", "—", "*", "·"})
    bullet_regex: re.Pattern = re.compile(r"^[•\-\–\—\*·]{1,2}$")

    # kolumny agregowane jeżeli istnieją
    agg_first_cols: tuple[str, ...] = ("section_raw", "section_numbering",
                                       "section_level", "font", "size", "color")
    sum_cols: tuple[str, ...] = ("word_count", "line_len")

    # docelowa kolejność kolumn (jeśli występują)
    desired_order: tuple[str, ...] = (
        "section_title",
        "section_numbering",
        "section_level",
        "section_raw",
        "pages",
        "section_text",
        "word_count",
        "line_len",
        "font",
        "size",
        "color",
    )


class DataFrameTransform(Protocol):
    """Pojedynczy krok przetwarzania DataFrame (SRP)."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class Aggregator(Protocol):
    """Agregacja do widoku sekcji."""

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class Sorter(Protocol):
    """Strategia sortowania wyniku."""

    def sort(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


# =========================
# Pomocnicze funkcje narzędziowe (SRP)
# =========================

def is_bullet_only(text: str | float, cfg: PreprocessConfig) -> bool:
    """True, jeśli wiersz to wyłącznie punktor (ew. z białymi znakami)."""
    if pd.isna(text):
        return False
    t = str(text).strip()
    return (t in cfg.bullet_tokens) or bool(cfg.bullet_regex.fullmatch(t))


def clean_fragment(text: str | float) -> str:
    """Zbij białe znaki wewnątrz fragmentu i przytnij brzegi."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def smart_join(acc: str, frag: str) -> str:
    """
    Łączenie fragmentów z „mądrym” odstępem:
    - jeśli `acc` kończy się myślnikiem, sklejamy bez spacji (łamane słowo).
    """
    if not acc:
        return frag
    if not frag:
        return acc
    if acc.endswith("-"):
        return acc[:-1] + frag
    return f"{acc} {frag}"


def uniq_preserve_order(values: Iterable) -> List:
    """Unikalne wartości z zachowaniem kolejności i bez NaN."""
    seen: set = set()
    out: List = []
    for v in values:
        if pd.isna(v):
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_numbering(num: str | float) -> list[int]:
    """'2.42.2' -> [2, 42, 2] — do sortowania numerycznego numeracji."""
    if pd.isna(num):
        return []
    parts = re.split(r"[^\d]+", str(num).strip())
    return [int(p) for p in parts if p.isdigit()]


# =========================
# Transformy (OCP: łatwo dodać nowe kroki)
# =========================

@dataclass
class NormaliseSchema(DataFrameTransform):
    """Normalizacja typów kluczowych kolumn do StringDtype (jeśli istnieją)."""
    text_like_cols: List[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for col in self.text_like_cols:
            if col in d.columns:
                d[col] = d[col].astype("string")
        return d


@dataclass
class DropBulletOnlyRows(DataFrameTransform):
    """Usuń wiersze z samym punktorem/bulletem."""
    cfg: PreprocessConfig

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if self.cfg.text_col not in d.columns:
            return d
        mask = ~d[self.cfg.text_col].apply(lambda t: is_bullet_only(t, self.cfg))
        return d.loc[mask].copy()


@dataclass
class KeepRowsWithTitle(DataFrameTransform):
    """Zostaw wyłącznie wiersze posiadające section_title."""
    cfg: PreprocessConfig

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if self.cfg.title_col not in d.columns:
            return d
        return d.loc[d[self.cfg.title_col].notna()].copy()


@dataclass
class StableReadingOrderSort(DataFrameTransform):
    """
    Stabilne sortowanie po section_title, page, y_top, x_left
    (wypełnia brakujące kolumny NaN, by zachować interfejs).
    """
    cfg: PreprocessConfig

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for col in (self.cfg.page_col, self.cfg.y_top_col, self.cfg.x_left_col):
            if col not in d.columns:
                d[col] = np.nan
        return d.sort_values(
            by=[self.cfg.title_col, self.cfg.page_col,
                self.cfg.y_top_col, self.cfg.x_left_col],
            kind="mergesort",
        )


# =========================
# Agregacja (SRP)
# =========================

@dataclass
class SectionAggregator(Aggregator):
    """Agreguje wiersze do poziomu sekcji i czyści tekst."""
    cfg: PreprocessConfig

    def _agg_text(self, series: pd.Series) -> str:
        acc = ""
        for frag in series.fillna("").map(clean_fragment):
            acc = smart_join(acc, frag)
        return re.sub(r"\s{2,}", " ", acc).strip()

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # słowniki agregacji tylko dla istniejących kolumn
        agg_dict: dict[str, object] = {}
        # pierwsza nie-NaN wartość z kolumn meta (jeśli istnieją)
        for col in self.cfg.agg_first_cols:
            if col in d.columns:
                agg_dict[col] = (
                    lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA
                )

        # strony jako lista unikalna z zachowaniem kolejności
        if self.cfg.page_col in d.columns:
            agg_dict[self.cfg.page_col] = uniq_preserve_order

        # tekst -> „section_text”
        if self.cfg.text_col in d.columns:
            agg_dict[self.cfg.text_col] = self._agg_text

        # sumowane kolumny metryczne (jeśli są)
        for col in self.cfg.sum_cols:
            if col in d.columns:
                agg_dict[col] = "sum"

        out = (
            d.groupby([self.cfg.title_col], dropna=False, sort=False)
            .agg(agg_dict)
            .reset_index()
            .rename(columns={self.cfg.text_col: "section_text",
                             self.cfg.page_col: "pages"})
        )

        # ustaw sensowną kolejność kolumn
        desired = list(self.cfg.desired_order)
        cols = [c for c in desired if c in out.columns] + [
            c for c in out.columns if c not in desired
        ]
        return out[cols]


# =========================
# Sortowanie końcowe (strategia — DIP)
# =========================

@dataclass
class NoopSorter(Sorter):
    """Brak dodatkowego sortowania."""
    def sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


@dataclass
class NumberingSorter(Sorter):
    """Sortuj po section_numbering interpretowanym numerycznie."""
    cfg: PreprocessConfig

    def sort(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.numbering_col not in df.columns:
            return df
        return (
            df.assign(_numkey=df[self.cfg.numbering_col].map(parse_numbering))
              .sort_values("_numkey", kind="mergesort")
              .drop(columns="_numkey")
        )


# =========================
# Pipeline orkiestrujący (ISP + DIP)
# =========================

@dataclass
class PreprocessingPipeline:
    """
    Wysokopoziomowy pipeline zależy od abstrakcji (transformy, agregator, sorter),
    a nie od implementacji (DIP).
    """
    transforms: list[DataFrameTransform]
    aggregator: Aggregator
    sorter: Sorter

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for step in self.transforms:
            d = step.transform(d)
        d = self.aggregator.aggregate(d)
        d = self.sorter.sort(d)
        return d


# =========================
# Fasada do wygodnego użycia (S — SRP)
# =========================

@dataclass
class CsvPreprocessingService:
    """
    Fasada upraszczająca użycie w kodzie aplikacji/skryptach/Notebookach.
    Brak argparse; wywołujesz metody bezpośrednio.
    """
    pipeline: PreprocessingPipeline

    @classmethod
    def default(cls) -> CsvPreprocessingService:
        """Domyślna konfiguracja SOLID-pipeline."""
        cfg = PreprocessConfig()
        transforms: list[DataFrameTransform] = [
            NormaliseSchema(
                text_like_cols=["text", "section_raw",
                                "section_title", "section_numbering"]
            ),
            DropBulletOnlyRows(cfg),
            KeepRowsWithTitle(cfg),
            StableReadingOrderSort(cfg),
        ]
        aggregator = SectionAggregator(cfg)
        sorter = NoopSorter()
        pipeline = PreprocessingPipeline(transforms, aggregator, sorter)
        return cls(pipeline=pipeline)

    @classmethod
    def with_numbering_sort(cls) -> CsvPreprocessingService:
        """Wariant z sortowaniem po section_numbering."""
        svc = cls.default()
        cfg = PreprocessConfig()
        svc.pipeline.sorter = NumberingSorter(cfg)
        return svc

    def process_dataframe(
        self,
        df: pd.DataFrame,
        sort_by_numbering: bool = False,
    ) -> pd.DataFrame:
        """Przetwórz DataFrame zgodnie z pipeline (opcjonalnie posortuj po numeracji)."""
        if sort_by_numbering:
            # nie modyfikujemy istniejącej instancji — zwracamy kopię z inną strategią
            svc = type(self).with_numbering_sort()
            return svc.pipeline.run(df)
        return self.pipeline.run(df)
