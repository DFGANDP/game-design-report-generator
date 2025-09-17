#!/usr/bin/env python3
"""
GDD Section Analysis (OpenAI only) — SOLID/PEP8

Funkcje-fasady kompatybilne z dotychczasowym main:
- concat_section_text_by_mapped_category(df, ...)
- create_prompt(filepath, mapped_category, section_text)
- run_openai_analysis(df, prompt_file, model, jsonl_path)
- add_keywords_column(df, top_n)
- pretty_print_row / pretty_print_df
- main()

Wewnątrz korzysta z:
- AnalysisConfig (konfiguracja)
- OpenAIResponsesClient (owijka nad OpenAI Responses)
- SectionAnalysisService (pętla analizy + zapis JSONL)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import re
import time

import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------
# Stałe domenowe (do podglądu)
# ---------------------------

CANONICAL_SECTIONS: List[str] = [
    "Intro / Executive Summary",
    "Gameplay (overview + flow)",
    "Game World & Levels",
    "Graphics / Art Style",
    "Mechanics / Systems",
    "Characters",
    "AI / Enemies",
    "Controls / Interface",
    "Story / Narrative",
    "Sound / Music",
    "Out of scope",
]


# ---------------------------
# Konfiguracja i klient OpenAI
# ---------------------------

@dataclass(frozen=True)
class AnalysisConfig:
    """Config dla wywołań analizy."""
    prompt_file: Path
    model: str = "gpt-5-nano"
    jsonl_path: Path = Path("outputs/openai_run.jsonl")
    sleep_seconds: float = 0.4  # proste odmierzanie między requestami


class OpenAIResponsesClient:
    """
    Cienka warstwa nad OpenAI Responses API.
    Oczekuje, że klucz będzie w zmiennych środowiskowych (np. OPENAI_API_KEY).
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        self._client = client or OpenAI()

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Usuwa ```json ... ``` jeśli model tak zwróci."""
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json\n"):
                raw = raw[5:]
        return raw

    def analyze(self, prompt_text: str, model: str) -> Dict[str, Any]:
        """
        Wywołanie Responses API. Zwraca zparsowany JSON lub {"error": "..."}.
        """
        try:
            resp = self._client.responses.create(
                model=model,
                instructions=(
                    "You are a strict JSON generator. "
                    "Output must be valid JSON only."
                ),
                input=prompt_text,
            )
            raw = self._strip_fences(resp.output_text)
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------
# Budowa promptu (fasada zgodna z Twoim kodem)
# ---------------------------

def create_prompt(
    filepath: str,
    mapped_category: str,
    section_text: str,
) -> str:
    """
    Ładuje szablon promptu i wstrzykuje {CATEGORY}, {SECTION_TEXT}.
    """
    core = Path(filepath).read_text(encoding="utf-8")
    return core.replace("{CATEGORY}", mapped_category).replace(
        "{SECTION_TEXT}", section_text
    )


# ---------------------------
# Agregacja i keywords (jak w Twoim main)
# ---------------------------

def concat_section_text_by_mapped_category(
    df: pd.DataFrame,
    text_sep: str = "\n",
    keep_first_cols: Iterable[str] = ("summary",),
) -> pd.DataFrame:
    """
    Grupowanie po mapped_category: sum metryk i join tekstu.
    Zostawia pierwszą wartość dla kolumn z keep_first_cols (jeśli istnieją).
    """
    agg_spec: Dict[str, Any] = {
        "line_len": "sum",
        "word_count": "sum",
        "section_text": lambda x: text_sep.join(x),
    }
    for col in keep_first_cols:
        if col in df.columns:
            agg_spec[col] = "first"

    return df.groupby("mapped_category", as_index=False).agg(agg_spec)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    TF-IDF keywords z jednego dokumentu (szybka wersja).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]
    order = scores.argsort()[::-1]
    return [feature_names[i] for i in order[:top_n]]


def add_keywords_column(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Dodaje kolumnę 'keywords' opartą o TF-IDF."""
    out = df.copy()
    out["keywords"] = out["section_text"].apply(lambda t: extract_keywords(t, top_n))
    return out


# ---------------------------
# Serwis analizy sekcji
# ---------------------------

@dataclass
class SectionAnalysisService:
    """
    Orkiestracja: tworzy prompt → wywołuje OpenAI → append do JSONL → rozszerza DF.
    """
    cfg: AnalysisConfig
    client: OpenAIResponsesClient

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wymagane kolumny: 'mapped_category', 'section_text'.
        Zwraca kopię DF poszerzoną o kolumny: analysis, summary, *_score, *_comment, etc.
        """
        df_out = df.copy()
        results: List[Dict[str, Any]] = []

        # Przygotuj świeży plik .jsonl
        self.cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.jsonl_path.exists():
            self.cfg.jsonl_path.unlink()

        for idx, row in df_out.iterrows():
            category = str(row["mapped_category"])
            text = str(row["section_text"])

            prompt_text = create_prompt(
                filepath=str(self.cfg.prompt_file),
                mapped_category=category,
                section_text=text,
            )

            analysis = self.client.analyze(prompt_text, model=self.cfg.model)
            results.append(analysis)

            record = {
                "row_index": int(idx),
                "mapped_category": category,
                "analysis": analysis,
            }
            with self.cfg.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            time.sleep(self.cfg.sleep_seconds)

        # kolumna z pełnym JSON
        df_out["analysis"] = results

        # bezpieczne pobieranie zagnieżdżonych kluczy
        def g(d: Optional[Dict[str, Any]], *keys: str, default=None):
            cur: Any = d or {}
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        df_out["summary"] = df_out["analysis"].apply(lambda x: g(x, "summary"))
        df_out["clarity_score"] = df_out["analysis"].apply(lambda x: g(x, "clarity", "score"))
        df_out["clarity_comment"] = df_out["analysis"].apply(lambda x: g(x, "clarity", "comment"))
        df_out["completeness_score"] = df_out["analysis"].apply(lambda x: g(x, "completeness", "score"))
        df_out["completeness_comment"] = df_out["analysis"].apply(lambda x: g(x, "completeness", "comment"))
        df_out["innovation_score"] = df_out["analysis"].apply(lambda x: g(x, "innovation", "score"))
        df_out["innovation_comment"] = df_out["analysis"].apply(lambda x: g(x, "innovation", "comment"))
        df_out["feasibility_score"] = df_out["analysis"].apply(lambda x: g(x, "feasibility", "score"))
        df_out["feasibility_comment"] = df_out["analysis"].apply(lambda x: g(x, "feasibility", "comment"))
        df_out["out_of_scope_reason"] = df_out["analysis"].apply(lambda x: g(x, "reason_not_fitting"))

        return df_out


# ---------------------------
# Fasady zgodne z Twoim użyciem
# ---------------------------

def run_openai_analysis(
    df: pd.DataFrame,
    prompt_file: str = "pormpt_openai.txt",
    model: str = "gpt-5-nano",
    jsonl_path: str = "outputs/openai_run.jsonl",
) -> pd.DataFrame:
    """
    Wygodna fasada: buduje config i klienta, uruchamia SectionAnalysisService.
    """
    cfg = AnalysisConfig(
        prompt_file=Path(prompt_file),
        model=model,
        jsonl_path=Path(jsonl_path),
        sleep_seconds=0.4,
    )
    service = SectionAnalysisService(cfg=cfg, client=OpenAIResponsesClient())
    return service.run(df)


def pretty_print_row(row: dict) -> None:
    """Prosty wydruk jednej sekcji z wynikami."""
    print("=" * 80)
    print(f"Category: {row.get('mapped_category', 'N/A')}")
    print(f"Line length: {row.get('line_len', 'N/A')}, Words: {row.get('word_count', 'N/A')}")
    print("\n--- Summary ---")
    print(row.get("summary", "No summary available"))

    print("\n--- Analysis ---")
    print(row.get("analysis", "No analysis available"))

    print("\n--- Scores ---")
    print(
        f"Clarity:       {row.get('clarity_score', 'N/A')}  "
        f"({row.get('clarity_comment', '')})"
    )
    print(
        f"Completeness:  {row.get('completeness_score', 'N/A')}  "
        f"({row.get('completeness_comment', '')})"
    )
    print(
        f"Innovation:    {row.get('innovation_score', 'N/A')}  "
        f"({row.get('innovation_comment', '')})"
    )
    print(
        f"Feasibility:   {row.get('feasibility_score', 'N/A')}  "
        f"({row.get('feasibility_comment', '')})"
    )

    if row.get("out_of_scope_reason"):
        print("\n⚠️ Out of scope:")
        print(row["out_of_scope_reason"])
    print("=" * 80)


def pretty_print_df(df: pd.DataFrame) -> None:
    """Iteracyjny wydruk DataFrame sekcja po sekcji."""
    for _, r in df.iterrows():
        pretty_print_row(r.to_dict())


# ---------------------------
# Minimalny main — identyczny przebieg jak u Ciebie
# ---------------------------

def main() -> None:
    filepath = "FinalData_mapped.csv"

    # 1) Wczytaj i zgrupuj tekst po mapped_category
    df = pd.read_csv(filepath)
    print(CANONICAL_SECTIONS)
    df_mapped = concat_section_text_by_mapped_category(df)
    print(df_mapped.head(20))

    # 2) Wywołaj OpenAI Responses na każdej sekcji
    df_analyzed = run_openai_analysis(
        df=df_mapped,
        prompt_file="gdd_analyzer\prompts\pormpt_openai.txt",
        model="gpt-5-nano",           # lub "gpt-5-mini"
        jsonl_path="outputs/openai_run.jsonl",
    )

    # 3) Zapis
    df_analyzed.to_csv("final_output_openaiNano.csv", index=False, encoding="utf-8")
    print(df_analyzed)


if __name__ == "__main__":
    main()
