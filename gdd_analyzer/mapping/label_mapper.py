#!/usr/bin/env python3
"""
GDD Section Mapper — SOLID/PEP8
Mapuje tytuły sekcji z PDF do kanonicznych kategorii przez LLM (np. Ollama).

Warstwy i odpowiedzialności:
- Config (MapperConfig) — parametry domenowe i modelowe.
- Model port (ChatModel) — interfejs do LLM (DIP).
- Adapter (OllamaChatModel) — implementacja dla ollama.Client.
- PromptBuilder — budowa promptu z listy sekcji.
- JsonExtractor — odporne wydobywanie pierwszego obiektu JSON z odpowiedzi.
- CategoryValidator — walidacja/normalizacja kategorii (tryb 'strict' lub 'closest').
- SectionMapperService — logika mapowania batchami (retry, backoff).
- DataFrameMapper — spinanie z Pandas (merge po kluczu deterministycznym).

Przykład użycia na końcu pliku.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, TypedDict

import pandas as pd

try:
    # opcjonalnie do fuzzy-normalizacji kategorii (nie wymagane w trybie strict)
    from rapidfuzz import process, fuzz  # type: ignore
    _HAS_FUZZ = True
except Exception:  # pragma: no cover
    _HAS_FUZZ = False


# =========================
# Domenowe stałe i typy
# =========================

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


@dataclass(frozen=True)
class SectionRow:
    """Minimalny wiersz wejściowy reprezentujący sekcję."""
    section_title: str
    section_numbering: Optional[str] = None


class MappingRecord(TypedDict):
    """Wyjściowy rekord mapowania z LLM (po walidacji)."""
    section_title: str
    section_numbering: Optional[str]
    target_category: str


# =========================
# Konfiguracje i polityki
# =========================

@dataclass(frozen=True)
class MapperConfig:
    """
    Konfiguracja mapowania.
    - prompt_template: tekst zawierający placeholder '{ITEMS_BLOCK}'.
    - categories: lista kanonicznych kategorii.
    - strategy: 'strict' (wymaga idealnej kategorii) lub 'closest' (zmapuj najbliższą).
    - min_similarity: próg podobieństwa dla 'closest' (0-100), jeśli rapidfuzz dostępny.
    - batch_size: ile sekcji w jednym wywołaniu LLM.
    - retries / backoff: prosta polityka ponowień.
    """
    prompt_template: str
    categories: List[str] = tuple(CANONICAL_SECTIONS)  # type: ignore[assignment]
    strategy: str = "strict"  # or "closest"
    min_similarity: int = 85
    batch_size: int = 40
    retries: int = 2
    backoff_seconds: float = 1.5
    model_name: str = "mistral-small:latest"
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if "{ITEMS_BLOCK}" not in self.prompt_template:
            raise ValueError("Prompt template must contain '{ITEMS_BLOCK}' placeholder.")
        if self.strategy not in {"strict", "closest"}:
            raise ValueError("strategy must be 'strict' or 'closest'.")


# =========================
# Port do modelu (DIP) + adapter Ollama
# =========================

class ChatModel(Protocol):
    """Abstrakcyjny port do modelu czatowego."""
    def chat(self, prompt: str, *, model: str, temperature: float) -> str:
        ...


@dataclass
class OllamaChatModel(ChatModel):
    """
    Adapter do ollama.Client (wstrzykujesz gotowego klienta).
    """
    client: Any  # oczekuje instancji ollama.Client

    def chat(self, prompt: str, *, model: str, temperature: float) -> str:
        resp = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return str(resp["message"]["content"])


# =========================
# Budowa promptu
# =========================

class PromptBuilder:
    """Składa prompt z szablonu i listy sekcji."""

    @staticmethod
    def rows_to_items_block(rows: Iterable[SectionRow]) -> str:
        lines: List[str] = []
        for idx, r in enumerate(rows, start=1):
            num = r.section_numbering or ""
            lines.append(
                f'{idx}) section_title="{r.section_title}", section_numbering="{num}"'
            )
        return "\n".join(lines)

    def build(self, template: str, rows: List[SectionRow]) -> str:
        return template.replace("{ITEMS_BLOCK}", self.rows_to_items_block(rows))


# =========================
# Ekstrakcja JSON z odpowiedzi
# =========================

class JsonExtractionError(RuntimeError):
    """Błąd wydobycia poprawnego JSON-a z treści modelu."""


class JsonExtractor:
    """Wydobywa pierwszy OBIEKT JSON z surowej odpowiedzi modelu (odpornie)."""

    _TRAILING_COMMAS = re.compile(r",\s*([}\]])")

    @classmethod
    def extract_object(cls, text: str) -> Dict[str, Any]:
        """
        Znajduje pierwszy obiekt JSON w tekście (preferuje blok na końcu).
        Czyści trailing commas i niełamliwe spacje.
        """
        clean = text.replace("\xa0", " ")
        clean = cls._TRAILING_COMMAS.sub(r"\1", clean)

        # 1) Kodowe bloki ```json ... ``` — jeśli są, spróbuj najpierw stamtąd
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", clean, flags=re.IGNORECASE)
        if fence:
            s = fence.group(1)
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass  # spróbujemy dalej

        # 2) Najpierw spróbuj ostatniego obiektu na końcu
        m = re.search(r"\{[\s\S]+\}\s*$", clean)
        if not m:
            # 3) fallback — pierwszy obiekt
            m = re.search(r"\{[\s\S]+?\}", clean)
        if not m:
            raise JsonExtractionError("No valid JSON object found in the text.")

        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise JsonExtractionError(f"Failed to parse JSON: {e}") from e


# =========================
# Walidacja/normalizacja kategorii
# =========================

class CategoryValidator:
    """
    Waliduje lub normalizuje kategorie do kanonicznych.
    - 'strict': każdy target_category MUSI wystąpić w liście kanonicznej.
    - 'closest': jeśli nie ma idealnego dopasowania, wybierz najbliższe (rapidfuzz).
    """

    def __init__(self, categories: List[str], strategy: str = "strict", min_similarity: int = 85) -> None:
        self.categories = list(categories)
        self.strategy = strategy
        self.min_similarity = int(min_similarity)

    def validate_all(self, records: List[MappingRecord]) -> List[MappingRecord]:
        out: List[MappingRecord] = []
        for rec in records:
            out.append(self._validate_one(rec))
        return out

    def _validate_one(self, rec: MappingRecord) -> MappingRecord:
        cat = rec.get("target_category") or ""
        if cat in self.categories:
            return rec

        if self.strategy == "strict":
            raise ValueError(f"Invalid category in model output: {cat!r}")

        # 'closest' — dopasuj najbliższą, jeśli to sensowne
        if not _HAS_FUZZ:
            # brak rapidfuzz → zachowuj się jak strict
            raise ValueError(
                f"Invalid category and rapidfuzz not available: {cat!r}"
            )

        best = process.extractOne(cat, self.categories, scorer=fuzz.WRatio)
        if not best:
            raise ValueError(f"Cannot normalize category: {cat!r}")

        best_label, score, _idx = best
        if int(score) < self.min_similarity:
            raise ValueError(
                f"Closest category '{best_label}' score {score} below threshold {self.min_similarity} for {cat!r}"
            )

        rec["target_category"] = str(best_label)
        return rec


# =========================
# Serwis mapujący (retry, batch)
# =========================

class MappingError(RuntimeError):
    """Błąd wywołania modelu / przetwarzania odpowiedzi."""


@dataclass
class SectionMapperService:
    """
    Orkiestruje:
    - build prompt for a batch
    - call ChatModel
    - extract JSON
    - validate/normalize categories
    """

    model: ChatModel
    cfg: MapperConfig
    prompt_builder: PromptBuilder = PromptBuilder()
    validator: CategoryValidator = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.validator is None:
            self.validator = CategoryValidator(
                categories=list(self.cfg.categories),
                strategy=self.cfg.strategy,
                min_similarity=self.cfg.min_similarity,
            )

    def _call_with_retry(self, prompt: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.retries + 1):
            try:
                raw = self.model.chat(
                    prompt,
                    model=self.cfg.model_name,
                    temperature=self.cfg.temperature,
                )
                return JsonExtractor.extract_object(raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.backoff_seconds * (2**attempt))
                else:
                    break
        raise MappingError(f"LLM call failed after retries: {last_err}") from last_err

    def map_batch(self, rows: List[SectionRow]) -> List[MappingRecord]:
        prompt = self.prompt_builder.build(self.cfg.prompt_template, rows)
        parsed = self._call_with_retry(prompt)

        mapping = parsed.get("mapping", [])
        if not isinstance(mapping, list):
            raise MappingError("Invalid response: 'mapping' is not a list.")

        # ensure structure & types
        cleaned: List[MappingRecord] = []
        for item in mapping:
            if not isinstance(item, dict):
                raise MappingError("Invalid mapping item, expected object.")
            rec: MappingRecord = {
                "section_title": str(item.get("section_title", "")).strip(),
                "section_numbering": (
                    None
                    if item.get("section_numbering") in (None, "", "None")
                    else str(item.get("section_numbering")).strip()
                ),
                "target_category": str(item.get("target_category", "")).strip(),
            }
            cleaned.append(rec)

        # validate/normalize categories
        return self.validator.validate_all(cleaned)

    def map_rows(self, rows: List[SectionRow]) -> List[MappingRecord]:
        out: List[MappingRecord] = []
        bs = int(self.cfg.batch_size)
        for start in range(0, len(rows), bs):
            batch = rows[start : start + bs]
            out.extend(self.map_batch(batch))
        return out


# =========================
# Warstwa Pandas (spinanie z Twoim DF)
# =========================

class DataFrameMapper:
    """Łączy wyniki mapowania z oryginalnym DataFrame w sposób stabilny."""

    @staticmethod
    def _merge_key(title: str, numbering: Optional[str]) -> str:
        n = "" if numbering is None else numbering
        return f"{title.strip()}||{n.strip()}"

    def map_dataframe(
        self,
        df: pd.DataFrame,
        mapper: SectionMapperService,
        *,
        title_col: str = "section_title",
        num_col: str = "section_numbering",
    ) -> pd.DataFrame:
        """Zwraca kopię DF z kolumną 'mapped_category' (join po kluczu)."""
        rows: List[SectionRow] = []
        for i in df.index:
            title = str(df.at[i, title_col]).strip()
            num_val = df.at[i, num_col] if num_col in df.columns else None
            numbering = None if pd.isna(num_val) else str(num_val).strip()
            rows.append(SectionRow(section_title=title, section_numbering=numbering))

        records = mapper.map_rows(rows)
        out_df = pd.DataFrame(records).rename(
            columns={"target_category": "mapped_category"}
        )

        # stabilny klucz łączący (title + numbering)
        df_copy = df.copy()
        df_copy["_merge_key"] = [
            self._merge_key(str(df_copy.at[i, title_col]), df_copy.at[i, num_col] if num_col in df.columns else None)
            for i in df_copy.index
        ]
        out_df["_merge_key"] = [
            self._merge_key(r["section_title"], r.get("section_numbering"))
            for r in records
        ]

        merged = df_copy.merge(out_df[["_merge_key", "mapped_category"]], on="_merge_key", how="left")
        merged.drop(columns=["_merge_key"], inplace=True)
        return merged


# =========================
# Wygodne helpery do wczytania promptu
# =========================

def load_prompt(path: Path) -> str:
    """Wczytaj prompt z pliku (UTF-8)."""
    return Path(path).read_text(encoding="utf-8")


# =========================
# Przykład użycia (programistyczny)
# =========================

'''
if __name__ == "__main__":
    # 1) Dane wejściowe
    df_in = pd.read_csv("FinalDataREFACTOR_SOLIDAFTERPREPROCESSING.csv")

    # 2) Wczytaj prompt (musi zawierać '{ITEMS_BLOCK}')
    template = load_prompt(Path("gdd_analyzer\prompts\prompt_ollama.txt"))

    # 3) Zbuduj konfigurację i model (Ollama)
    from ollama import Client  # import lokalny, by moduł był testowalny bez tej zależności

    ollama_client = Client(host="http://127.0.0.1:11434")
    chat_model = OllamaChatModel(client=ollama_client)

    cfg = MapperConfig(
        prompt_template=template,
        model_name="MHKetbi/Mistral-Small3.1-24B-Instruct-2503:q5_K_L",  # lub domyślny
        temperature=0.1,
        strategy="strict",     # zmień na 'closest' by auto-normalizować do kanonu
        min_similarity=88,     # używane tylko w 'closest'
        batch_size=40,
        retries=2,
        backoff_seconds=1.5,
    )

    # 4) Złóż usługę
    service = SectionMapperService(model=chat_model, cfg=cfg)

    # 5) Zmapuj i zintegruj z DataFrame
    df_mapper = DataFrameMapper()
    df_mapped = df_mapper.map_dataframe(
        df_in,
        service,
        title_col="section_title",
        num_col="section_numbering",
    )

    # 6) Zapis
    df_mapped.to_csv("FinalData_mappedSOLIDREFACTORSKURWYSYNUI.csv", index=False)
    print(df_mapped.head(10).to_string(index=False))
'''