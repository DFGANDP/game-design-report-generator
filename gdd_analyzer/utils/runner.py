from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from ollama import Client

from gdd_analyzer.extract.data_extractor import (
    PDFTableofContents,
    GddPdfInspector,
    extract_sections,
)
from gdd_analyzer.preprocess.section_preprocessor import (
    CsvPreprocessingService,
    NormaliseSchema,
)
from gdd_analyzer.mapping.label_mapper import (
    CANONICAL_SECTIONS,
    DataFrameMapper,
    MapperConfig,
    OllamaChatModel,
    SectionMapperService,
    load_prompt,
)
from gdd_analyzer.analyzer.openai_analyzer import (
    concat_section_text_by_mapped_category,
    run_openai_analysis,
    extract_keywords
)
from gdd_analyzer.report.gdd_report import ReportConfig, ReportGenerator, build_mapping

from gdd_analyzer.interfaces.log_handlers.console_handler import get_console_handler
from gdd_analyzer.interfaces.log_handlers.file_handler import get_file_handler
from gdd_analyzer.interfaces.log_handlers.setup_logging import setup_logger


@dataclass(frozen=True)
class PathsConfig:
    pdf_path: Path
    logs_path: Path
    prompt_path: Path
    tmp_dir: Path = Path("gdd_analyzer/output/tmp")
    out_dir: Path = Path("gdd_analyzer/output")

    # artefakty pośrednie/końcowe (domyślne, można nadpisać)
    spans_csv: Path = Path("gdd_analyzer/output/FinalData_spans.csv")
    lines_csv: Path = Path("gdd_analyzer/output/FinalData_lines.csv")
    assigned_csv: Path = Path("gdd_analyzer/output/FinalData_assigned.csv")
    mapped_csv: Path = Path("gdd_analyzer/output/FinalData_mapped.csv")
    openai_jsonl: Path = Path("gdd_analyzer/output/openai_run.jsonl")
    openai_csv: Path = Path("gdd_analyzer/output/final_output_openaiNano.csv")
    report_pdf: Path = Path("gdd_analyzer/output/GDD_Final_Report.pdf")


@dataclass(frozen=True)
class TocConfig:
    pages: Tuple[int, ...] = (2, 3)  # które strony zawierają spis treści
    drop_first_item: bool = True     # usuń nagłówek „CONTENTS” itp.


@dataclass(frozen=True)
class InspectorConfig:
    y_tol: float = 2.0
    space_gap_pt: Optional[float] = None
    min_header_score: int = 78
    content_min_page: int = 4  # strony <4 to front-matter


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model_name: str = "MHKetbi/Mistral-Small3.1-24B-Instruct-2503:q5_K_L"
    temperature: float = 0.1
    strategy: str = "strict"  # 'strict' lub 'closest'
    min_similarity: int = 88  # used in 'closest'
    batch_size: int = 40
    retries: int = 2
    backoff_seconds: float = 1.5


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = "gpt-5-nano"
    prompt_file: Optional[Path] = None  # if None, use domain from PathsConfig.prompt_path


class GddAnalyzerPipeline:
    """
    Orchestrates the full pipeline:
    1) Extract table of contents and sections
    2) Inspect PDF -> lines/sections
    3) CSV preprocessing
    4) Map sections to canonical structure (Ollama)
    5) Run OpenAI analysis
    6) Generate PDF report
    """

    def __init__(
        self,
        paths: PathsConfig,
        toc_cfg: TocConfig,
        inspector_cfg: InspectorConfig,
        ollama_cfg: OllamaConfig,
        openai_cfg: OpenAIConfig,
    ) -> None:
        self.paths = paths
        self.toc_cfg = toc_cfg
        self.inspector_cfg = inspector_cfg
        self.ollama_cfg = ollama_cfg
        self.openai_cfg = openai_cfg

        self.logger = setup_logger(
            name="gdd-analyzer",
            handlers=[
                get_console_handler("INFO"),
                get_file_handler(str(self.paths.logs_path), "DEBUG"),
            ],
        )

        self.paths.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.paths.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- Step 1: ToC -> sections ---------------------------------

    def extract_section_specs(self) -> List[dict]:
        """Reads the table of contents from selected pages and parses a list of sections/numbering."""
        extractor = PDFTableofContents(str(self.paths.pdf_path))
        toc_text = extractor.extract_pages(self.toc_cfg.pages).strip()

        sections = extract_sections(toc_text)
        if self.toc_cfg.drop_first_item and sections:
            sections = sections[1:]

        self.logger.info("Found %d items in the table of contents.", len(sections))
        self.logger.debug("TOC sections: %s", sections)
        return sections

    # --------------------- Step 2: inspecrt PDF and concat lines ---------------------

    def inspect_pdf(self, section_specs: List[dict]) -> pd.DataFrame:
        """Extracts lines, tags headers, and assigns them forward-filled to the text."""
        inspector = GddPdfInspector(
            y_tol=self.inspector_cfg.y_tol,
            space_gap_pt=self.inspector_cfg.space_gap_pt,
            min_header_score=self.inspector_cfg.min_header_score,
        )
        report = inspector.inspect_document(str(self.paths.pdf_path))

        all_spans = [
            {"page": i + 1, **span}
            for i, page in enumerate(report)
            for span in page["spans"]
        ]
        df_spans = inspector.spans_to_df(all_spans)
        df_lines = inspector.assemble_lines(df_spans)

        # odfiltruj front-matter
        df_lines = df_lines[df_lines["page"] >= self.inspector_cfg.content_min_page]

        df_headers_only = inspector.tag_confident_headers_simple(df_lines, section_specs)

        # przypisz sekcje do tekstu (ffill)
        cols = ["section_raw", "section_title", "section_numbering", "section_level", "section_confidence"]
        df_assigned = df_headers_only.copy()
        df_assigned[cols] = df_assigned[cols].ffill()

        # For debugging
        df_spans.to_csv(self.paths.spans_csv, index=False, encoding="utf-8")
        df_lines.to_csv(self.paths.lines_csv, index=False, encoding="utf-8")
        df_assigned.to_csv(self.paths.assigned_csv, index=False, encoding="utf-8")

        self.logger.info("PDF inspected and sections assigned.")
        return df_assigned

    # -------------------------- Step 3: Preprocessing CSV -----------------------------

    def preprocess(self, df_assigned: pd.DataFrame) -> pd.DataFrame:
        """Normalizes schema and sorts by section numbering."""
        NormaliseSchema(["text", "section_raw", "section_title", "section_numbering"]).transform(df_assigned)
        service = CsvPreprocessingService.default()
        df_out = service.process_dataframe(df_assigned, sort_by_numbering=True)
        self.logger.info("Preprocessing finished. Rows: %d", len(df_out))
        return df_out

    # ----------------------- Step 4: Mapping to our layout (Ollama) ---------------------

    def map_sections(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        """Maps titles to CANONICAL_SECTIONS using Local LLM (Ollama)."""
        prompt_template = load_prompt(self.paths.prompt_path)
        ollama_client = Client(host=self.ollama_cfg.host)
        chat_model = OllamaChatModel(client=ollama_client)

        cfg = MapperConfig(
            prompt_template=prompt_template,
            model_name=self.ollama_cfg.model_name,
            temperature=self.ollama_cfg.temperature,
            strategy=self.ollama_cfg.strategy,
            min_similarity=self.ollama_cfg.min_similarity,
            batch_size=self.ollama_cfg.batch_size,
            retries=self.ollama_cfg.retries,
            backoff_seconds=self.ollama_cfg.backoff_seconds,
        )

        mapper_service = SectionMapperService(model=chat_model, cfg=cfg)
        df_mapped = DataFrameMapper().map_dataframe(
            df_preprocessed,
            mapper_service,
            title_col="section_title",
            num_col="section_numbering",
        )
        df_mapped.to_csv(self.paths.mapped_csv, index=False, encoding="utf-8")
        self.logger.info("Section mapping finished.")
        return df_mapped

    # ----------------------------- Step 5: Analysis with OpenAI -----------------------------

    def analyze_with_openai(self, df_mapped: pd.DataFrame) -> pd.DataFrame:
        """Runs analysis of sections (summary/scores) using OpenAI."""
        # agregacja (opcjonalnie możesz użyć df_agg w raporcie)
        
        df_mapped_concat = concat_section_text_by_mapped_category(df_mapped)
        self.logger.info("Starting openAI calls.")
        prompt_file = self.openai_cfg.prompt_file or self.paths.prompt_path
        df_analyzed = run_openai_analysis(
            df=df_mapped_concat,
            prompt_file=str(prompt_file),
            model=self.openai_cfg.model,
            jsonl_path=str(self.paths.openai_jsonl),
        )
        df_analyzed.to_csv(self.paths.openai_csv, index=False, encoding="utf-8")
        self.logger.info("Analysis with OpenAI finished.")
        return df_analyzed

    # ------------------------------- Step 6: Gen PDF -------------------------------

    def generate_report(self, df_mapped: pd.DataFrame) -> Path:
        """Builds section-to-canonical mapping and generates the PDF report."""
        section_mapping = build_mapping(df_mapped, CANONICAL_SECTIONS)

        gen = ReportGenerator(cfg=ReportConfig(title="Game Design Document Analyzer — Final Report"))
        gen.generate(
            csv_path=self.paths.openai_csv,
            out_pdf=self.paths.report_pdf,
            section_mapping=section_mapping,
        )
        self.logger.info("PDF report generated: %s", self.paths.report_pdf)
        return self.paths.report_pdf

    # ----------------------------------- RUN ALL --------------------------------------

    def run(self) -> None:
        """Executes the full pipeline with basic error handling."""
        self.logger.info("Starting processing: %s", self.paths.pdf_path)
        try:
            section_specs = self.extract_section_specs()
            df_assigned = self.inspect_pdf(section_specs)
            df_pre = self.preprocess(df_assigned)
            df_mapped = self.map_sections(df_pre)
            df_analyzed = self.analyze_with_openai(df_mapped)
            self.generate_report(df_mapped)
        except Exception as exc:  # noqa: BLE001 (tu wystarczający handler „na końcu”)
            self.logger.exception("Pipeline error: %s", exc)
            raise




if __name__ == "__main__":

    out_dir = Path("gdd_analyzer/output")
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Path config
    paths = PathsConfig(
        pdf_path=Path("gdd_analyzer/input/Project_Vanishing_1.pdf"),
        logs_path=Path("gdd_analyzer/output/logs/pipeline.log"),
        prompt_path=Path("gdd_analyzer/prompts/prompt_ollama.txt"),
        # OUT
        openai_csv=Path("gdd_analyzer/output/final_output_openaiNano.csv"),
        report_pdf=Path("gdd_analyzer/output/GDD_Final_Report.pdf"),
    )

    pipeline = GddAnalyzerPipeline(
        paths=paths,
        toc_cfg=TocConfig(pages=(2, 3), drop_first_item=True),
        inspector_cfg=InspectorConfig(y_tol=2.0, space_gap_pt=None, min_header_score=78, content_min_page=4),
        ollama_cfg=OllamaConfig(
            host="http://127.0.0.1:11434",
            model_name="MHKetbi/Mistral-Small3.1-24B-Instruct-2503:q5_K_L",
            temperature=0.1,
            strategy="strict",
            min_similarity=88,
            batch_size=40,
            retries=2,
            backoff_seconds=1.5,
        ),
        openai_cfg=OpenAIConfig(
            model="gpt-5-nano",
            prompt_file=Path("gdd_analyzer/prompts/pormpt_openai.txt"),  # your openai prompt
        ),
    )

    # extract_keywords 
    # forgot add TODO

    pipeline.run()


