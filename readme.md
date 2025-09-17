### GDD Analyzer — Refactored Package Overview

This project now ships as a modular package `gdd_analyzer` with a single orchestrator you can run to execute the full pipeline: extract → preprocess → map → analyze → report.

The easiest way to understand and run the pipeline is via `gdd_analyzer/utils/runner.py`.

---

## Package Structure (high level)

```text
gdd_analyzer/
  analyzer/                # OpenAI-based analysis
  extract/                 # PDF inspection and ToC extraction
  interfaces/              # logging handlers and shared interfaces
  mapping/                 # Ollama-based section label mapping
  preprocess/              # CSV/section preprocessing utilities
  report/                  # PDF report generator
  utils/
    runner.py              # end-to-end orchestrator (entry point)
  output/                  # default artifacts (csv/jsonl/pdf)
  input/                   # place your PDF(s) here by default
  prompts/                 # mapping/analyzer prompt templates
```

Key coordination lives in `utils/runner.py` using small, focused services from each module.

---

## How the pipeline works (from `utils/runner.py`)

- Reads ToC pages and parses section specs.
- Inspects PDF pages, assembles lines, tags headers, and assigns sections.
- Normalizes and sorts data (preprocess).
- Maps section titles to canonical `CANONICAL_SECTIONS` (Ollama).
- Analyzes concatenated category text with OpenAI (summary + scores).
- Builds section mapping and generates the final PDF report.

Configuration objects you can tweak in one place:
- `PathsConfig`: input PDF path, prompts path, output artifact paths.
- `TocConfig`: which pages contain the table of contents.
- `InspectorConfig`: PDF line assembly and header scoring thresholds.
- `OllamaConfig`: host, model, batching, retries, mapping strategy.
- `OpenAIConfig`: model and optional prompt file.

---

## Setup

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- External services:
  - Ollama for the mapping step (ensure the model exists locally and the host matches your setup)
  - OpenAI for the analysis step (set your API key)

## ADD KEY OPENAI BEFORE RUN
```
$env:OPENAI_API_KEY = "sk-xxxxx"
```

---

## Quickstart

1) Put your input PDF in `gdd_analyzer/input/`, for example `Project_Vanishing_1.pdf`.
2) Adjust paths and configs in `gdd_analyzer/utils/runner.py` if needed (e.g., change `pdf_path`, `prompt_path`, OpenAI model name).
3) Run the orchestrator (PowerShell or cmd), as a module:

```bash
python -m gdd_analyzer.utils.runner
```

Artifacts (by default) will be saved in `gdd_analyzer/output/`:
- `FinalData_spans.csv` — raw spans
- `FinalData_lines.csv` — assembled lines
- `FinalData_assigned.csv` — lines with forward-filled section assignment
- `FinalData_mapped.csv` — canonical mapping results
- `openai_run.jsonl` — OpenAI request/response log
- `final_output_openaiNano.csv` — analyzed categories with scores and summary
- `GDD_Final_Report.pdf` — final PDF report

---

## Minimal config example (from `utils/runner.py`)

```python
paths = PathsConfig(
    pdf_path=Path("gdd_analyzer/input/Project_Vanishing_1.pdf"),
    logs_path=Path("gdd_analyzer/output/logs/pipeline.log"),
    prompt_path=Path("gdd_analyzer/prompts/prompt_ollama.txt"),
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
        prompt_file=Path("gdd_analyzer/prompts/pormpt_openai.txt"),
    ),
)

pipeline.run()
```

---

## Notes

- If your PDFs have different ToC pages or numbering styles, update `TocConfig` and `InspectorConfig` accordingly.
- To switch LLMs:
  - Mapping: change `OllamaConfig.model_name` and ensure the model is available in Ollama.
  - Analysis: change `OpenAIConfig.model` and (optionally) `prompt_file`.
- Logging is configured to output both to console and to `gdd_analyzer/output/logs/pipeline.log`.


## Example pdf output is GDD_Final_Report.pdf 