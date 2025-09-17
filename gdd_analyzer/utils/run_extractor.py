from pathlib import Path
import pandas as pd
from pprint import pprint
from ollama import Client  


from gdd_analyzer.extract.data_extractor import PDFTableofContents, extract_sections, GddPdfInspector
from gdd_analyzer.preprocess.section_preprocessor import NormaliseSchema, CsvPreprocessingService
from gdd_analyzer.mapping.label_mapper import OllamaChatModel, MapperConfig, SectionMapperService, load_prompt, DataFrameMapper, CANONICAL_SECTIONS
from gdd_analyzer.analyzer.openai_analyzer import concat_section_text_by_mapped_category, run_openai_analysis
from gdd_analyzer.report.gdd_report import ReportConfig, ReportGenerator, build_mapping

from gdd_analyzer.interfaces.log_handlers.console_handler import get_console_handler
from gdd_analyzer.interfaces.log_handlers.file_handler import get_file_handler
from gdd_analyzer.interfaces.log_handlers.setup_logging import setup_logger 

if __name__ == "__main__":
    pdf_path = r'gdd_analyzer\input\Project_Vanishing_1.pdf'
    log_path = Path("gdd_analyzer/output/logs/my_prod_label_logs_analiza.txt")

    logger = setup_logger(name = "review-analyzer", handlers=[get_console_handler('INFO'), get_file_handler(str(log_path), 'DEBUG')])
    logger.info("Start przetwarzania…")
    logger.info(pdf_path)

    toc_pages = [2, 3]  # przykładowo strony z spisem treści

    extractor = PDFTableofContents(pdf_path)
    spis_tresci = extractor.extract_pages(toc_pages).strip()

    #print("===== Spis tresci =====")
    #print(spis_tresci)

    result = extract_sections(spis_tresci)
    del result[0]
    pprint(result, sort_dicts=False, width=180)

    logger.info(result) # TRZEBA DODAC DO GENEROWANIA ANALIZY 


    inspector = GddPdfInspector(y_tol=2.0, space_gap_pt=None, min_header_score=78)
    report = inspector.inspect_document(pdf_path)

    # 2) Flatten lists with page index (1-based like u Ciebie)
    all_spans = [{"page": i + 1, **span} for i, page in enumerate(report) for span in page["spans"]]
    all_images = [{"page": i + 1, **img} for i, page in enumerate(report) for img in page["images"]]

    df_spans = inspector.spans_to_df(all_spans)
    df_lines = inspector.assemble_lines(df_spans)

    df_lines_filtered = df_lines[df_lines["page"] >= 4]

    df_headers_only = inspector.tag_confident_headers_simple(df_lines_filtered, result)

    print(df_headers_only)

    print(df_headers_only)
    print(df_headers_only[df_headers_only["section_title"].notna()])

    ## TEGP UZYC JUZ NALEZY W PDF PODSUMOWANIU ZEBY POKAZAC CO DO CZEGO POJDZIE ALBO DOPIERO PO PRZEMPAPOWANIU 


    # 6) FFill sekcje i zapis — jak u Ciebie:
    cols = ["section_raw", "section_title", "section_numbering", "section_level", "section_confidence"]
    df_assigned = df_headers_only.copy()
    df_assigned[cols] = df_assigned[cols].ffill()
    df_assigned.to_csv("FinalDataREFACTOR_SOLID.csv", index=False)

    #print(df_assigned)



    NormaliseSchema(["text", "section_raw", "section_title", "section_numbering"]).transform(df_assigned)

    service = CsvPreprocessingService.default()
    out = service.process_dataframe(df_assigned, sort_by_numbering=True)

    print(out)

    #out.to_csv("FinalDataREFACTOR_SOLIDAFTERPREPROCESSING.csv", index=False)
    template = load_prompt(Path("gdd_analyzer\prompts\prompt_ollama.txt"))
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
        out,
        service,
        title_col="section_title",
        num_col="section_numbering",
    )

    # 6) Zapis
    df_mapped.to_csv("FinalData_mappedSOLIDREFACTORSKURWYSYNUI.csv", index=False)
    print(df_mapped.head(10).to_string(index=False))



    ########### TEGO NALEZY UZYC DO PDF RAPROTU WSTEPNIE JAKO ILE ZREDUKOWANO DO ILU
    section_mapping_c = build_mapping(df_mapped, CANONICAL_SECTIONS)
    # Czyli tabelke pokazujaca 
    # Do tej kategorii zamppowano te 
    # Do tej te itd 
    # Podac tez section numbering



    # ANLAIZA OPENAI
    df_mapped_concat = concat_section_text_by_mapped_category(df_mapped)
    df_analyzed = run_openai_analysis(
        df=df_mapped,
        prompt_file="gdd_analyzer\prompts\pormpt_openai.txt",
        model="gpt-5-nano",           # lub "gpt-5-mini"
        jsonl_path="outputs/openai_run.jsonl",
    )
    df_analyzed.to_csv("final_output_openaiNanoSOLIDSKURWYSYNUasdsad.csv", index=False, encoding="utf-8") 


    # Generuj raport
    gen = ReportGenerator(
    cfg=ReportConfig(title="Game Design Document Analyzer — Final Report")
    )
    # Podmień ścieżki na swoje:
    result = gen.generate(
        csv_path=Path("final_output_openaiNano.csv"),
        out_pdf=Path("GDD_Final_Report.pdf"),
        section_mapping=section_mapping_c,  # NOWE: przekazujesz mapowanie
    )



