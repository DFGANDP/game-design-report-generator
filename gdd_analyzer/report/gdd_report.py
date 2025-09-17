#!/usr/bin/env python3
# gdd_report.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from xml.sax.saxutils import escape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SCORE_COLS = [
    "clarity_score",
    "completeness_score",
    "innovation_score",
    "feasibility_score",
]


@dataclass(frozen=True)
class ReportConfig:
    title: str = "Game Design Document Analyzer — Final Report"
    pagesize: tuple = A4
    left_margin_cm: float = 1.8
    right_margin_cm: float = 1.8
    top_margin_cm: float = 1.6
    bottom_margin_cm: float = 1.6
    radar_figsize_in: float = 4.0
    radar_image_size_cm: float = 10.0
    overview_col_widths_cm: tuple = (5.0, 9.0)
    cat_table_col_widths_cm: tuple = (5.0, 2.0, 2.5, 2.0, 2.5, 2.0, 2.0)
    mini_avg_col_widths_cm: tuple = (3.2, 2.0, 4.0, 2.0)


class DataLoader:
    @staticmethod
    def load(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for col in SCORE_COLS + ["line_len", "word_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["mapped_category"]).copy()


class StatsAggregator:
    @staticmethod
    def per_category(df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            df.groupby("mapped_category", dropna=False)
            .agg(
                sections=("mapped_category", "count"),
                words=("word_count", "sum"),
                clarity=("clarity_score", "mean"),
                completeness=("completeness_score", "mean"),
                innovation=("innovation_score", "mean"),
                feasibility=("feasibility_score", "mean"),
            )
            .reset_index()
            .sort_values("mapped_category")
        )
        return agg

    @staticmethod
    def overall_means(df: pd.DataFrame) -> Dict[str, float]:
        return {
            "Clarity": float(np.nanmean(df["clarity_score"])),
            "Completeness": float(np.nanmean(df["completeness_score"])),
            "Innovation": float(np.nanmean(df["innovation_score"])),
            "Feasibility": float(np.nanmean(df["feasibility_score"])),
        }


class RadarChartService:
    def __init__(self, figsize_in: float = 4.0) -> None:
        self.figsize_in = float(figsize_in)

    def save(self, values: Dict[str, float], out_path: Path) -> None:
        labels = list(values.keys())
        vals = [0 if (v is None or np.isnan(v)) else float(v) for v in values.values()]
        n = len(labels)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        vals += vals[:1]

        plt.figure(figsize=(self.figsize_in, self.figsize_in))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_ylim(0, 10)
        ax.set_title("Overall Quality Scores (Average)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()


class PdfStyles:
    def __init__(self) -> None:
        styles = getSampleStyleSheet()
        self.base = styles
        self.base.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18, spaceAfter=12))
        self.base.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6))
        self.base.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, textColor=colors.grey))
        self.comment = ParagraphStyle(
            "ScoreComment",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            spaceBefore=0,
            spaceAfter=0,
        )

    def para(self, txt: Optional[str]) -> Paragraph:
        return Paragraph(escape(str(txt or "")), self.comment)


class TableBuilders:
    @staticmethod
    def overview_table(styles: PdfStyles, totals: Dict[str, str], col_widths_cm: tuple[float, float]) -> Table:
        data = [
            ["Total sections", totals["sections"]],
            ["Total words (analyzed)", totals["words"]],
            ["Categories", totals["categories"]],
        ]
        tbl = Table(data, hAlign="LEFT", colWidths=[col_widths_cm[0] * cm, col_widths_cm[1] * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ]
            )
        )
        return tbl

    @staticmethod
    def categories_summary_table(styles: PdfStyles, agg: pd.DataFrame, col_widths_cm: tuple[float, ...]) -> Table:
        header = ["Category", "Sections", "Words", "Clarity", "Completeness", "Innovation", "Feasibility"]
        rows: List[List[str]] = []
        for _, r in agg.iterrows():
            rows.append(
                [
                    r["mapped_category"],
                    int(r["sections"]),
                    f"{int(r['words']):,}" if not pd.isna(r["words"]) else "-",
                    f"{r['clarity']:.1f}" if not pd.isna(r["clarity"]) else "-",
                    f"{r['completeness']:.1f}" if not pd.isna(r["completeness"]) else "-",
                    f"{r['innovation']:.1f}" if not pd.isna(r["innovation"]) else "-",
                    f"{r['feasibility']:.1f}" if not pd.isna(r["feasibility"]) else "-",
                ]
            )
        tbl = Table([header] + rows, repeatRows=1, hAlign="LEFT", colWidths=[w * cm for w in col_widths_cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                ]
            )
        )
        return tbl

    @staticmethod
    def mini_avg_table(styles: PdfStyles, g: pd.DataFrame, col_widths_cm: tuple[float, ...]) -> Table:
        c_avg = g[SCORE_COLS].mean(numeric_only=True)
        data = [
            ["Avg Clarity", fmt_num(c_avg.get("clarity_score")), "Avg Completeness", fmt_num(c_avg.get("completeness_score"))],
            ["Avg Innovation", fmt_num(c_avg.get("innovation_score")), "Avg Feasibility", fmt_num(c_avg.get("feasibility_score"))],
        ]
        tbl = Table(data, hAlign="LEFT", colWidths=[w * cm for w in col_widths_cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ]
            )
        )
        return tbl

    @staticmethod
    def score_table(styles: PdfStyles, row: pd.Series) -> Table:
        tbl = Table(
            [
                ["Clarity",       row.get("clarity_score", "-"),       styles.para(row.get("clarity_comment", ""))],
                ["Completeness",  row.get("completeness_score", "-"),  styles.para(row.get("completeness_comment", ""))],
                ["Innovation",    row.get("innovation_score", "-"),    styles.para(row.get("innovation_comment", ""))],
                ["Feasibility",   row.get("feasibility_score", "-"),   styles.para(row.get("feasibility_comment", ""))],
            ],
            hAlign="LEFT",
            colWidths=[3 * cm, 1.5 * cm, 10 * cm],
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        return tbl


# ---------- NEW: mapping renderer ----------

class SectionMappingRenderer:
    """Renderuje listę scalonych sekcji dla danej kategorii."""

    def __init__(self, styles: PdfStyles) -> None:
        self.styles = styles

    def build_block(self, category: str, mapping: Optional[Dict[str, List[str]]]) -> List[Flowable]:
        if not mapping:
            return []
        items = mapping.get(category) or []
        if not items:
            return []

        flows: List[Flowable] = []
        flows.append(Spacer(1, 0.05 * cm))
        flows.append(Paragraph("<b>Merged sections</b>", self.styles.base["Small"]))

        # Jednokolumnowa tabela z zawijaniem (bez punktorów, schludnie)
        data = [[Paragraph(escape(s), self.styles.base["Small"])] for s in items]
        tbl = Table(data, hAlign="LEFT", colWidths=[14.5 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("LEADING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        flows.append(tbl)
        flows.append(Spacer(1, 0.15 * cm))
        return flows


class PdfReportBuilder:
    def __init__(self, cfg: ReportConfig, styles: Optional[PdfStyles] = None) -> None:
        self.cfg = cfg
        self.styles = styles or PdfStyles()
        self.mapping_renderer = SectionMappingRenderer(self.styles)

    def build(
        self,
        df: pd.DataFrame,
        agg: pd.DataFrame,
        radar_path: Optional[Path],
        out_pdf: Path,
        section_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        doc = SimpleDocTemplate(
            str(out_pdf),
            pagesize=self.cfg.pagesize,
            leftMargin=self.cfg.left_margin_cm * cm,
            rightMargin=self.cfg.right_margin_cm * cm,
            topMargin=self.cfg.top_margin_cm * cm,
            bottomMargin=self.cfg.bottom_margin_cm * cm,
        )

        story: List = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Cover
        story.append(Paragraph(self.cfg.title, self.styles.base["H1"]))
        story.append(Paragraph(f"Generated: {timestamp}", self.styles.base["Small"]))
        story.append(Spacer(1, 0.4 * cm))

        # Overview
        total_sections = int(df.shape[0])
        total_words = int(np.nansum(df["word_count"])) if "word_count" in df.columns and not df["word_count"].dropna().empty else 0
        totals = {"sections": str(total_sections), "words": f"{total_words:,}", "categories": str(agg.shape[0])}
        story.append(Paragraph("<b>Overview</b>", self.styles.base["H2"]))
        story.append(TableBuilders.overview_table(self.styles, totals, col_widths_cm=self.cfg.overview_col_widths_cm))
        story.append(Spacer(1, 0.4 * cm))

        if radar_path and radar_path.exists():
            size = self.cfg.radar_image_size_cm * cm
            story.append(Image(str(radar_path), width=size, height=size))
            story.append(Spacer(1, 0.4 * cm))

        # Category summary
        story.append(Paragraph("<b>Category Score Summary</b>", self.styles.base["H2"]))
        story.append(TableBuilders.categories_summary_table(self.styles, agg, col_widths_cm=self.cfg.cat_table_col_widths_cm))
        story.append(PageBreak())

        # Detailed sections
        grouped = df.groupby("mapped_category", sort=False)
        for cat, g in grouped:
            story.append(Paragraph(cat, self.styles.base["H2"]))
            story.append(TableBuilders.mini_avg_table(self.styles, g, col_widths_cm=self.cfg.mini_avg_col_widths_cm))
            story.append(Spacer(1, 0.2 * cm))

            # NEW: merged sections for this category
            story.extend(self.mapping_renderer.build_block(category=cat, mapping=section_mapping))

            for _, row in g.iterrows():
                story.append(Paragraph("<b>Summary</b>", self.styles.base["Normal"]))
                story.append(Paragraph(str(row.get("summary", ""))[:4000], self.styles.base["Normal"]))
                story.append(Spacer(1, 0.15 * cm))
                story.append(TableBuilders.score_table(self.styles, row))

                oos = str(row.get("out_of_scope_reason", "") or "").strip()
                if oos and oos.lower() != "nan":
                    story.append(Spacer(1, 0.1 * cm))
                    story.append(Paragraph("<b>Out of scope</b>", self.styles.base["Small"]))
                    story.append(Paragraph(oos, self.styles.base["Small"]))

                story.append(Spacer(1, 0.35 * cm))

            story.append(Spacer(1, 0.4 * cm))

        doc.build(story)


class ReportGenerator:
    def __init__(
        self,
        cfg: ReportConfig | None = None,
        loader: Optional[DataLoader] = None,
        aggregator: Optional[StatsAggregator] = None,
        radar: Optional[RadarChartService] = None,
        builder: Optional[PdfReportBuilder] = None,
    ) -> None:
        self.cfg = cfg or ReportConfig()
        self.loader = loader or DataLoader()
        self.aggregator = aggregator or StatsAggregator()
        self.radar = radar or RadarChartService(figsize_in=self.cfg.radar_figsize_in)
        self.builder = builder or PdfReportBuilder(cfg=self.cfg)

    def generate(
        self,
        csv_path: Path,
        out_pdf: Path,
        section_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> dict:
        df = self.loader.load(csv_path)
        agg = self.aggregator.per_category(df)
        means = self.aggregator.overall_means(df)

        radar_path = out_pdf.with_suffix(".radar.png")
        self.radar.save(means, radar_path)

        self.builder.build(
            df=df,
            agg=agg,
            radar_path=radar_path,
            out_pdf=out_pdf,
            section_mapping=section_mapping,
        )

        return {"pdf": str(out_pdf), "radar": str(radar_path), "sections": int(df.shape[0]), "categories": int(agg.shape[0])}


def fmt_num(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{float(v):.1f}"

def build_mapping(
    df: pd.DataFrame,
    canonical_sections: List[str],
    drop_empty: bool = True,
) -> Dict[str, List[str]]:
    """
    Map each canonical section to a list of "numbering title" strings
    gathered from the dataframe in row order.

    Expected df columns: mapped_category, section_numbering, section_title.
    Missing numbering/title are gracefully skipped.

    Args:
        df: Source dataframe.
        canonical_sections: Ordered list of target categories.
        drop_empty: If True, remove categories with no mapped items.

    Returns:
        Dict mapping category -> list[str] like "1.2 Narrative Brief:".
    """
    result: Dict[str, List[str]] = {cat: [] for cat in canonical_sections}

    for row in df.itertuples(index=False):
        cat = getattr(row, "mapped_category", None)
        if cat not in result:
            continue

        num = getattr(row, "section_numbering", "") or ""
        title = getattr(row, "section_title", "") or ""

        num = str(num).strip()
        title = str(title).strip()

        if num and title:
            label = f"{num} {title}"
        elif title:
            label = title
        elif num:
            label = num
        else:
            label = "(missing section title)"

        result[cat].append(label)

    if drop_empty:
        result = {k: v for k, v in result.items() if v}

    return result

if __name__ == "__main__":
    # Przykład użycia:
    gen = ReportGenerator(cfg=ReportConfig(title="Game Design Document Analyzer — Final Report"))

    section_mapping_c = {
        "Intro / Executive Summary": ["1 GAME OBJECTIVES / BREIF", "1.1 Game Brief:", "1.3 Game Objectives:", "3.1 High Reach Brief:"],
        "Gameplay (overview + flow)": ["2 Gameplay", "2.1 Overview:"],
        "Game World & Levels": ["3 Game World", "3.1.1 Locations on High Reach Island"],
        "Graphics / Art Style": ["4 Graphics/ Art Style", "4.1 Storyboard and GUI development", "4.1.2 Storyboard", "6.2.1 Character Combat Animations", "6.4 Character Animation"],
        "Mechanics / Systems": ["3 6.1 Character Mechanics", "5 Mechanics", "5.1 Quest and Dialogue System", "5.1.2 Quest System", "5.1.3 Relationship System", "5.2 Skill trees and leveling up", "5.4 In-Game Economics", "6.2 Character Combat Mechanics"],
        "Characters": ["5.3.1 People of the town", "6 Characters", "6.3 Character Development"],
        "AI / Enemies": ["5.2.1 Enemy Progression", "5.3 AI and pathing", "5.3.2 Guard Patrol Routes", "5.3.3 Creatures of The Wild", "5.3.4 Bandits"],
        "Controls / Interface": ["2.2 Layout:", "4.1.1 GUI Development"],
        "Story / Narrative": ["1.2 Narrative Brief:", "4.2 Example Narrative Development", "4.2.1 Chapter 1: The Great Storm", "5.1.1 Dialogue System", "6.3.1 Character Narrative Development"],
        "Sound / Music": ["4.3 Sounds of High Reach"],
        "Out of scope": ["7 Project Plan", "7.1 Gantt Chart", "7.2 Scrum Reports", "8 References", "9 Bibliography"],
    }

    result = gen.generate(
        csv_path=Path("final_output_openaiNano.csv"),
        out_pdf=Path("GDD_Final_Report.pdf"),
        section_mapping=section_mapping_c,  # NOWE: przekazujesz mapowanie
    )
    print(f"Saved PDF:  {result['pdf']}")
    print(f"Saved radar:{result['radar']}")
