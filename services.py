import os
import io
import re
import hashlib
import logging
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from textwrap import wrap
from typing import List, Dict, Any

import cv2
import fitz
import numpy as np
import pytesseract
import torch
import torch.nn.functional as F
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import pdfplumber

from config import config
from models import (
    Base,
    Document,
    DepartmentContent,
    ExtractedText,
    DepartmentTable,  # ✅ department_tables model
)
from preprocessor import TextPreprocessor


# ------------------ Logging ------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kmrl_idms")

# ------------------ Thread Pool ------------------ #
executor = ThreadPoolExecutor(max_workers=4)
text_preprocessor = TextPreprocessor(lowercase=False)


# ------------------ Database Service ------------------ #
class DatabaseService:
    """Async SQLAlchemy engine and session factory."""

    def __init__(self) -> None:
        self.engine = create_async_engine(
            config.DATABASE_URL,
            echo=False,
            future=True,
        )
        self.SessionLocal = sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


db_service = DatabaseService()


# ------------------ File Converter ------------------ #
class FileConverter:
    """Convert any uploaded file to a PDF."""

    @staticmethod
    def convert_to_pdf(file_path: str) -> str:
        base, ext = os.path.splitext(file_path)
        ext = ext.lower()
        output_path = f"{base}_converted.pdf"

        if ext in [".pdf"]:
            return file_path

        elif ext in [".docx"]:
            from docx import Document as DocxDocument

            doc = DocxDocument(file_path)
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50

            for para in doc.paragraphs:
                if para.text.strip():
                    c.drawString(50, y, para.text)
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50

            c.save()
            return output_path

        elif ext in [".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50

            for line in lines:
                if line.strip():
                    c.drawString(50, y, line.strip())
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50

            c.save()
            return output_path

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(output_path, "PDF")
            return output_path

        elif ext in [".xls", ".xlsx"]:
            import pandas as pd

            xls = pd.ExcelFile(file_path)
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50

            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"Sheet: {sheet}")
                y -= 20
                c.setFont("Helvetica", 10)

                for row in df.values.tolist():
                    line = " | ".join(map(str, row))
                    c.drawString(50, y, line)
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50

                c.showPage()
                y = height - 50

            c.save()
            return output_path

        else:
            raise ValueError(f"Unsupported file type: {ext}")


# ------------------ Document Processor ------------------ #
class DocumentProcessor:
    """Utilities for document metadata and checksums."""

    def calculate_checksum(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_metadata(self, file_path: str) -> dict:
        doc = fitz.open(file_path)
        metadata = {
            "pages": doc.page_count,
            "file_size": os.path.getsize(file_path),
        }
        doc.close()
        return metadata


doc_processor = DocumentProcessor()


# ------------------ Paragraph Utilities ------------------ #
def split_into_paragraphs(raw_text: str) -> List[str]:
    """
    Smart paragraph splitter:
    - If the text already has blank-line paragraphs (\n\n), use those.
    - Otherwise, group lines based on bullets / headings.
    """
    if "\n\n" in raw_text:
        chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if len(chunks) > 1:
            return chunks

    lines = [ln.rstrip() for ln in raw_text.splitlines() if ln.strip()]
    paragraphs: List[str] = []
    current: List[str] = []

    for line in lines:
        is_bullet = bool(re.match(r"^\s*(?:[\u2022•·\-]|[0-9]+\.)\s+", line))
        is_heading = bool(re.match(r"^[A-Z][A-Za-z0-9 ,()&/-]{3,80}:$", line))

        if (is_bullet or is_heading) and current:
            paragraphs.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    return paragraphs


# ------------------ OCR Service ------------------ #
class OCRService:
    """Combine pdfplumber (structured) and Tesseract OCR (scanned)."""

    def __init__(self, config_obj) -> None:
        self.config = config_obj
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(
            gray,
            getattr(self.config, "PREPROCESS_THRESHOLD", 150),
            255,
            cv2.THRESH_BINARY,
        )
        denoised = cv2.medianBlur(thresh, 3)
        return Image.fromarray(denoised)

    def extract_readable_tables(self, tess_data: dict) -> List[str]:
        from collections import defaultdict

        lines_dict = defaultdict(list)
        n_boxes = len(tess_data["text"])

        for i in range(n_boxes):
            word = tess_data["text"][i].strip()
            if not word:
                continue
            key = (tess_data["block_num"][i], tess_data["line_num"][i])
            lines_dict[key].append(
                {
                    "text": word,
                    "left": tess_data["left"][i],
                    "width": tess_data["width"][i],
                }
            )

        table_lines: List[str] = []
        for _, words in sorted(lines_dict.items()):
            span = max(w["left"] + w["width"] for w in words) - min(
                w["left"] for w in words
            )
            if len(words) > 1 and span > 200:
                table_lines.append(" | ".join([w["text"] for w in words]))

        return table_lines

    def extract_text_and_tables(self, file_path: str) -> List[Dict[str, Any]]:
        extracted_pages: List[Dict[str, Any]] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables() or []
                    raw_text = page.extract_text() or ""

                    if not raw_text.strip() and not tables:
                        doc = fitz.open(file_path)
                        pix = doc[i].get_pixmap(
                            dpi=getattr(self.config, "OCR_DPI", 300)
                        )
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        preprocessed_img = self.preprocess_image(img)

                        lang_config = getattr(
                            self.config,
                            "TESSERACT_CONFIG",
                            "--oem 3 --psm 6 -l mal+eng",
                        )
                        ocr_text = pytesseract.image_to_string(
                            preprocessed_img, config=lang_config
                        )
                        raw_text = raw_text or ocr_text

                        raw_table_data = pytesseract.image_to_data(
                            preprocessed_img,
                            output_type=pytesseract.Output.DICT,
                        )
                        tables = [self.extract_readable_tables(raw_table_data)]

                        doc.close()

                    extracted_pages.append(
                        {
                            "page_number": i + 1,
                            "text": raw_text,
                            "tables": tables,
                            "confidence": 90,
                        }
                    )

        except Exception as e:
            logger.warning(f"pdfplumber failed, fallback to OCR only: {e}")
            extracted_pages = self.extract_text_and_tables_ocr_only(file_path)

        return extracted_pages

    def extract_text_and_tables_ocr_only(self, file_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        extracted_pages: List[Dict[str, Any]] = []

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=getattr(self.config, "OCR_DPI", 300))
                img_bytes = pix.tobytes()
                if not img_bytes:
                    logger.warning(
                        f"Page {page_num + 1} produced empty image. Skipping."
                    )
                    continue

                img = Image.open(io.BytesIO(img_bytes))
                preprocessed_img = self.preprocess_image(img)

                lang_config = getattr(
                    self.config,
                    "TESSERACT_CONFIG",
                    "--oem 3 --psm 6 -l mal+eng",
                )
                raw_text = pytesseract.image_to_string(
                    preprocessed_img, config=lang_config
                )

                if not raw_text.strip():
                    logger.warning(f"Page {page_num + 1} is empty after OCR. Skipping.")
                    continue

                raw_table_data = pytesseract.image_to_data(
                    preprocessed_img,
                    output_type=pytesseract.Output.DICT,
                )
                table_lines = self.extract_readable_tables(raw_table_data)

                extracted_pages.append(
                    {
                        "page_number": page_num + 1,
                        "text": raw_text,
                        "tables": table_lines,
                        "confidence": 90,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                continue

        doc.close()
        return extracted_pages


ocr_service = OCRService(config)


# ------------------ Summarizer ------------------ #
class Summarizer:
    """T5-based abstractive summarizer."""

    def __init__(self, model_name: str = "t5-small", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
    ) -> str:
        if not text.strip():
            return ""

        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


summarizer = Summarizer(model_name="t5-small")


# ------------------ BERT Classifier (RoBERTa) ------------------ #
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "bert_model", "saved_model")


class DepartmentClassifier:
    """Wraps a fine-tuned RoBERTa sequence classifier."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

    def classify_text(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze()

        best_idx = torch.argmax(probs).item()
        final_label = self.label_encoder.inverse_transform([best_idx])[0]
        best_prob = float(probs[best_idx])

        return {
            "final_department": final_label,
            "confidence": best_prob,
            "top_model_preds": [(final_label, best_prob)],
            "kw_scores": {},
        }


classifier = DepartmentClassifier()


# ------------------ TABLE HELPERS ------------------ #
def normalize_table(table) -> List[List[str]]:
    """Normalize pdfplumber / OCR table formats into a clean 2D list of strings."""
    if not table:
        return []

    if isinstance(table, list):
        rows = []
        for row in table:
            if isinstance(row, list):
                rows.append([str(cell).strip() for cell in row])
            else:
                rows.append([str(row).strip()])
        return rows

    return [[str(table).strip()]]


def table_to_text(rows: List[List[str]]) -> str:
    """Convert 2D rows to a plain-text representation for classification."""
    return "\n".join(" | ".join(r) for r in rows if any(c.strip() for c in r))


# --------------- PARAGRAPH vs TABLE FILTER HELPERS ---------------
def _normalize_str(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _para_is_mostly_table(para_norm: str, table_norms: List[str]) -> bool:
    """
    Heuristic to decide if a paragraph is basically table text.
    Used to avoid double-storing the same info as both text and table.
    """
    if not para_norm:
        return False

    for t in table_norms:
        if not t:
            continue

        # Strong containment in either direction
        if para_norm in t or t in para_norm:
            return True

        # Word-overlap heuristic
        p_words = set(para_norm.split())
        t_words = set(t.split())
        if not p_words:
            continue

        overlap = len(p_words & t_words) / len(p_words)
        if overlap >= 0.7:  # threshold can be tuned
            return True

    return False


# ------------------ PDF Generator ------------------ #
class PDFGenerator:
    """Generate department-wise PDFs with (optional) Malayalam font."""

    def __init__(self, config_obj):
        self.config = config_obj
        if self.config.MALAYALAM_FONT_PATH:
            pdfmetrics.registerFont(
                TTFont(
                    self.config.MALAYALAM_FONT_NAME,
                    self.config.MALAYALAM_FONT_PATH,
                )
            )

    def generate_department_pdf(
        self, content: str, department: str, document_id: str
    ) -> str:
        output_path = os.path.join(
            self.config.OUTPUT_DIR, f"{document_id}_{department}.pdf"
        )
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        c.setFont(self.config.MALAYALAM_FONT_NAME or "Helvetica", 12)
        textobject = c.beginText(50, height - 50)
        max_chars = 80

        for line in content.splitlines():
            for wrapped_line in wrap(line, max_chars):
                textobject.textLine(wrapped_line)

        c.drawText(textobject)
        c.showPage()
        c.save()
        return output_path


pdf_generator = PDFGenerator(config)


# ------------------ process_document ------------------ #
async def process_document(file_path: str, document_id: str) -> None:
    """
    Full pipeline:
    - Fetch document record
    - Compute metadata & checksum
    - Extract text/tables per page
    - Split into paragraphs (excluding table rows / table-like chunks)
    - Summarize + classify each paragraph
    - Classify tables separately & store in DepartmentTable
    - Generate per-department PDFs
    - Update document status/approval
    """
    async with db_service.SessionLocal() as session:
        doc: Document | None = None
        try:
            logger.info(f"Processing document: {document_id}")

            # ------------------ Fetch Document ------------------ #
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()

            if not doc:
                logger.error(f"Document {document_id} not found!")
                return

            # ------------------ Metadata & Checksum ------------------ #
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                executor, doc_processor.extract_metadata, file_path
            )
            checksum = await loop.run_in_executor(
                executor, doc_processor.calculate_checksum, file_path
            )

            doc.pages = metadata.get("pages")
            doc.file_size = metadata.get("file_size")
            doc.checksum = checksum
            doc.status = "processing"
            doc.original_path = file_path
            await session.commit()

            # ------------------ OCR / Text Extraction (RAW) ------------------ #
            ocr_results: List[Dict[str, Any]] = await loop.run_in_executor(
                executor, ocr_service.extract_text_and_tables, file_path
            )

            if not ocr_results:
                logger.warning(f"No text extracted from document {document_id}")
                doc.status = "failed"
                await session.commit()
                return

            all_dept_contents: Dict[str, List[str]] = {}
            extracted_entries: List[Any] = []

            # ------------------ Page Loop ------------------ #
            for page in ocr_results:
                page_num = page["page_number"]
                raw_text = page.get("text") or ""
                tables = page.get("tables", [])

                # Store cleaned full-page text
                processed_page_text = text_preprocessor.preprocess(raw_text)

                extracted_entries.append(
                    ExtractedText(
                        document_id=document_id,
                        page_number=page_num,
                        text_content=processed_page_text,
                        confidence=1.0,
                        ocr_engine="tesseract",
                        bbox=page.get("bbox", {}),
                        tables=tables,
                    )
                )

                # ------------------ Paragraph Split (RAW, table-aware) ------------------ #
                raw_text_for_paragraphs = raw_text

                # Precompute normalized table text for this page (for para vs table check)
                page_table_norms: List[str] = []

                if tables:
                    # Build a set of normalized table-row strings (for exact line removal)
                    table_row_keys = set()

                    for table in tables:
                        rows = normalize_table(table)
                        if not rows:
                            continue

                        # Full-table normalized text (for heuristic check)
                        table_text_full = table_to_text(rows)
                        norm_full = re.sub(r"\s+", " ", table_text_full).strip().lower()
                        if norm_full:
                            page_table_norms.append(norm_full)

                        # Row-level keys for strict removal from pdfplumber text
                        for row in rows:
                            row_text = " ".join(str(c) for c in row).strip()
                            if not row_text:
                                continue
                            key = re.sub(r"\s+", " ", row_text).strip().lower()
                            if key:
                                table_row_keys.add(key)

                    filtered_lines: List[str] = []
                    for line in raw_text.splitlines():
                        line_strip = line.strip()
                        if not line_strip:
                            continue

                        line_key = re.sub(r"\s+", " ", line_strip).strip().lower()

                        # 🚫 only skip lines that EXACTLY match a table row
                        if line_key in table_row_keys:
                            continue

                        filtered_lines.append(line_strip)

                    raw_text_for_paragraphs = "\n".join(filtered_lines)

                # Now split into paragraphs using ONLY non-table (or non-row) text
                paragraphs: List[str] = split_into_paragraphs(raw_text_for_paragraphs)

                for para in paragraphs:
                    # Use original para for table similarity check
                    para_norm = _normalize_str(para)

                    # 🚫 Skip paragraphs that are essentially table text
                    if _para_is_mostly_table(para_norm, page_table_norms):
                        logger.info(
                            f"[Doc {document_id} | Page {page_num}] "
                            "Skipping paragraph that matches table content."
                        )
                        continue

                    clean_para = text_preprocessor.preprocess(para)

                    if len(clean_para) < 30:
                        continue

                    # Summarization
                    summary = await loop.run_in_executor(
                        executor, summarizer.summarize, clean_para
                    )
                    summary = summary or clean_para

                    # Classification (TEXT)
                    prediction = await loop.run_in_executor(
                        executor, classifier.classify_text, clean_para
                    )
                    department = prediction["final_department"]

                    logger.info(
                        f"[Doc {document_id} | Page {page_num}] "
                        f"Paragraph → {department}"
                    )

                    # Accumulate TEXT Content
                    all_dept_contents.setdefault(department, [])
                    for line in summary.splitlines():
                        all_dept_contents[department].append(
                            f"{line} [Page {page_num}]"
                        )

                    # Store DepartmentContent (TEXT)
                    extracted_entries.append(
                        DepartmentContent(
                            document_id=document_id,
                            department=department,
                            content=summary,
                            page_start=page_num,
                            page_end=page_num,
                            confidence=prediction["confidence"],
                            keywords_matched=None,
                            pdf_path=None,
                            doc_priority=doc.priority,
                            approved=False,
                        )
                    )

                # ------------------ TABLE CLASSIFICATION (SEPARATE) ------------------ #
                for table in tables:
                    rows = normalize_table(table)
                    if not rows:
                        continue

                    table_text = table_to_text(rows)
                    if not table_text.strip():
                        continue

                    table_pred = await loop.run_in_executor(
                        executor, classifier.classify_text, table_text
                    )
                    table_dept = table_pred["final_department"]

                    logger.info(
                        f"[Doc {document_id} | Page {page_num}] "
                        f"Table → {table_dept}"
                    )

                    # Append table text only to that department (for PDF)
                    all_dept_contents.setdefault(table_dept, [])
                    all_dept_contents[table_dept].append(
                        f"{table_text} [Page {page_num}]"
                    )

                    # Persist in DepartmentTable
                    extracted_entries.append(
                        DepartmentTable(
                            document_id=str(document_id),
                            department=table_dept,
                            page_number=page_num,
                            rows=rows,
                            confidence=table_pred["confidence"],
                        )
                    )

            # ------------------ Bulk Insert ------------------ #
            session.add_all(extracted_entries)
            await session.commit()

            # ------------------ Generate PDFs per Department ------------------ #
            for department, lines in all_dept_contents.items():
                final_text = "\n".join(lines)

                pdf_path = await loop.run_in_executor(
                    executor,
                    pdf_generator.generate_department_pdf,
                    final_text,
                    department,
                    document_id,
                )

                result = await session.execute(
                    select(DepartmentContent).where(
                        and_(
                            DepartmentContent.document_id == document_id,
                            DepartmentContent.department == department,
                        )
                    )
                )

                for entry in result.scalars().all():
                    entry.pdf_path = pdf_path
                    session.add(entry)

                await session.commit()

            # ------------------ Final Document Status ------------------ #
            result = await session.execute(
                select(DepartmentContent).where(
                    DepartmentContent.document_id == document_id
                )
            )
            all_depts = list(result.scalars().all())

            doc.approved = all(d.approved for d in all_depts) if all_depts else False
            doc.status = "completed"
            session.add(doc)
            await session.commit()

            logger.info(f"✅ Document processing completed: {document_id}")

        except Exception as e:
            logger.exception(f"❌ Error processing document {document_id}: {e}")
            if doc:
                doc.status = "failed"
                session.add(doc)
                await session.commit()
            return
