# -*- coding: utf-8 -*-
import os
import time
import re
import base64
import requests
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import sys
import shutil
import asyncio
import urllib.parse
import json

# PDF Processing
import pypdf # For PDF page extraction

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Load Environment Variables ---
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")

# --- Check Essential Variables ---
if not all([DATALAB_API_KEY, DATALAB_MARKER_URL]):
    missing_vars = [var for var, val in {
        "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
    }.items() if not val]
    logging.critical(f"FATAL ERROR: Missing essential environment variables: {', '.join(missing_vars)}")
    sys.exit("Missing essential environment variables.")

# Directories
TEMP_UPLOAD_DIR = Path("temp_uploads")
PROCESSED_MARKDOWN_DIR = Path("processed_markdown")
EXTRACTED_IMAGES_DIR = Path("extracted_images")
# STATIC_DIR = Path("static") # Keep if you plan other static backend assets

# API Timeouts and Polling
DATALAB_POST_TIMEOUT = 60
DATALAB_POLL_TIMEOUT = 30
MAX_POLLS = 300
POLL_INTERVAL = 3
CHAPTER_EXTRACTION_LEVELS = [1, 2] # H1 and H2 for chapter detection
MIN_CHAPTER_CONTENT_LINES = 3 # Min lines (excluding header) for a chunk to be a chapter
MIN_CHAPTER_CONTENT_WORDS = 20 # Min words (excluding header) for a chunk to be a chapter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="PDF Chapter Extractor API")

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/extracted_images", StaticFiles(directory=EXTRACTED_IMAGES_DIR), name="extracted_images")
app.mount("/processed_markdown_files", StaticFiles(directory=PROCESSED_MARKDOWN_DIR), name="processed_markdown_files")

for dir_path in [TEMP_UPLOAD_DIR, PROCESSED_MARKDOWN_DIR, EXTRACTED_IMAGES_DIR]: # Removed STATIC_DIR and templates if not used
    dir_path.mkdir(parents=True, exist_ok=True)

job_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models (Unchanged) ---
class DocumentChapterInfo(BaseModel):
    doc_filename: str
    chapter_title: str

class ChapterExtractionJobDocResult(BaseModel):
    original_filename: str
    processed_markdown_file_url: Optional[str] = None
    chapters: Dict[str, str]
    image_urls: List[str] = []

class ChapterExtractionJobResultData(BaseModel):
    documents: List[ChapterExtractionJobDocResult] = []
    all_available_chapters: List[DocumentChapterInfo] = []

class ChapterExtractionJob(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    result: Optional[ChapterExtractionJobResultData] = None

class SelectedChaptersRequest(BaseModel):
    chapters_to_retrieve: List[DocumentChapterInfo] = Field(..., description="List of chapters to retrieve, identified by original document filename and chapter title.")

class SelectedChaptersResponse(BaseModel):
    job_id: str
    selected_markdown_content: str
    message: Optional[str] = None


# --- PDF Page Processing Helper ---
def parse_page_ranges_to_indices(range_str: str, total_pages: int) -> Optional[List[int]]:
    """
    Parses a page range string (e.g., "1-3,5,7-") into a list of 0-indexed page numbers.
    Returns None if parsing fails or range is invalid.
    """
    if not range_str.strip():
        return list(range(total_pages)) # Process all pages if range is empty

    indices = set()
    parts = range_str.split(',')
    try:
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                start, end = part.split('-', 1)
                start_idx = 0 if not start else int(start) - 1
                
                if not end: # e.g., "5-"
                    end_idx = total_pages - 1
                else: # e.g., "1-3" or "-3"
                    end_idx = int(end) - 1
                
                if start_idx < 0: start_idx = 0
                if end_idx >= total_pages: end_idx = total_pages - 1
                if start_idx > end_idx: continue # Invalid range part

                for i in range(start_idx, end_idx + 1):
                    indices.add(i)
            else:
                idx = int(part) - 1
                if 0 <= idx < total_pages:
                    indices.add(idx)
        return sorted(list(indices))
    except ValueError:
        logger.error(f"Invalid page range format: '{range_str}'")
        return None


def process_pdf_pages(job_id: str, original_pdf_path: Path, page_range_str: Optional[str]) -> Path:
    """
    Extracts specified pages from a PDF. If page_range_str is None or empty, returns original path.
    Saves the new PDF to a temporary location and returns its path.
    """
    if not page_range_str or not page_range_str.strip():
        logger.info(f"[{job_id}] No page range specified for {original_pdf_path.name}, using original PDF.")
        return original_pdf_path

    try:
        reader = pypdf.PdfReader(str(original_pdf_path))
        total_pages = len(reader.pages)
        
        selected_indices = parse_page_ranges_to_indices(page_range_str, total_pages)

        if selected_indices is None: # Parsing failed
            logger.warning(f"[{job_id}] Failed to parse page range '{page_range_str}' for {original_pdf_path.name}. Using original PDF.")
            return original_pdf_path
        
        if len(selected_indices) == total_pages: # All pages selected
             logger.info(f"[{job_id}] All pages selected for {original_pdf_path.name} via range '{page_range_str}'. Using original PDF.")
             return original_pdf_path

        if not selected_indices:
            logger.warning(f"[{job_id}] Page range '{page_range_str}' resulted in no pages for {original_pdf_path.name}. Using original PDF as fallback.")
            return original_pdf_path


        writer = pypdf.PdfWriter()
        for page_idx in selected_indices:
            writer.add_page(reader.pages[page_idx])

        clipped_pdf_dir = TEMP_UPLOAD_DIR / job_id / "clipped_pdfs"
        clipped_pdf_dir.mkdir(parents=True, exist_ok=True)
        clipped_pdf_path = clipped_pdf_dir / f"clipped_{original_pdf_path.stem}_{uuid.uuid4().hex[:4]}.pdf"
        
        with open(clipped_pdf_path, "wb") as f:
            writer.write(f)
        logger.info(f"[{job_id}] Clipped PDF '{original_pdf_path.name}' to {len(selected_indices)} pages. Saved to {clipped_pdf_path}")
        return clipped_pdf_path
    except Exception as e:
        logger.error(f"[{job_id}] Error processing PDF pages for {original_pdf_path.name} with range '{page_range_str}': {e}. Using original PDF.", exc_info=True)
        return original_pdf_path

# --- Helper Functions (Datalab, Image Saving, MD Rewriting - mostly unchanged) ---
def call_datalab_marker(file_path: Path) -> Dict:
    logger.info(f"Calling Datalab Marker API for {file_path.name}...")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        form_data = {
            "langs": (None, "English"), "force_ocr": (None, False), "paginate": (None, False),
            "output_format": (None, "markdown"), "use_llm": (None, False),
            "strip_existing_ocr": (None, False), "disable_image_extraction": (None, False)
        }
        headers = {"X-Api-Key": DATALAB_API_KEY}
        try:
            response = requests.post(DATALAB_MARKER_URL, files=files, data=form_data, headers=headers, timeout=DATALAB_POST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Datalab API request timed out for {file_path.name}")
            raise TimeoutError("Datalab API request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Datalab API request failed for {file_path.name}: {e}")
            raise Exception(f"Datalab API request failed: {e}")

    if not data.get("success"):
        err_msg = data.get('error', 'Unknown Datalab error')
        logger.error(f"Datalab API error for {file_path.name}: {err_msg}")
        raise Exception(f"Datalab API error: {err_msg}")

    check_url = data["request_check_url"]
    for i in range(MAX_POLLS): # Polling loop
        time.sleep(POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("status") == "complete": return poll_data
            if poll_data.get("status") == "error":
                 err_msg = poll_data.get('error', 'Unknown Datalab processing error')
                 raise Exception(f"Datalab processing failed: {err_msg}")
        except Exception as e: logger.warning(f"Polling Datalab error: {e}. Retrying...")
    raise TimeoutError("Polling timed out for Datalab.")


def save_extracted_images(images_dict: Dict[str, str], images_folder: Path, job_id: str, doc_safe_name: str) -> Tuple[Dict[str, str], List[str]]:
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files_map = {}
    web_image_urls = []
    for img_original_name, b64_data in images_dict.items():
        try:
            base_name = Path(img_original_name).stem
            suffix = Path(img_original_name).suffix or ".png"
            safe_img_filename_base = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in base_name]) or f"image_{uuid.uuid4().hex[:8]}"
            counter = 0
            safe_img_filename = f"{safe_img_filename_base}{suffix}"
            image_path_on_disk = images_folder / safe_img_filename
            while image_path_on_disk.exists():
                counter += 1; safe_img_filename = f"{safe_img_filename_base}_{counter}{suffix}"; image_path_on_disk = images_folder / safe_img_filename
            image_data = base64.b64decode(b64_data)
            with open(image_path_on_disk, "wb") as img_file: img_file.write(image_data)
            saved_files_map[img_original_name] = str(image_path_on_disk)
            web_url = f"/extracted_images/{job_id}/{doc_safe_name}/{safe_img_filename}"
            web_image_urls.append(web_url)
        except Exception as e: logger.warning(f"Could not save image '{img_original_name}': {e}")
    return saved_files_map, web_image_urls

def rewrite_markdown_image_paths(markdown_text: str, original_name_to_web_url_map: Dict[str, str], job_id: str, doc_safe_name: str) -> str:
    figure_pattern = re.compile(r"(!\[.*?\]\()([^\s\)]+)(\))")
    def replace_path(match):
        alt_text_and_opening_paren, original_path_in_md_encoded, closing_paren = match.groups()
        original_path_in_md_decoded = urllib.parse.unquote(original_path_in_md_encoded)
        new_web_url = original_name_to_web_url_map.get(original_path_in_md_decoded) or original_name_to_web_url_map.get(original_path_in_md_encoded)
        return f"{alt_text_and_opening_paren}{new_web_url}{closing_paren}" if new_web_url else match.group(0)
    return figure_pattern.sub(replace_path, markdown_text)

def extract_chapters_from_markdown(markdown_text: str, chapter_levels: List[int] = [1, 2]) -> Dict[str, str]:
    if not markdown_text.strip(): return {}
    
    header_patterns_str = [f"^#{{{level}}}\\s+" for level in sorted(list(set(chapter_levels)))]
    split_pattern_regex = re.compile("(?=" + "|".join(header_patterns_str) + ")", flags=re.MULTILINE)
    
    parts = split_pattern_regex.split(markdown_text)
    raw_chapters = {}

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped: continue
        lines = part_stripped.splitlines()
        first_line_stripped = lines[0].strip()
        
        current_title = ""
        is_header_part = False
        for level in sorted(chapter_levels):
            header_prefix = "#" * level + " "
            if first_line_stripped.startswith(header_prefix):
                current_title = first_line_stripped[len(header_prefix):].strip()
                is_header_part = True
                break
        
        if is_header_part and current_title:
            title_key = current_title
            counter = 1
            while title_key in raw_chapters: # Ensure unique title keys if duplicates exist
                counter += 1; title_key = f"{current_title} (part {counter})"
            raw_chapters[title_key] = part_stripped
        elif not raw_chapters and not is_header_part and len(parts) == 1: # Whole doc, no headers
             raw_chapters["Full Document"] = part_stripped


    # Filter out common non-chapter sections and very short sections
    final_chapters = {}
    common_toc_titles = ["contents", "table of contents"]
    
    for title, content in raw_chapters.items():
        # Filter based on title
        if title.lower() in common_toc_titles:
            # Keep if it's the only "chapter" or if content is substantial (meaning ToC itself is the content)
            # This logic is tricky: if "Contents" is the only H1, it might be the whole useful doc.
            # For now, if "Contents" is a title, we check its content length. If very short, likely just a header.
            content_lines_for_toc_check = content.splitlines()
            if len(content_lines_for_toc_check) <= (MIN_CHAPTER_CONTENT_LINES + 1) and len(raw_chapters) > 1: # +1 for header line
                logger.info(f"Filtering out suspected ToC header: '{title}' due to short content and other chapters existing.")
                continue
        
        # Filter based on content length (after removing header line from content for check)
        content_lines = content.splitlines()
        actual_content_text = "\n".join(content_lines[1:]).strip() # Content after the header
        
        if len(content_lines) < (MIN_CHAPTER_CONTENT_LINES +1) or len(actual_content_text.split()) < MIN_CHAPTER_CONTENT_WORDS:
            if len(raw_chapters) > 1 : # Don't filter if it's the only chunk
                logger.info(f"Filtering out short chapter: '{title}' (lines: {len(content_lines)-1}, words: {len(actual_content_text.split())})")
                continue
        
        final_chapters[title] = content

    if not final_chapters and markdown_text.strip(): # Fallback if all filtered or no headers
        final_chapters["Full Document"] = markdown_text.strip()
        
    return final_chapters

def cleanup_job_files(job_id: str):
    logger.info(f"[{job_id}] Cleaning up ALL temporary files and directories for job...")
    # Remove job-specific temp upload directory (contains original uploads and any clipped PDFs)
    job_temp_upload_dir = TEMP_UPLOAD_DIR / job_id
    if job_temp_upload_dir.exists():
        try: shutil.rmtree(job_temp_upload_dir); logger.info(f"Removed {job_temp_upload_dir}")
        except Exception as e: logger.warning(f"Error deleting {job_temp_upload_dir}: {e}")

    job_markdown_dir = PROCESSED_MARKDOWN_DIR / job_id
    if job_markdown_dir.exists():
        try: shutil.rmtree(job_markdown_dir); logger.info(f"Removed {job_markdown_dir}")
        except Exception as e: logger.warning(f"Error deleting {job_markdown_dir}: {e}")

    job_image_dir = EXTRACTED_IMAGES_DIR / job_id
    if job_image_dir.exists():
        try: shutil.rmtree(job_image_dir); logger.info(f"Removed {job_image_dir}")
        except Exception as e: logger.warning(f"Error deleting {job_image_dir}: {e}")
    logger.info(f"[{job_id}] Temporary file cleanup finished for job {job_id}.")

# --- Background Task Functions ---
def run_chapter_extraction_job(job_id: str, file_processing_details: List[Dict[str, Any]]):
    # file_processing_details: List of {"original_filename": str, "temp_disk_path": Path, "page_range_str": Optional[str]}
    logger.info(f"[{job_id}] Background chapter extraction job started for {len(file_processing_details)} file(s).")
    job_storage[job_id]["status"] = "processing"
    job_storage[job_id]["message"] = "Starting document processing..."

    job_result_data = ChapterExtractionJobResultData()
    # Note: temp_file_paths_on_disk in job_storage is less critical now as cleanup_job_files removes the whole job_id dir from TEMP_UPLOAD_DIR

    try:
        for i, file_detail in enumerate(file_processing_details):
            original_filename = file_detail["original_filename"]
            original_temp_disk_path = Path(file_detail["temp_disk_path"]) # Path to original full upload
            page_range_str = file_detail["page_range_str"]

            job_storage[job_id]["message"] = f"Processing file {i+1}/{len(file_processing_details)}: {original_filename} (Range: {page_range_str or 'All'})..."
            
            # 1. Process PDF pages (clipping)
            # process_pdf_pages returns original_temp_disk_path if no clipping, or path to new clipped PDF
            pdf_to_process_path = process_pdf_pages(job_id, original_temp_disk_path, page_range_str)

            safe_doc_name = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in Path(original_filename).stem]) or f"doc_{uuid.uuid4().hex[:8]}"
            doc_result_item = ChapterExtractionJobDocResult(original_filename=original_filename, chapters={}, image_urls=[])

            try:
                # 2. Call Datalab Marker on the (potentially clipped) PDF
                datalab_result = call_datalab_marker(pdf_to_process_path)
                raw_markdown_from_datalab = datalab_result.get("markdown", "")
                images_dict_from_datalab = datalab_result.get("images", {})

                doc_specific_images_dir = EXTRACTED_IMAGES_DIR / job_id / safe_doc_name
                saved_images_on_disk_map, web_image_urls_for_doc = save_extracted_images(
                    images_dict_from_datalab, doc_specific_images_dir, job_id, safe_doc_name
                )
                doc_result_item.image_urls = web_image_urls_for_doc
                
                original_name_to_web_url_map = {
                    orig_name: f"/extracted_images/{job_id}/{safe_doc_name}/{Path(disk_path).name}"
                    for orig_name, disk_path in saved_images_on_disk_map.items()
                }
                processed_markdown = rewrite_markdown_image_paths(
                    raw_markdown_from_datalab, original_name_to_web_url_map, job_id, safe_doc_name
                )
                
                doc_specific_markdown_dir = PROCESSED_MARKDOWN_DIR / job_id
                doc_specific_markdown_dir.mkdir(parents=True, exist_ok=True)
                processed_markdown_path_on_disk = doc_specific_markdown_dir / f"{safe_doc_name}_processed.md"
                processed_markdown_path_on_disk.write_text(processed_markdown, encoding="utf-8")
                doc_result_item.processed_markdown_file_url = f"/processed_markdown_files/{job_id}/{processed_markdown_path_on_disk.name}"

                chapters_content_map = extract_chapters_from_markdown(processed_markdown, CHAPTER_EXTRACTION_LEVELS)
                doc_result_item.chapters = chapters_content_map

                for title in chapters_content_map.keys():
                    job_result_data.all_available_chapters.append(
                        DocumentChapterInfo(doc_filename=original_filename, chapter_title=title)
                    )
                logger.info(f"[{job_id}] Successfully processed {original_filename}.")

            except Exception as e:
                error_message = f"Error processing document {original_filename}: {e}"
                logger.error(f"[{job_id}] {error_message}", exc_info=True)
                doc_result_item.chapters = {"Error": error_message}
            
            job_result_data.documents.append(doc_result_item)
        
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["message"] = "All documents processed. Chapters identified."
        job_storage[job_id]["result"] = job_result_data.model_dump() 
        logger.info(f"[{job_id}] Chapter extraction job completed.")

    except Exception as e:
        logger.exception(f"[{job_id}] Job failed during chapter extraction: {e}")
        job_storage[job_id]["status"] = "error"
        job_storage[job_id]["message"] = f"An unexpected error occurred: {e}"
        if "result" in job_storage[job_id]: job_storage[job_id]["result"] = None

# --- FastAPI Endpoints ---
@app.post("/extract-chapters", response_model=ChapterExtractionJob)
async def start_chapter_extraction(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    page_ranges: Optional[List[str]] = Form(None) # One range string per file, or None
):
    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Received /extract-chapters for {len(files)} file(s). Page ranges provided: {bool(page_ranges)}")

    if not files: raise HTTPException(status_code=400, detail="No files uploaded.")
    if page_ranges and len(page_ranges) != len(files):
        raise HTTPException(status_code=400, detail="If page_ranges are provided, there must be one range string for each file.")

    file_processing_details_for_task = []
    job_upload_dir = TEMP_UPLOAD_DIR / job_id # Job-specific temp dir for original uploads
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    job_storage[job_id] = {"status": "pending", "message": "Validating and saving files...", "result": None}

    try:
        for i, file in enumerate(files):
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                logger.warning(f"[{job_id}] Skipping non-PDF: {file.filename}")
                continue
            
            safe_fn_base = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in ('-','_','.')) or f"file_{uuid.uuid4().hex[:8]}"
            temp_file_path = job_upload_dir / f"{safe_fn_base}.pdf" # Original full PDF path

            try:
                with temp_file_path.open("wb") as buffer: shutil.copyfileobj(file.file, buffer)
                file_processing_details_for_task.append({
                    "original_filename": file.filename,
                    "temp_disk_path": str(temp_file_path), # Path to the original full uploaded PDF
                    "page_range_str": page_ranges[i] if page_ranges else None
                })
            except Exception as e:
                logger.error(f"Failed to save {file.filename}: {e}")
                # Basic cleanup if one file fails during initial save loop
                if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
                job_storage.pop(job_id, None)
                raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}.")
            finally: await file.close()
        
        if not file_processing_details_for_task:
            if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
            job_storage.pop(job_id, None)
            raise HTTPException(status_code=400, detail="No valid PDF files processed.")

        background_tasks.add_task(run_chapter_extraction_job, job_id=job_id, file_processing_details=file_processing_details_for_task)
        job_storage[job_id]["status"] = "queued"
        job_storage[job_id]["message"] = f"Job queued for {len(file_processing_details_for_task)} PDF(s)."
        return ChapterExtractionJob(job_id=job_id, status="queued", message=job_storage[job_id]["message"])

    except HTTPException as http_exc: # Catch explicitly raised HTTPExceptions
        if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
        job_storage.pop(job_id, None)
        raise http_exc
    except Exception as e: # Catch other unexpected errors
        logger.exception(f"Unexpected error starting job {job_id}")
        if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
        job_storage.pop(job_id, None)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Remaining endpoints (/get-selected-chapters, /status, /delete, /health) are largely the same
# Ensure they correctly handle the job_storage structure and call cleanup_job_files

@app.post("/get-selected-chapters/{job_id}", response_model=SelectedChaptersResponse)
async def get_selected_chapters_content(job_id: str, request_data: SelectedChaptersRequest):
    job_data = job_storage.get(job_id)
    if not job_data: raise HTTPException(status_code=404, detail="Job not found.")
    if job_data.get("status") != "completed": raise HTTPException(status_code=400, detail=f"Job not completed.")
    job_result_dict = job_data.get("result")
    if not job_result_dict: raise HTTPException(status_code=500, detail="Job result data missing.")
    try: job_result = ChapterExtractionJobResultData(**job_result_dict)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error parsing job result: {e}")

    selected_md_parts, found_count = [], 0
    for req_chap_info in request_data.chapters_to_retrieve:
        found = False
        for doc_res in job_result.documents:
            if doc_res.original_filename == req_chap_info.doc_filename and req_chap_info.chapter_title in doc_res.chapters:
                header = f"# From: {doc_res.original_filename}\n## Chapter: {req_chap_info.chapter_title}\n\n"
                selected_md_parts.append(header + doc_res.chapters[req_chap_info.chapter_title])
                found_count += 1; found = True; break
        if not found: selected_md_parts.append(f"\n---\n*Chapter '{req_chap_info.chapter_title}' from '{req_chap_info.doc_filename}' not found.*\n---\n")
    
    msg = f"Retrieved {found_count}/{len(request_data.chapters_to_retrieve)} chapters."
    return SelectedChaptersResponse(job_id=job_id, selected_markdown_content="\n\n---\n\n".join(selected_md_parts).strip(), message=msg)

@app.get("/status/{job_id}", response_model=ChapterExtractionJob)
async def get_job_status(job_id: str):
    job_info = job_storage.get(job_id)
    if not job_info: raise HTTPException(status_code=404, detail="Job not found")
    parsed_result = None
    if isinstance(job_info.get("result"), dict):
        try: parsed_result = ChapterExtractionJobResultData(**job_info["result"])
        except Exception: job_info["message"] = (job_info.get("message","") + " [Result parsing issue]").strip()
    return ChapterExtractionJob(job_id=job_id, status=job_info.get("status", "unknown"), message=job_info.get("message"), result=parsed_result)

@app.delete("/job/{job_id}", status_code=200)
async def delete_job_data(job_id: str):
    if not job_storage.get(job_id): raise HTTPException(status_code=404, detail="Job not found.")
    cleanup_job_files(job_id) 
    del job_storage[job_id]
    logger.info(f"[{job_id}] Job data and files deleted.")
    return {"message": f"Job {job_id} and associated files deleted."}

@app.get("/health")
async def health_check(): return {"status": "ok"}

# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001