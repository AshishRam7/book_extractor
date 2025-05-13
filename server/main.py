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

STATIC_DIR = Path("static") # Keep if you plan other static backend assets
# API Timeouts and Polling
DATALAB_POST_TIMEOUT = 120 # Increased timeout for potentially larger files/slower API
DATALAB_POLL_TIMEOUT = 60  # Increased poll timeout
MAX_POLLS = 300
POLL_INTERVAL = 5 # Slightly increased poll interval
CHAPTER_EXTRACTION_LEVELS = [1, 2, 3, 4] 
MIN_CHAPTER_CONTENT_LINES = 0 
MIN_CHAPTER_CONTENT_WORDS = 0 

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

for dir_path in [TEMP_UPLOAD_DIR, PROCESSED_MARKDOWN_DIR, EXTRACTED_IMAGES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

job_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
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
    if not range_str.strip():
        return list(range(total_pages)) 

    indices = set()
    parts = range_str.split(',')
    try:
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                start_idx = 0 if not start_str else int(start_str) - 1
                if not end_str: 
                    end_idx = total_pages - 1
                else: 
                    end_idx = int(end_str) - 1
                
                if start_idx < 0: start_idx = 0
                if end_idx >= total_pages: end_idx = total_pages - 1
                
                if start_idx > end_idx: 
                    logger.warning(f"Invalid page range part (start > end): '{part}'")
                    continue 
                for i in range(start_idx, end_idx + 1):
                    indices.add(i)
            else:
                idx = int(part) - 1
                if 0 <= idx < total_pages:
                    indices.add(idx)
                else:
                    logger.warning(f"Page number {int(part)} out of range (1-{total_pages}).")
        return sorted(list(indices)) if indices else None
    except ValueError:
        logger.error(f"Invalid page range format: '{range_str}'")
        return None

def process_pdf_pages(job_id: str, original_pdf_path: Path, page_range_str: Optional[str]) -> Path:
    if not page_range_str or not page_range_str.strip():
        logger.info(f"[{job_id}] No page range specified for {original_pdf_path.name}, using original PDF.")
        return original_pdf_path
    try:
        reader = pypdf.PdfReader(str(original_pdf_path))
        total_pages = len(reader.pages)
        selected_indices = parse_page_ranges_to_indices(page_range_str, total_pages)

        if selected_indices is None:
            logger.warning(f"[{job_id}] Failed to parse page range '{page_range_str}' or no valid pages for {original_pdf_path.name}. Using original PDF.")
            return original_pdf_path
        
        if len(selected_indices) == total_pages:
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
        clipped_pdf_path = clipped_pdf_dir / f"clipped_{original_pdf_path.stem}_{uuid.uuid4().hex[:8]}.pdf" 
        
        with open(clipped_pdf_path, "wb") as f:
            writer.write(f)
        logger.info(f"[{job_id}] Clipped PDF '{original_pdf_path.name}' to {len(selected_indices)} pages (indices: {selected_indices}). Saved to {clipped_pdf_path}")
        return clipped_pdf_path
    except Exception as e:
        logger.error(f"[{job_id}] Error processing PDF pages for {original_pdf_path.name} with range '{page_range_str}': {e}. Using original PDF.", exc_info=True)
        return original_pdf_path

# --- Helper Functions ---
def call_datalab_marker(file_path: Path) -> Dict:
    logger.info(f"Calling Datalab Marker API for {file_path.name}...")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        form_data = {
            "output_format": (None, "markdown"),
            "disable_image_extraction": (None, "false") 
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
            error_content = e.response.text if e.response else "No response content"
            logger.error(f"Datalab API error content: {error_content}")
            raise Exception(f"Datalab API request failed: {e} - {error_content}")

    if not data.get("success"):
        err_msg = data.get('error', 'Unknown Datalab error')
        logger.error(f"Datalab API error for {file_path.name}: {err_msg}")
        raise Exception(f"Datalab API error: {err_msg}")

    check_url = data["request_check_url"]
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("status") == "complete":
                logger.info(f"Datalab processing complete for {file_path.name}.")
                return {
                    "markdown": poll_data.get("markdown", ""),
                    "images": poll_data.get("images", {})
                }
            if poll_data.get("status") == "error":
                err_msg = poll_data.get('error', 'Unknown Datalab processing error')
                logger.error(f"Datalab processing failed for {file_path.name}: {err_msg}")
                raise Exception(f"Datalab processing failed: {err_msg}")
            logger.info(f"Polling Datalab for {file_path.name}: status {poll_data.get('status')}, attempt {i+1}/{MAX_POLLS}")
        except requests.exceptions.Timeout:
            logger.warning(f"Polling Datalab timed out for {file_path.name}. Retrying...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Polling Datalab error for {file_path.name}: {e}. Retrying...")
    raise TimeoutError(f"Polling timed out for Datalab processing of {file_path.name}.")

def save_extracted_images(images_dict: Dict[str, str], images_folder: Path, job_id: str, doc_safe_name: str) -> Tuple[Dict[str, str], List[str]]:
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files_map = {} 
    web_image_urls_list = [] 
    
    for img_original_name_in_md, b64_data in images_dict.items():
        try:
            base_name, suffix = Path(img_original_name_in_md).stem, Path(img_original_name_in_md).suffix
            if not suffix: suffix = ".png" 
            safe_img_filename_base = "".join([c for c in base_name if c.isalnum() or c in ('-', '_')]).strip() or f"image_{uuid.uuid4().hex[:8]}"
            counter = 0
            safe_img_filename_on_disk = f"{safe_img_filename_base}{suffix}"
            image_path_on_disk_obj = images_folder / safe_img_filename_on_disk
            
            while image_path_on_disk_obj.exists():
                counter += 1
                safe_img_filename_on_disk = f"{safe_img_filename_base}_{counter}{suffix}"
                image_path_on_disk_obj = images_folder / safe_img_filename_on_disk
            
            image_data = base64.b64decode(b64_data)
            with open(image_path_on_disk_obj, "wb") as img_file:
                img_file.write(image_data)
            
            web_url_for_md_and_list = f"/extracted_images/{job_id}/{doc_safe_name}/{safe_img_filename_on_disk}"
            saved_files_map[img_original_name_in_md] = web_url_for_md_and_list
            web_image_urls_list.append(web_url_for_md_and_list)
            logger.info(f"Saved image '{img_original_name_in_md}' to '{image_path_on_disk_obj}' (Web URL: '{web_url_for_md_and_list}')")
        except Exception as e:
            logger.warning(f"Could not save image '{img_original_name_in_md}': {e}", exc_info=True)
    return saved_files_map, web_image_urls_list

def rewrite_markdown_image_paths(markdown_text: str, original_name_to_web_url_map: Dict[str, str]) -> str:
    figure_pattern = re.compile(r"(!\[.*?\]\()([^)]+)(\))")
    def replace_path(match):
        alt_text_and_opening_paren = match.group(1)
        original_path_in_md_encoded_or_decoded = match.group(2) 
        closing_paren = match.group(3)
        path_decoded = urllib.parse.unquote(original_path_in_md_encoded_or_decoded)
        new_web_url = original_name_to_web_url_map.get(path_decoded)
        if not new_web_url:
            new_web_url = original_name_to_web_url_map.get(original_path_in_md_encoded_or_decoded)
        if new_web_url:
            logger.debug(f"Rewriting MD image path: '{original_path_in_md_encoded_or_decoded}' -> '{new_web_url}'")
            return f"{alt_text_and_opening_paren}{new_web_url}{closing_paren}"
        else:
            logger.warning(f"No mapping found for MD image path: '{original_path_in_md_encoded_or_decoded}'. Leaving original.")
            return match.group(0) 
    return figure_pattern.sub(replace_path, markdown_text)

def _clean_markdown_title(title: str) -> str:
    title = re.sub(r'\[([^\[\]]*)\]\(.*?\)', r'\1', title)
    title = re.sub(r'`(.*?)`', r'\1', title)
    title = re.sub(r'(\*\*|__)(.*?)(\1)', r'\2', title)
    title = re.sub(r'(\*|_)(.*?)(\1)', r'\2', title)
    title = ' '.join(title.strip().split())
    return title

def extract_chapters_from_markdown(markdown_text: str, chapter_levels: List[int] = [1, 2]) -> Dict[str, str]:
    if not markdown_text.strip(): return {}
    unique_sorted_levels = sorted(list(set(chapter_levels)))
    header_patterns_str = [f"^#{{{level}}}\\s+" for level in unique_sorted_levels]
    split_pattern_regex = re.compile("(?=" + "|".join(header_patterns_str) + ")", flags=re.MULTILINE)
    parts = split_pattern_regex.split(markdown_text)
    raw_chapters: Dict[str, str] = {}

    for i, part in enumerate(parts):
        part_stripped = part.strip()
        if not part_stripped: continue
        lines = part_stripped.splitlines()
        first_line_stripped = lines[0].strip()
        is_header_part = False
        extracted_title_from_header = ""
        for level in unique_sorted_levels:
            header_prefix = "#" * level + " "
            if first_line_stripped.startswith(header_prefix):
                extracted_title_from_header = _clean_markdown_title(first_line_stripped[len(header_prefix):].strip())
                is_header_part = True
                break
        if is_header_part and extracted_title_from_header:
            title_key = extracted_title_from_header
            counter = 1
            original_title_key_base = title_key
            while title_key in raw_chapters:
                counter += 1
                title_key = f"{original_title_key_base} (part {counter})"
            raw_chapters[title_key] = part_stripped 
        elif not raw_chapters and not is_header_part and len(parts) == 1: 
            title_key = "Full Document"
            raw_chapters[title_key] = part_stripped
            
    final_chapters = {}
    common_toc_titles = ["contents", "table of contents", "index"] 
    for title, content_with_header in raw_chapters.items():
        cleaned_title_lower = title.lower().strip()
        if any(toc_keyword in cleaned_title_lower for toc_keyword in common_toc_titles):
            content_lines_for_toc_check = content_with_header.splitlines()
            if len(content_lines_for_toc_check) <= (MIN_CHAPTER_CONTENT_LINES + 1) and len(raw_chapters) > 1:
                logger.info(f"Filtering out suspected ToC/Index titled: '{title}' due to short content and other chapters existing.")
                continue
        
        content_lines_after_header = content_with_header.splitlines()
        actual_content_text = ""
        if len(content_lines_after_header) > 1:
            actual_content_text = "\n".join(content_lines_after_header[1:]).strip()
        num_actual_content_lines = len(actual_content_text.splitlines())
        num_actual_content_words = len(actual_content_text.split())

        if (num_actual_content_lines < MIN_CHAPTER_CONTENT_LINES or \
            num_actual_content_words < MIN_CHAPTER_CONTENT_WORDS):
            if len(raw_chapters) > 1: 
                logger.info(f"Filtering out short chapter: '{title}' (actual content lines: {num_actual_content_lines}, words: {num_actual_content_words})")
                continue
        final_chapters[title] = content_with_header 
    if not final_chapters and markdown_text.strip():
        logger.info("No chapters extracted after filtering, or no headers found. Using 'Full Document' as fallback.")
        final_chapters["Full Document"] = markdown_text.strip()
    return final_chapters

def cleanup_job_files(job_id: str):
    logger.info(f"[{job_id}] Cleaning up ALL temporary files and directories for job...")
    for dir_to_remove in [TEMP_UPLOAD_DIR / job_id, PROCESSED_MARKDOWN_DIR / job_id, EXTRACTED_IMAGES_DIR / job_id]:
        if dir_to_remove.exists():
            try: 
                shutil.rmtree(dir_to_remove)
                logger.info(f"Removed {dir_to_remove}")
            except Exception as e: 
                logger.warning(f"Error deleting {dir_to_remove}: {e}")
    logger.info(f"[{job_id}] Temporary file cleanup finished for job {job_id}.")

# --- Background Task Functions ---
def run_chapter_extraction_job(job_id: str, file_processing_details: List[Dict[str, Any]]):
    logger.info(f"[{job_id}] Background chapter extraction job started for {len(file_processing_details)} file(s).")
    job_storage[job_id]["status"] = "processing"
    job_storage[job_id]["message"] = "Starting document processing..."
    job_result_data = ChapterExtractionJobResultData()
    try:
        for i, file_detail in enumerate(file_processing_details):
            original_filename = file_detail["original_filename"]
            original_temp_disk_path = Path(file_detail["temp_disk_path"])
            page_range_str = file_detail["page_range_str"]
            job_storage[job_id]["message"] = f"Processing file {i+1}/{len(file_processing_details)}: {original_filename} (Range: {page_range_str or 'All'})..."
            pdf_to_process_path = process_pdf_pages(job_id, original_temp_disk_path, page_range_str)
            safe_doc_name_base = Path(original_filename).stem
            safe_doc_name = "".join([c for c in safe_doc_name_base if c.isalnum() or c in ('-', '_')]).strip() or f"doc_{uuid.uuid4().hex[:8]}"
            doc_result_item = ChapterExtractionJobDocResult(original_filename=original_filename, chapters={}, image_urls=[])
            try:
                datalab_result = call_datalab_marker(pdf_to_process_path)
                raw_markdown_from_datalab = datalab_result.get("markdown", "")
                images_dict_from_datalab = datalab_result.get("images", {})
                doc_specific_images_dir = EXTRACTED_IMAGES_DIR / job_id / safe_doc_name
                saved_images_map, web_image_urls_for_doc = save_extracted_images(
                    images_dict_from_datalab, doc_specific_images_dir, job_id, safe_doc_name
                )
                doc_result_item.image_urls = web_image_urls_for_doc
                processed_markdown = rewrite_markdown_image_paths(raw_markdown_from_datalab, saved_images_map)
                doc_specific_markdown_dir = PROCESSED_MARKDOWN_DIR / job_id
                doc_specific_markdown_dir.mkdir(parents=True, exist_ok=True)
                processed_markdown_filename = f"{safe_doc_name}_processed.md"
                processed_markdown_path_on_disk = doc_specific_markdown_dir / processed_markdown_filename
                processed_markdown_path_on_disk.write_text(processed_markdown, encoding="utf-8")
                doc_result_item.processed_markdown_file_url = f"/processed_markdown_files/{job_id}/{processed_markdown_filename}"
                chapters_content_map = extract_chapters_from_markdown(processed_markdown, CHAPTER_EXTRACTION_LEVELS)
                doc_result_item.chapters = chapters_content_map
                for title in chapters_content_map.keys():
                    job_result_data.all_available_chapters.append(
                        DocumentChapterInfo(doc_filename=original_filename, chapter_title=title)
                    )
                logger.info(f"[{job_id}] Successfully processed {original_filename}. Found {len(chapters_content_map)} chapters.")
            except Exception as e:
                error_message = f"Error processing document {original_filename}: {str(e)}"
                logger.error(f"[{job_id}] {error_message}", exc_info=True)
                doc_result_item.chapters = {"Error processing this document": error_message}
            job_result_data.documents.append(doc_result_item)
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["message"] = "All documents processed. Chapter extraction complete."
        job_storage[job_id]["result"] = job_result_data.model_dump() 
        logger.info(f"[{job_id}] Chapter extraction job completed successfully.")
    except Exception as e:
        logger.exception(f"[{job_id}] Critical job failure during chapter extraction process: {e}")
        job_storage[job_id]["status"] = "error"
        job_storage[job_id]["message"] = f"An unexpected error occurred during the job: {str(e)}"
        if "result" in job_storage[job_id] : job_storage[job_id]["result"] = None

# --- FastAPI Endpoints ---
@app.post("/extract-chapters", response_model=ChapterExtractionJob)
async def start_chapter_extraction(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    page_ranges: Optional[List[str]] = Form(None)
):
    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Received /extract-chapters for {len(files)} file(s). Page ranges form data: {page_ranges}")

    if not files: 
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    if page_ranges is not None and not isinstance(page_ranges, list):
        if len(files) == 1 and isinstance(page_ranges, str):
            page_ranges = [page_ranges]
        else:
             raise HTTPException(status_code=400, detail=f"Page ranges format error. Expected a list of ranges, got {type(page_ranges)}.")

    if page_ranges and len(page_ranges) != len(files):
        raise HTTPException(status_code=400, detail=f"Mismatch: {len(files)} files uploaded but {len(page_ranges)} page range strings provided.")

    file_processing_details_for_task = []
    job_upload_dir = TEMP_UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_storage[job_id] = {"status": "pending", "message": "Validating and saving files...", "result": None}

    try:
        for i, file in enumerate(files):
            if not file.filename: 
                logger.warning(f"[{job_id}] Skipping file with no filename at index {i}.")
                continue
            if not file.filename.lower().endswith(".pdf"):
                logger.warning(f"[{job_id}] Skipping non-PDF file: {file.filename}")
                continue
            
            safe_fn_stem = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in ('-','_','.')) or f"file_{i}_{uuid.uuid4().hex[:4]}"
            temp_file_path = job_upload_dir / f"{safe_fn_stem}.pdf"

            try:
                # MODIFIED: Correctly read and write UploadFile content
                file_content = await file.read() 
                with temp_file_path.open("wb") as buffer:
                    buffer.write(file_content)
                
                current_page_range = None
                if page_ranges and i < len(page_ranges):
                    current_page_range = page_ranges[i]

                file_processing_details_for_task.append({
                    "original_filename": file.filename,
                    "temp_disk_path": str(temp_file_path),
                    "page_range_str": current_page_range
                })
                logger.info(f"[{job_id}] Saved '{file.filename}' to '{temp_file_path}' with page range '{current_page_range or 'All'}'.")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to save uploaded file {file.filename}: {e}", exc_info=True)
                if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
                job_storage.pop(job_id, None) 
                # MODIFIED: Added more error detail to the HTTP Exception
                raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename} to server. Error: {str(e)}")
            finally:
                await file.close()
        
        if not file_processing_details_for_task:
            logger.warning(f"[{job_id}] No valid PDF files were processed after initial validation.")
            if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
            job_storage.pop(job_id, None)
            raise HTTPException(status_code=400, detail="No valid PDF files found in upload.")

        background_tasks.add_task(run_chapter_extraction_job, job_id=job_id, file_processing_details=file_processing_details_for_task)
        job_storage[job_id]["status"] = "queued"
        job_storage[job_id]["message"] = f"Job queued for {len(file_processing_details_for_task)} PDF(s)."
        logger.info(f"[{job_id}] Job successfully queued.")
        return ChapterExtractionJob(job_id=job_id, status="queued", message=job_storage[job_id]["message"])

    except HTTPException as http_exc: 
        if job_id in job_storage and job_storage[job_id]["status"] == "pending":
             if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
             job_storage.pop(job_id, None)
        logger.error(f"[{job_id}] HTTPException during job submission: {http_exc.detail}")
        raise http_exc
    except Exception as e: 
        logger.exception(f"[{job_id}] Unexpected error during job submission process for job {job_id}")
        if job_id in job_storage and job_storage[job_id]["status"] == "pending":
            if job_upload_dir.exists(): shutil.rmtree(job_upload_dir)
            job_storage.pop(job_id, None)
        raise HTTPException(status_code=500, detail=f"Internal server error while starting job: {str(e)}")

@app.post("/get-selected-chapters/{job_id}", response_model=SelectedChaptersResponse)
async def get_selected_chapters_content(job_id: str, request_data: SelectedChaptersRequest):
    logger.info(f"[{job_id}] Received /get-selected-chapters request for {len(request_data.chapters_to_retrieve)} chapters.")
    job_data = job_storage.get(job_id)
    if not job_data:
        logger.warning(f"[{job_id}] Job ID not found for /get-selected-chapters.")
        raise HTTPException(status_code=404, detail="Job not found.")
    if job_data.get("status") != "completed":
        logger.warning(f"[{job_id}] Job status is '{job_data.get('status')}', not 'completed'.")
        raise HTTPException(status_code=400, detail=f"Job is not yet completed. Current status: {job_data.get('status')}")
    job_result_dict = job_data.get("result")
    if not job_result_dict:
        logger.error(f"[{job_id}] Job result data is missing for a completed job.")
        raise HTTPException(status_code=500, detail="Job result data missing for completed job.")
    try:
        job_result = ChapterExtractionJobResultData(**job_result_dict)
    except Exception as e:
        logger.error(f"[{job_id}] Error parsing stored job result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error parsing job result data: {str(e)}")

    selected_md_parts = []
    found_count = 0
    not_found_details = []
    for req_chap_info in request_data.chapters_to_retrieve:
        found_specific_chapter = False
        for doc_res in job_result.documents:
            if doc_res.original_filename == req_chap_info.doc_filename:
                if req_chap_info.chapter_title in doc_res.chapters:
                    chapter_header = f"# From Document: {doc_res.original_filename}\n## Chapter: {req_chap_info.chapter_title}\n\n"
                    selected_md_parts.append(chapter_header + doc_res.chapters[req_chap_info.chapter_title])
                    found_count += 1
                    found_specific_chapter = True
                    break 
        if not found_specific_chapter:
            not_found_details.append(f"Chapter '{req_chap_info.chapter_title}' from document '{req_chap_info.doc_filename}'")
    if not_found_details:
        logger.warning(f"[{job_id}] Could not find the following requested chapters: {'; '.join(not_found_details)}")
    message = f"Retrieved content for {found_count} out of {len(request_data.chapters_to_retrieve)} requested chapters."
    if not_found_details:
        message += f" Not found: {'; '.join(not_found_details)}."
    logger.info(f"[{job_id}] {message}")
    final_markdown_content = "\n\n---\n\n".join(selected_md_parts).strip()
    if not final_markdown_content and request_data.chapters_to_retrieve:
        final_markdown_content = "*No content found for the selected chapters.*"
        if not found_count: message = "No content could be retrieved for any of the selected chapters."
    return SelectedChaptersResponse(job_id=job_id, selected_markdown_content=final_markdown_content, message=message)

@app.get("/status/{job_id}", response_model=ChapterExtractionJob)
async def get_job_status(job_id: str):
    logger.debug(f"Status request for job ID: {job_id}")
    job_info = job_storage.get(job_id)
    if not job_info:
        logger.warning(f"Status request for unknown job ID: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    parsed_result_data = None
    if job_info.get("status") == "completed" and job_info.get("result"):
        if isinstance(job_info["result"], dict):
            try:
                parsed_result_data = ChapterExtractionJobResultData(**job_info["result"])
            except Exception as e:
                logger.error(f"[{job_id}] Error parsing job result data for status: {e}", exc_info=True)
                job_info["message"] = (job_info.get("message", "") + " [Warning: Result data parsing issue on server.]").strip()
                parsed_result_data = None 
        else: 
             logger.error(f"[{job_id}] Job result is not a dictionary for completed job. Type: {type(job_info['result'])}")
             parsed_result_data = None
    return ChapterExtractionJob(
        job_id=job_id,
        status=job_info.get("status", "unknown"),
        message=job_info.get("message"),
        result=parsed_result_data
    )

@app.delete("/job/{job_id}", status_code=200)
async def delete_job_data(job_id: str):
    logger.info(f"[{job_id}] Received request to delete job.")
    if job_id not in job_storage:
        logger.warning(f"[{job_id}] Attempted to delete non-existent job.")
        raise HTTPException(status_code=404, detail="Job not found.")
    cleanup_job_files(job_id)
    del job_storage[job_id]
    logger.info(f"[{job_id}] Job data and associated files deleted successfully.")
    return {"job_id": job_id, "message": f"Job {job_id} and associated files deleted successfully."}

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "ok", "message": "API is running."}

if __name__ == "__main__":
    print("To run this FastAPI application, use Uvicorn:")
    print("Example: uvicorn server.main:app --reload --host 0.0.0.0 --port 8001")