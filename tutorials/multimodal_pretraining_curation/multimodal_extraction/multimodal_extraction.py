# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import json
import os
import shutil
from typing import Any, Optional

import fitz  # PyMuPDF
import pandas as pd
import requests
from PIL import Image

# Constants for configuration and paths
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(SCRIPT_DIR_PATH, "source")
EXTRACTION_DIR = os.path.join(SCRIPT_DIR_PATH, "extraction_results")
SEPARATED_MODALITY_DIR = os.path.join(SCRIPT_DIR_PATH, "categorized_results")
NVAI_URL = "https://integrate.api.nvidia.com/v1"
PDF_DPI = 300

# Set your NVIDIA_API_KEY as an environment variable
API_KEY = os.environ.get("NVIDIA_API_KEY")
if not API_KEY:
    error_msg = "NVIDIA_API_KEY environment variable not set. Please set your API key before running this script."
    raise OSError(error_msg)

if os.path.exists(EXTRACTION_DIR):
    shutil.rmtree(EXTRACTION_DIR)
os.makedirs(EXTRACTION_DIR, exist_ok=True)

if os.path.exists(SEPARATED_MODALITY_DIR):
    shutil.rmtree(SEPARATED_MODALITY_DIR)
os.makedirs(SEPARATED_MODALITY_DIR, exist_ok=True)


def call_api(
    model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """
    Make an API call to the NVIDIA AI chat completions endpoint.

    Args:
        model (str): The model identifier to use for the API call (e.g., "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
        messages (list): List of message dictionaries containing the conversation history
        max_tokens (int): Maximum number of tokens to generate in the response
        tools (list, optional): List of tool definitions to enable function calling. Defaults to None.

    Returns:
        dict or None: JSON response from the API if successful, None if an error occurs

    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if model == "nvidia/llama-3.1-nemotron-nano-vl-8b-v1":
        payload["temperature"] = 0.2
        payload["top_p"] = 0.7
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(f"{NVAI_URL}/chat/completions", headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"API Error for {model}: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None


def call_nemoretriever_parse(base64_image: str) -> list[dict[str, Any]] | None:
    """
    Parse document layout and extract structured content using the nemoretriever-parse model.

    Args:
        base64_image (str): Base64-encoded string representation of the image to analyze

    Returns:
        list or None: List of extracted document objects with their metadata and bounding boxes
                     if successful, None if the API call fails or no content is extracted

    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
        }
    ]
    tools = [{"type": "function", "function": {"name": "markdown_bbox"}}]
    response_json = call_api(model="nvidia/nemoretriever-parse", messages=messages, max_tokens=3500, tools=tools)
    if response_json:
        tool_call = response_json.get("choices", [{}])[0].get("message", {}).get("tool_calls", [{}])[0]
        if not tool_call:
            return []
        arguments_str = tool_call.get("function", {}).get("arguments", "[]")
        parsed_args = json.loads(arguments_str)
        if isinstance(parsed_args, list) and len(parsed_args) > 0 and isinstance(parsed_args[0], list):
            return parsed_args[0]
        return parsed_args
    return None


def query_nemotron_for_image(base64_image: str, prompt: str) -> str:
    """
    Query the nemotron vision-language model with an image and text prompt for analysis.

    Args:
        base64_image (str): Base64-encoded string representation of the image to analyze
        prompt (str): Text prompt describing the analysis task or question to ask about the image

    Returns:
        str: The model's response text if successful, or an error message if the API call fails

    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]
    response_json = call_api(model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", messages=messages, max_tokens=1024)
    if response_json:
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "Error: No content.")
    return "Error during analysis."


def convert_pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image | None:
    """
    Convert a specific page from a PDF document to a PIL Image object.

    Args:
        pdf_path (str): Path to the PDF file to process
        page_num (int): Zero-based page number to convert (0 for first page)
        dpi (int, optional): Resolution in dots per inch for the output image. Defaults to 300.

    Returns:
        PIL.Image.Image or None: RGB image object if successful, None if an error occurs

    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except (fitz.FileDataError, fitz.PageError, OSError) as e:
        print(f"Error converting page {page_num} of {pdf_path}: {e}")
        return None


def encode_image_to_base64(image: Image.Image, img_format: str = "PNG") -> str:
    """
    Convert a PIL Image object to a base64-encoded string representation.

    This function takes a PIL Image object and converts it to a base64-encoded string
    that can be transmitted over HTTP or stored as text. The image is first saved to
    a memory buffer in the specified format, then encoded to base64.

    Args:
        image (PIL.Image.Image): The PIL Image object to encode
        img_format (str, optional): Image format for encoding (e.g., "PNG", "JPEG"). Defaults to "PNG".

    Returns:
        str: Base64-encoded string representation of the image

    """
    buffered = io.BytesIO()
    image.save(buffered, format=img_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_pdf(pdf_url: str) -> tuple[str, str]:
    """
    Process a single PDF file through the multimodal extraction pipeline.

    Args:
        pdf_url (str): URL of the PDF file to process

    Returns:
        tuple: (pdf_filename, basename)

    """
    pdf_filename = pdf_url.split("/")[-1].split("?")[0]
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    response = requests.get(pdf_url, timeout=120)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    basename = os.path.splitext(pdf_filename)[0]
    return pdf_path, basename


def pdf_to_page_images(pdf_filename: str, dpi: int) -> list[Image.Image | None]:
    """
    Convert each page of a PDF file into an image.

    Args:
        pdf_filename (str): Path to the PDF file to be converted.
        dpi (int): Resolution in dots per inch for the output images.

    Returns:
        list: A list of PIL Image objects, one for each page in the PDF.
    """
    page_images = []
    doc = fitz.open(pdf_filename)
    for page_idx in range(len(doc)):
        page_images.append(convert_pdf_page_to_image(pdf_filename, page_idx, dpi))
    return page_images


def save_results_to_output(file_results: dict[str, Any], basename: str) -> None:
    """
    Save the extracted results for a file to a JSON file in the extraction results directory.

    Args:
        file_results (dict): The results data to be saved, typically containing extracted content and metadata.
        basename (str): The base name (without extension) to use for the output JSON file.
    """
    json_output_path = os.path.join(EXTRACTION_DIR, f"{basename}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(file_results, f)
    print(f"\nResults saved to '{json_output_path}'")


def get_bbox_pixel_coords(bbox: dict[str, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Convert normalized bbox coordinates to pixel coordinates.

    Args:
        bbox (dict): Bounding box with keys 'xmin', 'ymin', 'xmax', 'ymax' (normalized 0-1).
        image_size (tuple): (width, height) of the image.

    Returns:
        tuple: (left, top, right, bottom) pixel coordinates.
    """
    width, height = image_size
    left = max(0, bbox["xmin"] * width)
    top = max(0, bbox["ymin"] * height)
    right = min(width, bbox["xmax"] * width)
    bottom = min(height, bbox["ymax"] * height)
    return left, top, right, bottom


def process_page_images_pipeline(page_images: list[Image.Image | None], file_results: dict[str, Any]) -> None:
    """
    Process each page image through the extraction and analysis pipeline.

    Args:
        page_images (list): List of PIL Image objects, one for each page in the PDF.
        file_results (dict): The results data to be updated, typically containing extracted content and metadata.

    Returns:
        None. Modifies file_results in place.
    """

    for page_idx, page_image in enumerate(page_images):
        if page_image is None:
            continue

        page_num = page_idx + 1
        print(f"\nAnalyzing Page {page_num}/{len(page_images)} ...")

        # Stage 1: Layout Analysis with nemoretriever-parse
        print("[Stage 1] Calling nemoretriever-parse for layout analysis...")
        b64_page_image = encode_image_to_base64(page_image)
        extracted_data = call_nemoretriever_parse(b64_page_image)

        page_entry = {"page_number": page_num, "status": "Layout extraction successful", "content": []}
        if extracted_data is None:
            page_entry["status"] = "Layout extraction failed"
            file_results["pages"].append(page_entry)
            continue

        print(f"[Stage 1] Found {len(extracted_data)} document objects.")

        # Stage 2: Deep Analysis with llama-3.1-nemotron-nano-vl-8b-v1
        for item_idx, item in enumerate(extracted_data):
            item_metadata = {
                "extraction_id": item_idx,
                "metadata": {
                    "source_page": page_num,
                    "type": item.get("type"),
                    "bbox": item.get("bbox")
                },
                "data": {}
            }
            item_type = item.get("type")

            if item_type == "Picture":
                print(f"[Stage 2] Found Picture (ID: {item_idx}). Triggering Specialist VLM...")
                bbox = item.get("bbox")
                if not bbox or bbox.get("xmax", 0) < bbox.get("xmin", 0):
                    item_metadata["data"] = {"error": "Invalid bounding box"}
                    page_entry["content"].append(item_metadata)
                    continue

                left, top, right, bottom = get_bbox_pixel_coords(bbox, page_image.size)

                if left >= right or top >= bottom:
                    item_metadata["data"] = {"error": "Zero-area bounding box"}
                    page_entry["content"].append(item_metadata)
                    continue

                # Crop the image and show it
                crop_box = (left, top, right, bottom)
                cropped_img = page_image.crop(crop_box)
                print("- Cropped image patch for analysis:")

                b64_cropped = encode_image_to_base64(cropped_img)

                # Step 2a: Triage/Classification
                classify_prompt = (
                    "Analyze the image patch and classify it as either 'Infographics' or 'Other'. "
                    "Infographics: charts, diagrams, graphs, tables, maps, or schematics that visually represent complex information and contain text. "
                    "Other: photographs, logos, icons, decorative images, or handwritten elements. "
                    "Strictly enforced, reflect on your analysis, your output should be a single category name of either Infographics or Other."
                )
                specific_classification = query_nemotron_for_image(
                    b64_cropped, classify_prompt
                ).lower().strip().replace("'", "").replace('"', '')
                print(f"- VLM Classification: '{specific_classification}'")

                analysis_path = "infographics" if "infographics" in specific_classification.lower() else "other"
                item["sub_type"] = analysis_path

                # Step 2b: Deep Analysis
                analysis_prompt = (
                    "Your task is to analyze the provided infographic image and describe its content. "
                    "If the information in the infographic is suitable for a tabular format, please represent it as a Markdown table. "
                    "Otherwise, provide a detailed summary."
                    if analysis_path == "infographics"
                    else "Describe this image in detail."
                )
                print("- Running deep analysis...")
                analysis_result = query_nemotron_for_image(b64_cropped, analysis_prompt)

                item_metadata["data"] = {
                    "content_classification": specific_classification,
                    "analysis_result": analysis_result
                }

            elif item_type == "Table":
                print(f"[Stage 2] Found Table (ID: {item_idx}). Parsing and reconstructing table...")

                # Show the detected table patch for context
                bbox = item.get("bbox")
                if bbox:
                    left, top, right, bottom = get_bbox_pixel_coords(bbox, page_image.size)
                    if left < right and top < bottom:
                        crop_box = (left, top, right, bottom)
                        print("- Cropped table patch for context:")

                latex_code = item.get("text", "")
                if not latex_code:
                    item_metadata["data"] = {"error": "Empty LaTeX content for table"}
                else:
                    try:
                        # Parse the LaTeX tabular content
                        content_str = latex_code.split("\\begin{tabular}")[1].split("\\end{tabular}")[0]
                        content_str = content_str.split("}", 1)[1].strip()
                        rows = [r.strip() for r in content_str.split("\\\\") if r.strip()]
                        table_data = [
                            [cell.replace("**", "").strip() for cell in row.split("&")]
                            for row in rows
                        ]

                        if not table_data:
                            error_msg = "No data could be parsed from the LaTeX string."
                            raise ValueError(error_msg)

                        # Normalize row lengths
                        header = table_data[0]
                        num_columns = len(header)
                        normalized_body = []
                        for current_row in table_data[1:]:
                            if len(current_row) != num_columns:
                                while len(current_row) < num_columns:
                                    current_row.append("")
                                if len(current_row) > num_columns:
                                    current_row = current_row[:num_columns]
                            normalized_body.append(current_row)

                        # Create a pandas DataFrame
                        df = pd.DataFrame(normalized_body, columns=header)

                        # Convert the clean DataFrame to HTML and display it
                        print("- Reconstructed HTML Table:")
                        html_table = df.to_html(index=False, border=1, classes="table table-striped")

                        # Store the final HTML in the results
                        item_metadata["data"] = {"type": "tabular", "content_html": html_table}

                    except (ValueError, IndexError) as e:
                        print(f"- ERROR: Failed to parse table data. Details: {e}")
                        item_metadata["data"] = {
                            "error": f"Could not parse content: {e}",
                            "raw_content": latex_code
                        }
            else:
                # Fallback for any other type, like 'Text'
                item_metadata["data"] = {"type": "textual", "content": item.get("text", "")}

            page_entry["content"].append(item_metadata)

        file_results["pages"].append(page_entry)


def separate_contents_on_modality(basename: str) -> None:
    """
    Separates and categorizes extracted content from all pages in the extraction results directory
    by modality (table, image, text, other), and saves the aggregated result as a JSON file.

    Args:
        basename (str): The base name (without extension) of the source file, used for naming the output JSON.

    Returns:
        None. The function writes the categorized results to disk as a JSON file.
    """
    json_output_path = os.path.join(SEPARATED_MODALITY_DIR, f"{basename}.json")
    separated_modality_results = {
        "source_filename": basename,
        "pages": []
    }

    # Predefine the modality keys and their extraction logic for efficiency
    modality_map = {
        "Table": lambda data: data.get("content_html", ""),
        "Picture": lambda data: data.get("analysis_result", ""),
        "Text": lambda data: data.get("content", "")
    }

    for extracted_file in os.listdir(EXTRACTION_DIR):
        extracted_file_path = os.path.join(EXTRACTION_DIR, extracted_file)
        with open(extracted_file_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
        for page in extracted_data.get("pages", []):
            if page.get("status") == "Layout extraction failed":
                continue  # Skip failed pages

            # Use dicts for clarity and to avoid repeated code
            modality_lists = {
                "table_text_extraction": [],
                "image_text_extraction": [],
                "text_text_extraction": [],
                "other_text_extraction": []
            }

            for item in page.get("content", []):
                item_type = item.get("metadata", {}).get("type")
                data = item.get("data", {})
                if item_type in modality_map:
                    if item_type == "Table":
                        modality_lists["table_text_extraction"].append(modality_map["Table"](data))
                    elif item_type == "Picture":
                        modality_lists["image_text_extraction"].append(modality_map["Picture"](data))
                    elif item_type == "Text":
                        modality_lists["text_text_extraction"].append(modality_map["Text"](data))
                else:
                    # For any other type, treat as "other"
                    modality_lists["other_text_extraction"].append(data.get("content", ""))

            separated_modality_results["pages"].append({
                "page_number": page.get("page_number"),
                "content": modality_lists
            })

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(separated_modality_results, f, ensure_ascii=False, indent=4)


def format_missing_file_error(file_path: str) -> str:
    """Format error message for missing file."""
    return f"Missing required file: {file_path}. Please ensure the file exists before running the script."


def main() -> None:
    """Main function to run the multimodal extraction pipeline."""
    pdf_urls_path = os.path.join(PDF_DIR, "pdf_urls.jsonl")
    if not os.path.exists(pdf_urls_path):
        error_msg = format_missing_file_error(pdf_urls_path)
        raise FileNotFoundError(error_msg)

    urls = pd.read_json(path_or_buf=pdf_urls_path, lines=True)[0].tolist()

    for pdf_url in urls:
        pdf_filename, basename = process_pdf(pdf_url)

        file_results = {
            "source_filename": pdf_filename,
            "pages": []
        }

        # Convert PDF to a list of page images
        print(f" Converting '{pdf_filename}' to images...")
        page_images = pdf_to_page_images(pdf_filename, PDF_DPI)
        print(f"Converted {len(page_images)} pages.")

        # Run through extraction pipeline
        process_page_images_pipeline(page_images, file_results)

        # Save results of extraction
        save_results_to_output(file_results, basename)

        # Separate and categorize results of extraction
        separate_contents_on_modality(basename)


if __name__ == "__main__":
    main()