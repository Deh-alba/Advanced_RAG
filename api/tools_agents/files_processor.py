import os
import io
import json
import shutil
import logging
import asyncio
import platform
import base64

from pathlib import Path
from PIL import Image
from typing import List

import aiofiles
import fitz  # PyMuPDF
import openai

from fastapi import UploadFile

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from datatabase import DBVector
from langchain.schema import Document



load_dotenv(dotenv_path="tools_agents/.env")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key=OPENAI_API_KEY)



logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

class files_processor:
    def __init__(self):
        self.upload_dir = "uploaded_files"
        self.output_dir = "output_json"
        self.temp_json_dir = "temp_output_json"
        self.image_dir = "extracted_images"
        self.page_images_dir = "page_images"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.db = DBVector()

    async def process_files(self, files: List[UploadFile]):
        
        #logging.info(f"Processing file: {files}")
        
        processed_files = []

        for file in files:
            
            input_path = os.path.join(self.upload_dir, file.filename)

            # Save uploaded file
            content = await file.read()
            async with aiofiles.open(input_path, "wb") as f_out:
                await f_out.write(content)

            filename = os.path.basename(input_path)
            filename_base = Path(input_path).stem
            logging.info(f"Processing: {input_path}")

            # Convert DOCX to PDF if needed
            if filename.lower().endswith(".docx"):
                print("🔄 Converting DOCX to PDF...")
                input_path = self._convert_docx_to_pdf(input_path)
                filename = os.path.basename(input_path)
                filename_base = Path(input_path).stem

            await self._extract_images_pdf(input_path, filename_base)
            image_paths = await self._render_pdf_to_images(input_path)

            os.makedirs(self.temp_json_dir, exist_ok=True)

            tasks = [
                self._process_page_async(page_num, image_path, filename_base)
                for page_num, image_path in image_paths
            ]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r]
            results.sort(key=lambda r: r["metadata"]["page"])

            shutil.rmtree(self.temp_json_dir, ignore_errors=True)

            logging.info(f"Temporary JSON directory cleaned up: {self.temp_json_dir}")
            output_path = os.path.join(self.output_dir, f"{filename_base}.json")
            logging.info(f"Saving combined output to: {output_path}")
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(results, ensure_ascii=False, indent=2))

            print(f"📦 Combined output saved to {output_path}")

            # Combine text from all pages
            logging.info(f"Combining text from {results} pages")
            combined_text = "\n\n".join([r["text"] for r in results])
            logging.info(f"Combined text length: {combined_text}")
            image_filenames = [r["metadata"]["image_path"] for r in results if "image_path" in r["metadata"]]
            logging.info(f"Extracted image paths: {image_filenames}")

            # Create and insert LangChain Document
            doc = Document(
                page_content=combined_text,
                metadata={
                    "source": file.filename,
                }
            )
            
            logging.info(f"Inserting document into DB: {doc}")

            self.db.insert_documents([doc])

            processed_files.append(file.filename)

        return processed_files

    def _convert_docx_to_pdf(self, docx_path: str, output_dir="converted_pdfs"):
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_path = os.path.join(output_dir, Path(docx_path).stem + ".pdf")
        system = platform.system()

        if system in ["Windows", "Darwin"]:
            from docx2pdf import convert
            convert(docx_path, output_pdf_path)
        elif system == "Linux":
            os.system(f'libreoffice --headless --convert-to pdf "{docx_path}" --outdir "{output_dir}"')
        else:
            raise RuntimeError("Unsupported OS for DOCX to PDF conversion.")

        if not os.path.exists(output_pdf_path):
            raise FileNotFoundError(f"Conversion failed: {output_pdf_path}")

        return output_pdf_path

    async def _render_pdf_to_images(self, pdf_path):
        os.makedirs(self.page_images_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_paths = []

        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            image_path = os.path.join(self.page_images_dir, f"image_page_{i}.png")
            pix.save(image_path)
            image_paths.append((i, image_path))

        return image_paths

    async def _extract_images_pdf(self, input_path, filename_base):
        doc = fitz.open(input_path)
        folder = os.path.join(self.image_dir, filename_base)
        os.makedirs(folder, exist_ok=True)

        for page_number, page in enumerate(doc, 1):
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                if ext.lower() in {"jpg", "jpeg"}:
                    image = Image.open(io.BytesIO(image_bytes))
                    output = io.BytesIO()
                    image.save(output, format="PNG")
                    image_bytes = output.getvalue()
                    ext = "png"

                out_path = os.path.join(folder, f"page_{page_number}_image_{img_index}.{ext}")
                with open(out_path, "wb") as f:
                    f.write(image_bytes)

    def _create_prompt(self, filename, page_num):
        return f"""
            You are an assistant that extracts structured content from a page image of a PDF document.

            Return a valid JSON object with this format:

            {{
            "text": "The full content of the page. For each image, insert an inline reference like [page_x_image_y] followed by the image description in brackets.",
            "metadata": {{
                "page": {page_num},
                "source": "{filename.replace('.pdf', 'docx')}",
                "images": ["page_x_image_y.png","page_x_image_y+1.png"]
            }},
            "image_descriptions": {{}}
            }}

            ⚠️ No markdown formatting. No trailing commas. JSON only.
            ⚠️ Use only double quotes and valid syntax.
            ⚠️ Put all image descriptions directly in the 'text' content.
            """

    def _merge_descriptions_into_text(self, parsed):
        text = parsed.get("text", "")
        image_descriptions = parsed.get("image_descriptions", {})
        for image_name, description in image_descriptions.items():
            placeholder = f"[{image_name.replace('.png', '')}]"
            if placeholder in text and f"{placeholder} [" not in text:
                text = text.replace(placeholder, f"{placeholder} [{description}]")
        parsed["text"] = text
        parsed["image_descriptions"] = {}
        return parsed

    async def _encode_image_to_base64_async(self, image_path):
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        return base64.b64encode(content).decode("utf-8")

    async def _process_page_async(self, page_num, image_path, filename_base):
        prompt = self._create_prompt(filename_base, page_num)
        try:
            base64_image = await self._encode_image_to_base64_async(image_path)

            response = await openai.AsyncClient().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You extract structured content from PDF page images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            parsed = self._merge_descriptions_into_text(parsed)

            async with aiofiles.open(f"{self.temp_json_dir}/page_{page_num}_output.json", "w", encoding="utf-8") as f:
                await f.write(json.dumps(parsed, ensure_ascii=False, indent=2))

            print(f"✅ Saved page {page_num}")
            return parsed

        except json.JSONDecodeError as e:
            print(f"⚠️ Page {page_num} - JSON error: {e}")
        except Exception as e:
            print(f"❌ Error on page {page_num}: {e}")
        return None
