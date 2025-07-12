from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List

from tools_agents.files_processor import files_processor
from pydantic import BaseModel
from typing import Dict
from tools_agents.qa_engine import QAGraphEngine, QAGraphEngineAgent
import logging


import asyncio

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = FastAPI()

process_files = files_processor()
#qa_engine = QAGraphEngine()
qa_engine_agent = QAGraphEngineAgent()


# Pydantic schema for incoming request
class QueryRequest(BaseModel):
    message: str
    user_id: str

# Response schema (optional but useful for docs)
class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = [] 



@app.post("/ask", response_model=QueryResponse)
"""
Handles POST requests to the "/ask" endpoint.
Receives a QueryRequest containing a user's question and user ID, then asynchronously generates a response using the qa_engine_agent.
Logs the generated result and returns a QueryResponse with the answer and sources.
Args:
    request (QueryRequest): The incoming request containing the user's message and user ID.
Returns:
    QueryResponse: An object containing the generated answer and its sources.
"""
async def ask_question(request: QueryRequest):
    
    result = await qa_engine_agent.generate_response(request.message, request.user_id)


    logging.info(f"input_message result: {result}")

    return QueryResponse(
        answer=result['answer'],
        sources=result['sources']
    )


@app.post("/upload_files/")
"""
Endpoint to upload multiple files.
Args:
    files (List[UploadFile]): List of files to be uploaded, received as form data.
Returns:
    dict: A dictionary containing the list of saved filenames under the key 'filenames'.
Raises:
    HTTPException: If file processing fails.
Example:
    Request:
        POST /upload_files/
        Form Data: files=[file1, file2, ...]
    Response:
        {
            "filenames": ["file1.txt", "file2.pdf"]
        }
"""
async def upload_files(files: List[UploadFile] = File(...)):
    
    saved_files = await process_files.process_files(files)

    return {"filenames": saved_files}