from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv
from app.services.RAG_Chroma import DocumentQA
from typing import List, Optional
import uvicorn
from pydantic import BaseModel

import os

# Add this to your startup code
os.makedirs("uploads", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)


load_dotenv()



app =FastAPI(
      title=settings.PROJECT_NAME
)
security = HTTPBearer()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
doc_qa = DocumentQA(model_id=settings.MODEL_NAME, use_auth=True, hf_token=HF_TOKEN)

os.makedirs(settings.UPLOAD_DIRECTORY,exist_ok=True)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DeleteRequest(BaseModel):
    ids: List[str] 

@app.get("/")
def root():
    return {"message": "Welcome to Document Q&A API"}

@app.get("/get_ids")
def display_doc_ids():
    return([doc_qa.collection.get()])





@app.get("/count")
def print_store():
    response = [doc_qa.collection.count()]
    return response

@app.get("/health")
def health_check():
      return{"status":"ok"} 


@app.post("/clear_db")
def delete_by_ids():
    all_doc = doc_qa.collection.get()
    doc_ids = all_doc["ids"]
    try:
        
        doc_qa.collection.delete(ids = doc_ids)
    except:
        print("error in deleting")
    return {"database_clear": "success"}

@app.post("/test-model")
def test_gpt2(prompt:str):
      response=doc_qa.generate_text(prompt) 
      return{"prompt":prompt,"response":response}

@app.post("/upload_document")
async def upload_document (file: UploadFile,chunk_size:Optional[int]=None):
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
    document_id = os.path.splitext(file.filename)[0]
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the document
        try:
            doc_qa.add_document(file_path, document_id,chunk_size=chunk_size)

    
            return {"message": "Document been added to database succesfully", "document_id": document_id}
        except Exception as e:
            print(f"Error uploading document to chroma: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error uploading document to chroma: {str(e)}"
            )
    except Exception as e:
        
        raise HTTPException(
            status_code=500, 
            detail=f"Error saving file: {str(e)}"
        )

    


@app.post("/query")
async def query_documents(
    
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 3
):
    
    try:
        similar_chunks,doc_ids = doc_qa.retrieve_relevant_document(query, top_k)
        return {
            "query": query,
            "results": similar_chunks,
            "ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ask")
async def ask_question(query: str, top_k: int = 3):
    try:
        print(f"Received query: {query}, top_k: {top_k}")  # Debug
        
        answer, similar_chunks = doc_qa.answer_question(query, top_k)
        
        print(f"Answer received: '{answer}'")  # Debug
        print(f"Type of answer: {type(answer)}")  # Check type
        
        response = {
            "query": query,
            "context_chunks": similar_chunks,
            "answer": answer
        }
        
        print(f"Final response: {response}")  # Debug
        
        return response
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    print("object has been created")

    


