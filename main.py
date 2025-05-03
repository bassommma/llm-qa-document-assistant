from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.services.RAG_Chroma import model_service
import os 


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.makedirs(settings.UPLOAD_DIRECTORY,exist_ok=True)
from typing import List, Optional
import uvicorn
from pydantic import BaseModel

app =FastAPI(
      title=settings.PROJECT_NAME
)

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
    return([model_service.collection.get()])





@app.get("/store")
def print_store():
    response = [model_service.collection.count()]
    return response

@app.get("/health")
def health_check():
      return{"status":"ok"}


@app.post("/clear_db")
def delete_by_ids():
    all_doc = model_service.collection.get()
    doc_ids = all_doc["ids"]
    try:
        
        model_service.collection.delete(ids = doc_ids)
    except:
        print("error in deleting")
    return {"database_clear": "success"}

@app.post("/test-model")
def test_gpt2(prompt:str):
      response=model_service.generate_text(prompt)
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
            model_service.add_document(file_path, document_id,chunk_size=chunk_size)

    
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
        similar_chunks,doc_ids = model_service.retrieve_relevant_document(query, top_k)
        return {
            "query": query,
            "results": similar_chunks,
            "ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ask")
async def ask_question(
    query: str,
    top_k: int = 3
):
    try:
        
        answer,similar_chunks = model_service.answer_question(query, top_k)
        
        
        
        return {
            "query": query,
            "context_chunks": similar_chunks,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

