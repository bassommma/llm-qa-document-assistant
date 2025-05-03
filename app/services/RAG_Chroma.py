from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline,BitsAndBytesConfig
from app.core.config import settings
import torch
import PyPDF2
import os
from typing import Union, List, Dict ,Optional
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
torch.set_grad_enabled(False)
class RAG_Chroma:
      
      

      def __init__(self,collection_name="documents", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None 
        self.tokenizer = None
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.pipeline = None
        self.client = chromadb.PersistentClient("./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name,embedding_function=self.embedding_model)
        self.model_path = r"F:\ai\local_projects\rag-with-chroma\models\MYllama"
      def load_models(self):
                if self.model is not None and self.tokenizer is not None:
                    print("Model is already loaded")
                    return
                quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                device_map="auto"
        )
        
                try:
                    print(f"Attempting to load model: {settings.MODEL_NAME}")
                    
                    # Load tokenizer first
                    print("Loading tokenizer...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    pad_token_id=self.tokenizer.eos_token_id
                    )
                    print("Tokenizer loaded successfully")
                    
        

                    
                    # Then load model
                    print("Loading model...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                    
                    # quantization_config=quantization_config,
                    self.model_path,
                    local_files_only=True
                    )
                    print("Model loaded successfully")
                    
                    # Save model

                    
                    # Create pipeline after both are loaded
                    print("Creating pipeline...")
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,  # Be explicit with parameter names
                        tokenizer=self.tokenizer,
                        max_new_tokens=255,
                        top_p=0.95,
                        temperature=0.1,
                        repetition_penalty=1.15,
                        do_sample=True
                    )
                    print("Pipeline created successfully")
                    
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    raise e
        
                print("Model is already loaded")
      def generate_text(self,formatted_prompt:str):
        self.load_models()
        result = self.pipeline(formatted_prompt)  
        full_text = result[0]['generated_text']
        return full_text

            
        
        
                
      

      
      def pdfreader(self,pdf_path:str) -> str:
          """read a pdf and extract it's text"""
          
          text=""
          with open(pdf_path,"rb")as file :
             pdf_reader=PyPDF2.PdfReader(file)
             for i in range(len(pdf_reader.pages)):
                 text += pdf_reader.pages[i].extract_text()+"\n"

          return text
      def chunk_text(self, text, chunk_size=1000, overlap=20):
        """Split text to chunks of chunk_size with debugging"""
        print(f"Starting chunking of text with length: {len(text)}")
        print(f"Initial memory usage: {self.get_memory_usage()} MB")
        
        chunks = []
        start = 0
        text_length = len(text)
        iteration_count = 0
        
        while start < text_length:
            iteration_count += 1
            
            if iteration_count % 100 == 0:
                print(f"Iteration {iteration_count}, processed {start}/{text_length} chars")
                print(f"Current memory usage: {self.get_memory_usage()} MB")
            
            # Calculate end position respecting text length
            end = min(start + chunk_size, text_length)
            
            # Try to find a natural breaking point
            if end < text_length:
                period_point = text.rfind('.', end-100, end)
                newline = text.rfind('\n', end-100, end)
                if period_point != -1 or newline != -1:
                    stop_point = max(period_point, newline)
                    end = stop_point + 1
            
            # Debug print for this chunk
            if iteration_count < 10 or iteration_count % 100 == 0:
                print(f"Chunk {iteration_count}: start={start}, end={end}, length={end-start}")
            
            chunk = text[start:end].strip()
            
            if chunk:
                chunk_size_bytes = len(chunk.encode('utf-8'))
                
                # Safety check for extremely large chunks
                if chunk_size_bytes > 1_000_000:  # 1MB safety check
                    print(f"Warning: Very large chunk detected ({chunk_size_bytes} bytes)")
                    # You could implement further splitting here if needed
                
                print(f"Adding chunk {len(chunks)+1}, size: {chunk_size_bytes} bytes")
                chunks.append(chunk)
            
            # Advance to next chunk position, ensuring we make progress
            new_start = end - overlap
            if new_start <= start:  # Ensure we're making progress
                new_start = start + 1
            start = new_start
            
            # Optional safety check to prevent infinite loops
            if iteration_count > 100000:  # A reasonable upper limit
                print("Warning: Excessive iterations, possible infinite loop. Breaking.")
                break
        
        print(f"Chunking complete. Created {len(chunks)} chunks.")
        print(f"Final memory usage: {self.get_memory_usage()} MB")
        return chunks



      def get_memory_usage(self):
            """Helper function to monitor memory usage"""
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Return in MB
                 

          
      def add_document(self,file_path,doc_id,chunk_size=2000):
         text = self.pdfreader(file_path)
         
         if chunk_size:
               chunks = self.chunk_text(text, chunk_size=chunk_size)
               print(f"Created {len(chunks)} chunks for document: {doc_id}")
               
               # Check if we need to handle batch size limits
               if len(chunks) > 5000:  # Safe value below Chroma's limit
                     print(f"Large number of chunks detected ({len(chunks)}). Will process in batches.")

               
               # Create unique IDs for each chunk
               chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
               
               # Create metadata for each chunk
               metadatas = []
               for i in range(len(chunks)):
                     metadatas.append({
                        "source": doc_id,  # Original document ID
                        "chunk_index": i,  # Which chunk this is
                        "date_added": datetime.now().isoformat(),
                        "file_type": doc_id.split('.')[-1] if '.' in doc_id else "text"
                     })
               
               # Add all chunks to the collection
               self.collection.add(
                     documents=chunks,  
                     ids=chunk_ids,     
                     metadatas=metadatas  
               )
         else:
               # If not chunking, add the entire document as one entry
               metadata = {
                     "source": doc_id,  
                     "date_added": datetime.now().isoformat(),
                     "file_type": doc_id.split('.')[-1] if '.' in doc_id else "text"
               }
               
               self.collection.add(
                     documents=[text],  
                     ids=[doc_id],      
                     metadatas=[metadata]  
               )
         





      def retrieve_relevant_document (self,query,n_results):
         
             
         response = self.collection.query(query_texts=[query],
                               n_results=n_results)
         
        
            
                               
         doc_ids = response['ids'][0]
         retrieved_docs=response['documents'][0]
         retrieved_docs="\n\n".join(retrieved_docs)
         return retrieved_docs,doc_ids
      


      def answer_question(self,query,n_results):
         context_text,doc_ids=self.retrieve_relevant_document (query,n_results)

         formatted_prompt = f"""
         please provide a good consice answer based on the given context and query
         ###context###
         {context_text}

         ###Question###
         {query}

        ###Answer###
        """
         full_answer = self.generate_text(formatted_prompt)
         if "<|answer|>" in full_answer :
             answer = full_answer.split("<|answer|>")[-1].strip()
             
         return answer,context_text
           

      
          
      


          
model_service = RAG_Chroma(collection_name='documents')
print(model_service.collection.name)
          
          



              
              

           


             


