from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile
from uvicorn import Config, Server
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

# load_dotenv()  

# GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")


# system = "You are an expert OCR and text reconstruction. Reconstruct to the original text the input. No commentary in response."
# human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
# chain = prompt | chat


app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True,lang="en",use_gpu=False)


def get_text(image_data):
        if image_data is None:
            raise ValueError("Image data is None")
        result=ocr.ocr(image_data)    
        txts = [line[1][0] for line in result[0]]
        final_output='  '.join(txts)
        # resp=chain.invoke({"text":final_output})
        return final_output



@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    detection_boxes = get_text(contents)
    return {"text": detection_boxes}



