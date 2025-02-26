import os
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# nos permite cargar las varaibles desde .env
dotenv.load_dotenv()

# instciantiate the FastAPI class
app = FastAPI(
    title="Mi primer BOT",
    description="""
        Mi primer bot basico
        
        1. * Consultas a un Modelo  Cohere
    """,
    version="v0.1.0"
    
)  

# model schema
class Agente(BaseModel):
    prompt: str


@app.post("/agente")
async def agente(request: Agente):
    
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    prompt = request.prompt
    
    # 1- instanciate the ChatCohere class (API_key)
    llm = ChatCohere(
        api_key=COHERE_API_KEY
    )
    
    # 2- create a list of messages in the conversation.
    list_message = [
        SystemMessage(content="Eres experto en IA, te llamaras PEPITO."),
        HumanMessage(content=prompt)
    ]
    
    # 3- invoke the model
    response = llm.invoke(list_message)
    
    list_message.append(response)
    
    return {
        "agente": list_message[-1].content,
    }

