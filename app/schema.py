from pydantic import BaseModel, Field
from typing  import  Optional, List
from enum import Enum

class IntentType(str,Enum):
    """
    Mögliche Absichtstypen eines Support-Tickets
    """
    COMPLAINT = "complaint"
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    UNKNOW = "unknow"


class ChatRequest(BaseModel):
    """
    Request Model für Chat-Endpunkt
    """
    message:str = Field(...,min_length=1, max_length=2000, description="Benutzernachricht")
    thread_id:Optional[str] = Field(None, description="Thread ID für Koversations-Persistenz")


    class Config:
        json_schema_extra = {
            "example":{
                "message":"Mein Internet funktioniert seit 2 Tagen nicht mehr!",
                "thread_id":"user_12345"
            }
        }


class ChatResponse(BaseModel):
    """
    Response Model für Chat-Endpunkt
    """
    response:str = Field(...,description="Agent-Antwort")
    intent:str = Field(...,description="Erkannte Absicht")
    confidence:float  = Field(...,ge=0.0,le=1.0, description="Konfidenz-Score")
    needs_human:bool = Field(...,description="Ob menschliche Intervention nötig ist")
    thread_id:str = Field(...,description="Thread ID der Konversation")



    class Config:
        json_schema_extra = {
            "example":{
                "response": "Es tut mir leid zu hören, dass Ihr Internet nicht funktioniert...",
                "intent":"technical",
                "confidence":0.85,
                "needs_human":False,
                "thread_id":"user_12345"
            }
        }


class HealthResponse(BaseModel):
    """
    Health Check Response
    """
    status: str = "ok"
    model:str
    version:str = "1.0.0"

