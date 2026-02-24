from typing import Annotated, Optional
from langgraph.graph import  MessagesState
from langchain_core.messages import BaseMessage
from operator import  add

class SupportState(MessagesState):
    """
    Erweiterter State für unser Support-System.

    Erbt von MessageState (enthält bereits 'messages' Liste),
    unf fügt zusätzliche Metadaten hinzu

    Attributes:
        messages: Liste von Nachrichten (inherited von MessagesState)
        intent: Erkannte Absicht des Users (complaint/technical/billing/general)
        confidence: Konfidenz-Score der Intent-Klassifizierung (0.0 bis 1.0)
        needs_human: Flag ob menschliche Intervention erforderlich ist
        ticket_id: Eindeutige Ticket-Identifikation
        knowledge_base_results: Ergebnisse aus der Wissenbank-Suche (falls RAG aktiviert)
    """

    # Intent classification
    intent: Optional[str] = None
    condifence: float = 0.0

    # Human-in-the-lopp
    needs_human:bool = False

    # Tracking
    ticket_id: Optional[str] = None

    # RAG (für zukunftige Erweiterung)
    knowledge_base_results: Annotated[list[str], add] = []

    # Debugging/Logging
    processing_steps: Annotated[list[str],add] = []



def get_initial_state() -> dict:
    """
    Factory-Funktion für einen leeren State
    """
    return {
        "messages":[],
        "intent":None,
        "confidence":0.0,
        "needs_human":False,
        "ticket_id":None,
        "knowledge_base_results":[],
        "processing_steps":[]
    }
