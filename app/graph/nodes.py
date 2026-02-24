from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.messages import  SystemMessage, HumanMessage, AIMessage
from app.graph.state import SupportState
from app.config import get_settings
import json
import uuid

settings = get_settings()

# Gemini LLM initialisieren
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    temperature=settings.temperature,
    max_tokens=settings.max_tokens,
    google_api_key=settings.google_api_key
)


def classify_intent(state:SupportState) -> dict:
    """
    Node 1: Intent Klassifizierung

    Analysiert die letzte User-Nachricht und klassifiziert sie in:
    - complaint: Beschwerde über Service/Produkt
    - technical: Technisches Problem
    - billing: Rechnungs-/zahlungsfrage
    - general: Allgemeine Anfrage

    Returns:
        dict mit intent, confidence und processing_steps
    """

    # Letzte User-Nachricht holen
    last_message = state["messages"][-1].content

    # Spezialisierter Prompt für Intent Classification
    classification_prompt = SystemMessage(content="""
    Du bist ein Intent-Classifier für ein Support-System.
    
    Analysiere die Nachricht und Klassifiziere sie in GENAU eine dieser Kategorien:
    - complaint: Beschwerde, Unzufriedenheit, Ärger
    - technical: Technisches Problem, Fehler, funktioniert nicht
    - billing: Rechnung, Zahlung, Preise, Abonnement
    - general: Allgemeine Frage, Information
    
    Antworte NUR mit einem JSON-Objekt ib diesem Format (KEINE zusätzlichen Texte):
    {"intent":"category","confidence":0.95,"reasoning":"kurze Begründung"}
    
    Beispiele:
    User: "Mein WLAN funktioniert nicht !"
    {"intent":"technical","confidence":0.95,"reasoning":"Technisches Problem mit WLAN"}
    
    User: "Warum wurde mir doppelt abgebucht ?"
    {"intent":"billing","confidence":0.90,"reasoning":"Frage zur Abbuchung"}
    
    User: "Ich bin sehr unzufrieden mit eurem Service!"
    {"intent":"complaint","confidence":0.95, "reasoning":"Klare Beschwerde über Service"}
    """)

    user_message = HumanMessage(content=last_message)

    # LLM aufrufen
    response = llm.invoke([classification_prompt,user_message])

    try:
        # JSON parsen
        result = json.loads(response.content.strip())
        intent = result.get("intent", "general")
        confidence = result.get("confidence",0.5)
        reasoning = result.get("reasoning", "No reasoning provided")


    except json.JSONDecodeError:
        # Fallback wenn JSON-Parsing fehlschlägt
        print(f"JSON Parse Error. Raw response: {response.content}")
        intent = "general"
        confidence = 0.3
        reasoning = "Parsing failed"

    print(f"Intent: {intent} | Confidence; {confidence:.2f} | Reasoning: {reasoning}")

    return {
        "intent":intent,
        "confidence":confidence,
        "processing_steps":[f"Intent classified as '{intent}' with {confidence:.2f} confidence"]

    }


def search_knowledge_base(state:SupportState) -> dict:
    """
    Node 2: Wissenbank-Suche (Mock-Implementation)

    In einer echten Anwendung würde hier:
    - Eine Vektor-DB (Qdrant/Pinecone) durchsucht
    - RAG (Retrieval Augmented Generation) durchgeführt
    - Relevante Dokumente/FAQs zurückgegeben

    Für Demo-Zwecke: Mock-Daten basierend auf Intent
    """

    intent = state["intent"]

    # Mock Knowledge Base
    knowledge_base = {
        "technical": [
            "Troubleshooting: Starten Sie Ihren Router neu und warten Sie 30 Sekunden.",
            "Überprüfen Sie, ob alle Kabel korrekt angeschlossen sind.",
            "Testen Sie die Verbindung mit einem anderen Gerät."
        ],
        "billing": [
            "Rechnungen werden monatlich am 1. des Monats erstellt.",
            "Sie können Ihre Zahlungsmethode im Kundenportal ändern.",
            "Bei Abbuchungsfehlern kontaktieren Sie bitte unsere Buchhaltung."
        ],
        "complaint": [
            "Wir nehmen Ihre Beschwerde sehr ernst.",
            "Ein Manager wird sich innerhalb von 24 Stunden bei Ihnen melden.",
            "Sie können Feedback auch an feedback@company.com senden."
        ],
        "general": [
            "Unsere Geschäftszeiten sind Mo-Fr 9-18 Uhr.",
            "Support ist per Chat, E-Mail und Telefon erreichbar.",
            "FAQ finden Sie auf unserer Website unter /support/faq"
        ]
    }

    results = knowledge_base.get(intent,knowledge_base["general"])

    print(f"Knowledge Base Search: Found {len(results)} relevant documents")

    return {
        "knowledge_base_results":results,
        "processing_steps":[f"Retrieved {len(results)} documents from knowledge base"]
    }


def generate_response(state:SupportState) -> dict:
    """
    Node 3: Response-Generierung

    Nutzt den Intent, die KB-Ergebnisse und die Koversationshistorie,
    um eine hilfreiche und empathische Antwort zu generieren
    """
    intent = state["intent"]
    confidence = state["confidence"]
    kb_results = state.get("knowledge_base_results",[])

    # Context aus Knowledge Base
    kb_context = "\n".join([f"-{doc}" for doc in kb_results])

    # System Prompt für Response Generation
    system_prompt = SystemMessage(content=f"""
    Du bist ein professioneller Support-Agent für ein Telekommunikationsunternehmen.

    KONTEXT:
    -   Intent: {intent}
    -   Confidence: {confidence:.2f}
    WISSENBASIS:
    {kb_context}
    
    ANWEISUNGEN:
    1. Sei empathisch und professionell
    2. Nutze die Informationen aus der Wissensbasis
    3. Biete konkrete Lösungsschritte an
    4. Halte dich kurz (2-4 Sätze)
    5. Bei technischen Problemen: Schritt-für-Schritt Anleitung
    6. Bei Beschwerden: Entschuldigung + Lösungsweg
    7. Antworte auf Deutsch
    
    WICHTIG: Schreibe NUR die Antwort für den Kunden, keine Meta--Kommentare.

"""
                                  )
    # Letzte Nachricht
    last_message = HumanMessage(content=state["messages"][-1].content)

    # Response generieren
    response = llm.invoke([system_prompt,last_message])

    # Ticket ID generieren (falls noch nicht vorhanden)
    ticket_id = state.get("ticket_id") or f"TKT-{uuid.uuid4().hex[:8].upper()}"

    print(f"Response generated | Ticket: {ticket_id}")

    # AI Message zur Historie hinzufügen
    return {
        "messages":[AIMessage(content=response.content)],
        "ticket_id":ticket_id,
        "processing_steps":{"Generated final response"}
    }




def human_handoff(state:SupportState)-> dict:
    """
    Node 4: Eskalation zu menschlichem Agen

    Wird aufgerufen wenn:
    - Confidence < threshold
    - Komplexe/sensible Anfragen
    - User explizit nach menschlichem Agent fragt
    """
    ticket_id = state.get("ticket_id") or f"TKT-{uuid.uuid4().hex[:8].upper()}"

    handoff_message = f"""
    Ich habe Ihre Anfrage an einer unserer Spezialisten weitergeleitet.
    
    Ticket-Nummer: {ticket_id}
    Voraussichtliche Wartezeit: 5 -10 Minuten
    Sie erhalten eine E-Mail-Bestätigung
    
    Ein menschlicher Agent wird sich in kürze Zeit bei Ihenen melden, um Ihre Anfragen
    persönlich zu bearbeiten.
    
    
    """

    print(f"Human Handoff triggered | Ticket: {ticket_id}")

    return {
        "messages": [AIMessage(content=handoff_message)],
        "ticket_id":ticket_id,
        "needs_human":True,
        "processing_steps":["Escalated to human agent"]
    }