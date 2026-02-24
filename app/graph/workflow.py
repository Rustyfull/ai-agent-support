from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import  SupportState
from app.graph.nodes import (
    classify_intent,
    search_knowledge_base,
    generate_response,
    human_handoff
)
from app.config import get_settings


settings = get_settings()

def should_continue(state:SupportState) -> str:
    """
    Conditional Edge: Routing-Logik

    Entscheidet basierend auf State, welcher Node als nächstes kommt:

    1. Confidence < Threshold  -> Human Handoff
    2. Intent = technical -> Search Knowledge Base
    3. Sonst -> generate response
    """
    confidence = state.get('confidence',0.0)
    intent  = state.get('intent','general')

    # Regel 1 : Niedrige Confidence -> Eskalation
    if confidence < settings.confidence_threshold:
        print(f"Low confidence ({confidence:.2f}) -> Human handoff")
        return "human_handoff"

    # Regel 2 : Technische Probleme -> Knowledge Base
    if intent == "technical":
        print(f"Technical Issue -> Search knowledge base")
        return "search_knowledge_base"

    # Regel 3: Alle anderen -> Direkt response
    print(f"{intent} intent -> Generate Response")
    return "generate_response"




def create_workflow() -> StateGraph:
    """
    Erstellt den Langgraph Workflow
    """

    # Workflow initialisieren
    workflow = StateGraph(SupportState)

    # === NODES HINZUFÜGEN ====
    workflow.add_node("classify_intent",classify_intent)
    workflow.add_node("search_knowledge_base",search_knowledge_base)
    workflow.add_node("generate_response",generate_response)
    workflow.add_node("human_handoff",human_handoff)

    # === EDGES DEFINIEREN ===

    # 1. Start -> Intent Classification
    workflow.set_entry_point("classify_intent")

    # 2. Intent Classification -> COnditional Router
    workflow.add_conditional_edges(
        "classify_intent", should_continue,{
            "search_knowledge_base":"search_knowledge_base",
            "generate_response":"generate_response",
            "human_handoff":"human_handoff"
        }
    )


    # 3. Knowledge Base -> Response Generation
    workflow.add_edge("search_knowledge_base","generate_response")

    # 4. Response Generation -> END
    workflow.add_edge("generate_response",END)

    # 5. Human Handoff -> END
    workflow.add_edge("human_handoff",END)

    return workflow

# Memory für Persistent (speichert Konversationshistorie)
memory = MemorySaver()

# Graph kompilieren
workflow = create_workflow()
graph = workflow.compile(checkpointer=memory)



def get_graph():
    """
    Getter-Funktion für den kompilierten Graph
    """
    return graph



# Für Debugging. Graph-Visualisierung ausgeben
if __name__ == "__main__":
    try:
        import  graphviz
        from IPython.display import display, Image
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("Graph-Visualisierung nicht verfügbar (benötigt IPython + graphviz)")
        print("\nGraph-Struktur")
        print(graph.get_graph().to_json())
        raise e

