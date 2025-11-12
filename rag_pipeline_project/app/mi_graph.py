# app/mi_graph.py
from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from app.rag_pipeline import RAGPipeline

class ConversationState(TypedDict, total=False):
    query: str
    response: str
    user_stance: Literal["resistant", "curious", "ready", "neutral"]
    chat_history: List[str]

class MIConversationGraph:
    def __init__(self):
        self.llm = Ollama(model="llama3.1:8b")  # swap to qwen2.5:14b later if you want
        self.rag = RAGPipeline()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(ConversationState)
        workflow.add_node("analyze_stance", self.analyze_user_stance)
        workflow.add_node("resistant_response", self.handle_resistance)
        workflow.add_node("curious_response", self.provide_information)
        workflow.add_node("ready_response", self.reinforce_understanding)

        workflow.set_entry_point("analyze_stance")
        workflow.add_conditional_edges(
            "analyze_stance",
            self.route_based_on_stance,
            {
                "resistant": "resistant_response",
                "curious": "curious_response",
                "ready": "ready_response",
                "neutral": "curious_response",
            },
        )
        workflow.add_edge("resistant_response", END)
        workflow.add_edge("curious_response", END)
        workflow.add_edge("ready_response", END)
        return workflow.compile()

    def analyze_user_stance(self, state: ConversationState):
        prompt = f"""
Klassifiziere die Haltung des Nutzers mit GENAU EINEM Wort.
Nachricht: {state['query']}

Erlaubte Kategorien:
- resistant  (defensiv, wütend, abweisend)
- curious    (fragt nach, offen)
- ready      (akzeptiert, will Infos)
- neutral    (unklar)

Gib NUR ein Wort zurück: resistant | curious | ready | neutral
"""
        stance = self.llm.invoke(prompt).strip().lower()
        state["user_stance"] = stance if stance in {"resistant","curious","ready","neutral"} else "neutral"
        return state

    def route_based_on_stance(self, state: ConversationState):
        return state["user_stance"]

    def _format_docs(self, docs):
        parts = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            snippet = d.page_content[:600].replace("\n", " ").strip()
            parts.append(f"[Quelle {i} | {src} | Seite {page}]\n{snippet}")
        return "\n\n".join(parts) if parts else "Keine relevanten Dokumente gefunden."

    def handle_resistance(self, state: ConversationState):
        docs = self.rag.retrieve(state["query"])
        ctx = self._format_docs(docs)
        prompt = f"""
Der Nutzer wirkt widerständig/defensiv. Antworte im MI-Stil:
- Gefühle anerkennen (Reflexion)
- Keine Debatten
- 1–2 offene Fragen
- Kurze, ruhige Sätze
- Quellen am Ende (Format: [Dokument, Seite X])

Nutzer: {state['query']}
Kontext:
{ctx}

Antwort (Deutsch, MI-Stil, mit Quellen am Ende):
"""
        state["response"] = self.llm.invoke(prompt)
        return state

    def provide_information(self, state: ConversationState):
        docs = self.rag.retrieve(state["query"])
        ctx = self._format_docs(docs)
        prompt = f"""
Der Nutzer ist neugierig/offen. Antworte im MI-Stil:
- Kurze, sachliche Infos
- 1 offene Frage am Schluss
- Quellen am Ende (Format: [Dokument, Seite X])

Frage: {state['query']}
Kontext:
{ctx}

Antwort (Deutsch, MI-Stil, mit Quellen am Ende):
"""
        state["response"] = self.llm.invoke(prompt)
        return state

    def reinforce_understanding(self, state: ConversationState):
        docs = self.rag.retrieve(state["query"])
        ctx = self._format_docs(docs)
        prompt = f"""
Der Nutzer ist bereit für Infos. Antworte im MI-Stil:
- Klare, belegte Fakten
- Kurze Zusammenfassung (Reflexion)
- Autonomie betonen
- Quellen am Ende (Format: [Dokument, Seite X])

Aussage/Frage: {state['query']}
Kontext:
{ctx}

Antwort (Deutsch, MI-Stil, mit Quellen am Ende):
"""
        state["response"] = self.llm.invoke(prompt)
        return state

    def process(self, query: str, history: List[str] | None = None):
        initial_state: ConversationState = {
            "query": query,
            "response": "",
            "user_stance": "neutral",
            "chat_history": history or [],
        }
        result = self.graph.invoke(initial_state)
        return result.get("response", "")
