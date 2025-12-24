#  orchestrator worker 
# a central dynamically breakdown task, assgn them to worker llm and synthesizes their results.
# Agentic AI system using an orchestrator–worker pattern where an LLM dynamically plans and 
# executes analysis steps via a state-driven workflow.”

#  AI Research Assistant
#  For Example User asks: Give me a short market analysis for an AI startup in healthcare.”


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import START,StateGraph,END

load_dotenv()
os.environ["GROQ_API_kEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama-3.1-8b-instant")


#This state stores:
class State(TypedDict):
    query:str
    market:str
    risks:str
    opportunity:str
    next_step:str
    final_report:str 
    
    
def orchestrator(state: State):
    prompt = f"""
    Query: {state['query']}
    
    Completed analysis:
    - Market: {'Done' if state.get('market') else 'Not done'}
    - Risks: {'Done' if state.get('risks') else 'Not done'}
    - Opportunity: {'Done' if state.get('opportunity') else 'Not done'}
    
    What should we do next? Choose ONE:
    - "market" - if market analysis needed
    - "risk" - if risk analysis needed
    - "opportunity" - if opportunity analysis needed
    - "synthesize" - if all analysis is complete
    
    Respond with ONLY one word: market, risk, opportunity, or synthesize
    """
    
    decision = llm.invoke(prompt)
    
    #Cleans and normalizes the LLM response. .strip() Removes whitespace and newlines and .lower() Normalizes casing Ensures "Market", "MARKET", "market" all behave the same
    next_step = decision.content.strip().lower()
    
    print(f" Orchestrator decided: {next_step}")
    return {"next_step": next_step}


def route_next(state: State):
    next_step = state.get("next_step","")
    
    if next_step == "market":
        return "market_worker"
    elif next_step == "risk":
        return "risk_worker"
    elif next_step == "opportunity":
        return "opportunity_worker"
    elif next_step == "synthesize":
        return "synthesizer"
    else:
        return "end"
    
    
def market_worker(state:State):
    msg=llm.invoke(f"Analyze the market for: , {state["query"]}")
    return {"market":msg.content}

def risk_worker(state:State):
    msg=llm.invoke(f"Identify risk for: ,{state["query"]}")
    return {"risks":msg.content}

def opportunity_worker(state:State):
    msg=llm.invoke(f"Find the opportunity for: ,{state["query"]}")
    return {"opportunity":msg.content}

def synthesizer(state: State):
    result = llm.invoke(
        f"""
        Create a short market report using:
        Market: {state['market']}
        Risks: {state['risks']}
        Opportunity: {state['opportunity']}
        """
    )
    return {"final_report": result.content}


graph=StateGraph(State)


# Add all nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("market_worker", market_worker)
graph.add_node("risk_worker", risk_worker)
graph.add_node("opportunity_worker", opportunity_worker)
graph.add_node("synthesizer", synthesizer)


# Set entry point - everything starts at orchestrator
graph.add_edge(START , "orchestrator")

# Orchestrator uses conditional routing to decide what happens next
graph.add_conditional_edges(
    "orchestrator",  # From orchestrator
    route_next,      # Use this function to decide where to go
    {
        "market_worker": "market_worker",
        "risk_worker": "risk_worker",
        "opportunity_worker": "opportunity_worker",
        "synthesizer": "synthesizer",
        "end": END
    }
)

# After each worker finishes, go BACK to orchestrator for next decision
graph.add_edge("market_worker", "orchestrator")
graph.add_edge("risk_worker", "orchestrator")
graph.add_edge("opportunity_worker", "orchestrator")

# After synthesis, we're done
graph.add_edge("synthesizer", END)

compiled_graph=graph.compile()

result=compiled_graph.invoke(
    {
        "query":"Give me a short analysis for an AI startup in healthcare."
    }
)

print(result["final_report"])   
