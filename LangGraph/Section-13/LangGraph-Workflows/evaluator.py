# Evaluator Optimzer Workflow


import os
from dotenv import load_dotenv
from typing_extensions import TypedDict,Literal
from pydantic import BaseModel , Field
from langchain_groq import ChatGroq
from langgraph.graph import START,StateGraph,END

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_aPI_KEY")

llm=ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    fact:str
    topic:str
    feedback:str
    real_orNot:str
    
    
class Feedback(BaseModel):
    grade:Literal["real","not_real"]=Field(
        description="tell me if the fact  is real or not?",
    )
    feedaback:str = Field(
        description= "if the fact is not real give me feedback about it",
    )

evaluator=llm.with_structured_output(Feedback)

    
def llm_generator(state:State):
    if state.get("feedback"):
        msg=llm.invoke(f"Regenerate a fact about: {state["topic"]} and also see the feedback for improvement {state["feedback"]}")
        return{"fact":msg.content.strip()}
    else:
        msg=llm.invoke(f"generate a fact of this topic: {state["topic"]}")
        return{"fact":msg.content.strip()}
    
    
def llm_evaluator(state:State):
    grade=evaluator.invoke(f"Grade the fact: {state["fact"]}")
    return{"real_orNot":grade.grade , "feedback":grade.feedaback}

    
def router(state:State):
    if state["real_orNot"] == "real":
        return "Approved"
    
    elif state["real_orNot"] == "not_real":
        return "Rejected + Feedback"
    

graph=StateGraph(State)

graph.add_node("llm_generator" , llm_generator)
graph.add_node("llm_evaluator" , llm_evaluator)

graph.add_edge(START, "llm_generator")           
graph.add_edge("llm_generator", "llm_evaluator")

graph.add_conditional_edges(
    "llm_evaluator", 
    router,
    {
        "Approved" : END,
        "Rejected + Feedback" : "llm_generator",
    },
    )            


optimzer=graph.compile()

result=optimzer.invoke(
    {
        "topic" : "Write real fact about hailey Bieber."
    }
)

print("Fact:", result["fact"])
print("Real or Not:", result.get("real_orNot"))
print("Feedback:", result.get("feedback"))

    
    
        