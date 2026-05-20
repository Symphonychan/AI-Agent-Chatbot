from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage

from langgraph.prebuilt import create_react_agent


openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
       llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else [] 
    agent=create_react_agent( 
    model=groq_llm,
    tools=tools,
    prompt= "Act as an AI chatbot is smart friendly"
    
    )

    state={"messages":query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]
    return ai_messages[-1]
        