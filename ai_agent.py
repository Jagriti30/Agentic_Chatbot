#1st Step: Setting-up API Keys for Groq & Travily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#2nd Step: Setting-up LLM & tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq("llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#3rd Step: Setting-up AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent

system_prompt="Act as an AI chatbot who is smart and friendly"

agent=create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    state_modifier=system_prompt
)

query="Tell me about th trends in crypto markets"
state={"messages": query}
response=agent.invoke(state)
messages=response.get("messages")
ai_messages=[messages.content for message in messages if isinstance(message, AIMessage)]
print(ai_messages[-1])
