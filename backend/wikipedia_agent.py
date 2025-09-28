from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document

# --- Tool Initialization ---
# Using a wrapper to query Wikipedia
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2500))

def get_tool_router_node(llm):
    """
    Factory to create the router node with more sophisticated routing logic.
    """
    def tool_router_node(state):
        """
        Routes the question to 'wikipedia', 'vectorstore', or 'fallback'.
        """
        print("---NODE: TOOL ROUTER---")
        question = state["condensed_question"]
        
        # This new prompt includes the specific logic you requested.
        prompt = PromptTemplate.from_template(
            """You are an expert at routing a user's question to the best data source.
            Given the question, decide whether it is best answered by:

            1. 'vectorstore': Use this for specific questions that seem to be about the user's private, uploaded study documents. These might be detailed, technical, or specific to a narrow domain. This is the default for document-related queries.

            2. 'wikipedia': Use this ONLY if the question is a factual query (e.g., starts with "Who is...", "What is...") AND the user explicitly asks to use Wikipedia (e.g., includes "on wiki", "from wikipedia", "using wikipedia").

            3. 'fallback': Use this for general conversation, greetings, or any general knowledge question where the user has NOT explicitly asked for Wikipedia.

            Return a single word: 'vectorstore', 'wikipedia', or 'fallback'.

            User Question: {question}
            """
        )
        
        router_chain = prompt | llm | StrOutputParser()
        source = router_chain.invoke({"question": question})
        
        # Default to fallback for safety if the model returns an unexpected value
        if source.lower() not in ["vectorstore", "wikipedia", "fallback"]:
            source = "fallback"
            
        print(f"---DECISION: Router chose '{source}'---")
        return {"source": source.lower()}
    return tool_router_node

def run_wikipedia_node(state, llm):
    """
    Extracts the core topic, runs the Wikipedia tool with error handling, 
    and adds the source URL to the metadata.
    """
    print("---NODE: RUN WIKIPEDIA---")
    question = state["condensed_question"]

    # Chain to extract the actual search query from the user's question
    prompt = PromptTemplate.from_template(
        """You are an expert at extracting search terms. From the following user question, extract only the core subject or topic the user wants to search for on Wikipedia.
        For example, if the user asks 'tell me about the solar system on wikipedia', you should extract 'solar system'.
        
        User Question: {question}
        
        Search Term:"""
    )
    
    query_extractor_chain = prompt | llm | StrOutputParser()
    search_query = query_extractor_chain.invoke({"question": question})
    
    print(f"---INFO: Extracted search query '{search_query}' for Wikipedia---")
    
    # --- ADDED ERROR HANDLING ---
    try:
        # Run the tool with the cleaned search query
        wiki_result = wikipedia_tool.invoke(search_query)
        # Create the source URL
        url_query = search_query.replace(" ", "_")
        source_url = f"https://en.wikipedia.org/wiki/{url_query}"
        metadata = {"source": "wikipedia", "source_url": source_url}
    except Exception as e:
        print(f"---ERROR: Wikipedia tool failed with error: {e}---")
        wiki_result = f"Sorry, I couldn't find anything on Wikipedia for '{search_query}'. It might be a disambiguation page or the topic may not exist."
        metadata = {"source": "wikipedia", "source_url": None}
    
    # Create the Document with the result and metadata
    documents = [Document(page_content=wiki_result, metadata=metadata)]
    
    return {"documents": documents}