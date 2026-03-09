import os
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.graphs import FalkorDBGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Environment Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- FalkorDB Connection ---
DB_HOST = "localhost"
DB_PORT = 6379
DB_PASSWORD = None # No password by default for falkordb docker
FALKOR_GRAPH_ID = "imdb"

graph = FalkorDBGraph(
    database=FALKOR_GRAPH_ID,
    host=DB_HOST,
    port=DB_PORT,
    password=DB_PASSWORD
)

# --- LLM and Embeddings ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# --- Data Loading and Preprocessing ---
def load_and_clean_data(file_path="data/imdb_top_1000.csv"):
    """Loads and cleans the IMDB dataset."""
    print("Loading and cleaning data...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please download it first.")
        
    df = pd.read_csv(file_path)
    df['Year'] = df['Released_Year'].replace('PG', None)
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df[df['Year'] >= 2000]
    df = df.head(100) # Smaller subset for faster processing
    print("Data loaded and cleaned.")
    return df

# --- Graph Population ---
def populate_graph(df):
    """Populates the FalkorDB graph with movie data."""
    print("Populating graph...")
    
    # Create Indexes (skip if they already exist)
    try:
        graph.query("CREATE INDEX FOR (m:Movie) ON (m.title)")
    except Exception as e:
        print(f"Movie title index already exists: {e}")
    
    try:
        graph.query("CREATE INDEX FOR (p:Person) ON (p.name)")
    except Exception as e:
        print(f"Person name index already exists: {e}")
    
    try:
        graph.query("CREATE INDEX FOR (g:Genre) ON (g.name)")
    except Exception as e:
        print(f"Genre name index already exists: {e}")

    for index, row in df.iterrows():
        # Create Movie node
        graph.query(
            "CREATE (m:Movie {title: $title, overview: $overview, year: $year})",
            {"title": row["Series_Title"], "overview": row["Overview"], "year": int(row["Year"])},
        )
        # Create Genre nodes and relationships
        genres = row["Genre"].split(", ")
        for genre_name in genres:
            graph.query("MERGE (g:Genre {name: $name})", {"name": genre_name})
            graph.query(
                "MATCH (m:Movie {title: $movie_title}), (g:Genre {name: $genre_name}) CREATE (m)-[:IN_GENRE]->(g)",
                {"movie_title": row["Series_Title"], "genre_name": genre_name},
            )
        # Create Director nodes and relationships
        director_name = row["Director"]
        graph.query("MERGE (p:Person {name: $name})", {"name": director_name})
        graph.query(
            "MATCH (m:Movie {title: $movie_title}), (p:Person {name: $director_name}) CREATE (p)-[:DIRECTED]->(m)",
            {"movie_title": row["Series_Title"], "director_name": director_name},
        )
        # Create Actor nodes and relationships
        actors = [row["Star1"], row["Star2"], row["Star3"], row["Star4"]]
        for actor_name in actors:
            graph.query("MERGE (p:Person {name: $name})", {"name": actor_name})
            graph.query(
                "MATCH (m:Movie {title: $movie_title}), (p:Person {name: $actor_name}) CREATE (p)-[:ACTED_IN]->(m)",
                {"movie_title": row["Series_Title"], "actor_name": actor_name},
            )
    print("Graph populated.")

# --- Vector Index ---
def create_vector_index():
    """Creates a vector index on movie overviews."""
    print("Creating vector index...")
    try:
        graph.query(
            "CREATE VECTOR INDEX FOR (m:Movie) ON (m.embedding) OPTIONS {dimension: 1536, similarityFunction: 'cosine'}"
        )
    except Exception as e:
        print(f"Index likely already exists: {e}")

    # Populate embeddings
    result = graph.query("MATCH (m:Movie) WHERE m.embedding IS NULL RETURN m.overview, m.title")
    # Result is now a list directly, not an object with result_set
    for row in result:
        overview = row[0]
        title = row[1]
        embedding = embeddings.embed_query(overview)
        graph.query(
            "MATCH (m:Movie {title: $title}) SET m.embedding = vecf32($embedding)",
            {"title": title, "embedding": embedding},
        )
    print("Vector index created and embeddings populated.")


# --- LangGraph State ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    documents: List[Document]
    
# --- LangGraph Nodes ---

def router(state):
    """
    Routes question to graph_qa or vector_search.
    """
    print("---ROUTING QUESTION---")
    # Simple routing logic for demonstration
    if "actor" in state["question"].lower() or "director" in state["question"].lower() or "movie" in state["question"].lower() and "overview" not in state["question"].lower() :
        return "graph_qa"
    else:
        return "vector_search"


def decomposer(state):
    """
    Decomposes a complex question into sub-questions.
    """
    print("---DECOMPOSING QUESTION---")
    
    prompt = PromptTemplate(
        template="""
        You are an expert in decomposing complex questions into sub-questions.
        Decompose the following question into sub-questions that can be answered by a graph or vector store.
        Question: {question}
        Sub-questions:
        """,
        input_variables=["question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    sub_questions = chain.invoke({"question": state["question"]}).split('\n')
    
    # For simplicity, we'll just process the first sub-question in this implementation
    # A more advanced implementation would loop through all sub-questions
    state['question'] = sub_questions[0]
    
    return state
    
def vector_search(state):
    """
    Performs a vector search on the movie overviews.
    """
    print("---PERFORMING VECTOR SEARCH---")
    question = state["question"]
    
    query_embedding = embeddings.embed_query(question)
    
    # Find similar movie overviews using the correct FalkorDB syntax
    result = graph.query("""
        CALL db.idx.vector.queryNodes('Movie', 'embedding', $k, vecf32($query_embedding)) 
        YIELD node
        RETURN node.title, node.overview, node.year
    """, {"k": 3, "query_embedding": query_embedding})
    
    # Result is now a list directly
    docs = [Document(page_content=f"Title: {row[0]}, Year: {row[2]}\nOverview: {row[1]}", metadata={'source': 'vector_search'}) for row in result]

    state['documents'] = docs
    state['context'] = "\n\n".join([doc.page_content for doc in docs])
    
    return state

def graph_qa(state):
    """
    Answers a question by querying the graph structure.
    """
    print("---PERFORMING GRAPH QA---")
    question = state["question"]
    
    # Generate a Cypher query from the question
    prompt = PromptTemplate(
        template="""
        You are an expert in generating Cypher queries for a movie database.
        The schema is as follows:
        - Nodes: Movie, Person, Genre
        - Relationships: ACTED_IN, DIRECTED, IN_GENRE
        - Movie properties: title, overview, year, embedding
        - Person properties: name
        - Genre properties: name
        
        Based on the user's question, generate a Cypher query to retrieve the answer.
        Only return the Cypher query, no other text.
        
        Question: {question}
        Cypher Query:
        """,
        input_variables=["question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    cypher_query = chain.invoke({"question": question})
    
    print(f"Generated Cypher: {cypher_query}")
    
    cleaned_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

    try:
        result = graph.query(cleaned_query)
        # Result is now a list directly
        answer = str(result)
    except Exception as e:
        answer = f"Error executing query: {e}"
        
    state['answer'] = answer
    
    return state

def graph_qa_with_context(state):
    """
    Augments graph queries with information from the vector search.
    """
    print("---PERFORMING GRAPH QA WITH CONTEXT---")
    question = state["question"]
    context = state["context"]
    
    prompt = PromptTemplate(
        template="""
        You are an expert in answering questions about movies.
        Use the provided context to help answer the question.
        
        Context:
        {context}
        
        Question: {question}
        Answer:
        """,
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    state['answer'] = answer
    
    return state
    
def final_answer(state):
    """
    Generates a final answer from the state.
    """
    print("---GENERATING FINAL ANSWER---")
    if 'answer' in state and state['answer']:
        # Answer from graph_qa
        pass
    elif 'context' in state and state['context']:
        # Answer from vector search (needs summarization/generation)
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Summarize the following context to answer the user's question.
            
            Context:
            {context}
            
            Question: {question}
            Answer:
            """,
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm | StrOutputParser()
        state['answer'] = chain.invoke({"context": state['context'], "question": state['question']})
    else:
        state['answer'] = "Sorry, I couldn't find an answer to your question."
        
    return state

# --- Main Execution ---
def main():
    """Main function to run the GraphRAG workflow."""

    # --- Initial Data and Graph Setup ---
    try:
        # Check if data already exists
        result = graph.query("MATCH (m:Movie) RETURN count(m) as count")
        movie_count = result[0][0] if result else 0
        
        if movie_count > 0:
            print(f"Found {movie_count} existing movies in database.")
            user_input = input("Do you want to clear and reload the data? (y/n): ").lower()
            if user_input == 'y':
                print("Clearing existing data...")
                graph.query("MATCH (n) DETACH DELETE n")
                print("Data cleared.")
                df = load_and_clean_data()
                populate_graph(df)
                create_vector_index()
            else:
                print("Using existing data.")
        else:
            df = load_and_clean_data()
            populate_graph(df)
            create_vector_index()
    except FileNotFoundError:
        print("Please make sure the 'imdb_top_1000.csv' file is located in the 'data' directory.")
        print("You can find it on Kaggle: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
        return
        
    # --- Assemble LangGraph ---
    workflow = StateGraph(GraphState)

    workflow.add_node("decomposer", decomposer)
    workflow.add_node("vector_search", vector_search)
    workflow.add_node("graph_qa", graph_qa)
    workflow.add_node("graph_qa_with_context", graph_qa_with_context)
    workflow.add_node("final_answer", final_answer)

    workflow.set_entry_point("decomposer")

    workflow.add_conditional_edges(
        "decomposer",
        router,
        {
            "graph_qa": "graph_qa",
            "vector_search": "vector_search",
        }
    )
    
    workflow.add_edge("graph_qa", "final_answer")
    workflow.add_edge("vector_search", "graph_qa_with_context")
    workflow.add_edge("graph_qa_with_context", "final_answer")
    workflow.add_edge("final_answer", END)
    
    app = workflow.compile()
    
    # --- Interactive Q&A ---
    print("\n--- GraphRAG is ready. Ask a question! (type 'exit' to quit) ---")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
        
        inputs = {"question": question}
        result = app.invoke(inputs)
        
        print(f"Agent: {result['answer']}")

if __name__ == "__main__":
    main()
