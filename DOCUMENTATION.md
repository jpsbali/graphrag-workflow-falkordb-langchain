# Code Documentation

This document provides a detailed explanation of the code in `src/graphrag.py`.

## Overview

The `src/graphrag.py` script is the heart of the GraphRAG application. It sets up the environment, connects to the FalkorDB database, loads and processes data, builds the knowledge graph, and runs an interactive question-answering session powered by a LangGraph pipeline.

## Imports

The script imports necessary libraries, including:
- `os` and `dotenv` for environment variable management.
- `pandas` for data manipulation.
- `langchain_community.graphs.FalkorDBGraph` for interacting with FalkorDB.
- `langchain_openai` for using OpenAI's chat models and embeddings.
- `langchain_core` for prompts and output parsers.
- `langgraph` for building the stateful graph pipeline.

## Environment Setup

- The script loads the `OPENAI_API_KEY` from a `.env` file.
- It defines the connection details for the FalkorDB instance.

## `load_and_clean_data(file_path)`

- **Purpose:** Loads the IMDb dataset from the specified CSV file, cleans it, and returns a pandas DataFrame.
- **Cleaning Steps:**
    - Replaces 'PG' in the 'Released_Year' column with `None`.
    - Drops rows with no year.
    - Converts the 'Year' column to integer type.
    - Filters the DataFrame to include only movies released in or after the year 2000.
    - Takes the first 100 rows for faster processing during development.

## `populate_graph(df)`

- **Purpose:** Populates the FalkorDB graph with data from the DataFrame.
- **Actions:**
    - Creates indexes on `Movie(title)`, `Person(name)`, and `Genre(name)` for faster queries.
    - Iterates through the DataFrame and for each movie:
        - Creates a `Movie` node with `title`, `overview`, and `year` properties.
        - Creates `Genre` nodes and `IN_GENRE` relationships.
        - Creates a `Person` node for the director and a `DIRECTED` relationship.
        - Creates `Person` nodes for the actors and `ACTED_IN` relationships.

## `create_vector_index()`

- **Purpose:** Creates a vector index in FalkorDB and populates it with embeddings of movie overviews.
- **Actions:**
    - Creates a vector index on the `embedding` property of `Movie` nodes.
    - Retrieves movies that don't have an embedding.
    - For each of these movies, it generates an embedding for the `overview` using OpenAI's embedding model and stores it in the `embedding` property of the movie node.

## LangGraph State and Nodes

The script defines a `GraphState` TypedDict to manage the state of the RAG pipeline. The pipeline consists of several nodes, each performing a specific task.

### `router(state)`

- Routes the question to either `graph_qa` or `vector_search` based on keywords in the question.

### `decomposer(state)`

- Decomposes a complex question into simpler sub-questions using an LLM.

### `vector_search(state)`

- Performs a vector similarity search on movie overviews.

### `graph_qa(state)`

- Generates and executes a Cypher query against the knowledge graph to answer a question.
- It now includes a step to clean the generated Cypher query by removing markdown formatting.

### `graph_qa_with_context(state)`

- Augments graph queries with context from the vector search to answer questions that require both structured and unstructured data.

### `final_answer(state)`

- Generates a final, human-readable answer from the information gathered in the previous steps.

## `main()`

- The main function that orchestrates the entire workflow.
- It checks if the graph is already populated and asks the user if they want to reload the data.
- It assembles the LangGraph workflow by adding the nodes and defining the edges and conditional edges.
- It compiles the workflow into a runnable application.
- It starts an interactive Q&A loop where the user can ask questions.

## Execution Block

- The `if __name__ == "__main__":` block ensures that the `main()` function is called only when the script is executed directly.
