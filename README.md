# GraphRAG with FalkorDB, LangChain, and LangGraph

This project demonstrates a Retrieval-Augmented Generation (RAG) workflow using a graph database (FalkorDB) combined with LangChain and LangGraph. It uses a dataset of the top 1000 IMDB movies to build a knowledge graph and then answers questions about the movies, actors, and directors.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Python >=3.12,<4.0](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv)
- [An OpenAI API Key](https://platform.openai.com/account/api-keys)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jpsbali/graphrag-workflow-falkordb-langchain.git
cd graphrag-workflow-falkordb-langchain
```

### 2. Set up your Environment

**a. OpenAI API Key:**

Copy `.env_example`  to `.env` file in the root of the project and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

**b. FalkorDB:**

Start the FalkorDB instance using Docker Compose. This will also create a volume to persist the data.

```bash
docker-compose up -d
```

Docker misc commands
```bash
docker ps
docker logs -f falkordb
docker exec -it falkordb /bin/bash
docker exec -it falkordb redis-cli PING
docker debug falkordb
```

You can check if the database is running by connecting to it with a Redis client at `localhost:6379` or via the Web UI (see section 6 below)

**c. Python Dependencies:**

This project uses `uv` to manage dependencies in a bash like shell (Git Bash on Windows). 
After installing `uv`, you can sync and source your environment:

```bash
uv sync
source .venv/Scripts/activate
```

### 3. Dataset used in this project

This project uses the "IMDB Top 1000" dataset. The dataset for this project is already downloaded to data/imdb_top_1000.csv file. .

- **Dataset:** [IMDB Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- **File Name:** `data/imdb_top_1000.csv`

## 4. Running the Application

Once the setup is complete, you can run the GraphRAG application:

```bash
graphrag
```

The script will first:
1. Load and clean the data from `imdb_top_1000.csv`.
2. Populate the FalkorDB graph with movies, actors, directors, and genres.
3. Create a vector index on the movie overviews.

After the setup is complete, you will be prompted to ask questions. If the DB is already populated you will be asked to reuse the same DB or repopulate it.

## 5. Example Questions

Here are some questions you can ask the GraphRAG agent:

- "Which movies did Christopher Nolan direct?"
    `MATCH (p:Person {name: "Christopher Nolan"})-[:DIRECTED]->(m:Movie) 
    RETURN m.title`
- "Who acted in the movie 'The Dark Knight'?"
    `MATCH (m:Movie {title: "The Dark Knight"})-[:ACTED_IN]->(a:Person) RETURN a.name`
    `MATCH (m:Movie {title: "The Dark Knight"})<-[:ACTED_IN]-(a:Person) RETURN a.name`
    `MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Dark Knight'}) RETURN p.name AS actor`
- "What is the overview of the movie 'Inception'?"
    `MATCH (m:Movie {title: "Inception"}) RETURN m.overview`
- "Can you recommend a movie from the 'Action' genre?"
    `MATCH (g:Genre {name: "Action"})-[:IN_GENRE]->(m:Movie) RETURN m.title`
    `MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Action'}) RETURN m.title, m.overview, m.year ORDER BY m.year DESC LIMIT 10 ORDER BY m.year DESC LIMIT 10`
- "Tell me about a movie with a complex plot." (This will likely use vector search)
    `MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = "Thriller" OR g.name = "Mystery" OR g.name = "Drama" RETURN m.title, m.overview, m.year ORDER BY m.year DESC LIMIT 10`

## 6. Web Interface for the Falkor DB

Once the GraphRAG application is up you can access the Web UI

```bash
http://localhost:3000
```
## 7. Cleaning Up

To exit the application, type `exit` or `Ctrl-C`.

To stop and remove the FalkorDB container and the associated volume, run:

```bash
docker-compose down -v
```
