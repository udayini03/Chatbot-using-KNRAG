import streamlit as st
from langchain_community.graphs import Neo4jGraph
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import create_react_agent
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from typing import Any, Dict, Union, List
# from serpapi import GoogleSearch
import serpapi
from langchain.tools.base import Tool

def llm_agent():

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = "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"
    # Connect to Neo4j database

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(neo4j_uri, neo4j_username, neo4j_password)
    # Declare df globally as an empty DataFrame
    df = pd.DataFrame()
    uploaded_file = st.file_uploader("Upload a file:")
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)

    # Initialize conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        
        # movies_query = f"""
        # LOAD CSV WITH HEADERS FROM text = '{uploaded_file.getvalue().decode("utf-8")}'
        # AS row
        # MERGE (m:Movie {{id:row.movieId}})
        # SET m.released = date(row.released),
        #     m.title = row.title,
        #     m.imdbRating = toFloat(row.imdbRating)
        # FOREACH (director in split(replace(row.director, "'", "\'"), '|') |
        #     MERGE (p:Person {{name:trim(director)}})
        #     MERGE (p)-[:DIRECTED]->(m))
        # FOREACH (actor in split(replace(row.actors, "'", "\'"), '|') |
        #     MERGE (p:Person {{name:trim(actor)}})
        #     MERGE (p)-[:ACTED_IN]->(m))
        # FOREACH (genre in split(row.genres, '|') |
        #     MERGE (g:Genre {{name:trim(genre)}})
        #     MERGE (m)-[:IN_GENRE]->(g))
        # """
        # graph.query(movies_query)
        # graph.refresh_schema()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    # chain2 = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, validate_cypher=True,top_k=100)

    os.environ["SERPAPI_API_KEY"] = "b28b058c39f2cf579250a2dd8b4c683178ed3edf15a6b7ec9c01531dc7a9cb40"
    serp = "b28b058c39f2cf579250a2dd8b4c683178ed3edf15a6b7ec9c01531dc7a9cb40"
    
    tool_DA = Tool(
        name="Pandas Dataframe Agent",
        func=create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors": True},
        ),
        description="A tool that can answer questions about a pandas dataframe.",
        # Add arguments dictionary
        output_schema={"arguments": {}},
    )

    Graph_tool = Tool(
        name="Graph Cypher QA Chain",
        func=GraphCypherQAChain.from_llm(graph=graph, llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), validate_cypher=True, top_k=50),
        description="A tool that can answer questions using a graph database and Cypher queries.",
    )

    def search_with_serpapi(query: str, num_results: int = 1) -> dict:
        """Search for information using the SERP API.

        Args:
            query: The search query.
            num_results: The number of results to return.

        Returns:
            A dictionary containing the search results.
        """
        params = {
            "api_key": "b28b058c39f2cf579250a2dd8b4c683178ed3edf15a6b7ec9c01531dc7a9cb40",
            "engine": "google",
            "q": query,
            "num": num_results,
            "location": "India",
            "google_domain": "google.co.in",
            "gl": "in",
            "hl": "en"
        }
        # Create a GoogleSearch object
        search1 = serpapi.search(params)
        results = search1.as_dict()
        return results

    tool_SERP = Tool(
        name="SERP API",
        func=search_with_serpapi,
        description="A tool that can search for information using the SERP API.",
    )

    # Switch button and code execution
    switch_state = st.checkbox("Use Knowledge Graph RAG")

    user_query = st.text_input("Ask me anything about movies or uploaded file for analysis:")
    button_clicked = st.button("Ask the Chatbot!")

    if button_clicked:
        if user_query is not None:
            st.session_state.conversation_history.append(user_query)

            if switch_state:  # Without GS
                tools = [tool_DA, Graph_tool]

                agent_without_GS = initialize_agent(
                    tools,
                    llm,
                    agent="zero-shot-react-description",
                    agent_executor_kwargs={
                        "handle_parsing_errors": True,
                    },
                    verbose=True,
                )

                response = agent_without_GS.run(user_query)
            else:  # With GS
                tools = [tool_DA, Graph_tool, tool_SERP]

                agent_with_GS = initialize_agent(
                    tools,
                    llm,
                    agent="zero-shot-react-description",
                    agent_executor_kwargs={
                        "handle_parsing_errors": True,
                    },
                    verbose=True,
                )

                response = agent_with_GS.run(user_query)
            st.write("Result:")
            st.write(response)

            st.session_state.conversation_history.append(response)

            # Update the conversation history in the sidebar
            with st.sidebar:
                st.title("Conversation History")
                for i, entry in enumerate(st.session_state.conversation_history):
                    if i % 2 == 0:
                        st.write(f"You: {entry}")
                    else:
                        st.write(f"Bot: {entry}")
        else:
            st.warning("Enter text before asking Chatbot")