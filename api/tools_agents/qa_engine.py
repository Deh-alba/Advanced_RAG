# Workflow
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver


from langchain_core.messages import SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

import os
import logging
from dotenv import load_dotenv

# agent
import getpass
import os
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END

from typing import List

from langchain_core.documents import Document

from langgraph.prebuilt import ToolNode, tools_condition

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver


from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError


from langchain_openai import OpenAIEmbeddings

from datatabase import DBVector


from langchain_chroma import Chroma


# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

load_dotenv(dotenv_path="tools_agents/.env")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY



class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    messages: MessagesState
    sources: list

class OverallState(TypedDict):
    messages: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str


class State(MessagesState):
    context: List[Document]


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )



@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant information based on the query."""
    logging.info('retrieve-> ')

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # PGVector connection settings

    collection_name = "my_docs"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db_cosine",
    )

    retrieved_docs = vector_store.similarity_search_with_relevance_scores(
        query, 
        k=10
    )

    logging.info(f"Retrieved documents: {retrieved_docs}")

    docs = []
    for i in retrieved_docs:
 
        docs.append(i[0])

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in docs
    )
    
    return serialized, docs


class QAGraphEngineAgent:

    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    def __init__(self):

        self.llm = init_chat_model("gpt-4o", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.response_model = init_chat_model("openai:gpt-4.1", temperature=0)

        self.grader_model = init_chat_model("openai:gpt-4.1", temperature=0.7)

        self.checkpointer = InMemorySaver()
    
        #self.vector_store = self._start_db()

        self.graph = self._build_graph()


    def _start_db(self):
        return DBVector()


    def query_or_respond(self, state: MessagesState) -> MessagesState:
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])

        logging.info('query_or_respond-> ')
        logging.info(state['messages'])
        logging.info(response)


        return {"messages": [response]}


    def generate(self, state: MessagesState) -> OutputState:
        """
        Generates an answer to a user's question based on the provided conversation state and retrieved tool messages.
        This method performs the following steps:
        1. Extracts the most recent tool messages from the conversation state.
        2. Formats the content of these tool messages into a system prompt, instructing the assistant to answer using the provided context and to include image paths where relevant.
        3. Constructs the conversation prompt by combining the system message with relevant human, system, and AI messages (excluding tool calls).
        4. Invokes the language model to generate a response based on the constructed prompt.
        5. Collects artifact metadata from the tool messages to extract unique source identifiers.
        6. Returns the generated response along with the list of unique sources used in the answer.
        Args:
            state (MessagesState): The current state of the conversation, including all messages exchanged.
        Returns:
            OutputState: A dictionary containing the generated messages and a list of unique sources referenced in the answer.
        """
        """Generate answer."""
        # Get generated ToolMessages
        logging.info('generate-> ')

        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks based about standard operating procedure."
            "The documentos about standard operating procedure are in "
            "Use the following pieces of retrieved context to answer the question. "
            "You must add the images paths to answer like this: [<image_path>]. "
            "if you don't have an answer in the context write 'I don't know. Can you rephrase the question?'."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)

        
        context = []
        for tool_message in tool_messages:
            context.extend(tool_message.artifact)



        sources = set()
        for i in context:
            if 'source' in i.metadata:
                sources.add(i.metadata['source'])
        
        logging.info(f"Sources: {sources}")

        return {"messages": [response],"sources":list(sources)}


    def grade_documents(self, state: MessagesState) -> Literal["generate", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        #question = state["user_input"]
        context = state["messages"][-1].content
        
        
        
        prompt = self.GRADE_PROMPT.format(question=context, context=context)
        response = (
            self.grader_model
            .with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score


        if score == "yes":
            return "generate"
        else:
            return "rewrite_question"


    def rewrite_question(self, state: MessagesState):
        """
        Rewrites the original user question using a predefined prompt and a response model.
        Args:
            state (MessagesState): The current state containing the conversation messages.
        Returns:
            dict: A dictionary with a single key "messages", containing a list with the rewritten user question.
        """
        """Rewrite the original user question."""
        logging.info('rewrite_question-> ')

        messages = state["messages"]
        #question = state["user_input"]
        prompt = self.REWRITE_PROMPT.format(question=messages)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])

        
        return {"messages": [{"role": "user", "content": response.content}]}


    def _build_graph(self):
        """
        Constructs and compiles a state graph for the QA engine workflow.
        This method defines the nodes and edges of the state graph, representing the flow of operations such as querying, retrieving, rewriting questions, and generating responses. Conditional edges are used to determine the next step based on tool decisions and agent assessments. The resulting graph is compiled with an optional checkpointer for state management.
        Returns:
            StateGraph: The compiled state graph representing the QA engine's workflow.
        """

        builder = StateGraph(MessagesState, output_schema=OutputState)
        builder.add_node("query_or_respond", self.query_or_respond)
        builder.add_node("retrieve" ,ToolNode([retrieve]))

        builder.add_node('rewrite_question',self.rewrite_question)
        builder.add_node('generate',self.generate)


        builder.add_edge(START, "query_or_respond")


        builder.add_conditional_edges(
            "query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        builder.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )



        builder.add_edge("rewrite_question", "query_or_respond")
        

        #builder.add_edge("generic_ret", END)
        builder.add_edge("generate", END)

        graph = builder.compile(checkpointer=self.checkpointer)


        return graph



    async def generate_response(self, input_message, user_id):
        """
        Asynchronously generates a response to a user's input message using a graph-based QA engine.
        Args:
            input_message (str): The user's input message to process.
            user_id (str): The unique identifier for the user, used to configure the session/thread.
        Returns:
            dict: A dictionary containing:
                - "answer" (str): The generated answer or a fallback message if no answer is found.
                - "sources" (list): A list of sources related to the answer (empty if none).
        Raises:
            None: Handles GraphRecursionError internally and returns a fallback response.
        """
        config = {"configurable": {"thread_id": user_id}}
        
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": user_id})



        try:

            result = self.graph.invoke({"messages": [{"role": "user", "content": input_message}]}, config=config)


        except GraphRecursionError:
            snapshot = self.graph.get_state(config)  # returns StateSnapshot
            state_dict = snapshot.values         # extract the actual state dict
            partial = state_dict.get("graph_output")

            result = partial
            
            if isinstance(result, str):
                return {"answer": 'I didn’t find a top answer in my knowledge base. Would you mind rephrasing your question or providing more context?', "sources": []}


        if 'sources' not in result:
            result['sources'] = []

        return {"answer": result['messages'][-1].content, "sources": result['sources']}


