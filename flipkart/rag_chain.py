from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL , temperature=0.5)
        self.history_store={}

    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        # Rewrite the user question using the context of previous conversations
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
Return your response in plain HTML (not markdown), using user-friendly formatting such as bullet points (<ul><li>), short paragraphs (<p>), or tables (<table>) if appropriate.
Stick to the provided context. Be concise and helpful.

CONTEXT:
{context}

QUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  # Original input given by the user
        ])

        # qa_prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
        #                   Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""), # This input has been modified by the context_prompt
        #     MessagesPlaceholder(variable_name="chat_history"), 
        #     ("human", "{input}")  # Original input given by the user
        # ])        

        # The history_aware_retriever is responsible for taking into account the previous conversation history
        # when retrieving relevant documents. It uses the context_prompt to rewrite the user's current question
        # as a standalone query, incorporating information from the chat history. This helps the retriever
        # fetch more contextually relevant documents from the vector store, especially when the user's question
        # depends on earlier parts of the conversation.
        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        )

        # The question_answer_chain is responsible for taking the retrieved documents (context)
        # and generating a final answer to the user's question. It uses the provided language model (self.model)
        # and the qa_prompt, which instructs the model to answer as an e-commerce bot using only the given context.
        # The chain "stuffs" (concatenates) all retrieved documents into the prompt, so the model can reference them
        # directly when formulating its answer. This ensures that the response is grounded in the actual product reviews and titles.
        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            # input_messages_key specifies the key in the input dictionary that contains the user's message.
            input_messages_key="input",
            # history_messages_key specifies the key used to store and retrieve the chat history for the session.
            history_messages_key="chat_history",
            # output_messages_key specifies the key in the output dictionary where the model's answer will be stored.
            output_messages_key="answer"
        )