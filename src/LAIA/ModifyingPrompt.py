from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.4,
        max_tokens= 200
    )
    prompt = ChatPromptTemplate.from_template("""
    You are impersonating the identity and personality of El Clot. The local communities have given you data of their knowledge about their neighbourhood. You have to answer any question or query, taking into consideration the data given. If someone makes a question, answer. If someone gives a suggestion, say thanks. If someone has a complaint, give solutions. Always under answer it in less than 200 characters:
    Context: {context}
    Question: {input}
    """)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever, 
        create_stuff_documents_chain(llm=model, prompt=prompt)
    )
    return retrieval_chain
