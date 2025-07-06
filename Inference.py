from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from requests_aws4auth import AWS4Auth
import boto3

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

region = "ap-south-1"
host = "search-findomain1-lgyucsnynjo3aejlv5cmxnp64q.ap-south-1.es.amazonaws.com"

# Get AWS credentials (frozen)
credentials = boto3.Session().get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

#Connect to OpenSearch
vectorstore = OpenSearchVectorSearch(
    index_name="financialinfo",
    opensearch_url=f"https://{host}",
    opensearch_client=client,
    embedding_function=embedding_function
     
)

#Integrate RAG with LangChain

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="mistral", temperature=0.0)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=3),
    return_source_documents=True
)

query = "What should I invest in for the next 5 years?"
result = qa_chain(query)

print("Answer:", result["result"])