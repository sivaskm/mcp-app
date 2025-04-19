import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

mcp = FastMCP("Kural")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "thirukural"
index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")

@mcp.tool()
def get_relevant_kurals(user_input: str) -> str:
    """Get relevant Thirurkurals for the User's question/input

    Args:
        user_input: User's question or input
    """
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    results = vector_store.similarity_search_with_score(user_input, k=10)
    context_list = []

    for doc in results:
        context_list.append((f"Number: {doc[0].metadata["Number"]}\n"
                f"Thirukural: {doc[0].metadata["kural"]}\n"
                f"Short Explanation: {doc[0].page_content}\n"
                f"Adhikaram: {doc[0].metadata["adikaram_name"]}\n"
                f"Iyal: {doc[0].metadata["iyal_name"]}\n"
                f"Paal: {doc[0].metadata["paul_name"]}\n"
                f"Paal in English: {doc[0].metadata["paul_translation"]}\n"
                f"Explanation by Mu Varadarasanar: {doc[0].metadata["Mu Varadarasanar"]}\n"
                f"Explanation by Solomon Pappaiah {doc[0].metadata["Solomon Pappaiah"]}\n"
                f"Explanation by Muthuvel Karunanidhi: {doc[0].metadata["Muthuvel Karunanidhi"]}\n"))
    return "\n---\n".join(context_list)

if __name__ == "__main__":
    mcp.run(transport='stdio')
