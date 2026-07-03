from bson import ObjectId
from pymongo import MongoClient

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from presidio_analyzer import AnalyzerEngine

class UnstructuredDataAgent:
    """
    MongoDB + Chroma + Azure OpenAI RAG pipeline
    """

    # ============================================
    # INIT
    # ============================================
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        chroma_path: str,
        azure_endpoint: str,
        azure_key: str,
        embedding_deployment: str,
        chat_deployment: str,
        embedding_api_version: str = "2023-06-01-preview",
        chat_api_version: str = "2024-12-01-preview",
        collection_label: str = "rag_collection",
        temperature: float = 0.0,
    ):
        # Mongo
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[db_name]
        self.mongo_coll = self.mongo_db[collection_name]

        # Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=embedding_api_version,
            api_key=azure_key,
        )

        # Vector DB
        self.vector_db = Chroma(
            collection_name=collection_label,
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
        )

        # Chat model
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=chat_api_version,
            deployment_name=chat_deployment,
            temperature=temperature,
        )

    # ============================================
    # BUILD VECTOR INDEX
    # ============================================
    def build_index(self):
        print("Loading documents from MongoDB...")

        docs = list(self.mongo_coll.find({}, {"_id": 1, "content": 1}))

        texts = []
        ids = []
        metadatas = []

        for d in docs:
            doc_id = str(d["_id"])
            text = d.get("content", "").strip()

            if not text:
                continue

            texts.append(text)
            ids.append(doc_id)
            metadatas.append({"mongo_id": doc_id})

        if not texts:
            print("No documents found to index.")
            return

        print(f"Embedding {len(texts)} documents...")

        self.vector_db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

        print("Vector index built.")
        print("Vector DB size:", self.vector_db._collection.count())

    # ============================================
    # RAG QUERY
    # ============================================
    def ask(self, question: str, k: int = 3, run_pii_audit=True):
        results = self.vector_db.similarity_search(question, k=k)

        context_chunks = []
        source_docs = []

        for doc in results:
            mongo_id = doc.metadata.get("mongo_id")

            try:
                mongo_doc = self.mongo_coll.find_one({"_id": ObjectId(mongo_id)})
            except Exception:
                mongo_doc = None

            if mongo_doc and mongo_doc.get("content"):
                content = mongo_doc["content"]
                context_chunks.append(content)

                # store full document metadata for transparency
                source_docs.append({
                    "mongo_id": str(mongo_doc["_id"]),
                    "content": content
                })

        if not context_chunks:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
Use the following documents as context:

{context_text}

Question: {question}

Answer clearly and concisely:
"""

        response = self.llm.invoke(prompt)

        if run_pii_audit:
            print("\n" + "=" * 80)
            print("Using Unstrucutred Data stored in MongoDB database to answer question")
            print("Seraching for PII data in source documents")
            print("=" * 80)
            self.__contains_pii(str(source_docs))

        return {
            "question": question,
            "response": response.content,
            "sources": source_docs,
        }


    # ============================================
    # OPTIONAL — RESET VECTOR DB
    # ============================================
    def reset_index(self):
        print("Clearing vector database...")
        self.vector_db.delete_collection()
        print("Vector DB cleared.")

    # ============================================
    # PRIVATE METHODS 
    # ============================================

    def __contains_pii(self,text: str) -> bool:
        analyzer = AnalyzerEngine()

        results = analyzer.analyze(
            text=text,
            language="en"
        )

        if results:
            seen = set()  # track unique PII
            print("\nWARNING: PII detected:")
            for r in results:
                # skip URLs
                if r.entity_type == "URL" or r.score < 0.85:
                    continue
                pii_text = text[r.start:r.end]
                if pii_text not in seen:
                    seen.add(pii_text)
                    print(f"- {r.entity_type}: '{pii_text}' (confidence={r.score:.2f})")
            print("\n")
            return True

        print("\n No PII detected.\n")
        return False
