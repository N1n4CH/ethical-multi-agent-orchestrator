import os
from pymongo import MongoClient
from pypdf import PdfReader

import os
from pymongo import MongoClient
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def load_pdfs_to_mongo(folder_path="./data/unstructured"):
    # Connect to MongoDB (Cosmos DB for MongoDB)
    # Credential is read from an environment variable — never hardcode
    # real connection strings in this file, this repo is public.
    client = MongoClient(os.environ["COSMOS_CONNECTION_STRING"])

    # Create or access the database and collection
    db = client[os.environ.get("COSMOS_DBNAME", "documents")]
    collection_name = os.environ.get("COSMOS_COLLECTION", "permits")

    # Explicitly create collection if it doesn't exist
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    collection = db[collection_name]

    # Ensure folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Loop through PDFs
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

            # Insert into MongoDB
            doc = {
                "filename": filename,
                "content": text
            }

            collection.insert_one(doc)

    print("All PDFs loaded into MongoDB (documents.permits).")


if __name__ == "__main__":
    load_pdfs_to_mongo()

