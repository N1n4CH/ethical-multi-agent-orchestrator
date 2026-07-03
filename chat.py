import warnings
warnings.filterwarnings("ignore", message="Field.*conflict with protected namespace")

from azure.keyvault.secrets import SecretClient
from azure.identity import DeviceCodeCredential

from agents.structured_data_agent import StructuredDataAgent
from agents.unstructured_data_agent import UnstructuredDataAgent
from agents.multimodal_data_agent import MultimodalDataAgent

# ============================================================
#  Secrets
# ============================================================

keyVaultName = "ethical-orchestrator-kv"
KVUri = f"https://{keyVaultName}.vault.azure.net/"

print("Connecting to Azure for authentication.")

credential = DeviceCodeCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

# ============================================================
# Structured Data Agent Secrets
# ============================================================

# Each secret name below corresponds to the name of the secret as stored
# in Azure Key Vault. No credentials are hardcoded anywhere in this file --
# every value is pulled at runtime from Key Vault.

# NOTE: secret names below match what was actually created in Key Vault
# (ethical-orchestrator-kv). The Foundry resource hosts both the chat and
# embedding deployments, so structured + unstructured share the same
# foundry-endpoint / foundry-api-key secrets.

structuredazureendpoint = client.get_secret("foundry-endpoint").value
structuredazureapikey = client.get_secret("foundry-api-key").value
structuredpostgresqlpassword = client.get_secret("postgres-password").value
structuredpostgresqluser = client.get_secret("postgres-user").value
structuredpostgresqldbname = client.get_secret("postgres-dbname").value
structuredpostgresqlhost = client.get_secret("postgres-host").value

unstructuredmongourl = client.get_secret("cosmos-connection-string").value
unstructureddbname = client.get_secret("cosmos-dbname").value
unstructuredcollectionname = client.get_secret("cosmos-collection").value
unstructuredazureendpoint = client.get_secret("foundry-endpoint").value
unstructuredazurekey = client.get_secret("foundry-api-key").value

multimodalazureconnstring = client.get_secret("storage-connection-string").value
multimodalazurecontentsafetyendpoint = client.get_secret("safety-endpoint").value
multimodalazurecontentsafetykey = client.get_secret("safety-key").value

# ============================================================
#  Chat Manager
# ============================================================

class AgentChatManager:
    def __init__(self):
        self._load_agents()

    # -------------------------
    #  Agents
    # -------------------------
    def _load_agents(self):

        print("\nLoading Structured Data Agent")
        print("-------------------------------")
        self.structured = StructuredDataAgent(
            azure_endpoint=structuredazureendpoint,
            api_key=structuredazureapikey,
            deployment="gpt-5.4-mini",
            db_config={
                "host": structuredpostgresqlhost,
                "dbname": structuredpostgresqldbname,
                "user": structuredpostgresqluser,
                "password": structuredpostgresqlpassword,
                "port": 5432,
                "sslmode": "require",
            }
        )

        print("\nLoading Unstructured Data Agent")
        print("---------------------------------")
        self.unstructured = UnstructuredDataAgent(
            mongo_uri=unstructuredmongourl,
            db_name=unstructureddbname,
            collection_name=unstructuredcollectionname,
            chroma_path="./data/chroma_db_storage",
            azure_endpoint=unstructuredazureendpoint,
            azure_key=unstructuredazurekey,
            embedding_api_version="2023-06-01-preview",
            chat_api_version="2024-12-01-preview",
            embedding_deployment="text-embedding-3-small",
            chat_deployment="gpt-5.4-mini",
        )
        self.unstructured.build_index()

        print("\nLoading Multimodal Data Agent")
        print("-------------------------------")
        self.multimodal = MultimodalDataAgent(
            azure_conn_str=multimodalazureconnstring,
            container_name="houses",
            content_safety_endpoint=multimodalazurecontentsafetyendpoint,
            content_safety_key=multimodalazurecontentsafetykey
        )

    # -------------------------
    #  Routing
    # -------------------------
    def _route_query(self, user_message: str) -> str:
        """
        Route user query to one of:
        - structured
        - unstructured
        - multimodal

        This is a rule-based router. You can extend it later.
        """
        msg = user_message.lower().strip()

        multimodal_keywords = [
            "house like mine",
            "find me a house like mine",
            "show me a house like mine",
            "similar house",
            "similar image",
            "image",
            "photo",
            "picture",
            "looks like",
            "visual",
        ]

        structured_keywords = [
            "demographic",
            "demographics",
            "price",
            "home price",
            "most expensive",
            "average",
            "median",
            "count",
            "how many",
            "highest",
            "lowest",
            "neighborhood",
            "income",
            "population",
            "sql",
            "database",
        ]

        unstructured_keywords = [
            "permit",
            "approved",
            "document",
            "documents",
            "text",
            "restaurant",
            "permit approved",
            "notes",
            "report",
            "permit document",
        ]

        # Priority 1: multimodal
        if any(keyword in msg for keyword in multimodal_keywords):
            return "multimodal"

        # Priority 2: unstructured
        if any(keyword in msg for keyword in unstructured_keywords):
            return "unstructured"

        # Priority 3: structured
        if any(keyword in msg for keyword in structured_keywords):
            return "structured"

        # Default fallback
        return "unstructured"

    # -------------------------
    #  Agent wrappers
    # -------------------------
    def _run_structured(self, prompt: str) -> str:
        result = self.structured.ask(prompt, verbose=False, run_bias_audit=True)

        if isinstance(result, dict):
            return result.get("response", str(result))

        return str(result)

    def _run_unstructured(self, prompt: str) -> str:
        result = self.unstructured.ask(prompt, run_pii_audit=True)

        if isinstance(result, dict):
            return result.get("response", str(result))

        return str(result)

    def _run_multimodal(self, prompt: str) -> str:
        """
        Right now your original code ignored the prompt and always used
        query-house-2.jpg. Keeping that behavior here for parity.
        """
        query_image = "query-house-1.jpg" # You can change this to "query-house-1.jpg" to test with the other image
        matches, query_img = self.multimodal.find_similar(query_image)

        print("\nTop matches:")
        for score, path, address in matches:
            print(f"{address} | similarity: {score:.4f}")

        self.multimodal.show_results(query_img, matches)

        if not matches:
            return "I could not find similar houses."

        lines = ["Here are the top similar houses I found:"]
        for score, path, address in matches[:5]:
            lines.append(f"- {address} (similarity: {score:.4f})")

        return "\n".join(lines)

    # -------------------------
    #  Chat Interface
    # -------------------------
    def chat(self, user_message: str) -> str:
        route = self._route_query(user_message)
        print(f"[Router] Selected agent: {route}")

        try:
            if route == "structured":
                return self._run_structured(user_message)

            if route == "unstructured":
                return self._run_unstructured(user_message)

            if route == "multimodal":
                return self._run_multimodal(user_message)

            return "I could not determine the correct agent."
        except Exception as e:
            return f"An error occurred while processing your request with the {route} agent: {e}"


# ============================================================
#  Run Chat
# ============================================================

if __name__ == "__main__":
    banner = r"""
                                   /\ 
                                  /  \ 
                                 /____\ 
                ________________/______\_______________________
               /                                               \
              /_________________________________________________\
             |   ____      ____      ____      ____      ____   |
             |  | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |  |
             |  | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |  |
             |  |____|    |____|    |____|    |____|    |____|  |
             |                                                  |
             |        NEIGHBORHOOD INSIGHTS & DATA ASSISTANT    |
             |__________________________________________________|
    """
    print(banner)
    print("\n This AI-powered assistant provides automatically generated responses. "
          "Please use discretion, as answers may contain inaccuracies or errors.\n")
    print("Data comes from a structured demographics database, permit documents in NoSQL, and images stored in Azure.")
    print("Data remains in its original systems and is not copied into the assistant.")
    print("Embeddings and other AI artifacts are generated at runtime and are not permanently stored.")

    chat = AgentChatManager()

    print("\n\n--------------------------------")
    print("\n Agent Chat Ready.\n")
    print("Here are some examples of questions you can ask:")
    print("  - Which demographic group has the most expensive homes?")
    print("  - Was a permit approved for a restaurant in any of the neighborhoods?")
    print("  - Find me a house like mine")
    print("\n Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        answer = chat.chat(user_input)
        print(f"Agent: {answer}\n")
