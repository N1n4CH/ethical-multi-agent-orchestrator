from azure.storage.blob import BlobServiceClient
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", message="Field.*conflict with protected namespace")
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential

class MultimodalDataAgent:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        azure_conn_str = None,
        container_name = None,
        content_safety_endpoint = None,
        content_safety_key = None
    ):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Azure Blob Storage
        self.azure_conn_str = azure_conn_str
        self.container_name = container_name

        if azure_conn_str and container_name:
            self.blob_service = BlobServiceClient.from_connection_string(azure_conn_str)
            self.container_client = self.blob_service.get_container_client(container_name)
        else:
            self.container_client = None

        # Azure Content Safety
        self.content_safety_client = ContentSafetyClient(endpoint=content_safety_endpoint, credential=AzureKeyCredential(content_safety_key))

    # ============================================
    # PRIVATE METHODS
    # ============================================

    def __load_image_from_blob(self, blob_name):

        print("\n" + "=" * 80)
        print(f"Analysing image: {blob_name}")

        blob = self.container_client.get_blob_client(blob_name)
        image_bytes = blob.download_blob().readall()

        # --- RUN CONTENT SAFETY ---
        request = AnalyzeImageOptions(image=ImageData(content=image_bytes))
        response = self.content_safety_client.analyze_image(request)

        print(f"Analyzing Blob Image: {blob_name}")
        inappropriate_content = False
        for c in response.categories_analysis:
            print(c.category, c.severity)
            if c.severity != 0:
                inappropriate_content = True

        if inappropriate_content:
            print(f"Warning: Harmful content detected in blob image: {blob_name}\n")
        else:
            print(f"No harmful content detected in blob image: {blob_name}\n")

        # Return the image
        return Image.open(BytesIO(image_bytes)).convert("RGB")


    def __load_image(self, path):
        return Image.open(path).convert("RGB")

    def __compute_similarity_blob(self, query_image_path, blob_name):
        img1 = self.__load_image(query_image_path)
        img2 = self.__load_image_from_blob(blob_name)

        inputs = self.processor(images=[img1, img2], return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        if not isinstance(outputs, torch.Tensor):
            image_embeds = outputs.pooler_output
        else:
            image_embeds = outputs

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        similarity = torch.dot(image_embeds[0], image_embeds[1]).item()
        return similarity

    # ============================================
    # PUBLIC METHOD — FIND SIMILAR
    # ============================================

    def find_similar(self, query_image_path, prefix="", top_k=3):
        if not self.container_client:
            raise ValueError("Azure Blob Storage is not configured.")

        results = []

        print("\n" + "=" * 80)
        print("Using Multimodal (Image files) Data stored in Azure Data Blob to answer question")
        print("Scan images for harmful contente")
        print("=" * 80)

        blob_list = list(self.container_client.list_blobs(name_starts_with=prefix))
        print("Blob count:", len(blob_list))

        blob_list = self.container_client.list_blobs(name_starts_with=prefix)

        for blob in blob_list:
            fname = blob.name
            print(fname)

            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                similarity = self.__compute_similarity_blob(query_image_path, fname)
                address = os.path.splitext(os.path.basename(fname))[0].replace("_", " ")
                results.append((similarity, fname, address))

            except Exception as e:
                print("Skipping:", fname, e)

        results.sort(reverse=True, key=lambda x: x[0])

        query_img = self.__load_image(query_image_path)
        return results[:top_k], query_img

    # ============================================
    # PUBLIC METHOD — SHOW RESULTS
    # ============================================

    def show_results(self, query_img, matches):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")

        for i, (score, blob_name, address) in enumerate(matches, start=2):
            img = self.__load_image_from_blob(blob_name)

            plt.subplot(1, 4, i)
            plt.imshow(img)
            plt.title(f"{address}\n{score:.3f}")
            plt.axis("off")

        plt.show()
