import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tempfile
import os
from tqdm import tqdm

# Initialize Qdrant client and load collection
client = QdrantClient(":memory:")
collection_name = "img_collection"
client.create_collection(
   collection_name="img_collection",
   vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)

# Set temporary directory globally
tempfile.tempdir = "/home/akriti/Notebooks/fashion-product-images-dataset/fashion-dataset/fashion-dataset/images"

# Initialize tokenizer, processor, and model
model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

# List to store images
image_list = []

# Define the directory containing the images
directory = "./fashion-product-images-dataset/fashion-dataset/fashion-dataset/images"

# List all files in the directory
files = os.listdir(directory)

# Select only the first 100 image files
selected_files = files[:100]

# Loop through each selected file
for file_name in selected_files:
    # Check if the file is an image
    if file_name.endswith(".jpg"):
        # Open the image
        image_path = os.path.join(directory, file_name)
        img = Image.open(image_path)
        # Append the image to the list
        image_list.append(img)

# Upload image records to Qdrant collection
records = []
for idx, sample in tqdm(enumerate(image_list), total=len(image_list)):
    processed_img = processor(text=None, images=sample, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    img_px = list(sample.getdata())
    img_size = sample.size
    records.append(models.Record(id=idx, vector=img_embds, payload={"pixel_lst": img_px, "img_size": img_size}))

# Upload records in batches of 30
for i in range(30, len(records), 30):
    print(f"finished {i}")
    client.upload_records(
        collection_name="img_collection",
        records=records[i-30:i],
    )


# Function to process uploaded image
def process_image(input_image):
    # Convert PIL image to NumPy array
    input_image_np = input_image.astype('uint8')

    # Convert NumPy array to PIL image
    img = Image.fromarray(input_image_np)

    # Process the saved image
    processed_img = processor(text=None, images=img, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]

    # Search for similar images in Qdrant collection
    search_result = client.search(
        collection_name=collection_name,
        query_vector=img_embds,  # Specify the query vector
        limit=1  # Retrieve only the top result
    )

    # Check if any results are returned
    if search_result:
        # Retrieve and return the similar image
        similar_image_payload = search_result[0].payload
        similar_image_data = similar_image_payload["pixel_lst"]
        similar_image_size = similar_image_payload["img_size"]
        similar_image = Image.new("RGB", similar_image_size)
        similar_image.putdata(similar_image_data)
    else:
        # If no results are found, return None
        similar_image = None

    return similar_image


# Define the Gradio interface
gr.Interface(
    fn=process_image,
    inputs="image",
    outputs="image",
    title="Similar Image Search",
    description="Upload an image to find a similar image from the collection.",
    examples=[["fashion-product-images-dataset/fashion-dataset/fashion-dataset/images/1163.jpg"]],
).launch(share=True)
