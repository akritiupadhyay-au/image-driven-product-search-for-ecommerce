import random
import torch
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tempfile
import os
from tqdm import tqdm

client = QdrantClient(":memory:")
collection_name = "Products"
client.create_collection(
   collection_name="Products",
   vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)

tempfile.tempdir = "/home/akriti/Notebooks/fashion-images/data"

model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

image_list = []

def process_directory(directory):
    files = os.listdir(directory)
    selected_files = files[:75]
    for file_name in selected_files:
        if file_name.endswith(".jpg"):
            image_path = os.path.join(directory, file_name)
            img = Image.open(image_path)
            image_list.append(img)

# Process directories
process_directory("/home/akriti/Notebooks/fashion-images/data/Apparel_Boys/Images/images_with_product_ids/")
process_directory("/home/akriti/Notebooks/fashion-images/data/Apparel_Girls/Images/images_with_product_ids/")
process_directory("/home/akriti/Notebooks/fashion-images/data/Footwear_Men/Images/images_with_product_ids/")
process_directory("/home/akriti/Notebooks/fashion-images/data/Footwear_Women/Images/images_with_product_ids/")

records = []
for idx, sample in tqdm(enumerate(image_list), total=len(image_list)):
    processed_img = processor(text=None, images=sample, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    img_px = list(sample.getdata())
    img_size = sample.size
    records.append(models.Record(id=idx, vector=img_embds, payload={"pixel_lst": img_px, "img_size": img_size}))

for i in range(30, len(records), 30):
    print(f"finished {i}")
    client.upload_records(
        collection_name="Products",
        records=records[i-30:i],
    )


def process_image(input_image, product_category):
    input_image_np = input_image.astype('uint8')
    img = Image.fromarray(input_image_np)

    processed_img = processor(text=None, images=img, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]

    search_result = client.search(
        collection_name=collection_name,
        query_vector=img_embds,  
        limit=1  
    )

    if search_result:
        reverse_image_payload = search_result[0].payload
        reverse_image_data = reverse_image_payload["pixel_lst"]
        reverse_image_size = reverse_image_payload["img_size"]
        reverse_image = Image.new("RGB", reverse_image_size)
        reverse_image.putdata(reverse_image_data)
    else:
        reverse_image = None

    return reverse_image

# Define function to get 5 images of selected category
def get_images_from_category(category):
    # Convert category to string
    category_str = str(category)
    # Directory path for selected category
    category_dir = f"/home/akriti/Notebooks/fashion-images/data/{category_str.replace(' ', '_')}/Images/images_with_product_ids/"
    # List of image paths
    image_paths = os.listdir(category_dir)[:5]
    # Open and return images
    images = [Image.open(os.path.join(category_dir, img_path)) for img_path in image_paths]
    return images


# Define your product categories
product_categories = ["Apparel Boys", "Apparel Girls", "Footwear Men", "Footwear Women"]

# Define function to handle category selection
def select_category(category):
    # Get images corresponding to the selected category
    images = get_images_from_category(category)
    # Return a random image from the list
    return random.choice(images)
    

# Create interface components for the image-driven search
image_input = gr.Image(label="Upload an image")

# Create interface components for the category selection
category_dropdown = gr.Dropdown(product_categories, label="Select a product category")
submit_button = gr.Button()
images_output = gr.Image(label="Images of Selected Category")

# Create Gradio interfaces
image_search_interface = gr.Interface(
    fn=process_image,
    inputs=image_input,
    outputs=gr.Image(),
    title="Image-driven Product Search for Ecommerce",
    description="Upload an image to perform a reverse image search from the collection.",
    examples=[["/home/akriti/Notebooks/fashion-images/data/Apparel_Boys/Images/images_with_product_ids/2691.jpg"],["/home/akriti/Notebooks/fashion-images/data/Apparel_Girls/Images/images_with_product_ids/2697.jpg"],
               ["/home/akriti/Notebooks/fashion-images/data/Footwear_Men/Images/images_with_product_ids/1636.jpg"],["/home/akriti/Notebooks/fashion-images/data/Footwear_Women/Images/images_with_product_ids/2610.jpg"]],
)

category_search_interface = gr.Interface(
    fn=select_category,
    inputs=category_dropdown,
    outputs=images_output,
    title="Category-driven Product Search for Ecommerce",
    description="Select a product category to view a random image from the corresponding directory.",
)

# Combine both interfaces into the same API
combined_interface = gr.TabbedInterface([image_search_interface, category_search_interface])

# Launch the combined interface
combined_interface.launch(share=True)
