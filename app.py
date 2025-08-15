import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from PIL import Image
from flask import Flask, request, jsonify
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import kagglehub
import requests
import gunicorn

# --- 1. Load your models and data once when the app starts ---
app = Flask(__name__)

# Download the dataset from Kaggle Hub and get the paths
try:
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    csv_path = os.path.join(dataset_path, "styles.csv")
    images_path = os.path.join(dataset_path, "images")
    print("Dataset downloaded successfully!")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit()

# Load processed embeddings and IDs
try:
    embeddings_array = np.load(os.path.join('processed_data', 'fashion_embeddings.npy'))
    product_ids_array = np.load(os.path.join('processed_data', 'fashion_ids.npy'))
    print("Embeddings loaded successfully.")
except FileNotFoundError:
    print("Error: Processed data files not found. Please run the embedding extraction script first.")
    exit()

# Load the full styles.csv for metadata lookup
try:
    df_full = pd.read_csv(csv_path, on_bad_lines='skip')
    df_full = df_full.set_index('id')
    print("Metadata loaded successfully.")
except FileNotFoundError:
    print("Error: styles.csv not found.")
    exit()

# Load the feature extraction model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
feature_extractor = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
print("Feature extraction model loaded successfully.")

# --- CHANGE: Load a smaller LLM that fits within the memory limit ---
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True # Add this for better memory management
)
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
print(f"LLM {model_id} loaded successfully.")

# --- Helper functions ---
def get_image_embedding(image, model):
    img = image.resize((224, 224))
    img_array = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    processed_img = preprocess_input(img_array)
    embedding = model.predict(processed_img, verbose=0).flatten()
    return embedding / np.linalg.norm(embedding)

def find_similar_items(target_embedding, embeddings_array, product_ids_array, df_full, top_n=5):
    similarities = cosine_similarity(target_embedding.reshape(1, -1), embeddings_array)[0]
    top_indices = np.argsort(similarities)[::-1]
    similar_ids = product_ids_array[top_indices][:top_n]
    return df_full.loc[similar_ids]

def generate_gpt_prompt(target_metadata, similar_metadata):
    prompt = f"You are a professional fashion stylist. A user has presented the following item: {target_metadata.get('productDisplayName', 'Unknown Item')}. It is a {target_metadata.get('masterCategory', 'Unknown Category')} item with sub-category: {target_metadata.get('subCategory', 'Unknown Sub-Category')}. Visually similar items for context are: {', '.join(similar_metadata['productDisplayName'].tolist())}. Based on this, provide a detailed breakdown of how to style the original item. Suggest complementary colors, accessories, footwear, and suitable occasions. Provide a unique, creative, and confident response."
    return prompt

def get_llm_recommendation(prompt, llm_pipe):
    messages = [{"role": "user", "content": prompt}]
    generation_kwargs = {"max_new_tokens": 200, "do_sample": True, "temperature": 0.7, "top_p": 0.95}
    outputs = llm_pipe(messages, **generation_kwargs)
    stylist_response = "Could not generate a styling recommendation."
    if outputs and outputs[0] and 'generated_text' in outputs[0] and outputs[0]['generated_text']:
        for message in reversed(outputs[0]['generated_text']):
            if 'content' in message:
                stylist_response = message['content']
                break
    return stylist_response

# --- 2. Define your API endpoint ---
@app.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream)
        target_embedding = get_image_embedding(image, feature_extractor)
        similar_items = find_similar_items(target_embedding, embeddings_array, product_ids_array, df_full)
        
        if similar_items.empty:
            return jsonify({"error": "Could not find any similar items."}), 404
        
        target_item_metadata = similar_items.iloc[0].to_dict()

        llm_prompt = generate_gpt_prompt(target_item_metadata, similar_items)
        recommendation = get_llm_recommendation(llm_prompt, llm_pipe)

        return jsonify({
            "target_item": target_item_metadata.get('productDisplayName'),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))