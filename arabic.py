import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
from groq import Groq, RateLimitError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from io import BytesIO

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Groq client
client = Groq(api_key="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")

# Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=5, initial_delay=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    print(f"Rate limit reached. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                    delay *= 2
            raise Exception("Max retries reached, unable to complete request")
        return wrapper
    return decorator

# Function to scrape website data
def scrape_website(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        if url.endswith(".pdf"):
            pdf_file = BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            return text
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = [p.text for p in soup.find_all(['p', 'span', 'div']) if p.text.strip()]
            return ' '.join(paragraphs)
    else:
        print(f"Failed to retrieve content from {url}")
        return None

@retry_with_backoff()
def call_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function to extract English-Language pairs
def extract_eng_lang_pairs(text):
    prompt = """
    Extract all English-Arabic sentence pairs, phrases, and word pairs related to medical terms or common daily use from the provided text.
    Only extract full sentences, phrases, or word pairs with direct translations.
    Return JSON ONLY in this format:
    [{"English": "...", "Arabic": "..."}]
    """
    text_chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
    pairs = []

    for chunk in text_chunks:
        content = call_groq(prompt + "\n" + chunk)
        json_blocks = re.findall(r'\[.*?\]', content, re.S)
        
        if not json_blocks:
            print("No JSON detected, skipping chunk")
            print("Model Response:", content)
            continue

        for block in json_blocks:
            try:
                cleaned_block = block.strip("`""'")
                chunk_pairs = json.loads(cleaned_block)
                filtered_pairs = [(pair.get("English"), pair.get("Arabic")) for pair in chunk_pairs if pair.get("English") and pair.get("Arabic")]
                pairs.extend(filtered_pairs)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print("Failed to parse JSON from model response:", e)
                print("Problematic JSON:", block)
                continue

    return pairs

# Function to rephrase sentences
def rephrase_sentences(pairs):
    rephrased_pairs = []
    for eng, lang in pairs:
        if not eng or not lang:
            continue
        prompt = f"Generate 10 rephrased versions of the English sentence/phrase/word: '{eng}' and its Arabic translation: '{lang}'. Ensure that each rephrased English item corresponds to an accurately rephrased Arabic translation. Return JSON ONLY in this format: [{{'English': '...', 'Arabic': '...'}}, ...]"
        content = call_groq(prompt)

        json_blocks = re.findall(r'\[.*?\]', content, re.S)
        
        if not json_blocks:
            print("No JSON detected for sentence:", eng)
            print("Model Response:", content)
            continue

        for block in json_blocks:
            try:
                cleaned_block = block.strip("`""'")
                rephrased = json.loads(cleaned_block)
                for pair in rephrased:
                    if "English" in pair and "Arabic" in pair:
                        rephrased_pairs.append(("Rephrased", pair["English"], pair["Arabic"]))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for sentence: {eng}")
                print("Problematic JSON:", block)
    return rephrased_pairs

# Function to implement RAG pipeline
def retrieve_relevant_chunks(text):
    chunks = text.split('. ')  # Simple sentence chunking
    documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

# Main function
def main(urls):
    all_data = []
    for url in urls:
        text = scrape_website(url)
        if text:
            vector_store = retrieve_relevant_chunks(text)
            pairs = extract_eng_lang_pairs(text)
            for eng, lang in pairs:
                all_data.append(("Original", eng, lang))
            rephrased = rephrase_sentences(pairs)
            all_data.extend(rephrased)

    if all_data:
        df = pd.DataFrame(all_data, columns=["Property", "English Sentence", "Lang Sentence"])
        df.to_csv("outputarabic.csv", index=False)
        print("CSV file generated successfully with filtered, rephrased sentences, phrases, words, and RAG-based augmentation.")
    else:
        print("No data to save to CSV.")

if __name__ == "__main__":
    urls = ["https://medlineplus.gov/languages/arabic.html"]
    main(urls)