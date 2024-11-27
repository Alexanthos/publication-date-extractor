import os
import requests
import fitz  
from PIL import Image
from datasets import load_dataset
import pandas as pd



# Function to download a PDF file
def download_pdf(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)
        print(f"Downloaded: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")

# Function to extract the first page as an image
def extract_first_page_as_image(pdf_path, output_image_path):
    try:
        pdf_document = fitz.open(pdf_path)
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image.save(output_image_path)
        print(f"Extracted image: {output_image_path}")
    except Exception as e:
        print(f"Failed to extract image from {pdf_path}: {e}")



if __name__=='__main__':
    # Directories for storage
    pdf_dir = "pdf_documents"
    images_dir = "extracted_images"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    dataset = load_dataset("maribr/publication_dates_fr")
    dataset = dataset['train'].to_pandas()
    # pdf_urls = dataset['train']['url']
    print(dataset)
    
    df_cache_urls = pd.read_csv("NLP_in_industry-original_data.csv")[['url','cache']]
    
    pdf_urls = dataset.merge(df_cache_urls)['cache'].to_list()
    
    image_paths = []
    
    # Process PDFs
    for i, url in enumerate(pdf_urls):
        pdf_path = os.path.join(pdf_dir, f"doc_{i}.pdf")
        image_path = os.path.join(images_dir, f"{i}.png")

        # Download PDF
        download_pdf(url, pdf_path)

        # Extract first page as image
        extract_first_page_as_image(pdf_path, image_path)
        
        """
        TODO

            - save a csv file with the columns : url date image_path
            - try to predict things for a couple rows
            - push code
        """

        if not os.path.isfile(image_path): 
            image_path = None
            
        image_paths.append(image_path)
    
    df_cache_urls['path'] = image_paths
    df_cache_urls.to_csv('data_VLM.csv')
    
    print("Processing complete. Images are saved in:", images_dir)
    