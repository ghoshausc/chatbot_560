import bs4
from bs4 import BeautifulSoup
import re,time,ast, requests
import pandas as pd
import numpy as np 
import os
import requests
import spacy

import socket
hostname = socket.gethostname()
print(hostname)

from googlesearch import search   

import cv2, sys
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import csv

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from io import BytesIO
from PIL import Image    #install pyspellcher, fpdf, langchain
import os

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def process_query(query):
    
    start_X = time.time()

    print('Got query : ',query,'\n\n')
    
    # to search 
    # query = "Public safety picture of Manhattan Beach, LA, how safe for students?"

    links = []
    for j in search(query): 
        links.append(j) 

    ## scraping each website, getting the texts and the images in them..

    all_website_content = []

    for url in links:

        try:
            response = requests.get(url,timeout=3)
            # print('After response...')
            soup = BeautifulSoup(response.text, 'html.parser')
            # print('After soup...')
            # Extract text content from the parsed HTML
            text_content = soup.get_text().strip()
            all_website_content.append(text_content)

            # print("Text content from", url, ":\n", text_content.strip()[:100])
            print("-" * 50)

            # else:
            #     print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")

        except Exception as e:
            print(f"An error occurred while processing {url}: {str(e)}")


    #saving the scraped data to the file

    # Name of the output PDF file
    pdf_file = "/Users/soumyarn/USC/Fall_2023/DSCI_560/Project/localGPT/SOURCE_DOCUMENTS/scraped_data.pdf"

    doc = SimpleDocTemplate(pdf_file, pagesize=letter)

    story = []

    # Defining a style
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    spacy_link_content = all_website_content[:]

    spacy_link_content = [i.replace('\n','').strip() for i in spacy_link_content]

    spacy_link_content = [i.replace('\t','') for i in spacy_link_content]

    for string in spacy_link_content:
        try:
            p = Paragraph(string, style)
            story.append(p)
            story.append(Paragraph("<br/>", style))
        except:
            print('Inside except')
            # print(f"String content: {string[:100]}")

    doc.build(story)

    print(f"PDF saved as {pdf_file}")

    ### extracting the image URLs and storing them in a lsit


    def extract_image_urls_from_webpage(url):
        response = requests.get(url)
        image_urls = []

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')  # Find all image tags in the HTML

            for img in img_tags:
                img_url = img.get('src')  # Get the 'src' attribute of the image tag
                if img_url and (img_url.startswith('http') or img_url.startswith('https')):
                    image_urls.append(img_url)

        return image_urls


    all_image_urls = []
    for url in links[:3]:
        image_urls = extract_image_urls_from_webpage(url)
        all_image_urls.extend(image_urls)


    ##The code below will scrape all images from the web pages and store them inside fodler named X. Please don't touch this

    def download_images(image_urls, folder_name):
        # Create the folder if it doesn't exist
        print('Folder name where images will be stored : ',folder_name,'\n\n')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for index, url in enumerate(image_urls, start=1):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image_name = f'image_{index}.png'  # Modify the file extension if needed
                    image_path = os.path.join(folder_name, image_name)

                    with open(image_path, 'wb') as image_file:
                        image_file.write(response.content)

                    print(f"Downloaded: {image_name}")
                else:
                    print(f"Failed to download: {url}")

            except Exception as e:
                print(f"Error downloading {url}: {e}")


    start = time.time()
    folder_name = 'X'
    download_images(all_image_urls, folder_name)
    end = time.time()

    print('Time taken to download images : ',end - start,'\n\n')

    ### now writing a code which will iterate through the imags in the folder and say which image is most closely related to the user query (for images without text)

    ###code to find matching images

    model = VGG16(weights='imagenet')

    # provide the folder path accordingly..

    folder_path = 'X/'

    image_objects = {}  # Dictionary to store image objects

    # Iterate through all files in the folder

    #for the images without texts

    start = time.time()

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter by file extensions
            img_path = os.path.join(folder_path, filename)
            # print('Image path:', img_path)

            try:
                img = Image.open(img_path)
                img.verify()  # Check if the file is a valid image
                img = image.load_img(img_path, target_size=(224, 224))  # VGG16 input size
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)

                decoded_preds = decode_predictions(preds, top=5)[0]  # Get top 3 predictions
                # print("Predictions:","for image : ",img_path)
                for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                    print(f"{i + 1}: {label} ({score:.2f})")

                #code to store images as keys and values as the objects in it

                objects_detected = [label for (_, label, _) in decoded_preds]  # Extract labels

                # Store detected objects in the dictionary with filename as key
                image_objects[filename] = objects_detected

            except (IOError, SyntaxError) as e:
                # Skip over files that are not valid images
                # print(f"Skipped: {img_path} - Error: {e}")
                pass

    end = time.time()

    print('Time taken to process images without texts : ',end - start,'\n\n')

    ### the functiosn below all together show images most related to the user search query, remember this would work for images with texts


    #there can be images with texts, this is the code below for that..


    # Loading pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet', include_top=True)

    # Function to extract text using Tesseract OCR
    def extract_text_from_image(img):
        return pytesseract.image_to_string(img)

    def img_process(img_path):

        # Load and ?preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict objects in the image
        predictions = model.predict(x)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

        # Store recognized objects and extracted text as sets
        recognized_objects = set()
        extracted_text = set()

        for _, label, _ in decoded_predictions:
            recognized_objects.add(label)

        # Convert image to OpenCV format for text extraction
        img_cv = cv2.imread(img_path)

        # Extract text from the image using Tesseract OCR
        text = extract_text_from_image(img_cv)
        extracted_text.add(text)

        # Print the sets of recognized objects and extracted text
        # print("Recognized Objects:")
        # print(recognized_objects)
        # print("\nExtracted Text:")
        # print(extracted_text)
        return extracted_text


    def discard_wrong_spellings(text):
        spell = SpellChecker()

        # Tokenize the text into words
        words = text.strip().split()

        # Get the set of misspelled words
        misspelled = spell.unknown(words)

        # print('Misspelled words are : ',misspelled)

        # Filter out words that are not misspelled
        correct_words = [word for word in words if word not in misspelled]

        return ' '.join(correct_words)

    def get_image_filenames(directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        return image_files

    def get_text_for_image(folder, user_query):

        dict_img_words = dict()

            # Replace 'images' with the path to the directory
        image_directory = folder
        # image_directory = '/Users/soumyarn/USC/Fall_2023/DSCI_560/Project/downloaded_images/'

        image_filenames = get_image_filenames(image_directory)


        start = time.time()
        # Print the collected image filenames

        for filename in image_filenames:
            try:
                if 'checkpoints' not in filename:
                    extracted_text = img_process(filename)
                    valid_text = remove_special_alphanumeric_words(extracted_text)
                    string = " ".join(i for i in valid_text)
                    fnal_ans = discard_wrong_spellings(string)
                    dict_img_words[filename] = fnal_ans
            except:
                pass
        end = time.time()

        print('Time taken to process all images : ',end - start,'\n\n')

        for k,v in dict_img_words.items():
            if v:
                v = v.lower()
                dict_img_words[k] = v

        print('User query : ',user_query)

        values = list(dict_img_words.values())

        values.append(user_query)

        # Load a pre-trained Sentence Transformer model (you can choose other models too)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Encode all values into embeddings
        value_embeddings = model.encode(values, convert_to_tensor=True)

        # Encode the user query into an embedding
        query_embedding = model.encode([user_query], convert_to_tensor=True)

        # Calculate cosine similarity between the query embedding and all value embeddings
        cosine_scores = util.pytorch_cos_sim(query_embedding, value_embeddings)

        # Convert cosine similarity scores to numpy array
        cosine_scores = cosine_scores.cpu().numpy()

        # Get indices of values sorted by similarity (in descending order)
        similar_indices = cosine_scores.argsort(axis=1)[0][::-1]

        # Set a similarity threshold (adjust as needed)
        threshold = 0.05

        similar_keys = []

        for i in list(similar_indices):
            if cosine_scores[0][i] > 0.03:
                try:
                    if 'checkpoints' not in list(dict_img_words.keys())[i]:
                        similar_keys.append((list(dict_img_words.keys())[i],cosine_scores[0][i]))
                except:
                    continue

        displayed_images = []
        x = set()

        for img_filename, sim_score in similar_keys[:10]:
            try:
                img = Image.open(img_filename)

                image_content = tuple(img.getdata())

                if image_content not in x:
                    # plt.imshow(img)
                    # plt.title(f"Similarity Score: {sim_score}")
                    # plt.axis('off') 
                    # plt.show()
                    x.add(image_content)
                    displayed_images.append((img_filename,sim_score))
            except FileNotFoundError:
                print(f"Image file {key} not found.")

        return displayed_images

    folder_path = '/Users/soumyarn/USC/Fall_2023/Project/X/'
    x = get_text_for_image(folder_path, query)   #provide the folder path accordingly

    #saving the images with detected injects to SOURCE_DOCUMENTS

    csv_filename = '/Users/soumyarn/USC/Fall_2023/DSCI_560/Project/localGPT/SOURCE_DOCUMENTS/image_objects.csv'

    print('Image objects dict : ',image_objects)


    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename', 'Detected Objects'])  # Write header

        for img_file, objects in image_objects.items():
            csv_writer.writerow([img_file, ', '.join(objects)])

    print(f"Data has been written to {csv_filename}")



    ### iterate through the dict and find the image top-related to the query

    data = pd.read_csv('image_objects.csv')

    # User query
    user_query = "Detailed statistics of crimes near Marriott DTLA"

    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # Downloading the 'en_core_web_sm' model first
        spacy.cli.download("en_core_web_sm")

    # Function to calculate similarity between the query and detected objects
    def calculate_similarity(query, detected_objects):
        query_doc = nlp(query)
        detected_doc = nlp(detected_objects)
        return query_doc.similarity(detected_doc)

    # Calculate similarity for each row in the DataFrame
    data['similarity'] = data.apply(lambda x: calculate_similarity(user_query, x['Detected Objects']), axis=1)

    # Sort the DataFrame by similarity in descending order
    sorted_data = data.sort_values(by='similarity', ascending=False)

    # Get the image filenames most related to the user query
    top_related_images = sorted_data['Image Filename'].tolist()

    # Display the top related image filenames
    print("Image filenames most related to the user query:")
    print(top_related_images)


    end_X = time.time()

    print('Finished populating SOURCE_DOCUMENTS and fiding the most related ones....\n')

    print('TIme taken : ',end_X - start_X,'\n\n')