import os
import logging
import click
import torch, time
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import ingest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS, SOURCE_DIRECTORY, DOCUMENT_MAP, INGEST_THREADS
)

import re,time,ast, requests
import pandas as pd
import numpy as np 

import requests
from bs4 import BeautifulSoup
from googlesearch import search
from requests_html import HTMLSession
import socket
socket.gethostbyname("")
import spacy


#from ingest.py

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


#incorporating ingest code here


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
           file_log(name + ' failed to submit')
           return None
        else:
           data_list = [future.result() for future in futures]
           # return data and file paths
           return (data_list, filepaths)
    
def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
       file_extension = os.path.splitext(file_path)[1]
       loader_class = DOCUMENT_MAP.get(file_extension)
       if loader_class:
           file_log(file_path + ' loaded.')
           loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()[0]
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None 
    
    
def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")
    
    
    
    
#storing scraped content to a PDF





def scrape_and_store_data(query):
    
    # to search 
    query = "How many crimes took place in Glendale this week? "

    links = []
    for j in search(query): 
        links.append(j) 
    
    ## scraping each website, getting the texts and the images in them..
    print('Links list length : ',len(links),'\n\n')
    
    all_website_content = []

    for url in links:

        # print(response = requests.get(url).status_code)
        try:
            # Send a GET request to the URL

            response = requests.get(url,timeout=10)
            # print('After response...')
            soup = BeautifulSoup(response.text, 'html.parser')
            # print('After soup...')
            # Extract text content from the parsed HTML
            text_content = soup.get_text().strip()
            all_website_content.append(text_content)


            # You can now process and print or save the text content as needed
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

    for string in spacy_link_content:
        p = Paragraph(string, style)
        story.append(p)
        story.append(Paragraph("<br/>", style))  # Add a line break after each paragraph

    doc.build(story)

    print(f"PDF saved as {pdf_file}")
    

def ingest(query, qa, show_sources, save_qa):
    
    def file_log(logentry):
       file1 = open("file_ingest.log","a")
       file1.write(logentry + "\n")
       file1.close()
       print(logentry + "\n")

    def load_documents(source_dir: str) -> list[Document]:
        # Loads all documents from the source documents directory, including nested folders
        
        print('The source directory is : ',source_dir)
        paths = []
        for root, _, files in os.walk(source_dir):
            # print('Number of files : ',len(files))
            for file_name in files:
                # print('Importing: ' + file_name,'\n\n')
                if 'checkpoint' not in file_name and 'DS_Store' not in file_name:
                    print('Valid file : ',file_name,'\n\n')
                    file_extension = os.path.splitext(file_name)[1]
                    source_file_path = os.path.join(root, file_name)
                    if file_extension in DOCUMENT_MAP.keys():
                        paths.append(source_file_path)
        # Have at least one worker and at most INGEST_THREADS workers
        n_workers = min(INGEST_THREADS, max(len(paths), 1))
        chunksize = round(len(paths) / n_workers)
        docs = []
        
        with ProcessPoolExecutor(n_workers) as executor:
            
            futures = []
            # split the load operations into chunks
            for i in range(0, len(paths), chunksize):
                # select a chunk of filenames
                filepaths = paths[i : (i + chunksize)]
                # submit the task
                # print('Length of filepaths is : ',len(filepaths),'\n\n')
                try:
                    future = executor.submit(load_document_batch, filepaths)
                except Exception as ex:
                    file_log('executor task failed: %s' % (ex))
                    # print('Inside load document except \n\n')
                    future = None
                if future is not None:
                    futures.append(future)
            # process all results
            
            # print('Future length is : ',len(futures))
            for future in as_completed(futures):
                # print('Future type : ',type(future),'\n\n')
                # try:
                contents, _ = future.result()
                docs.extend(contents)
                # except Exception as ex:
                #     file_log('Exception: %s' % (ex))
        
        print('Docs length : ',len(docs),'\n\n')
        return docs


    def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
        # Splits documents for correct Text Splitter
        # print('Inside split docs...\n\n')
        text_docs, python_docs = [], []
        
        # print()
        for doc in documents:
            if doc is not None:
               file_extension = os.path.splitext(doc.metadata["source"])[1]
               # print('File extension : ',file_extension,'\n\n')
               if file_extension == ".py":
                    python_docs.append(doc)
               else:
                    text_docs.append(doc)
                    # print('Not a .py file',doc,'\n\n')
        return text_docs, python_docs


    @click.command()
    @click.option(
        "--device_type",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=click.Choice(
            [
                "cpu",
                "cuda",
                "ipu",
                "xpu",
                "mkldnn",
                "opengl",
                "opencl",
                "ideep",
                "hip",
                "ve",
                "fpga",
                "ort",
                "xla",
                "lazy",
                "vulkan",
                "mps",
                "meta",
                "hpu",
                "mtia",
            ],
        ),
        help="Device to run on. (Default is cuda)",
    )
    def main(device_type):
        # Load documents and split in chunks

        start = time.time()
        print('device type custom : ',device_type)
        print('Source directory : ',SOURCE_DIRECTORY,'\n\n')
        logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
        documents = load_documents(SOURCE_DIRECTORY)
        end = time.time()
        print('TIme taken to load docs : ',end - start, '\n\n')
        start = time.time()
        text_documents, python_documents = split_documents(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=880, chunk_overlap=200
        )
        end = time.time()
        print('TIme taken for RecursiveSplitter : ',end - start, '\n\n')
        
        print('Length of text_documents is : ',len(text_documents),'\n\n')
        texts = text_splitter.split_documents(text_documents)
        texts.extend(python_splitter.split_documents(python_documents))
        # logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
        # logging.info(f"Split into {len(texts)} chunks of text")

        start = time.time()
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device_type},
        )
        
        # These are much smaller embeddings and will work for most appications
        # If you use HuggingFaceEmbeddings, make sure to also use the same in the
        # run_localGPT.py file.

        end = time.time()
        
        print('TIme taken to generate the embeddings : ',end - start,'\n\n')

        print('Texts type is : ',type(texts),len(texts),'    ',type(embeddings),'\n\n')

        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )

        end = time.time()

        print('Time taken to ingest is : ',end - start,'\n\n')
        
        start = time.time()
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]
        end = time.time()
        
        print('TIme taken to generate : ',end - start,'\n\n')
        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
        
        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)
    main()
    
    

### end of ingesting data code

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """
    start = time.time()
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device_type})
    
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name="allenai/longformer-base-4096")
    
    end = time.time()
    
    print('Time takn to load HuggingFaceTrabnsformer : ',end - start,'\n\n')
    
    # load the vectorstore
    
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    print('Device details : ',device_type, MODEL_ID, MODEL_BASENAME,'\n\n')
    # load the llm pipeline
    start = time.time()
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    
    print('LLM is : ',llm,'\n\n')
    
    if use_history:
        
        print('Inside if... \n')
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        print('Inside if... \n')
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
    end = time.time()
    
    print('Time takn to load llm : ',end - start,'\n\n')
    
    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)

def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    
    start = time.time()
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    end = time.time()
    
    print('TIme taken to load qa pipeline : ',end - start,'\n\n')
    
    # Interactive questions and answers
    while True:
        start = time.time()
        query = input("\nEnter a query: ")
        if query == "exit":
            break
            
        scrape_and_store_data(query)        
        end = time.time()

        print('Time taken for my scraping thing : ',end - start,'\n\n')

        # feeding data to bot

        ingest(query, qa, show_sources, save_qa)

        print('Ingest returns : \n\n')

        print('Right after ingest....','\n\n')
        
        # Get the answer from the chain
        
        #query, qa, doocs, show_sources, save_qa 
#         res = qa(query)
#         answer, docs = res["result"], res["source_documents"]

#         # Print the result
#         print("\n\n> Question:")
#         print(query)
#         print("\n> Answer:")
#         print(answer)

#         if show_sources:  # this is a flag that you can set to disable showing answers.
#             # # Print the relevant sources used for the answer
#             print("----------------------------------SOURCE DOCUMENTS---------------------------")
#             for document in docs:
#                 print("\n> " + document.metadata["source"] + ":")
#                 print(document.page_content)
#             print("----------------------------------SOURCE DOCUMENTS---------------------------")
        
#         # Log the Q&A to CSV only if save_qa is True
#         if save_qa:
#             utils.log_to_csv(query, answer)
        
        

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    print('Main in beginning...\n\n')
    main()
