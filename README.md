This is a Public Safety Chatbot that answers questions of public safety in different areas of Los Angeles. 

This project was inspired by the original [localGPT](https://github.com/PromtEngineer/localGPT). The code asks for a user query related to public safety in Los Angeles, scrapes websites of top-related searches from Google and answers questions basd on the latest update. It will also show most relevant images (if any ) related to the user query. 

# Process of working : Run the code run_localGPT_copy2.py. It will ask for a user query. Once the query is entered : 

1. Scrapes top-related websites
2. Creates a PDF from the scraped content
3. Stores the file under directory `SOURCE_DIRECTORY`
4. Downloads the images found in these URls (downloads top 40 ones) and stores them inside `images` directory.
5. Detects objects/texts in these images
6. Picks the top 5 related images based on the user query (can not pick an image if not related to query)
7. Displays the image files top-related
8. Shows the response of the query



# Environment Setup 🌍

1. 📥 Please install Python 3.11 or create with Conda : conda create -n environment_name python=3.11


2. 📥 Clone the repo using git:

```shell
git clone https://github.com/ghoshausc/public_safety_chatbot_560.git
```

3. 🛠️ Install the dependencies using pip

To set up your environment to run the code, first install all requirements:

```shell
pip install -r requirements.txt
```

***Installing LLAMA-CPP :***


For `NVIDIA` GPUs support, use `cuBLAS`

```shell
# Example: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

For Apple Metal (`M1/M2`) support, use

```shell
# Example: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

***Please specify the pip versions according to the pip version on your system

### Support file formats:
LocalGPT currently supports the following file formats. LocalGPT uses `LangChain` for loading these file formats. The code in `constants.py` uses a `DOCUMENT_MAP` dictionary to map a file format to the corresponding loader. In order to add support for another file format, simply add this dictionary with the file format and the corresponding loader from [LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/).

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```


Note: When you run this for the first time, it will need internet access to download the embedding model (default: `Instructor Embedding`). In the subsequent runs, no data will leave your local environment and you can ingest data without internet connection.

## Ask questions about public safety in LA!

In order to chat with your documents, run the following command (by default, it will run on `cuda`).

```shell
python3 run_localGPT_copy2.py
```
You can also specify the device type like : 

```shell
python3 run_localGPT_copy2.py --device_type cuda # to run on GPU
```

You will be presented with a prompt:

```shell
> Enter a query:
```

After typing your question, hit enter, the bot will start scraping and creating the files inside folder SOURCE_DOCUMENTS. This will take some time based on your hardware. 

For bot scraping top-related results, you would get a response like this : 

<img width="1312" alt="Screenshot 12-11-2023" src="https://github.com/ghoshausc/public_safety_chatbot_560/blob/fce685ce3cace0b9b2c8c778d0d85e0e249ef103/bot_scraping.png">


For bot responding to your queris, you will get a response like this below.
<img width="1312" alt="Screenshot 12-11-2023" src="https://github.com/ghoshausc/public_safety_chatbot_560/blob/b33a80dec8ad7d6a15ab1c456cc45eb049df87fb/bot_working_demo.png">

Once the answer is generated, you can then ask another question without re-running the script, just wait for the prompt again.


For seeing most related images, you can take a look a the folder images/ which will contain all images downloaded from the top-related websites. 

It then identifies the objects/texts in the images. Below is a screen shot for that : 

<img width="1312" alt="Screenshot 12-11-2023" src="https://github.com/ghoshausc/public_safety_chatbot_560/blob/27e896e777faccbb7923dce77eaebbf2e982e7de/image_object_detection.png">


You can check the related images from the `/images` folder, below is where you can get the related image filenames. 

<img width="1312" alt="Screenshot 12-11-2023" src="https://github.com/ghoshausc/public_safety_chatbot_560/blob/7d4ef6a630add288efcbac6111d14ab867dec0d8/query_related_images.png">

***Note:*** When you run this for the first time, it will need internet connection to download the LLM (default: `TheBloke/Llama-2-7b-Chat-GGUF`). After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

## There are times when scraping results from websites take too long due to website issues. To address this, we have a request timeout but if it still gets stuck, you can try out with some other queries to verify that the bot is working. 



# GPU and VRAM Requirements

Below is the VRAM requirement for different models depending on their size (Billions of parameters). The estimates in the table does not include VRAM used by the Embedding models - which use an additional 2GB-7GB of VRAM depending on the model.

| Mode Size (B) | float32   | float16   | GPTQ 8bit      | GPTQ 4bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7B      | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3.5 GB - 5 GB      |
| 13B     | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6.5 GB - 8 GB      |
| 32B     | 130 GB    | 65 GB     | 32.5 GB - 35 GB| 16.25 GB - 19 GB   |
| 65B     | 260.8 GB  | 130.4 GB  | 65.2 GB - 67 GB| 32.6 GB - 35 GB    |


# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. Vicuna-7B is based on the Llama model so that has the original Llama license.

# Common Errors

 - [Torch not compatible with CUDA enabled](https://github.com/pytorch/pytorch/issues/30664)

   -  Get CUDA version
      ```shell
      nvcc --version
      ```
      ```shell
      nvidia-smi
      ```
   - Try installing PyTorch depending on your CUDA version
      ```shell
         conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
      ```
   - If it doesn't work, try reinstalling
      ```shell
         pip uninstall torch
         pip cache purge
         pip install torch -f https://download.pytorch.org/whl/torch_stable.html
      ```

- [ERROR: pip's dependency resolver does not currently take into account all the packages that are installed](https://stackoverflow.com/questions/72672196/error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-pa/76604141#76604141)
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```
- [Failed to import transformers](https://github.com/huggingface/transformers/issues/11262)
  - Try re-install
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```
