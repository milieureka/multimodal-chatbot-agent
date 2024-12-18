# LLaMA-Powered Multimodal Video Understanding Chatbot Agent:

This project provides a chatbot interface for video content captioning and visual Q&A cing video content using the LongVU model. It incorporates Retrieval-Augmented Generation (RAG) to enhance response accuracy and integrates a basic fact-checking mechanism to improve the reliability of the answers.

## Demo

<div align="center">
    <a href='https://vision-cair.github.io/LongVU'><img src="https://github.com/milieureka/multimodal-chatbot-agent/blob/main/demo-2.gif" alt="Demo GIF" style="width: 100%; max-width: 800px;"></a>
</div>

## Product Diagram

- Infrastructure Diagram

![diagram](https://github.com/milieureka/multimodal-chatbot-agent/blob/main/Product%20Diagram.png)

- RAG flow

![rag](https://github.com/milieureka/multimodal-chatbot-agent/blob/main/RAG.png)

## Overview

This application leverages the LongVU model for understanding video input in combination with a text-based question and leverages RAG by retrieve relevant document to help LongVU model to produce better output. It offers the following features:

- **Video and Image Input:** Users can provide video or image as input.
- **Chatbot Interface:** A Gradio interface is implemented to provide a chatbot experience with video input and text-based questions.
- **LongVU Integration:** Spatiotemporal Adaptive Compression for Long Video-Language Understanding. Basically, the model drop repeated frames from video to reduce the token by using SigLIP and DINOv2. Then feed remainding token to LLM model with training dataset from image-text stage ([LLaVA-OneVision-Single](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)) and video-text stage ([VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT))
    + The model architecture of LongVU follows [LLaVA](https://github.com/haotian-liu/LLaVA) and [Cambrian](https://github.com/cambrian-mllm/cambrian)
    + LLM backbone: [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
    + Vision encoder [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [DINOv2](https://huggingface.co/facebook/dinov2-giant) 
- **Retrieval-Augmented Generation (RAG):** Leverages a vector database and embedding model to retrieve additional relevant information based on the user's question to help enhance the quality of the answer.
- **Fact-Checking Mechanism:** Attempts to cross-validate answers with the retrieved information.

## Code Infrastructure

The project is structured with the following components:

1.  **`main.py`**:
    *   **Setup**: Clones the LongVU repository from GitHub, creates a conda environment, installs the required packages, and adds the LongVU source directory to the system path.
    *   **`Chat` Class**: Contains the core logic for loading and interacting with the LongVU model, and performs RAG with text embeddings.
    *   **`generate` Function**: Processes input (video, images, and text), generates responses using RAG and LongVU, and integrates fact-checking results.
    *   **Gradio Interface**: Sets up the chatbot using the Gradio framework.

2. **`factcheck.py`** (Placeholder):
    *  This file currently has a basic fact_check function, but the function can be enhanced to check the answers using more sophisticated algorithms.

3.  **`requirements.txt`**: Lists all the necessary Python packages.

## LongVU Model Reference

This project integrates the LongVU model, which is a Vision-Language model that can understand multimodal input (including images and videos). The model used here is a `qwen` model, from the checkpoint that is downloaded from the GitHub repo.

The LongVU model's components include:

-   **Tokenizer:** Converts input text into tokens that the model understands.
-   **Processor:** Handles pre-processing of input data for the model.
-   **Model:** The core LongVU model architecture for generating answers based on video and text input.

The model loading and usage is handled in `longvu/builder.py`, the checkpoint is loaded from `checkpoints/longvu_qwen`, all the relevant code is imported from `longvu` package after the environment setup in `main.py`.

## Setup Instructions

Follow these steps to set up the project:

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Run `main.py`**: Run the following command to clone the LongVU repository, install dependencies, and start the chatbot application:

    ```bash
    python main.py
    ```

    This script will perform the following actions:
    * Clone the LongVU repository from GitHub to the `LongVU` folder.
    * Creates the virtual environment called `longvu`.
    * Installs the requirements defined in `requirements.txt`.
    * Launch the Gradio app.

    The first time you run it, it may take longer because the script downloads and sets up LongVU and the environment.

3.  **Access the Application:**
    *   Once the script is finished, you'll see a Gradio interface link in the terminal. Open that link in your web browser.

## Usage

1.  **Input Video/Image:** Upload a video or image in the provided input window.
2.  **Enter Question:** Type your question about the video in the text box.
3.  **Send Query:** Press the "Send" button or hit "Enter" in the text box.
4.  **View Response:** The chatbot will provide an answer below the input box. You will also see fact-check information.

## Key Files

-   `main.py`: The core Python script for the project.
-   `requirements.txt`: The text file listing all the required Python packages.
-  `factcheck.py`: The python file with the fact_check function for response verification.

## Reference

```
@article{shen2024longvu,
  author ={Shen, Xiaoqian and Xiong, Yunyang and Zhao, Changsheng and Wu, Lemeng and Chen, Jun and Zhu, Chenchen and Liu, Zechun and Xiao, Fanyi and Varadarajan, Balakrishnan and Bordes, Florian and Liu, Zhuang and Xu, Hu and J. Kim, Hyunwoo and Soran, Bilge and Krishnamoorthi, Raghuraman and Elhoseiny, Mohamed and Chandra, Vikas},
  title = {LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding},
  journal = {arXiv preprint arXiv:2410.17434},
  year = {2024},
}
```
