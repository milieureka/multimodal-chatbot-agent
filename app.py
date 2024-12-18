import os
import re
import traceback
import torch
import gradio as gr
import sys
import numpy as np
import cv2
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

# RAG and Fact-Checking imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from factcheck import fact_check

def setup_longvu():
    """Clones the LongVU repository, sets up the conda environment, and installs dependencies."""
    repo_url = "https://github.com/Vision-CAIR/LongVU"
    repo_dir = "LongVU"  # The directory where it will be cloned

    # Check if the directory already exists
    if os.path.exists(repo_dir):
        logging.info(f"Directory '{repo_dir}' already exists. Skipping cloning.")
    else:
        logging.info("Cloning LongVU repository...")
        subprocess.run(["git", "clone", repo_url], check=True)

    logging.info("Creating and activating conda environment...")
    try:
         subprocess.run(["conda", "create", "-n", "longvu", "python=3.10", "-y"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating conda env: {e}")
        return False
    
    # Use the full path to ensure the env is activated
    conda_env_path = os.path.join(os.getcwd(), "venv")
    activate_command = os.path.join(conda_env_path, "Scripts", "activate") if os.name == 'nt' else os.path.join(conda_env_path, "bin", "activate")

    if os.name == 'nt':
        subprocess.run([activate_command, "longvu"], check=True, shell=True)
    else:
         subprocess.run(["source", activate_command, "longvu"], check=True, shell=True)

    os.chdir(repo_dir)
    logging.info("Installing requirements...")

    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing requirements: {e}")
        return False

    logging.info("LongVU setup complete.")
    return True


# Execute LongVU setup
if not setup_longvu():
    print("LongVU setup failed. Exiting.")
    sys.exit(1)

# Add the directory containing the longvu package to the Python path
sys.path.append(os.path.join(os.getcwd(),'LongVU'))

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

class Chat:

    def __init__(self):
        self.version = "qwen"
        model_name = "cambrian_qwen"
        model_path = "./checkpoints/longvu_qwen"
        device = "cuda:7"

        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, model_name, device=device)
        self.model.eval()

        # Initialize embeddings, vector store, RAG chain and fact check model
        self.embedding_model_name = "all-mpnet-base-v2"
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = self._load_vector_store()  # Load or create vector store

    def _load_vector_store(self, persist_directory="db"):
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logging.info("Loading vector store from disk...")
            return Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            logging.info("Creating vector store...")
            # Replace with your data loading
            documents = self._create_dummy_documents()
            vector_store = Chroma.from_documents(documents, self.embedding, persist_directory=persist_directory)
            vector_store.persist()
            return vector_store

    def _create_dummy_documents(self):
         # Replace with your data loading
        return [
            Document(page_content="Video one is about cats", metadata={"source": "video_one"}),
            Document(page_content="Video two is about dogs", metadata={"source": "video_two"})
        ]

    def remove_after_last_dot(self, s):
        last_dot_index = s.rfind('.')
        if last_dot_index == -1:
            return s
        return s[:last_dot_index + 1]

    # @spaces.GPU(duration=120)
    @torch.inference_mode()
    def generate(self, data: list, message, temperature, top_p, max_output_tokens):
        # TODO: support multiple turns of conversation.
        assert len(data) == 1

        tensor, image_sizes, modal = data[0]

        conv = conv_templates[self.version].copy()

        if isinstance(message, str):
            conv.append_message("user", DEFAULT_IMAGE_TOKEN + '\n' + message)
        elif isinstance(message, list):
            if DEFAULT_IMAGE_TOKEN not in message[0]['content']:
                message[0]['content'] = DEFAULT_IMAGE_TOKEN + '\n' + message[0]['content']
            for mes in message:
                conv.append_message(mes["role"], mes["content"])

        conv.append_message("assistant", None)

        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        if "llama3" in self.version:
            input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_output_tokens,
                use_cache=True,
                top_p=top_p,
                stopping_criteria=[stopping_criteria],
            )

        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return self.remove_after_last_dot(pred)

    def generate_with_rag(self, data, message, temperature, top_p, max_output_tokens):
        if isinstance(message, str):
            query = message
        elif isinstance(message, list):
            query = message[-1]['content'] # consider the last user message
        else:
            query = ""

        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)

        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context:{context}\n\nUser question: {query}"

        # Call the base model
        return self.generate(data, prompt, temperature, top_p, max_output_tokens)
    
    def fact_check_response(self, response, context, query):
        try:
            fact_result = fact_check(response, context, query)
            logging.info(f"Fact check result: {fact_result}")
        except Exception as e:
            logging.error(f"Error in fact checking: {e}")
            fact_result = "Not able to fact check"
        return fact_result


# @spaces.GPU(duration=120)
def generate(image, video, message, chatbot, textbox_in, temperature, top_p, max_output_tokens, dtype=torch.float16):
    if textbox_in is None:
        raise gr.Error("Chat messages cannot be empty")
        return (
            gr.update(value=image, interactive=True),
            gr.update(value=video, interactive=True),
            message,
            chatbot,
            None,
        )

    data = []
    processor = handler.processor
    try:
        if image is not None:
            data.append((processor['image'](image).to(handler.model.device, dtype=dtype), None, '<image>'))
        elif video is not None:
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.arange(0, frame_count, round(fps))
            video_tensor = []
            for i in frame_indices:
                ret, frame = cap.read()
                if not ret:
                    break
                video_tensor.append(frame)
            cap.release()
            video_tensor = np.stack(video_tensor)
            image_sizes = [video_tensor[0].shape[:2]]
            video_tensor = process_images(video_tensor, processor, handler.model.config)
            video_tensor = [item.unsqueeze(0).to(handler.model.device, dtype=dtype) for item in video_tensor]
            data.append((video_tensor, image_sizes, '<video>'))
        elif image is None and video is None:
            data.append((None, None, '<text>'))
        else:
            raise NotImplementedError("Not support image and video at the same time")
    except Exception as e:
        traceback.print_exc()
        return gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), message, chatbot, None

    assert len(message) % 2 == 0, "The message should be a pair of user and system message."

    show_images = ""
    if image is not None:
        show_images += f'<img src="./file={image}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if video is not None:
        show_images += f'<video controls playsinline width="300" style="display: inline-block;"  src="./file={video}"></video>'

    one_turn_chat = [textbox_in, None]

    # 1. first run case
    if len(chatbot) == 0:
        one_turn_chat[0] += "\n" + show_images
    # 2. not first run case
    else:
        # scanning the last image or video
        length = len(chatbot)
        for i in range(length - 1, -1, -1):
            previous_image = re.findall(r'<img src="./file=(.+?)"', chatbot[i][0])
            previous_video = re.findall(r'<video controls playsinline width="500" style="display: inline-block;"  src="./file=(.+?)"', chatbot[i][0])

            if len(previous_image) > 0:
                previous_image = previous_image[-1]
                # 2.1 new image append or pure text input will start a new conversation
                if (video is not None) or (image is not None and os.path.basename(previous_image) != os.path.basename(image)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break
            elif len(previous_video) > 0:
                previous_video = previous_video[-1]
                # 2.2 new video append or pure text input will start a new conversation
                if image is not None or (video is not None and os.path.basename(previous_video) != os.path.basename(video)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break

    message.append({'role': 'user', 'content': textbox_in})

    # Generate response using RAG
    text_en_out = handler.generate_with_rag(data, message, temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens)

    # Fact Check
    if isinstance(message, str):
        query_to_check = message
    elif isinstance(message, list):
        query_to_check = message[-1]['content']
    else:
        query_to_check = ""

    # get context for fact check from RAG
    retriever = handler.vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query_to_check)
    context_to_check = "\n".join([doc.page_content for doc in relevant_docs])

    fact_result = handler.fact_check_response(text_en_out, context_to_check, query_to_check)
    
    text_en_out = f"{text_en_out}\n\nFact Check Result:{fact_result}"
    message.append({'role': 'assistant', 'content': text_en_out})
    one_turn_chat[1] = text_en_out
    chatbot.append(one_turn_chat)

    return gr.update(value=image, interactive=True), gr.update(value=video, interactive=True), message, chatbot, None


def regenerate(message, chatbot):
    message.pop(-1), message.pop(-1)
    chatbot.pop(-1)
    return message, chatbot


def clear_history(message, chatbot):
    message.clear(), chatbot.clear()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True),
            message, chatbot,
            gr.update(value=None, interactive=True))

if __name__ == "__main__":
    handler = Chat()

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)

    theme = gr.themes.Default()

    with gr.Blocks(title='Team-5', theme=theme, css="") as demo:
        gr.Markdown("""
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div>
            <h1 >Video Content Generating</h1>
        </div>
        </div>
        """)
        message = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.State(None)
                video = gr.Video(label="Input Video")

                with gr.Accordion("Parameters", open=True) as parameter_row:

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )

                    top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            interactive=True,
                            label="Top P",
                    )

                    max_output_tokens = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=128,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="Team-5", bubble_full_width=True, height=420)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary", interactive=True)
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn     = gr.Button(value="👍  Upvote", interactive=True)
                    downvote_btn   = gr.Button(value="👎  Downvote", interactive=True)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn      = gr.Button(value="🗑️  Clear history", interactive=True)

        submit_btn.click(
            generate, 
            [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens],
            [image, video, message, chatbot])

        regenerate_btn.click(
            regenerate, 
            [message, chatbot], 
            [message, chatbot]).then(
            generate, 
            [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens], 
            [image, video, message, chatbot, textbox])

        textbox.submit(
            generate,
            [
                image,
                video,
                message,
                chatbot,
                textbox,
                temperature,
                top_p,
                max_output_tokens,
            ],
            [image, video, message, chatbot, textbox],
        )
        
        clear_btn.click(
            clear_history, 
            [message, chatbot],
            [image, video, message, chatbot, textbox])

    demo.launch(share=True)