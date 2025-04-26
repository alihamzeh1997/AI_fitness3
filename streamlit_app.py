import streamlit as st
import cv2
import os
import time
import numpy as np
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import tempfile
import toml
import json
from PIL import Image, ImageDraw, ImageFont
import io

# LangChain imports
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# Install required packages
# pip install streamlit opencv-python numpy langchain openai faiss-cpu pillow

# Custom imports for OpenRouter
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Dict, Any, Mapping, Union
import requests

# Custom OpenRouter LLM class
class OpenRouterLLM(LLM):
    model_name: str
    api_key: str
    system_prompt: str = "You are a helpful AI assistant."
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        
        if stop:
            payload["stop"] = stop
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from OpenRouter API: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

from langchain_core.language_models import ChatResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration
from typing import List, Optional, Any, Union, Mapping
import requests
# Custom OpenRouter Chat Model
class OpenRouterChatModel(BaseChatModel):
    model_name: str
    api_key: str
    
    @property
    def _llm_type(self) -> str:
        return "openrouter_chat"
    
    def _generate(
        self,
        messages: List[Union[HumanMessage, SystemMessage, AIMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "YOUR_APP_URL",  # Required by OpenRouter
            "X-Title": "YOUR_APP_NAME"       # Recommended
        }
        
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
            
            formatted_messages.append({
                "role": role,
                "content": message.content
            })
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"OpenRouter API Error {response.status_code}: {response.text}")
        
        response_data = response.json()
        message = response_data["choices"][0]["message"]
        
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=message["content"]),
                    generation_info=response_data.get("usage", {})
                )
            ],
            llm_output={
                "model": self.model_name,
                "usage": response_data.get("usage", {})
            }
        )
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
    
# Image handling functions
def resize_image(image, max_dimension=300):
    """Resize image with the longest dimension set to max_dimension"""
    height, width = image.shape[:2]
    
    if height > width:
        ratio = max_dimension / height
        new_height = max_dimension
        new_width = int(width * ratio)
    else:
        ratio = max_dimension / width
        new_width = max_dimension
        new_height = int(height * ratio)
    
    resized = cv2.resize(image, (new_width, new_height))
    return resized

def annotate_frame(frame, timestamp):
    """Add timestamp annotation to frame"""
    # Convert to PIL Image for better text rendering
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw timestamp at bottom right
    draw.text((10, 10), timestamp, font=font, fill=(255, 255, 255))
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def extract_frames(video_path, fps=1):
    """Extract frames from video at specified FPS"""
    frames = []
    timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return [], []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    # Calculate frame extraction interval
    interval = int(video_fps / fps)
    if interval < 1:
        interval = 1
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Calculate timestamp
            current_time = frame_count / video_fps
            mins = int(current_time // 60)
            secs = int(current_time % 60)
            ms = int((current_time - int(current_time)) * 1000)
            timestamp = f"{mins:02d}:{secs:02d}.{ms:03d}"
            
            # Resize and annotate frame
            resized_frame = resize_image(frame)
            annotated_frame = annotate_frame(resized_frame, timestamp)
            
            frames.append(annotated_frame)
            timestamps.append(timestamp)
        
        frame_count += 1
    
    cap.release()
    return frames, timestamps

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def group_frames_by_time(frames, timestamps, interval=5):
    """Group frames into specified time intervals"""
    groups = []
    current_group = []
    current_group_timestamps = []
    
    for i, timestamp in enumerate(timestamps):
        time_parts = timestamp.split(':')
        mins = int(time_parts[0])
        secs = float(time_parts[1])
        total_secs = mins * 60 + secs
        
        group_index = int(total_secs // interval)
        
        if not current_group or int(float(current_group_timestamps[0].split(':')[0]) * 60 + 
                               float(current_group_timestamps[0].split(':')[1])) // interval == group_index:
            current_group.append(frames[i])
            current_group_timestamps.append(timestamp)
        else:
            groups.append((current_group.copy(), current_group_timestamps.copy()))
            current_group = [frames[i]]
            current_group_timestamps = [timestamp]
    
    if current_group:
        groups.append((current_group, current_group_timestamps))
    
    return groups

def create_vision_prompt_with_images(frames, system_prompt, user_prompt, previous_summary=None):
    """Create a prompt with base64 encoded images for vision models"""
    base64_images = [encode_image_to_base64(frame) for frame in frames]
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add previous summary if available
    if previous_summary:
        messages.append({
            "role": "user", 
            "content": f"Here is the summary of the previous segment: {previous_summary}"
        })
        messages.append({
            "role": "assistant", 
            "content": "I understand. I'll consider this previous information when analyzing the next frames."
        })
    
    # Add user prompt with images
    user_content = [{"type": "text", "text": user_prompt}]
    
    # Add images
    for img in base64_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        })
    
    messages.append({"role": "user", "content": user_content})
    
    return messages

def analyze_video_segment(frames, timestamps, model, api_key, system_prompt, user_prompt, previous_summary=None):
    """Analyze a segment of the video using the selected model"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = create_vision_prompt_with_images(frames, system_prompt, user_prompt, previous_summary)
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            st.error(f"Error from OpenRouter API: {response.text}")
            return None
        
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error analyzing video segment: {str(e)}")
        return None

def create_rag_system(segment_analyses, timestamps):
    """Create RAG system from segment analyses"""
    documents = []
    
    for i, (analysis, timestamp_group) in enumerate(zip(segment_analyses, timestamps)):
        start_time = timestamp_group[0]
        end_time = timestamp_group[-1]
        
        doc = Document(
            page_content=analysis,
            metadata={
                "segment": i,
                "start_time": start_time,
                "end_time": end_time,
                "time_range": f"{start_time} - {end_time}"
            }
        )
        documents.append(doc)
    
    # Create embedding function
    # embedding_function = OpenAIEmbeddings(openai_api_key=st.secrets["openrouter_api_key"], openai_api_base="https://openrouter.ai/api/v1")
    model_name = "BAAI/bge-small-en-v1.5"  # Top open-source model
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # or "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
        )


    # Create vector store
    vector_store = FAISS.from_documents(documents, embedding_function)
    
    return vector_store

def create_chat_model():
    """Create LangChain chat model for conversation"""
    chat_model = OpenRouterChatModel(
        model_name="deepseek/deepseek-v3-base:free",
        api_key=st.secrets["openrouter_api_key"]
    )
    return chat_model

def get_system_prompt():
    """Get system prompt for the vision model"""
    return """You are a professional fitness trainer and video analyzer specialized in exercise form analysis. 
    Your task is to analyze fitness videos and provide detailed feedback on exercise form, technique, and performance.
    
    For each segment of video you receive, you should:
    1. Identify the exercise being performed
    2. Track the joints and body parts involved in the exercise
    3. Count the number of repetitions with exact timestamps
    4. Analyze the tempo of each repetition (how fast or slow each phase is performed)
    5. Provide detailed form analysis for each repetition
    
    Be precise with your observations and always provide specific timestamps when referring to events in the video.
    Apply your expert knowledge of proper exercise form to provide constructive feedback.
    """

def get_initial_user_prompt():
    """Get initial user prompt for the first video segment"""
    return """Analyze these frames from a fitness video and provide the following information:
    1. Identify the specific exercise being performed
    2. Track key joints and body parts involved in the movement
    3. Count the number of complete repetitions with timestamps
    4. Analyze the tempo of each repetition
    5. Provide form analysis for each repetition
    
    Be specific about what you observe in these exact frames.
    """

def get_continuation_user_prompt():
    """Get user prompt for subsequent video segments"""
    return """Continuing from the previous segment, analyze these new frames and provide updated information:
    1. Continue tracking the exercise being performed
    2. Track key joints and body parts involved in the movement
    3. Count any additional repetitions with timestamps
    4. Continue tempo analysis for new repetitions
    5. Provide form analysis for new repetitions
    
    Remember to consider what happened in the previous segment when counting repetitions.
    """

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Fitness Video Analyzer",
        page_icon="💪",
        layout="wide"
    )
    
    # CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        color: #333333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        border: 2px dashed #CCCCCC;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    .chat-container {
        border: 1px solid #EEEEEE;
        border-radius: 5px;
        padding: 10px;
        height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #E8F0FE;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-message {
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("<h1 class='main-header'>🏋️ Fitness Video Analyzer</h1>", unsafe_allow_html=True)
    
    # Check for password in secrets
    if "app_password" in st.secrets:
        # Session state for authentication
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            password = st.text_input("Enter password to access the app", type="password")
            if password:
                if password == st.secrets["app_password"]:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
            return
    else:
        # No password required
        st.session_state.authenticated = True
    
    # Initialize session state variables
    if "frames" not in st.session_state:
        st.session_state.frames = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "segment_analyses" not in st.session_state:
        st.session_state.segment_analyses = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "video_analyzed" not in st.session_state:
        st.session_state.video_analyzed = False
    if "timestamp_groups" not in st.session_state:
        st.session_state.timestamp_groups = []
    
    # App sidebar for configuration
    with st.sidebar:
        st.markdown("<h2 class='subheader'>Configuration</h2>", unsafe_allow_html=True)
        
        # FPS selection
        fps = st.slider("Frames per second", 1, 30, 5)
        
        # Model selection
        model_options = {
            "Free Models": {
                "Google": [
                    "google/gemma-3-27b-it:free",
                    "google/gemini-2.0-flash-thinking-exp:free",
                    "google/gemini-2.5-pro-exp-03-25:free"
                ],
                "Meta": [
                    "meta-llama/llama-3.2-11b-vision-instruct:free",
                    "meta-llama/llama-4-maverick:free"
                ],
                "Mistral AI": [
                    "mistralai/mistral-small-3.1-24b-instruct:free"
                ],
                "Qwen": [
                    "qwen/qwen2.5-vl-32b-instruct:free"
                ],
                "ByteDance": [
                    "bytedance-research/ui-tars-72b:free"
                ],
                "Allen AI": [
                    "allenai/molmo-7b-d:free"
                ],
                "Moonshot AI": [
                    "moonshotai/kimi-vl-a3b-thinking:free"
                ]
            },
            "Premium Models": {
                "Google": [
                    "google/gemini-2.5-flash-preview",
                    "google/gemini-2.5-pro-preview-03-25"
                ],
                "Anthropic": [
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3.5-haiku:beta",
                    "anthropic/claude-3.7-sonnet:thinking",
                    "anthropic/claude-3.7-sonnet"
                ],
                "OpenAI": [
                    "openai/gpt-4o-mini-2024-07-18",
                    "openai/chatgpt-4o-latest",
                    "openai/o3",
                    "openai/o4-mini-high",
                    "openai/gpt-4.1-mini",
                    "openai/gpt-4.1"
                ],
                "X AI": [
                    "x-ai/grok-vision-beta",
                    "x-ai/grok-2-vision-1212"
                ],
                "Mistral AI": [
                    "mistralai/pixtral-large-2411"
                ],
                "Microsoft": [
                    "microsoft/phi-4-multimodal-instruct"
                ]
            }
        }
        
        # First select category
        model_category = st.radio("Model Category", list(model_options.keys()))
        
        # Then select provider
        providers = list(model_options[model_category].keys())
        model_provider = st.selectbox("Model Provider", providers)
        
        # Finally select specific model
        available_models = model_options[model_category][model_provider]
        selected_model = st.selectbox("Select Model", available_models)
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            segment_duration = st.slider("Segment duration (seconds)", 1, 15, 5)
    
    # Main area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("<h2 class='subheader'>Upload Video</h2>", unsafe_allow_html=True)
        
        # Video upload section
        with st.container():
            st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a fitness video file", type=["mp4", "mov", "avi", "mkv"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                # Button to process video
                if st.button("Process Video"):
                    with st.spinner("Extracting frames..."):
                        # Extract frames
                        frames, timestamps = extract_frames(video_path, fps)
                        
                        if frames:
                            st.session_state.frames = frames
                            st.session_state.timestamps = timestamps
                            st.session_state.timestamp_groups = []
                            
                            # Group frames by time segments
                            frame_groups = group_frames_by_time(frames, timestamps, segment_duration)
                            
                            for frames_group, timestamps_group in frame_groups:
                                st.session_state.timestamp_groups.append(timestamps_group)
                            
                            # Display sample frames
                            st.success(f"Extracted {len(frames)} frames from video!")
                            
                            # Show sample frame
                            if len(frames) > 0:
                                st.image(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), caption="Sample Frame")
                        else:
                            st.error("Failed to extract frames from the video.")
            
            # Button to analyze video after frames are extracted
            if st.session_state.frames and not st.session_state.video_analyzed:
                if st.button("Analyze Video"):
                    with st.spinner("Analyzing video segments..."):
                        # Group frames by time segments
                        frame_groups = group_frames_by_time(st.session_state.frames, st.session_state.timestamps, segment_duration)
                        
                        # Get prompts
                        system_prompt = get_system_prompt()
                        initial_prompt = get_initial_user_prompt()
                        continuation_prompt = get_continuation_user_prompt()
                        
                        # Process each segment
                        segment_analyses = []
                        previous_summary = None
                        timestamp_groups = []
                        
                        for i, (frames_group, timestamps_group) in enumerate(frame_groups):
                            # Update progress bar
                            progress_text = f"Analyzing segment {i+1}/{len(frame_groups)}"
                            progress = st.progress(0)
                            
                            # Use initial prompt for first segment, continuation prompt for others
                            current_prompt = initial_prompt if i == 0 else continuation_prompt
                            
                            # Analyze segment
                            analysis = analyze_video_segment(
                                frames_group, 
                                timestamps_group,
                                selected_model,
                                st.secrets["openrouter_api_key"],
                                system_prompt,
                                current_prompt,
                                previous_summary
                            )
                            
                            if analysis:
                                segment_analyses.append(analysis)
                                timestamp_groups.append(timestamps_group)
                                previous_summary = analysis
                                
                                # Update progress
                                progress.progress((i + 1) / len(frame_groups))
                            else:
                                st.error(f"Failed to analyze segment {i+1}")
                                break
                        
                        # Create RAG system if all segments were analyzed
                        if len(segment_analyses) == len(frame_groups):
                            st.session_state.segment_analyses = segment_analyses
                            st.session_state.timestamp_groups = timestamp_groups
                            
                            with st.spinner("Building knowledge base..."):
                                rag = create_rag_system(segment_analyses, timestamp_groups)
                                st.session_state.rag_system = rag
                                st.session_state.video_analyzed = True
                                
                                st.success("Video analysis complete! You can now chat about the video.")
                        else:
                            st.warning("Video analysis was not completed. Please try again.")
    
    with col2:
        st.markdown("<h2 class='subheader'>Chat with Your Video</h2>", unsafe_allow_html=True)
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
        
        # Input for new messages
        if st.session_state.video_analyzed:
            user_input = st.text_input("Ask about the video:", key="user_input")
            
            if user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get response
                with st.spinner("Thinking..."):
                    # Create chat model
                    chat_model = create_chat_model()
                    
                    # Create memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    
                    # Add previous messages to memory
                    for i in range(0, len(st.session_state.messages) - 1, 2):
                        if i + 1 < len(st.session_state.messages):
                            memory.chat_memory.add_user_message(st.session_state.messages[i]["content"])
                            memory.chat_memory.add_ai_message(st.session_state.messages[i + 1]["content"])
                    
                    # Create retriever
                    retriever = st.session_state.rag_system.as_retriever(
                        search_type="mmr",  # Maximum Marginal Relevance
                        search_kwargs={"k": 3}  # Return top 3 most relevant documents
                    )
                    
                    # Create chain
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=chat_model,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True
                    )
                    
                    # Get response
                    response = qa_chain({"question": user_input})
                    answer = response["answer"]
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Clear input and rerun to update chat
                st.rerun()
        else:
            st.info("Upload and analyze a video to start chatting!")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, LangChain, and OpenRouter.")

if __name__ == "__main__":
    main()