from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from openai import OpenAI
import requests
import os
import time
import cloudinary
import cloudinary.uploader
import io
import streamlit as st

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# Retrieve API keys from .env
# IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME")
# CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
# CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Retrieve API keys from secrets
IMGUR_CLIENT_ID = st.secrets["IMGUR_CLIENT_ID"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
CLOUDINARY_NAME = st.secrets["CLOUDINARY_NAME"]
CLOUDINARY_API_KEY = st.secrets["CLOUDINARY_API_KEY"]
CLOUDINARY_API_SECRET = st.secrets["CLOUDINARY_API_SECRET"]

# upload image to cloud
def upload_to_imgur(image):
    """Upload an image to Imgur and return the URL."""
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    url = "https://api.imgur.com/3/image"
    
    # Send the image to Imgur
    response = requests.post(url, headers=headers, files={"image": image})
    
    if response.status_code == 200:
        # Get the URL of the uploaded image
        image_url = response.json()["data"]["link"]
        return image_url
    else:
        st.error("Failed to upload image")
        return None


## img2text 
def img2text(image_url):
    """Use OpenRouter's API to process the image and return the description."""
    # if not OPENROUTER_API_KEY:
    #     st.error("API Key not found. Make sure OPENROUTER_API_KEY is set in your environment.")
    #     return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    # Make the API request to generate text from the uploaded image URL
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": os.getenv("YOUR_SITE_URL"),  # Optional, for rankings on OpenRouter
            "X-Title": os.getenv("YOUR_APP_NAME"),  # Optional, for app title in OpenRouter rankings
        },
        model="qwen/qwen-2-vl-7b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )

    # Return the response from OpenRouter
    return completion.choices[0].message.content


# llm
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short horror story based on a single narrative,
    the story should be no more than 2000 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="microsoft/mai-ds-r1:free",
    )

    story_llm = LLMChain(llm=llm,
        prompt=prompt, verbose=True
        )
    
    story = story_llm.predict(scenario=scenario)

    return story


# text to speech
from gtts import gTTS
import cloudinary
import cloudinary.uploader
import io
import time
import streamlit as st

# text to speech
def text2speech(message, lang="en-us", slow=False):
    try:
        # Initialize Cloudinary
        cloudinary.config(
            cloud_name=CLOUDINARY_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET
        )

        # Generate speech using gTTS
        tts = gTTS(text=message, lang=lang, slow=slow)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)  # Reset file pointer to the start

        # Upload to Cloudinary
        upload_response = cloudinary.uploader.upload(
            audio_data,
            resource_type='raw',  # raw for non-image files like audio
            public_id="generated_audio",
            format="mp3"  # gTTS outputs MP3 format
        )

        # Retrieve the URL for the uploaded audio
        audio_url = upload_response.get('url')
        print("Audio file uploaded successfully.")
        return audio_url

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None



# streamlit gui
def main():

    st.set_page_config(page_title="Orro Stories", page_icon="🎈")

    page_image = "orro_stories.webp"
    st.image(page_image)
    st.header("Orro Stories")
    st.write("Generate audio Horror story based on the image you upload.")
    st.info("Please ensure you're not uploading any image that reveals your privacy.")
    uploaded_file = st.file_uploader("Choose an image...", type=
    ["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Upload the image to Imgur and get the URL
        image_url = upload_to_imgur(uploaded_file)

        if image_url:
            st.image(image_url, caption="Uploaded Image", use_container_width=True)
            
            # Generate text from the uploaded image
            with st.spinner("Generating scenario from image..."):
                scenario = img2text(image_url)

            # display scenario
            with st.expander("scenario"):
                st.write(scenario)

            # generate story from scenario
            with st.spinner("Generating story..."):
                story = generate_story(scenario)

            # display story
            with st.expander("story"):
                st.write(story)

            # Convert the story to audio
            with st.spinner("Converting story to audio..."): 
                audio_file = text2speech(story)
            if audio_file:
                st.audio(audio_file)

if __name__ == '__main__':
    main()