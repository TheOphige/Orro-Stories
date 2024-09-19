from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from openai import OpenAI
import requests
import os
import time
import streamlit as st

# from dotenv import find_dotenv, load_dotenv
# load_dotenv(find_dotenv())

# Retrieve API keys from .env
# IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Retrieve API keys from secrets
IMGUR_CLIENT_ID = st.secrets["IMGUR_CLIENT_ID"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

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
    if not OPENROUTER_API_KEY:
        st.error("API Key not found. Make sure OPENROUTER_API_KEY is set in your environment.")
        return

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
    model_name="mattshumer/reflection-70b:free",
    )

    story_llm = LLMChain(llm=llm,
        prompt=prompt, verbose=True
        )
    
    story = story_llm.predict(scenario=scenario)

    return story


# text to speech
def text2speech(message, max_retries=5, retry_delay=10):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {"inputs": message}

    retries = 0
    while retries < max_retries:
        response = requests.post(API_URL, headers=headers, json=payloads)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            print(f"Content-Type: {content_type}")

            if 'audio/flac' in content_type:
                with open('audio.flac', 'wb') as file:
                    file.write(response.content)
                print("Audio file saved successfully.")
                break

        elif response.status_code == 503:
            # Model is loading, wait for the estimated time
            response_data = response.json()
            estimated_time = response_data.get("estimated_time", retry_delay)
            st.info(f"Model is loading. Retrying in {estimated_time} seconds...")
            time.sleep(estimated_time)
            retries += 1

        elif response.status_code == 500 and 'Model too busy' in response.text:
            st.info(f"Model too busy. Retrying in {retry_delay} seconds... ({retries+1}/{max_retries})")
            retries += 1
            time.sleep(retry_delay)
        
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            break
    else:
        st.error("Max retries reached. Unable to process the request.")


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
            st.image(image_url, caption="Uploaded Image", use_column_width=True)
            
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