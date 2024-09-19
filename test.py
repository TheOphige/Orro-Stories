# from dotenv import find_dotenv, load_dotenv
# from transformers import pipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI

# import requests
# import os

# import streamlit as st

# load_dotenv(find_dotenv())

# # Retrieve the Hugging Face API token from the environment
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ## img2text 
# def img2text(filename):
#     # image_to_text = pipeline("image-to-text", model= "")

#     # text = image_to_text(url)[0]['generated_text']

#     # print(text)
#     # return text


#     API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
#     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)

#     print(response)
#     print(response.json())
#     print(response.content)
#     #return output
#     #return response.json()

    
    

# img2text("cats.jpg")

# # # llm
# # def generate_story(scenario):
# #     template = """
# #     You are a story teller;
# #     You can generate a short story based on a single narrative,
# #     the story should be no more than 20 words;

# #     CONTEXT: {scenario}
# #     STORY:
# #     """

# #     prompt = PromptTemplate(template=template, input_variables=["scenario"])

# #     story_llm = LLMChain(llm=OpenAI(
# #         model_name="gpt-3.5-turbo", temperature=1),
# #         prompt=prompt, verbose=True
# #         )
    
# #     story = story_llm.predict(scenario=scenario)

# #     print(story)
# #     return story

# # # text to speech
# # def text2speech(message):
# #     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
# #     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
# #     payloads = {
# #         "inputs": message
# #     }
# #     response = requests.post(API_URL, headers=headers, json=payloads)
# #     with open('audio.flac', 'wb') as file:
# #         file.write(response.content)



# # def main():

# #     st.set_page_config(page_title="img 2 audio story", page_icon="🎈")

# #     st.header("Turn img into audio story")
# #     uploaded_file = st.file_uploader("Choose an image...", type=
# #     "jpg")

# #     if uploaded_file is not None:
# #         print(uploaded_file)
# #         bytes_data = uploaded_file.getvalue()
# #         with open(uploaded_file.name, "wb") as file:
# #             file.write(bytes_data)
# #         st.image(uploaded_file, caption="Uploaded Image.",
# #                  use_column_width=True)
# #         scenario = img2text(uploaded_file.name)
# #         story = generate_story(scenario)
# #         text2speech(story)

# #         with st.expander("scenario"):
# #             st.write(scenario)
# #         with st.expander("story"):
# #             st.write(story)

# #         st.audio("audio.flac")

# # if __name__ == '__main__':
# #     main()



import requests
import streamlit as st
import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Retrieve API keys from .env
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

def generate_image_text(image_url):
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

# text = generate_image_text("https://imgur.com/ZzNtnuG"+".jpeg")
# print(text)

# Streamlit GUI for image upload and processing
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Upload the image to Imgur and get the URL
    image_url = upload_to_imgur(uploaded_file)

    if image_url:
        st.write(f"Image URL: {image_url}")
        st.image(image_url, caption="Uploaded Image", use_column_width=True)
        
        # Generate text from the uploaded image
        result = generate_image_text(image_url)
        if result:
            st.write("Image Description:", result)


# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from os import getenv
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())

# def generate_story(scenario):
#     template = """
#     You are a story teller;
#     You can generate a short horror story based on a single narrative,
#     the story should be no more than 1500 words;

#     CONTEXT: {scenario}
#     STORY:
#     """

#     prompt = PromptTemplate(template=template, input_variables=["scenario"])

#     llm = ChatOpenAI(
#     openai_api_key=getenv("OPENROUTER_API_KEY"),
#     openai_api_base=getenv("OPENROUTER_BASE_URL"),
#     model_name="mattshumer/reflection-70b:free",
#     )

#     story_llm = LLMChain(llm=llm,
#         prompt=prompt, verbose=True
#         )
    
#     story = story_llm.predict(scenario=scenario)

#     print(story)
#     return story

# # Example usage
# scenario = "A knight fighting a dragon to save a princess"
# generate_story(scenario)


# from openai import OpenAI
# from os import getenv

# # Initialize the OpenAI client with your API key
# api_key = getenv("OPENROUTER_API_KEY")
# if not api_key:
#     raise ValueError("API Key not found. Make sure OPENROUTER_API_KEY is set in your environment.")

# # gets API Key from environment variable OPENAI_API_KEY
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key=api_key,
# )

# def generate_story(scenario):
#     response =  client.chat.completions.create(

#         model="mattshumer/reflection-70b:free",
 
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Generate a short story based on this narrative:"},
#                     {"type": "text", "text": scenario}
#                 ]
#             }
#         ]
#     )

#     story = response.choices[0].message.content
#     print(story)
#     return story

# # Example usage
# scenario = "A knight fighting a dragon to save a princess"
# generate_story(scenario)


# import requests
# from dotenv import find_dotenv, load_dotenv
# import os

# load_dotenv(find_dotenv())

# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# import time


# def text2speech(message, max_retries=5, retry_delay=10):
#     API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
#     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
#     payloads = {"inputs": message}

#     retries = 0
#     while retries < max_retries:
#         response = requests.post(API_URL, headers=headers, json=payloads)

#         if response.status_code == 200:
#             content_type = response.headers.get('Content-Type')
#             print(f"Content-Type: {content_type}")

#             if 'audio/flac' in content_type:
#                 with open('audio.flac', 'wb') as file:
#                     file.write(response.content)
#                 print("Audio file saved successfully.")
#                 break

#         elif response.status_code == 503:
#             # Model is loading, wait for the estimated time
#             response_data = response.json()
#             estimated_time = response_data.get("estimated_time", retry_delay)
#             print(f"Model is loading. Retrying in {estimated_time} seconds...")
#             time.sleep(estimated_time)
#             retries += 1

#         elif response.status_code == 500 and 'Model too busy' in response.text:
#             print(f"Model too busy. Retrying in {retry_delay} seconds... ({retries+1}/{max_retries})")
#             retries += 1
#             time.sleep(retry_delay)
        
#         else:
#             print(f"Error: {response.status_code} - {response.text}")
#             break
#     else:
#         print("Max retries reached. Unable to process the request.")






# Example usage
# text2speech("Hello man be okay")


# def text2speech(message):
#     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
#     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
#     payloads = {
#         "inputs": message
#     }
#     response = requests.post(API_URL, headers=headers, json=payloads)
#     with open('audio.flac', 'wb') as file:
#         file.write(response.content)
#     print("Audio file saved successfully.")



# story = """
# In the land of Eridoria, a beautiful princess named Sophia was kidnapped by a fierce dragon named Tharros. The dragon took Sophia deep into its lair, a cave system hidden in the mountains. The king, desperate to save his daughter, called upon his bravest knight, Sir Edward.

# Sir Edward set off immediately to rescue Sophia. As he approached the dragon's lair, he could feel the ground shaking beneath his feet from Tharros's powerful movements. He drew his sword and charged into battle, ready to face whatever dangers lay ahead.

# Tharros, sensing the knight's presence, emerged from the shadows. Its scales glinted in the dim light of the cave, and its eyes burned with fierce intensity. Sir Edward stood tall, his armor gleaming in the faint light.

# The battle raged on, with Sir Edward dodging the dragon's flames and striking at its scales. But Tharros was relentless, its claws swiping at the knight with precision. Sir Edward parried each blow, his sword clanging against the dragon's claws.

# Just as the knight thought he had the upper hand, Tharros unleashed a powerful blast of fire. Sir Edward raised his shield, but the flames were too intense. He stumbled backward, his armor scorched and smoldering.

# Sophia, seeing her rescuer in danger, found the strength to break free from her chains. She rushed to Sir Edward's side, using her knowledge of the cave system to guide him to a hidden chamber filled with reflective crystals. The knight used these crystals to redirect the dragon's fire back at itself, weakening Tharros significantly.

# With the dragon weakened, Sir Edward launched a final attack. Sophia, filled with newfound courage, stood by his side, offering words of encouragement as she watched the knight fight for her life. Together, they managed to defeat Tharros and free the kingdom from its terror.

# As they made their way back to the castle, Sophia realized that she had fallen in love with Sir Edward, not just as her rescuer, but as the man who had shown her her own strength and bravery. The kingdom celebrated their victory, and the king, grateful for his daughter's safe return, asked Sir Edward to be his son-in-law. The knight, honored by the request, accepted, and the two were married in a grand ceremony, marking the beginning of a new era of peace and prosperity for the kingdom of Eridoria.
# """



# text2speech(story)



################# FULL WORKING CODE #######################
# from dotenv import find_dotenv, load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# from openai import OpenAI
# import requests
# import os
# import time
# import streamlit as st


# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# def upload_to_imgur(image):
#     """Upload an image to Imgur and return the URL."""
#     headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
#     url = "https://api.imgur.com/3/image"
    
#     # Send the image to Imgur
#     response = requests.post(url, headers=headers, files={"image": image})
    
#     if response.status_code == 200:
#         # Get the URL of the uploaded image
#         image_url = response.json()["data"]["link"]
#         return image_url
#     else:
#         st.error("Failed to upload image")
#         return None


# ## img2text 
# def img2text(image_url):
#     """Use OpenRouter's API to process the image and return the description."""
#     if not OPENROUTER_API_KEY:
#         print("API Key not found. Make sure OPENROUTER_API_KEY is set in your environment.")
#         return

#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENROUTER_API_KEY
#     )

#     # Make the API request to generate text from the uploaded image URL
#     completion = client.chat.completions.create(
#         extra_headers={
#             "HTTP-Referer": os.getenv("YOUR_SITE_URL"),  # Optional, for rankings on OpenRouter
#             "X-Title": os.getenv("YOUR_APP_NAME"),  # Optional, for app title in OpenRouter rankings
#         },
#         model="qwen/qwen-2-vl-7b-instruct:free",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "What's in this image?"
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url
#                         }
#                     }
#                 ]
#             }
#         ]
#     )

#     # Return the response from OpenRouter
#     return completion.choices[0].message.content


# # llm
# def generate_story(scenario):
#     template = """
#     You are a story teller;
#     You can generate a short horror story based on a single narrative,
#     the story should be no more than 2000 words;

#     CONTEXT: {scenario}
#     STORY:
#     """

#     prompt = PromptTemplate(template=template, input_variables=["scenario"])

#     llm = ChatOpenAI(
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
#     model_name="mattshumer/reflection-70b:free",
#     )

#     story_llm = LLMChain(llm=llm,
#         prompt=prompt, verbose=True
#         )
    
#     story = story_llm.predict(scenario=scenario)

#     return story


# # text to speech
# def text2speech(message, max_retries=5, retry_delay=10):
#     API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
#     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
#     payloads = {"inputs": message}

#     retries = 0
#     while retries < max_retries:
#         response = requests.post(API_URL, headers=headers, json=payloads)

#         if response.status_code == 200:
#             content_type = response.headers.get('Content-Type')
#             print(f"Content-Type: {content_type}")

#             if 'audio/flac' in content_type:
#                 with open('audio.flac', 'wb') as file:
#                     file.write(response.content)
#                 print("Audio file saved successfully.")
#                 break

#         elif response.status_code == 503:
#             # Model is loading, wait for the estimated time
#             response_data = response.json()
#             estimated_time = response_data.get("estimated_time", retry_delay)
#             print(f"Model is loading. Retrying in {estimated_time} seconds...")
#             time.sleep(estimated_time)
#             retries += 1

#         elif response.status_code == 500 and 'Model too busy' in response.text:
#             print(f"Model too busy. Retrying in {retry_delay} seconds... ({retries+1}/{max_retries})")
#             retries += 1
#             time.sleep(retry_delay)
        
#         else:
#             print(f"Error: {response.status_code} - {response.text}")
#             break
#     else:
#         print("Max retries reached. Unable to process the request.")


# # streamlit gui
# def main():

#     st.set_page_config(page_title="Orro Stories", page_icon="🎈")

#     st.header("Orro Stories")
#     st.write("Generate audio Horror story based on the image you upload.")
#     uploaded_file = st.file_uploader("Choose an image...", type=
#     ["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         # Upload the image to Imgur and get the URL
#         image_url = upload_to_imgur(uploaded_file)

#         if image_url:
#             print(f"Image URL: {image_url}")
#             st.image(image_url, caption="Uploaded Image", use_column_width=True)
            
#             # Generate text from the uploaded image
#             scenario = img2text(image_url)
#             story = generate_story(scenario)
#             text2speech(story)

#             with st.expander("scenario"):
#                 st.write("Image Description:", scenario)
#             with st.expander("story"):
#                 st.write(story)

#             st.audio("audio.flac")

# if __name__ == '__main__':
#     main()