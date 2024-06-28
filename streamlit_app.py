import hmac
import streamlit as st


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
import streamlit as st
import json
import base64
import os
import requests
from together import Together

# Chatbot App Code
def save_history(messages, file_path="chat_history.json"):
    with open(file_path, "w") as f:
        json.dump(messages, f)

def load_history(file_path="chat_history.json"):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def reset_history(file_path="chat_history.json"):
    with open(file_path, "w") as f:
        json.dump([], f)
    st.session_state.messages = []

client = Together(api_key=st.secrets["api-key"])



def chatbot_app():
    st.title("Chat with AI")

    # if "chat_model" not in st.session_state:
    #     st.session_state["chat_model"] = "Qwen/Qwen2-72B-Instruct"

    if "messages" not in st.session_state:
        st.session_state.messages = load_history()  # Load chat history

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Input..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        max_tokens = 512

        with st.chat_message("assistant") as message_container:
            stream = client.chat.completions.create(
                model=st.session_state["chat_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens,
                stream=True,
            )

            response = ""
            words = ""
            message_placeholder = st.empty()  # Create a placeholder for the streaming message

            for chunk in stream:
                part = chunk.choices[0].delta.content or ""
                words += part
                message_placeholder.markdown(words.strip(), unsafe_allow_html=True)  # Update the placeholder with the response so far
                response += part

            st.session_state.messages.append({"role": "assistant", "content": response})

        save_history(st.session_state.messages)  # Save chat history

def generate_image(prompt, negative_prompt="", steps=25):
    # Use the together API to generate images
    response = client.images.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        steps=steps,
        n=1,  # Generate 1 image
    )

    # Extract the base64 encoded image data
    image_data = response.data[0].b64_json

    # Decode the base64 encoded image data
    image_bytes = base64.b64decode(image_data)

    return image_bytes



def image_gen_app():
    st.title("Image Generation")

    if "latest_image" not in st.session_state:
        st.session_state.latest_image = None

    prompt = st.text_input("Enter a prompt for image generation")
    negative_prompt = st.text_input("Negative prompt (optional)")

    if st.button("Generate Image"):
        try:
            image_bytes = generate_image(prompt, negative_prompt, steps)
            st.session_state.latest_image = image_bytes
            st.session_state.latest_steps = steps
            st.image(image_bytes, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to generate image: {str(e)}")

    if st.session_state.latest_image:
        st.image(st.session_state.latest_image, caption=f"Latest Generated Image | 1024x1024 | {st.session_state.latest_steps} steps", use_column_width=True)

# Create a dictionary that maps page names to page functions
pages = {
    "Chatbot": chatbot_app,
    "Image Generation": image_gen_app,
}

# Add a sidebar with the page selection
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

if selection == "Chatbot":
    max_tokens = st.sidebar.number_input("Max tokens for chat", min_value=1, max_value=32768, value=512, step=1)
    st.session_state["chat_model"] = st.sidebar.selectbox(
    "Select model",
    ["Qwen/Qwen2-72B-Instruct", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.3"],
    index=0
    )
    if st.sidebar.button("Reset chat history"):
        reset_history()
elif selection == "Image Generation":
    # Number of steps input for Image Generation page
    steps = st.sidebar.number_input("Number of steps for image generation", min_value=1, max_value=100, value=25, step=1)

# Call the function for the selected page
pages[selection]()
