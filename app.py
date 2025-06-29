print("Starting Diyaan Bot...")
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from duckduckgo_search import DDGS
import torch
import urllib.parse

# Load model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_history_ids = None

# Safe content filter
def is_safe(text):
    banned = ["sex", "nudity"]
    return not any(word in text.lower() for word in banned)

# Search using DuckDuckGo
def web_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        for r in results:
            if "wikipedia.org" not in r["href"].lower():
                return f"{r['title']}\n{r['body']}\n{r['href']}"
    return "No safe results found."

# Generate image
def image_url(prompt):
    if not is_safe(prompt):
        return None
    encoded = urllib.parse.quote_plus(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}"

# Chatbot handler
def handle_input(user_input, history):
    global chat_history_ids
    user_lower = user_input.lower()

    # 🔍 Auto web search
    search_keywords = ["who is", "what is", "where is", "how to", "tell me about", "when did", "define"]
    if any(word in user_lower for word in search_keywords):
        result = web_search(user_input)
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": result}]

    # 🖼️ Auto image generation
    image_keywords = ["show me a picture of", "generate image of", "draw", "create an image of", "i want to see"]
    if any(word in user_lower for word in image_keywords):
        prompt = user_input
        for key in image_keywords:
            prompt = prompt.replace(key, "")
        prompt = prompt.strip()
        url = image_url(prompt)
        msg = f"![Image]({url})" if url else "⚠️ Blocked image prompt (unsafe)."
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": msg}]

    # ⚠️ Blocked input
    if not is_safe(user_input):
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": "⚠️ This message is blocked due to safety settings."}]

    # 🤖 Normal chatbot
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply.strip()}]

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Diyaan Bot — Free Chat + Search + Image")
    chatbot = gr.Chatbot(label="Diyaan Bot", type="messages")
    msg = gr.Textbox(label="Ask me anything...")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")
    state = gr.State([])

    send_btn.click(fn=handle_input, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(fn=handle_input, inputs=[msg, state], outputs=[chatbot, state])
    clear_btn.click(lambda: [], None, chatbot)

demo.queue().launch(server_name="0.0.0.0", server_port=8080)
