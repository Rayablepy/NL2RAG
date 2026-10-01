import os
import shutil

import gradio as gr

from config import ACTUAL_FILE_PATH
from loader import delete_data, list_data, save_data
from main import getresponse

DIR = ACTUAL_FILE_PATH
os.makedirs(DIR, exist_ok=True)

ALLOWED_TYPES = [".pdf", ".csv", ".txt", ".docx", ".xlsx"]


def store_uploads(paths: list[str] | None) -> tuple[gr.update, str]:
    if not paths:
        return gr.Dropdown(choices=list_data()), "No files selected."
    saved = []
    for path in paths:
        name = os.path.basename(path)
        shutil.copyfile(path, os.path.join(DIR, name))
        save_data(name)
        saved.append(name)
    return gr.Dropdown(choices=list_data()), f"Saved {len(saved)} file(s) to {DIR}"


def remove_selected(names: list[str] | None) -> tuple[gr.Dropdown, str]:
    if not names:
        return gr.Dropdown(choices=list_data()), "No files selected."
    for name in names:
        delete_data(name)
    return gr.Dropdown(choices=list_data()), f"Deleted {len(names)} file(s)"


async def respond(message: str, history: list[dict] | None) -> tuple[str, list[dict]]:
    history = history or []
    if not message.strip():
        return "", history
    reply = await getresponse(message)
    return "", [*history, {"role": "user", "content": message}, {"role": "assistant", "content": reply}]


with gr.Blocks(title="NL2SQL") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Upload Data")
            upload = gr.File(
                file_count="multiple",
                file_types=ALLOWED_TYPES,
                type="filepath",
                label="Upload files",
            )
            upload_status = gr.Textbox(label="Upload status", interactive=False)
            gr.Markdown("### Saved Files")
            saved_files = gr.Dropdown(choices=[], multiselect=True, label="Select files to delete")
            delete_status = gr.Textbox(label="Delete status", interactive=False)
            gr.Button("Delete Selected", variant="stop").click(
                remove_selected, inputs=saved_files, outputs=[saved_files, delete_status]
            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat",
                height="100%",
                placeholder="Your input here...",
                layout="bubble",
            )
            prompt = gr.Textbox(label="Your input here...", show_label=False)
            gr.Button("Send").click(
                respond, inputs=[prompt, chatbot], outputs=[prompt, chatbot]
            )
            prompt.submit(respond, inputs=[prompt, chatbot], outputs=[prompt, chatbot])

    demo.load(lambda: gr.Dropdown(choices=list_data()), outputs=saved_files)
    upload.change(
        store_uploads, inputs=upload, outputs=[saved_files, upload_status]
    )


if __name__ == "__main__":
    demo.launch()