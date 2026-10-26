import os
import tempfile
from functools import lru_cache

import gradio as gr
from huggingface_hub import hf_hub_download

from main import analyze_video


@lru_cache(maxsize=1)
def get_model_path():
    return hf_hub_download(
        repo_id=os.getenv("MODEL_REPO_ID"),
        filename="best.pt",
    )

def process_video(video_path):
    if not video_path:
        raise gr.Error("Please upload a video.")

    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ).name

    analyze_video(
        input_path=video_path,
        output_path=output_path,
        model_path=get_model_path(),
        use_stubs=False,
    )

    return output_path


demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        sources=["upload"],
        type="filepath",
        label="Upload football video",
    ),
    outputs=gr.Video(label="Analyzed video"),
    title="Football Video Analysis",
    description=(
        "YOLO player, referee, and ball tracking with team assignment, "
        "ball possession, camera-motion compensation, speed, and distance estimation."
    ),
)

if __name__ == "__main__":
    demo.launch()