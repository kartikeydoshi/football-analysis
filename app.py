import os
import tempfile
from functools import lru_cache

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

from main import analyze_video


SAMPLE_VIDEO_PATH = os.path.join(
    os.path.dirname(__file__),
    "input_videos",
    "08fd33_4.mp4",
)


@lru_cache(maxsize=1)
def get_model_path():
    return hf_hub_download(
        repo_id=os.getenv("MODEL_REPO_ID"),
        filename="best.pt",
    )

@spaces.GPU
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


with gr.Blocks(title="Football Video Analysis") as demo:
    gr.Markdown(
        "# Football Video Analysis\n"
        "YOLO player, referee, and ball tracking with team assignment, "
        "ball possession, camera-motion compensation, speed, and distance estimation."
    )

    video_input = gr.Video(
        sources=["upload"],
        label="Upload football video",
    )
    with gr.Row():
        add_sample_button = gr.Button("Add sample")
        analyze_button = gr.Button("Analyze video", variant="primary")
    analyzed_video = gr.Video(label="Analyzed video")

    add_sample_button.click(
        fn=lambda: SAMPLE_VIDEO_PATH,
        outputs=video_input,
    )
    analyze_button.click(
        fn=process_video,
        inputs=video_input,
        outputs=analyzed_video,
    )

if __name__ == "__main__":
    demo.launch()