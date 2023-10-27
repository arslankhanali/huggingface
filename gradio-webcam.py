import gradio as gr

def webcam(video0,video1):
    return video0,video1

interface = gr.Interface(
    fn=webcam,
    inputs=[gr.Webcam(mirror_webcam=False,streaming=True)],
    outputs=["image","image"],
    )

if __name__ == "__main__":
    interface.launch()