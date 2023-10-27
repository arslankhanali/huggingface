from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import gradio as gr

def predict(image):
    print(image)
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    print(results)
    # Initialize an empty list to store dictionaries
    dict_results = {}

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # Create dictionaries in a loop
        dict_results[model.config.id2label[label.item()]] = round(score.item(), 3)
        print(dict_results)
    return dict_results

    # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    # box = [round(i, 2) for i in box.tolist()]
    # print(
    #         f"Detected {model.config.id2label[label.item()]} with confidence "
    #         f"{round(score.item(), 3)} at location {box}"
    # )

demo = gr.Interface(predict, gr.inputs.Image(type="pil"), "label")
# demo = gr.Interface(predict, gr.Image(source="webcam", streaming=True,type="pil"), "label",live=True)
# demo = gr.Interface(predict, gr.Image(source="webcam", streaming=True,type="pil"), "label",live=False)
demo.launch(server_name="192.168.1.106",server_port=7680)