import base64
import requests

from config import OPENAI_API_KEY
from utils import convert_pil_to_base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt_explain(task, original_image, explanation_image, category, ground_truth=None):
    # Getting the base64 string
    base64_image = convert_pil_to_base64(original_image)
    base64_image_xai = convert_pil_to_base64(explanation_image)
    prompt = (
            f"The first image is the original input image. The second image is the heatmap of category {category} in the original image." +
            f"First, you look at the original image to identify which parts belong to category {category} and which parts do not." +
            f"Then, you look at the second image to see the overall heatmap of category {category}." +
            f"Can you describe the concentrated regions on the image for the task {task} of category {category}?" +
            f"Your final answer should be correct, intuitive, compact, and simple for end-users to understand the meaning of heatmap." +
            f"Your final answer should describe the regions where the model is focusing on to predict the {category}." +
            f"Your final answer should be separated by bullet points." +
            f"First, shortly describe the heatmap in one sentence." +
            f"Secondly, describe the most concentrated region of {category} in the heatmap." +
            f"Thirdly, describe the least concentrated region of {category} in the heatmap." +
            f"Fourthly, assess the localization quality if the concentrated region aligns with the {category} in the original image."
    )
    if ground_truth is not None:
        prompt += (
                f"The fourth bullet point should show is the model's category {category} is correct with the ground-truth {ground_truth} or not? Must be accurate. For example, category 'purse' is different from ground truth 'shopping bag'" +
                f"Is the most concentrated region of predict {category} in the heatmap supportive for model to correct predict the ground truth {ground_truth}?"
        )

    print(prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_xai}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 10000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    final_response = response.json()['choices'][0]['message']['content']
    return final_response

def seg_gpt_explain(original_image, explanation_image, segmentation_image, ground_truth, category):
    # Getting the base64 string
    base64_image = convert_pil_to_base64(original_image)
    base64_image_xai = convert_pil_to_base64(explanation_image)
    base64_image_seg = convert_pil_to_base64(segmentation_image)
    base64_image_gt = convert_pil_to_base64(ground_truth)

    prompt = (
            f"You are a Explainable AI expert for semantic segmentation models." +
            f"The first image is the original image. " +
            f"The second image is the explanation map of category {category} in the original image." +
            f"The third image is the prediction of an AI model for category {category} in the original image." +
            f"The fourth image is the ground truth of category {category} in the original image."
            f"First, capture the image context in the original image." +
            f"Secondly, identify which parts belong to category {category} in the ground truth." +
            f"Then, you look at the explanation map to see the saliency map for the segmentation mask of category {category}." +
            f"Your task is to check the focused region in the explanation map supports the prediction for category {category}" +
            f"Think step by step to understand explanation map, prediction aligned with the ground truth." +
            f"Your final answer must be concise, simple and separated by bullet points." +
            f"First, shortly describe the explanation map." +
            f"Secondly, describe the most focused region of {category} in the explanation." +
            f"Thirdly, describe the least focused region of {category} in the explanation." +
            f"Fourthly, assess the localization quality if the focused region aligns with the prediction and ground truth for {category}."
    )

    print(prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_xai}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_seg}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_gt}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 10000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    final_response = response.json()['choices'][0]['message']['content']
    return final_response
