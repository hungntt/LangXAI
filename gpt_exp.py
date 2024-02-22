import base64
import requests

from config import OPENAI_API_KEY
from utils import convert_pil_to_base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt_explain(task, original_image, explanation_image, prediction, ground_truth=None):
    # Getting the base64 string
    base64_image = convert_pil_to_base64(original_image)
    base64_image_xai = convert_pil_to_base64(explanation_image)
    prompt = (
            f"The first image is the original input image. The second image is the heatmap of prediction {prediction} in the original image." +
            f"First, you look at the original image to identify which parts belong to prediction {prediction} and which parts do not." +
            f"Then, you look at the second image to see the overall heatmap of prediction {prediction}." +
            f"Can you describe the concentrated regions on the image for the task {task} of prediction {prediction}?" +
            f"Your final answer should be correct, intuitive, compact, and simple for end-users to understand the meaning of heatmap." +
            f"Your final answer should describe the regions where the model is focusing on to predict the {prediction}." +
            f"Your final answer should be separated by bullet points." +
            f"The first bullet point should give a brief overview of the heatmap." +
            f"The second bullet point should show the most concentrated region of {prediction} in the heatmap." +
            f"The third bullet point should show the least concentrated region of {prediction} in the heatmap."
    )
    if ground_truth is not None:
        prompt += (
                f"The fourth bullet point should show is the model's prediction {prediction} is correct with the ground-truth {ground_truth} or not? Must be accurate. For example, prediction 'purse' is different from ground truth 'shopping bag'" +
                f"Is the most concentrated region of predict {prediction} in the heatmap supportive for model to correct predict the ground truth {ground_truth}?"
        )

    print(prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
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
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    final_response = response.json()['choices'][0]['message']['content']
    return final_response
