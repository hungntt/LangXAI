import gradio as gr

from config import SERVER_PORT, SEGMENTATION_XAI, DETECTION_XAI, CLASSIFICATION_XAI
from dataloader.imagenet_loader import load_all_images_in_subfolders_in_a_folder
from text_based_exp import gpt_explain
from model_wrapper.classification import model_prediction, clf_run_xai
from model_wrapper.detection import det_run_xai
from model_wrapper.segmentation import seg_run_xai


# from xai_llm import get_text_span_label, generate_llm_explain, update_output_image


def main():
    with gr.Blocks(title="LangXAI", theme=gr.themes.Base()).queue() as demo:
        gr.Markdown("""# LangXAI: XAI Explanations in Human Language""")
        with gr.Row():
            # UI for Classification task
            with gr.Tab(label="Classification"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            clf_img = gr.Image(type="pil", label="Sample Image")
                        with gr.Row():
                            clf_gt = gr.Text(label="Ground Truth")
                        with gr.Row():
                            clf_model = gr.Dropdown(["SwinV2-Tiny", "SwinV2-Small", "SwinV2-Base"],
                                                    value="SwinV2-Tiny",
                                                    label="Choose the classification model")
                        with gr.Row():
                            run_clf_btn = gr.Button("Run classification")
                    with gr.Column():
                        clf_pred = gr.Label(label="Top-3 Predictions")
                        clf_xai_cat = gr.Dropdown(label="Choose the category to be explained", choices=[])
                        clf_xai = gr.Dropdown(CLASSIFICATION_XAI,
                                              label="Choose the XAI method")
                        clf_run_xai_btn = gr.Button("Run Explainable AI")

                    with gr.Column():
                        clf_xai_img = gr.Image(type="pil", label="Explanation Image")
                        clf_xai_llm_btn = gr.Button(label="Explain in human language")
                    with gr.Column():
                        clf_xai_llm_txt = gr.Text(label="Generated Description")

                with gr.Row():
                    with gr.Column():
                        gr.Examples(label="ImageNet dataset",
                                    examples=load_all_images_in_subfolders_in_a_folder(),
                                    inputs=[clf_img, clf_gt],
                                    examples_per_page=10)

            # UI for Segmentation task
            with gr.Tab(label="VQI Segmentation"):
                with gr.Row():
                    with gr.Column():
                        seg_model = gr.Dropdown(["ResNet50", "ResNet101"],
                                                value="ResNet50", label="Choose the segmentation model")
                        seg_xai = gr.Dropdown(SEGMENTATION_XAI,
                                              value="GradCAM",
                                              label="XAI Method")
                        seg_cat = gr.Dropdown(['cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'],
                                              value="cable",
                                              label="Category")
                        seg_img = gr.Image(type="pil", label="Sample Image")
                        seg_xai_btn = gr.Button("Run Explainable AI")
                    with gr.Column():
                        seg_pred = gr.Image(type="pil", label="Segmentation")
                        seg_xai_img = gr.Image(type="pil", label="Explanation")
                    with gr.Column():
                        seg_xai_llm_btn = gr.Button("Explain in human language")
                        seg_xai_llm_text = gr.Text(label="Generated Description")
                    with gr.Column():
                        gr.Examples(examples=[
                            ["images/ttpla/cable/1_00217.jpg", "cable"],
                            ["images/ttpla/cable/6_00015.jpg", "cable"],
                            ["images/ttpla/tower_wooden/3_00092.jpg", "tower_wooden"],
                        ], inputs=[seg_img, seg_cat])

            # UI for Detection task
            with gr.Tab(label="Detection"):
                with gr.Row():
                    with gr.Column():
                        det_model = gr.Dropdown(["FasterRCNN", "YOLOX"])
                        det_xai = gr.Dropdown(DETECTION_XAI, label="XAI Method")
                        det_img = gr.Image(type="pil", label="Sample Image")
                        det_xai_btn = gr.Button("Run Explainable AI")
                    with gr.Column():
                        det_pred = gr.Image(type="pil", label="Detection")
                        det_pred_xai = gr.Image(type="pil", label="Explanation")
                    with gr.Column():
                        det_xai_llm_btn = gr.Button("Explain in human language")
                        det_xai_llm_output = gr.Text(label="Generated Description")

        # Classification task functions
        run_clf_btn.click(fn=model_prediction,
                          inputs=[clf_model, clf_img],
                          outputs=[clf_pred, clf_xai_cat])

        # Run XAI model for classification
        clf_run_xai_btn.click(fn=clf_run_xai,
                              inputs=[clf_model, clf_img, clf_xai_cat, clf_xai],
                              outputs=[clf_xai_img])

        # Run LLM model for classification
        clf_xai_llm_btn.click(fn=gpt_explain,
                              inputs=[clf_xai_img, clf_img, clf_xai_cat, clf_gt],
                              outputs=[clf_xai_llm_txt],
                              queue=True)

        # Segmentation task functions
        seg_xai_btn.click(fn=seg_run_xai,
                          inputs=[seg_model, seg_xai, seg_cat, seg_img],
                          outputs=[seg_pred, seg_xai_img])

        # Run LLM for segmentation
        seg_xai_llm_btn.click(fn=gpt_explain,
                              inputs=[seg_img, seg_xai_img, seg_cat],
                              outputs=[seg_xai_llm_text],
                              queue=True)

        # Detection task functions
        det_xai_btn.click(fn=det_run_xai,
                          inputs=[det_model, det_xai, det_img],
                          outputs=[det_pred, det_pred_xai])

        # Run LLM for detection
        det_xai_llm_btn.click(fn=gpt_explain,
                              inputs=[det_img, det_pred_xai],
                              outputs=[det_xai_llm_output],
                              queue=True)

        demo.launch(share=False, server_port=SERVER_PORT)


if __name__ == "__main__":
    main()
