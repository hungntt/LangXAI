import gradio as gr
from gradio.themes.utils.sizes import text_md, text_lg

from config import SERVER_PORT, SEGMENTATION_XAI, DETECTION_XAI, CLASSIFICATION_XAI, DESCRIPTION_TEMPLATE, \
    CLASSIFICATION_MODEL, SEGMENTATION_MODEL, DESCRIPTION_TEMPLATE_CLF
from dataloader.coco_loader import load_coco_images, load_coco_annotations
from dataloader.ttpla_loader import ttpla_images_loader
from dataloader.imagenet_loader import imagenetv2_images_loader
from gpt_exp import gpt_explain, seg_gpt_explain
from model_wrapper.classification import clf_pred, clf_run_xai
from model_wrapper.detection import det_run_xai, det_pred
from model_wrapper.segmentation import seg_run_xai, seg_run_pred, seg_get_label
from utils import extract_index_from_object_detection_output

# from xai_llm import get_text_span_label, generate_llm_explain, update_output_image

css = """
h1 {
    text-align: center;
    display:block;
}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


def main():
    with gr.Blocks(title="LangXAI", js=js_func, css=css, theme='shivi/calm_seafoam').queue() as demo:
        # Add list of logo
        gr.HTML("""
        <div style="display: ruby-text; justify-content: center; align-items: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/FPT_Software_logo.svg/1200px-FPT_Software_logo.svg.png" style="height: 80px;">
            <img src="https://quynhon.ai/quynhonAI.png" style="height: 70px;">
            <img src="https://i.postimg.cc/nLPSVZ6T/AELab-Logo-Aligned-Transparent.png", style="height: 80px">
            <img src="https://schulichleaders.com/wp-content/uploads/New-Brunswick-1.png" style="height: 80px;">
            <img src="https://www.tf.fau.de/files/2022/10/FAU-Logo.png" style="height: 80px;">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRPGn3oNoRJxZ07LXw9k9jNo7sjPk-Ft7s3w&s" style="height: 55px;">
        </div>
        </br>
        """)
        gr.Markdown("""# LangXAI: Integrating Large Vision Models for Generating Textual Explanations to
        # Enhance Explainability in Visual Perception Tasks
        """)
        with gr.Row():
            # UI for Classification task
            with gr.Tab(label="Classification"):
                clf_tsk = gr.Textbox(label="Classification", visible=False)
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            clf_img = gr.Image(type="pil", label="Input image", show_label=False)
                        with gr.Row():
                            clf_gt = gr.Text(label="Ground Truth", show_label=False)
                            clf_model = gr.Dropdown(CLASSIFICATION_MODEL,
                                                    value="SwinV2-Tiny",
                                                    label="Choose model", show_label=False)
                        with gr.Row():
                            clf_pred_btn = gr.Button("Run classification")
                    with gr.Column():
                        clf_preds = gr.Label(label="Top-3 Predictions", show_label=False)
                        with gr.Row():
                            clf_xai_cat = gr.Dropdown(label="Category", choices=[])
                            clf_xai = gr.Dropdown(CLASSIFICATION_XAI,
                                                  label="XAI Technique")
                        with gr.Row():
                            clf_run_xai_btn = gr.Button("Run Explainable AI")
                    with gr.Column():
                        clf_xai_img = gr.Image(type="pil", label="Explanation Image", show_label=False)
                        clf_xai_llm_btn = gr.Button("Textual Explanation", variant="primary")
                    with gr.Column():
                        clf_xai_llm_txt = gr.Text(label="Textual Explanation", show_label=False)
                        # clf_xai_desc = gr.Textbox(label="Description Format",
                        #                           value=DESCRIPTION_TEMPLATE_CLF,
                        #                           show_label=False)
                with gr.Row():
                    with gr.Column():
                        gr.Examples(label="ImageNet dataset",
                                    examples=imagenetv2_images_loader(),
                                    inputs=[clf_img, clf_gt],
                                    examples_per_page=10)

            # UI for Segmentation task
            with gr.Tab(label="Semantic Segmentation"):
                seg_tsk = gr.Text(label="semantic segmentation", visible=False)
                with gr.Row():
                    with gr.Column():
                        seg_img = gr.Image(type="pil", label="Input image")
                        seg_model = gr.Dropdown(SEGMENTATION_MODEL,
                                                value="ResNet50", label="Choose the segmentation model")
                        seg_cat = gr.Dropdown(['cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'],
                                              value="cable",
                                              label="Category")
                    with gr.Column():
                        seg_gt_img = gr.Image(type="pil", label="Ground Truth (Optional)")
                        seg_pred_btn = gr.Button("Run segmentation")
                        seg_img_id = gr.Textbox(label="Image ID", visible=False)
                        seg_current_model = gr.State()
                    with gr.Column():
                        seg_pred_img = gr.Image(type="pil", label="Segmentation")
                        seg_xai = gr.Dropdown(SEGMENTATION_XAI,
                                              value="GradCAM",
                                              label="XAI Method")
                        seg_xai_btn = gr.Button("Run Explainable AI")
                    with gr.Column():
                        seg_xai_img = gr.Image(type="pil", label="Explanation Image")
                        seg_xai_llm_btn = gr.Button("Textual Explanation", variant="primary")
                # with gr.Row():
                #     seg_xai_desc = gr.Textbox(label="Textual Explanation",
                #                               value=DESCRIPTION_TEMPLATE,
                #                               lines=6)
                with gr.Row():
                    seg_xai_llm_text = gr.Text(label="Textual Explanation")

                with gr.Row():
                    with gr.Column():
                        gr.Examples(label="TTPLA dataset",
                                    examples=ttpla_images_loader(),
                                    inputs=[seg_img, seg_cat, seg_img_id],
                                    examples_per_page=20)

            # UI for Detection task
            with gr.Tab(label="Object Detection"):
                det_tsk = gr.Textbox("object detection", visible=False)
                with gr.Row():
                    with gr.Column():
                        det_img = gr.Image(type="pil", label="Input image")
                        det_img_id = gr.Textbox(label="Image ID", visible=False)
                        det_model = gr.Dropdown(["FasterRCNN"],
                                                value="FasterRCNN",
                                                label="Detection Model")
                        det_gt = gr.Image(type="pil", label="Ground Truth (Optional)")
                        det_pred_btn = gr.Button("Run Detection")
                    with gr.Column():
                        det_pred_img = gr.AnnotatedImage(label="Detection")
                        det_xai = gr.Dropdown(DETECTION_XAI, label="XAI Method", value="GCAME")
                        det_xai_idx = gr.Textbox(label="Object Index")
                        det_xai_btn = gr.Button("Run Explainable AI")
                    with gr.Column():
                        det_pred_xai_img = gr.Image(type="pil", label="Explanation Image")
                        det_xai_llm_btn = gr.Button("Textual Explanation", variant="primary")
                    with gr.Column():
                        det_xai_llm_output = gr.Text(label="Textual Explanation")
                        # det_xai_desc = gr.Textbox(label="Description Format",
                        #                           value=DESCRIPTION_TEMPLATE,
                        #                           lines=10)
                        det_xai_cls = gr.Textbox(label="Predicted Classes", visible=False)
                with gr.Row():
                    gr.Examples(label="COCO dataset",
                                examples=load_coco_images(),
                                inputs=[det_img, det_img_id],
                                examples_per_page=20)

        # ============== Classification task functions
        # Run prediction for classification
        clf_pred_btn.click(fn=clf_pred,
                           inputs=[clf_model, clf_img],
                           outputs=[clf_preds, clf_xai_cat])

        # Run XAI model for classification
        clf_run_xai_btn.click(fn=clf_run_xai,
                              inputs=[clf_model, clf_img, clf_xai_cat, clf_xai],
                              outputs=[clf_xai_img])

        # Run LLM model for classification
        clf_xai_llm_btn.click(fn=gpt_explain,
                              inputs=[clf_tsk, clf_xai_img, clf_img, clf_xai_cat, clf_gt],
                              outputs=[clf_xai_llm_txt],
                              queue=True)

        # ============== Segmentation task functions
        seg_img.change(fn=seg_get_label,
                       inputs=[seg_model, seg_img, seg_img_id, seg_cat],
                       outputs=[seg_gt_img, seg_current_model])
        # Run prediction for segmentation
        seg_pred_btn.click(fn=seg_run_pred,
                           inputs=[seg_current_model, seg_cat],
                           outputs=[seg_pred_img])

        # Run XAI model for segmentation
        seg_xai_btn.click(fn=seg_run_xai,
                          inputs=[seg_current_model, seg_xai],
                          outputs=[seg_xai_img])

        # Run LLM for segmentation
        seg_xai_llm_btn.click(fn=seg_gpt_explain,
                              inputs=[seg_img, seg_xai_img, seg_pred_img, seg_gt_img, seg_cat],
                              outputs=[seg_xai_llm_text],
                              queue=True)

        # ============== Detection task functions
        det_img.change(fn=load_coco_annotations,
                       inputs=[det_img_id],
                       outputs=[det_gt])

        # Run prediction for detection
        det_pred_btn.click(fn=det_pred,
                           inputs=[det_model, det_img],
                           outputs=[det_pred_img])

        # Select the object to be explained
        det_pred_img.select(fn=extract_index_from_object_detection_output,
                            inputs=None,
                            outputs=[det_xai_idx])

        # Run XAI model for detection
        det_xai_btn.click(fn=det_run_xai,
                          inputs=[det_model, det_xai, det_img, det_xai_idx],
                          outputs=[det_pred_xai_img, det_xai_cls])

        # Run LLM for detection
        det_xai_llm_btn.click(fn=gpt_explain,
                              inputs=[det_tsk, det_img, det_pred_xai_img, det_xai_cls],
                              outputs=[det_xai_llm_output],
                              queue=True)

        demo.launch(share=False, server_port=SERVER_PORT)


if __name__ == "__main__":
    main()
