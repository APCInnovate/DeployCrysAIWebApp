from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import plotly.figure_factory as ff
import io
import tempfile  
import base64    
import mimetypes
import requests
import os
import subprocess

def install_detectron2():
    try:
        import detectron2
        print("Detectron2 is already installed.")
    except ImportError:
        print("Installing Detectron2...")
        subprocess.run([
            "pip", "install",
            "git+https://github.com/facebookresearch/detectron2.git"
        ], check=True)
        print("Detectron2 installed successfully.")

install_detectron2()


# Run post_install.sh on first startup (only if not already executed)
if not os.path.exists("/tmp/post_install_done"):
    subprocess.run(["bash", "post_install.sh"], check=True)
    open("/tmp/post_install_done", "w").close()


url = "hhttps://github.com/AKCNN-Repo/DeployShinyApp/releases/tag/V0.0.1/model_final.pth"
response = requests.get(url)
with open("output/model_final.pth", "wb") as f:
    f.write(response.content)


# Shiny synchronous mode
os.environ["SHINY_SYNC_MODE"] = "1"

# Define UI layout 
app_ui = ui.page_fluid(
    ui.navset_bar(
        ui.nav_panel(
            "Multiclass Prediction & Measurement",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_file("file_upload", "Upload Images", multiple=True, accept=[".jpg", ".png", ".jpeg"]),
                    ui.input_checkbox("show_plots", "Show plots?", value=True),
                    ui.input_checkbox("show_metrics", "Show metrics?", value=True),
                    ui.input_checkbox("show_histogram", "Show histogram?", value=True),
                    ui.input_checkbox("show_kde", "Show KDE curve?", value=True),
                    ui.input_checkbox("show_rug", "Show rug?", value=True),
                    ui.input_numeric("pixels_per_metric", "Pixels per metric:", value=0.85),
                    ui.input_numeric("threshold", "Prediction threshold:", value=0.50),
                    ui.input_action_button("start_analysis", "Start Analysis")
                ),
                # Main content
                ui.output_text_verbatim("results"),
                ui.navset_tab(
                    ui.nav_panel(
                        "Processed Images",
                        ui.output_ui("processed_images_ui")  # Renamed to avoid conflict
                    ),
                    ui.nav_panel(
                        "Metrics",
                        ui.output_table("metrics_table")  # Metrics table
                    ),
                    ui.nav_panel(
                        "Plots",
                        output_widget("aspect_ratio_plot"),
                        output_widget("feret_diameter_plot"),
                        output_widget("roundness_plot"),
                        output_widget("circularity_plot"),
                        output_widget("sphericity_plot")
                    )
                )
            )
        ),
        title="APCNN: A DL-based Offline Image Analysis Toolkit (Beta)"
    )
)

# Define Server Logic
def server(input: Inputs, output: Outputs, session: Session):
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    # Register the datasets
    Data_Register_training = "manittol_s_train"
    Data_Register_valid = "manittol_s_test"
    register_coco_instances(
        "manittol_s_train",
        {},'./DATASETS/Train/COCO_Train_up.json','./DATASETS/Train/'
    )
    register_coco_instances(
        "manittol_s_test",
        {},'./DATASETS/Test/COCO_Test_up.json', './DATASETS/Test/'
    )
    metadata = MetadataCatalog.get(Data_Register_training)
    metadata_valid = MetadataCatalog.get(Data_Register_valid)
    dataset_train = DatasetCatalog.get(Data_Register_training)
    dataset_valid = DatasetCatalog.get(Data_Register_valid)

    @reactive.Calc
    def setup_predictor():
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("manittol_s_train",)
        cfg.DATASETS.TEST = ("manittol_s_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 3000
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR,'model_final.pth'
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = input.threshold()
        cfg.MODEL.DEVICE = "cuda"
        MetadataCatalog.get("manittol_s_test").set(things_classes=["Particle", "Bubble", "Droplet"])
        MetadataCatalog.get("manittol_s_test").set(things_colors=[(0, 0, 225), (0, 255, 0), (255, 0, 0)])
        manittol_s_test_metadata = MetadataCatalog.get("manittol_s_test")
        return DefaultPredictor(cfg)

    def compute_metrics(contours, pixels_per_metric):
        metrics_data = {
            "Aspect Ratio": [],
            "Feret Diameter": [],
            "Roundness": [],
            "Circularity": [],
            "Sphericity": [],
            "Length": [],
            "Width": [],
            "CircularED": []
        }
        for c in contours:
            if cv2.contourArea(c) < 1:
                continue
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if cv2.__version__.startswith("3") else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            dimA, dimB = dA / pixels_per_metric, dB / pixels_per_metric
            diameter = max(dimA, dimB)
            aspect_ratio = dimB / dimA if dimA != 0 else 0
            roundness = 1 / aspect_ratio if aspect_ratio != 0 else 0
            sphericity = (2 * np.sqrt(np.pi * area)) / perimeter if perimeter != 0 else 0
            circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
            metrics_data["Aspect Ratio"].append(aspect_ratio)
            metrics_data["Feret Diameter"].append(diameter)
            metrics_data["Roundness"].append(roundness)
            metrics_data["Circularity"].append(circularity)
            metrics_data["Sphericity"].append(sphericity)
            metrics_data["Length"].append(max(dimA, dimB))
            metrics_data["Width"].append(min(dimA, dimB))
            metrics_data["CircularED"].append(np.sqrt(4 * area / np.pi))
        return pd.DataFrame(metrics_data)

    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # Store list of Processed Images
    processed_images_list = []  # Renamed variable

    @output
    @render.text
    def results():
        if input.start_analysis() == 0:
            return "Press 'Start Analysis' to begin."
        elif not input.file_upload():
            return "No images uploaded."
        else:
            metrics_df_list = []
            predictor = setup_predictor()
            total_particle_count = total_bubble_count = total_droplet_count = 0
            # Clear previous processed images
            processed_images_list.clear()
            for img_info in input.file_upload():
                img_path = img_info["datapath"]
                im = cv2.imread(img_path)
                outputs = predictor(im)
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=MetadataCatalog.get("manittol_s_test"),
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                processed_img = Image.fromarray(out.get_image()[:, :, ::-1])
                # Save Processed Images to Temp file
                temp_dir = tempfile.gettempdir()
                processed_image_filename = os.path.join(temp_dir, f"output_{img_info['name']}")
                processed_img.save(processed_image_filename)
                # Add Processed Img_Path to the list 
                processed_images_list.append(processed_image_filename)
                mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
                output = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)

                # Indexing of Mask_Array
                for i in range(mask_array.shape[0]):
                    mask = mask_array[i].astype(np.uint8) * 255  # Corrected indexing

                    # Resize mask
                    if mask.shape != output.shape:
                        mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Merge masks
                    output = np.maximum(output, mask)

                # Apply a Binary Tthreshold (0 or 255)
                _, thresholded_output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)
                # Save Thresholded output to Temp file
                thresholded_output_filename = os.path.join(temp_dir, f"thresholded_output_{img_info['name']}.jpg")
                cv2.imwrite(thresholded_output_filename, thresholded_output)  # Save for debugging

                gray = thresholded_output if thresholded_output.ndim == 2 else cv2.cvtColor(thresholded_output, cv2.COLOR_BGR2GRAY)
                cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                print(f"Number of contours found: {len(cnts)}")  # Debug statement to check number of contours
                if len(cnts) == 0:
                    continue
                (cnts, _) = contours.sort_contours(cnts)
                df_metrics = compute_metrics(cnts, input.pixels_per_metric())
                if not df_metrics.empty:
                    metrics_df_list.append(df_metrics)
                classes = outputs["instances"].pred_classes.to("cpu").numpy()
                total_particle_count += sum(classes == 0)
                total_bubble_count += sum(classes == 1)
                total_droplet_count += sum(classes == 2)
            if metrics_df_list:
                all_metrics_df = pd.concat(metrics_df_list, ignore_index=True)
                all_metrics_df.to_csv(
                    'Results.csv',
                    index=False
                )
                return (
                    f"Analysis Complete\n"
                    f"Total particles detected: {total_particle_count}\n"
                    f"Total bubbles detected: {total_bubble_count}\n"
                    f"Total droplets detected: {total_droplet_count}"
                )
            else:
                return "No valid contours found in the uploaded images."

    def plot_distribution(data, feature_name, show_hist, show_kde, show_rug):
        fig = ff.create_distplot(
            [data[feature_name]],
            [feature_name],
            show_hist=show_hist,
            show_curve=show_kde,
            show_rug=show_rug
        )
        return fig

    def check_and_load_results():
        results_path = 'Results.csv'
        if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
            return pd.read_csv(results_path)
        return None

    @output
    @render.ui
    def processed_images_ui():  # Renamed function
        if input.start_analysis() == 0:
            return ui.div("No images processed yet.")
        elif not input.file_upload():
            return ui.div("No images uploaded.")
        else:
            if not processed_images_list:
                return ui.div("No images processed.")
            # UI elements for Processed Images
            image_elements = []
            for processed_image_path in processed_images_list:
                if os.path.exists(processed_image_path):
                    # Read Image & Encode as base64
                    with open(processed_image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    # Determine MIME type based on File Extension
                    mime_type, _ = mimetypes.guess_type(processed_image_path)
                    if mime_type is None:
                        mime_type = 'image/jpeg'  # default
                    data_uri = f"data:{mime_type};base64,{encoded_string}"
                    image_elements.append(
                        ui.div(
                            ui.img(
                                src=data_uri,
                                style="width:100%; height:auto; margin-bottom: 20px;",
                                alt="Processed Image"
                            ),
                            style="margin-bottom: 20px;"
                        )
                    )
            return ui.div(*image_elements)

    @output
    @render.table
    def metrics_table():
        if input.start_analysis() == 0 or not input.show_metrics():
            return None
        else:
            df = check_and_load_results()
            if df is not None:
                return df
            else:
                return None

    @output
    @render_widget
    def aspect_ratio_plot():
        df = check_and_load_results()
        if df is not None and input.show_plots():
            return plot_distribution(df, 'Aspect Ratio', input.show_histogram(), input.show_kde(), input.show_rug())

    @output
    @render_widget
    def feret_diameter_plot():
        df = check_and_load_results()
        if df is not None and input.show_plots():
            return plot_distribution(df, 'Feret Diameter', input.show_histogram(), input.show_kde(), input.show_rug())

    @output
    @render_widget
    def roundness_plot():
        df = check_and_load_results()
        if df is not None and input.show_plots():
            return plot_distribution(df, 'Roundness', input.show_histogram(), input.show_kde(), input.show_rug())

    @output
    @render_widget
    def circularity_plot():
        df = check_and_load_results()
        if df is not None and input.show_plots():
            return plot_distribution(df, 'Circularity', input.show_histogram(), input.show_kde(), input.show_rug())

    @output
    @render_widget
    def sphericity_plot():
        df = check_and_load_results()
        if df is not None and input.show_plots():
            return plot_distribution(df, 'Sphericity', input.show_histogram(), input.show_kde(), input.show_rug())

# Instantiate Shiny App
app = App(app_ui, server)
