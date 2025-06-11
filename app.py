##### Baseline Imports START (Do Not Edit) #####
from pathlib import Path
from datetime import datetime
from shiny import App, Inputs, Outputs, Session, ui, render, reactive, req
from shinywidgets import output_widget, render_widget, render_plotly
from shiny.types import ImgData, FileInfo
import os
from faicons import icon_svg
##### Baseline Imports END #####
#################### Additional Libraries #################
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import plotly.figure_factory as ff
import io
import tempfile  
import base64    
import mimetypes
import plotly.express as px
import plotly.graph_objects as go
import shutil
import asyncio
import random
import json
import requests
import os
import subprocess

url = 'https://github.com/APCInnovate/DeployCrysAIWebApp/releases/download/v0.0.1/model_final.pth'
#url = 'https://github.com/AKCNN-Repo/DeployShinyApp/releases/download/V0.0.1/model_final.pth'
response = requests.get(url)

response = requests.get(url)
with open("output/model_final.pth", "wb") as f:
    f.write(response.content)


# Shiny synchronous mode
os.environ["SHINY_SYNC_MODE"] = "1"

# Icons for percentile boxes
ICONS = {
    "d10": icon_svg("percent"),
    "d50": icon_svg("chart-line"),
    "d90": icon_svg("percent"),
}

##### Resource and Version Info START #####
resource_dir = Path(__file__).parent / "www" # Do Not Edit
Web_app_version = "Version: 0.0.0" # Edit as Required
##### Resource and Version Info END #####

home_tab_content = ui.div(
    ui.card(ui.card_header(ui.h2("CrysAI ™: A DLIA for Offline Image Analysis")),
        ui.card(ui.card_header(ui.h3("Data Upload & Input Specification", style="font-size:1.8rem; font-weight:bold; color:#12375b;")),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header(
                            ui.h3("1. Upload & Settings", style="font-size:1.6rem; font-weight:bold; color:#12375b;")
                        ),
                        ui.card_body(
                            ui.input_file(
                                "file_upload",
                                "Upload Images",
                                multiple=True,
                                accept=[".jpg", ".png", ".jpeg"]
                            ),
                            ui.input_numeric(
                                "pixels_per_metric",
                                "Pixels per metric:",
                                value=1.52
                            ),
                            ui.input_numeric(
                                "threshold",
                                "Prediction threshold:",
                                value=0.50
                            ),
                            ui.input_action_button(
                                "start_analysis",
                                "Start Analysis",
                                style="font-size:1.2rem; padding:0.5rem 1rem;"
                            )
                        )
                    ),
                    ui.card(
                        ui.card_header(
                            ui.h3("2. Display & Plot Options", style="font-size:1.6rem; font-weight:bold; color:#12375b;")
                        ),
                        ui.card_body(
                         
                            ui.input_checkbox(
                                "show_plots",
                                "Show Plots ?",
                                value=True
                            ),
                            ui.input_checkbox(
                                "show_metrics",
                                "Show Metrics ?",
                                value=True
                            ),

                            ui.h5("Percentile Metrics", style="font-size:1.3rem; font-weight:600;"),
                            ui.input_select(
                                "selected_metric",
                                "",
                                choices=[
                                    "Feret Diameter", "Length", "Width", "CircularED", "CircularEP",
                                    "SphericalED", "SphericalEP", "Aspect Ratio", "Circularity",
                                    "Sphericity", "Roundness", "Roughness", "Solidity", "Compactness"
                                ],
                                selected="Feret Diameter",
                                width="100%"
                            )
                        )
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header(ui.h3("3. Shape & Size-Based Metrics", style="font-size:1.6rem; font-weight:bold; color:#12375b;")),
                        ui.card_body(
                            ui.h5("Metrics for Data Table", style="font-size:1.3rem; font-weight:600;"),
                            ui.input_checkbox_group(
                                "selected_metrics_to_show", 
                                "",
                                choices=[
                                    "Feret Diameter", "Length", "Width", "CircularED", "CircularEP",
                                    "SphericalED", "SphericalEP", "Aspect Ratio", "Circularity",
                                    "Sphericity", "Roundness", "Roughness", "Solidity", "Compactness"
                                ],
                                selected=["Feret Diameter", "Aspect Ratio"],
                                width="100%"
                            ),
                            #ui.hr(),
                            ui.h5("Distribution Plot Settings", style="font-size:1.3rem; font-weight:600;"),
                            ui.input_checkbox(
                                "show_histogram",
                                "Show histogram",
                                value=True
                            ),
                            ui.input_checkbox(
                                "show_kde",
                                "Show KDE curve",
                                value=True
                            ),
                            ui.input_checkbox(
                                "show_rug",
                                "Show rug",
                                value=True
                            )
                        )
                    )
                )
            )
        )    
    ),
    
    ui.hr(),
    ui.output_text("results")
)               

settings_tab_content = ui.div(
    ui.card(
        ui.card_header(ui.h2("Image Segmentation & Particle Characterization")),
        # Segmentation Results card
        ui.card(
            ui.card_header(
                ui.h3(
                    "Segmentation Results",
                    style="font-size:1.8rem; font-weight:bold; color:#12375b;"
                )
            ),
            ui.row(
                # Processed Images (scrollable)
                ui.column(
                    6,
                    ui.card(
                        ui.card_header(
                            ui.h3(
                                "Processed Images",
                                style="font-size:1.6rem; font-weight:bold; color:#12375b;"
                            )
                        ),
                        ui.card_body(
                            ui.div(
                                ui.output_ui("processed_images_ui"),
                                style="max-height:600px; overflow-y:auto; padding-right:10px;"
                            )
                        )
                    )
                ),
                # Metrics Table (scrollable & stuck below its header)
                ui.column(
                    6,
                    ui.card(
                        ui.card_header(
                            ui.h3(
                                "Metrics Table",
                                style="font-size:1.6rem; font-weight:bold; color:#12375b;"
                            )
                        ),
                        ui.card_body(
                            ui.div(
                                ui.download_button(
                                    "download_metrics", 
                                    "Download Metrics CSV",
                                    style="font-size:1.2rem; margin-bottom:10px;"
                                ),
                                ui.output_table("metrics_table"),
                                style="max-height:600px; overflow-y:auto; padding-right:10px;"
                            )
                        )
                    )
                )
            )
        )
    )
)

reports_tab_content = ui.div(
    ui.card(
        ui.card_header(ui.h2("Shape & Size-Based Metrics Evaluation")),
        
        # 2a) Parameter selectors
        ui.card(
            ui.card_header(
                ui.h3("Choose Metrics to Plot", style="font-size:1.6rem; font-weight:bold; color:#12375b;")
            ),
            ui.card_body(
                ui.input_selectize(
                    "size_metrics_plot",
                    "Size-based Metrics:",
                    choices=[
                        "Feret Diameter", "Length", "Width",
                        "CircularED", "CircularEP", "SphericalED", "SphericalEP"
                    ],
                    multiple=True,
                    options={"placeholder": "Select size metrics"}
                ),
                ui.input_selectize(
                    "shape_metrics_plot",
                    "Shape-based Metrics:",
                    choices=[
                        "Aspect Ratio", "Circularity", "Sphericity",
                        "Roundness", "Roughness", "Solidity", "Compactness"
                    ],
                    multiple=True,
                    options={"placeholder": "Select shape metrics"}
                )
            )
        ),

        # 2b) Percentile summary card
        ui.card(
            ui.card_header(
                ui.h3("Percentile Summary", style="font-size:1.6rem; font-weight:bold; color:#12375b;")
            ),
            ui.card_body(
                ui.layout_columns(
                    ui.value_box("D[10]", ui.output_ui("d10"), showcase=ICONS["d10"]),
                    ui.value_box("D[50]", ui.output_ui("d50"), showcase=ICONS["d50"]),
                    ui.value_box("D[90]", ui.output_ui("d90"), showcase=ICONS["d90"]),
                    fill=False,
                )
            )
        ),

        # 2c) Distribution plots in two columns
        ui.card(
            ui.card_header(
                ui.h3("Distribution Plots", style="font-size:1.6rem; font-weight:bold; color:#12375b;")
            ),
            ui.card_body(
                ui.output_ui("dynamic_dist_plots")   # <— here’s your @render.ui helper
            )
        )
    )    
)        

userinfo_tab_content = ui.div(
                 ui.navset_pill(
                            ui.nav_panel("CrysAI User Guide", ui.tags.iframe(src = "CrysAI_User_Guide.pdf", style="width: 100%; height: 100vh; border: none;" ) ), 
                            ui.nav_panel("Metrics used for Evaluation", ui.tags.iframe(src = "CrysAI Metrics for Evaluating Crystal Shape and Size Descriptors.pdf", style="width: 100%; height: 100vh; border: none;" )), 
                            ui.nav_panel("Standard Operating Procedure", ui.tags.iframe(src = "SOP for CrysAI Model Training.pdf", style="width: 100%; height: 100vh; border: none;" ))
                 )
)

feedback_tab_content = ui.div(
    ui.card(
        ui.card_header(ui.h3("WebApp Feedback", style="color: #12375b;font-weight:bold;")),
        ui.p("If you encounter an issue or have suggestions for how to improve this WebApp, please fill in the form below and click Submit. This will open a prepared email, for you to send. Thank you!"),
        ui.input_select(id="feedback_type_id", label="Please select:", choices=["Report an Issue", "Log a Suggestion"], multiple=False, width="25%"),
        ui.input_text(id="feedback_webapp_name_id", label="Provide the WebApp name:", value="", width="50%"),
        ui.row(
            ui.column(9,
                ui.input_text_area(id="feedback_details_id", label="Describe the Issue or Suggestion:", value="", width="100%", height="140px")
            ),
            ui.column(3,
                ui.output_ui(id="submit_feedback_ui_id")
            )     
        )
    )
)

##### UI Definition START #####
app_ui = ui.page_fillable(
    ##### WebApp Styling START (Do Not Edit) #####
    # External stylesheets
    ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Montserrat"),
    ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"),
    ui.tags.link(rel="stylesheet", href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"),
    ui.head_content(ui.tags.link(rel="stylesheet", href="www/style.css")),

    # JavaScript for sidebar toggling behavior
    ui.head_content(
        ui.tags.script("""
            Shiny.addCustomMessageHandler("sidebar_toggled", (message) => {
                const layoutEl = document.querySelector(".bslib-sidebar-layout");
                if (layoutEl) {
                    layoutEl.style.setProperty("grid-template-columns", `${message.width} 1fr`, "important");
                }

                const isCollapsed = message.collapsed;

                /* NEW: push the width into a CSS variable that the stylesheet will pick up */
                document.documentElement.style.setProperty("--sidebar-width", message.width);

                document.querySelectorAll(".nav-pills .nav-link").forEach(linkEl => {
                    if (!linkEl.dataset.origHTML) {
                        linkEl.dataset.origHTML = linkEl.innerHTML;
                    }

                    if (isCollapsed) {
                        const iTag = linkEl.querySelector("i");
                        if (iTag) {
                            linkEl.innerHTML = iTag.outerHTML;
                        }
                    } else {
                        linkEl.innerHTML = linkEl.dataset.origHTML;
                    }
                });

                const btn = document.getElementById("toggle_sidebar");
                if (btn) {
                    btn.style.marginLeft = message.collapsed ? "63px" : "192px";
                }
            });            
        """)
    ),
    ##### WebApp Styling END #####

 #### Header Panel START #####
    ui.row(
        ui.div(
            ui.div(
                ui.output_image("app_title_id", height="100%"),
                style="margin: 5px; margin-left:12px;"
            ),
            ui.input_action_button(
                "toggle_sidebar", "", width="34px;", class_="Toggle-Button",
                style="height: 40px; margin-top: -58px; margin-bottom: 0px; margin-left: 192px; border-radius: 0px; padding:0px; padding-top:0px; padding-bottom: 0px; box-shadow: none;",
                icon=ui.tags.i(class_="fa fa-bars")
            ),
            style="height: 38px; overflow: hidden; margin-bottom: -15px; margin-top: -15px; position: relative; z-index: 10; box-shadow: 5px 5px 10px #00000012 !important; background-color: #fff;"
        ), style = "border: 2px #fff !important"
    ),
   ##### Header Panel END #####
    ui.output_ui("body_ui"),  # Login panel
    
)

##### Server Definition START  #####
def server(input: Inputs, output: Outputs, session: Session):
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    
    metrics_store = reactive.Value(None)
    has_results = reactive.Value(False)
    is_logged_in = reactive.Value(False)

    @output
    @render.text
    def is_logged_in_txt():
        return "true" if is_logged_in.get() else "false"

    @reactive.Effect
    @reactive.event(input.login_btn)
    def on_login():
        if input.username() == "admin" and input.password() == "admin":
            is_logged_in.set(True)

    @reactive.Effect
    @reactive.event(input.logout_btn)
    def on_logout():
        is_logged_in.set(False)

    @output
    @render.ui
    def body_ui():
        if not is_logged_in.get():
            return ui.div(
                ui.h3("Please log in", style="text-align:center;"),
                ui.input_text("username", "Username:"),
                ui.input_password("password", "Password:"),
                ui.input_action_button("login_btn", "Log in",
                                       class_="btn-primary",
                                       style="width:100%;"),
                style=(
                    "max-width:300px; margin:100px auto; "
                    "padding:20px; border:1px solid #ddd; border-radius:8px;"
                )
            )
        else:
            return ui.div(
                ui.div(
                    ui.input_action_button(
                        "logout_btn", "Logout",
                        icon=icon_svg("power-off"),
                        class_="btn-warning",
                        style="float:right; margin:10px;"
                    )
                ),
                ui.div(
                    ui.navset_pill(
                        ui.nav_panel(ui.HTML('<i class="fas fa-home"></i> Parameters'),     home_tab_content),
                        ui.nav_panel(ui.HTML('<i class="fas fa-cogs"></i> DLIA Segmentation'), settings_tab_content),
                        ui.nav_panel(ui.HTML('<i class="fas fa-chart-bar"></i> Metrics Evaluation'), reports_tab_content),
                        ui.nav_panel(ui.HTML('<i class="fas fa-user"></i> User Information'), userinfo_tab_content),
                        ui.nav_panel(ui.HTML('<i class="fas fa-comment-dots"></i> Feedback'), feedback_tab_content),
                        id="tabs"
                    ),
                    class_="layout-container"
                ),
                ui.row(
                    ui.column(6, ui.output_ui("copyright_company_URL_id")),
                    ui.column(2, ui.output_text("Version_track_id"),
                              offset=4, style="text-align:right;"),
                    style="margin-top:-15px; padding:10px; background:#fff;"
                )
            )

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
        cfg.MODEL.DEVICE = "cpu"
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
        cfg.TEST.DETECTIONS_PER_IMAGE = 200
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, 'model_final.pth'
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = input.threshold()
        cfg.MODEL.DEVICE = "cpu"
        MetadataCatalog.get("manittol_s_test").set(things_classes=["Particle", "Bubble", "Droplet"])
        MetadataCatalog.get("manittol_s_test").set(things_colors=[(0, 0, 225), (0, 255, 0), (255, 0, 0)])
        manittol_s_test_metadata = MetadataCatalog.get("manittol_s_test")
        return DefaultPredictor(cfg)

    def compute_metrics(contours, pixels_per_metric):
        metrics_data = {
            "Feret Diameter": [],
            "Length": [],
            "Width": [],
            "CircularED": [],
            "CircularEP": [],
            "SphericalEP": [],
            "SphericalED": [],
            "Aspect Ratio": [],
            "Circularity": [],
            "Sphericity": [],
            "Roundness": [],
            "Roughness": [],
            "Solidity": [],
            "Compactness": [],   


        }
        for c in contours:
            if cv2.contourArea(c) < 1:
                continue
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            convexHull = cv2.convexHull(c)
            hull_area = cv2.contourArea(convexHull)
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
            dimArea = area/pixels_per_metric
            dimPerimeter = perimeter/pixels_per_metric
            dimHull_Area = hull_area/pixels_per_metric
            diaFeret = max(dimA, dimB)

            Feret_diam = diaFeret
            Length = max(dimA, dimB)
            Width = min(dimA, dimB)
            aspect_ratio = Length / Width if Width != 0 else 0
            #aspect_ratio = max(dimB,dimA)/min(dimB,dimA), if dimA != 0 else 0
            CircularED = np.sqrt(4*dimArea/np.pi)
            #CircularEP = np.sqrt(dimPerimeter/np.pi)
            CircularEP = dimPerimeter/np.pi
            SphericalED = 2*np.sqrt(dimArea/np.pi)
            SphericalEP = 2*np.pi*np.sqrt(dimArea/(4*np.pi))                        
            Circularity = 4*np.pi*(dimArea/(dimPerimeter)**2) if perimeter != 0 else 0
            Sphericity = Sphericity = (2*np.sqrt(np.pi*dimArea)) /dimPerimeter if dimPerimeter != 0 else 0
            Roundness = 4*dimArea/(np.pi*(diaFeret)**2) if diaFeret > 0 else 0
            Roughness = CircularEP / SphericalED if SphericalED > 0 else 0            
            Solidity = dimArea / dimHull_Area if dimHull_Area > 0 else 0
            Compactness = np.sqrt(4*dimArea/(np.pi*(diaFeret)**2))
            

            metrics_data["Feret Diameter"].append(Feret_diam)
            metrics_data["Length"].append(Length)
            metrics_data["Width"].append(Width)
            metrics_data["CircularED"].append(CircularED)
            metrics_data["CircularEP"].append(CircularEP)
            metrics_data["SphericalEP"].append(SphericalEP)
            metrics_data["SphericalED"].append(SphericalED)
            metrics_data["Aspect Ratio"].append(aspect_ratio)
            metrics_data["Circularity"].append(Circularity)
            metrics_data["Sphericity"].append(Sphericity)
            metrics_data["Roundness"].append(Roundness)
            metrics_data["Roughness"].append(Roughness)
            metrics_data["Solidity"].append(Solidity)
            metrics_data["Compactness"].append(Compactness)
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
                # put it into our in-memory store
                metrics_store.set(all_metrics_df)
                has_results.set(True)
                return (
                    "Analysis Complete | \n"
                    f"Total particles detected: {total_particle_count}\n"
                    #f"Total bubbles detected: {total_bubble_count}\n"
                    #f"Total droplets detected: {total_droplet_count}"
                )


    # First, add this debug function
    @reactive.Effect
    def debug_plot_data():
        if input.show_plots():
            df = check_and_load_results()
     
    def plot_distribution(data, feature_name, show_hist, show_kde, show_rug):
        try:
            print(f"Creating plot for {feature_name}")
            fig = go.Figure()
            if show_hist:
                fig.add_trace(go.Histogram(
                    x=data[feature_name],
                    name='Histogram',
                    nbinsx=30,
                    opacity=0.7
                ))
            
            # Add KDE (violin plot)
            if show_kde:
                fig.add_trace(go.Violin(
                    x=data[feature_name],
                    name='KDE',
                    side='positive',
                    line_color='black',
                    meanline_visible=True,
                    fillcolor='rgba(0,0,0,0)',
                    points=False,
                    yaxis='y2'
                ))
            
            # Add rug plot 
            if show_rug:
                fig.add_trace(go.Scatter(
                    x=data[feature_name],
                    y=[-0.15] * len(data[feature_name]),  # Moved down for visibility
                    mode='markers',
                    name='Rug',
                    marker=dict(
                        symbol='line-ns-open',  # Changed symbol
                        size=15,  # Increased size
                        color='rgba(0,0,0,0.7)',  # More opaque
                        line=dict(width=2)  # Added line width
                    ),
                    yaxis='y2'
                ))
            
            # Update layout with adjusted ranges
            fig.update_layout(
                title=f'Distribution of {feature_name}',
                height=400,
                width=600,
                template='plotly_white',
                showlegend=True,
                title_x=0.5,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(title=feature_name),
                yaxis=dict(title='Count', side='left'),
                yaxis2=dict(
                    title='Density',
                    side='right',
                    overlaying='y',
                    showgrid=False,
                    range=[-0.2, 1],  # Adjusted range to show rug plot
                    showticklabels=False  # Hide secondary axis labels
                ),
                barmode='overlay'
            )
            
            #print(f"Plot created successfully for {feature_name}")
            return fig
            
        except Exception as e:
            print(f"Error in plot_distribution for {feature_name}: {e}")
            return None


    # Modify the check_and_load_results function
    def check_and_load_results():
        df = metrics_store.get()
        return df if (df is not None and not df.empty) else None

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
        req(metrics_store.get() is not None)
        req(not metrics_store.get().empty)

        df = metrics_store.get()
        selected_columns = list(input.selected_metrics_to_show())
        available_id_columns = []
        for col in ['Image', 'Object ID', 'image', 'object_id', 'Image Name', 'Object', 'id']:
            if col in df.columns:
                available_id_columns.append(col)
        display_columns = available_id_columns + selected_columns
        existing_columns = [col for col in display_columns if col in df.columns]
        if not existing_columns:
            return df
        filtered_df = df[existing_columns]
        
        return filtered_df

    @output
    @render.ui
    def d10():
        metric = input.selected_metric()
        df = metrics_store.get()
        req(df is not None); req(not df.empty)
        if metric in df.columns:
            p10 = df[metric].quantile(0.1)
            return ui.h4(f"{metric}: {p10:.2f}")
        else:
            return ui.p("N/A")

    @output
    @render.ui
    def d50():
        metric = input.selected_metric()
        df = metrics_store.get()
        req(df is not None); req(not df.empty)
        if metric in df.columns:
            p50 = df[metric].quantile(0.5)
            return ui.h4(f"{metric}: {p50:.2f}")
        else:
            return ui.p("N/A")

    @output
    @render.ui
    def d90():
        metric = input.selected_metric()
        df = metrics_store.get()
        req(df is not None); req(not df.empty)
        if metric in df.columns:
            p90 = df[metric].quantile(0.9)
            return ui.h4(f"{metric}: {p90:.2f}")
        else:
            return ui.p("N/A")

    @output
    @render.download(filename=lambda: f"metrics-{date.today().isoformat()}-{random.randint(100, 999)}.csv")
    def download_metrics():
        df = metrics_store.get()
        if df is None or df.empty:
            return None
        return df.to_csv(index=False).encode("utf-8")

    @output
    @render_plotly
    def feret_diameter_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Feret Diameter',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def width_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Width',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def circulared_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='CircularED',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def length_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Length',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def circularep_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='CircularEP',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def sphericaled_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='SphericalED',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def sphericalep_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='SphericalEP',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig

    @output
    @render_plotly
    def aspect_ratio_plot():
        if not input.show_plots():
            print("Show plots is disabled")
            return None
            
        df = check_and_load_results()
        if df is None:
            print("No data available for plotting")
            return None
            
        print("Creating aspect ratio plot")
        fig = plot_distribution(
            data=df,
            feature_name='Aspect Ratio',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        #print("Aspect ratio plot created:", fig is not None)
        return fig

    @output
    @render_plotly
    def circularity_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Circularity',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig 

    @output
    @render_plotly
    def sphericity_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Sphericity',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig 

    @output
    @render_plotly
    def roundness_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Roundness',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig    

    @output
    @render_plotly
    def roughness_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Roughness',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig     

    @output
    @render_plotly
    def solidity_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Solidity',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig 
    
    @output
    @render_plotly
    def compactness_plot():
        if not input.show_plots():
            return None
        df = check_and_load_results()
        if df is None:
            return None
        return plot_distribution(
            data=df,
            feature_name='Compactness',
            show_hist=input.show_histogram(),
            show_kde=input.show_kde(),
            show_rug=input.show_rug()
        )
        return fig
    
    @output
    @render.ui
    def dynamic_dist_plots():
        size = input.size_metrics_plot() or []
        shape = input.shape_metrics_plot() or []
        # build a Shiny UI object
        return ui.row(
            ui.column(
                6,
                ui.div(
                    *[output_widget(f"{m.lower().replace(' ', '_')}_plot") for m in size],
                    style="max-height:500px; overflow-y:auto;"
                )
            ),
            ui.column(
                6,
                ui.div(
                    *[output_widget(f"{m.lower().replace(' ', '_')}_plot") for m in shape],
                    style="max-height:500px; overflow-y:auto;"
                )
            )
        )


    ##### Toggle Sidebar Functionality START (Do Not Edit) #####
    # Track the state of the toggled side bar
    toggled = reactive.Value(False)
    
    # Toggle the iAchieve logo between the full and toggled state
    @output
    @render.image
    def app_title_id():
        if toggled.get():
            img_file = resource_dir / "iAchieve_Logo_Only.png"
            width_css = "52px"
        else:
            img_file = resource_dir / "iAchieve_Logo.png"
            width_css = "175px"
        return {"src": img_file, "style": f"width:{width_css}; margin-left: -3px; margin-right: 0px;"}

    # Toggle the sidebar width and toggled button position between the expanded and collapsed state 
    # (also removes tab names and leaves just icons)
    @reactive.Effect
    @reactive.event(input.toggle_sidebar)
    async def _toggle_layout_columns():
        toggled.set(not toggled.get())
        new_width = "55px" if toggled.get() else "185px"
        await session.send_custom_message("sidebar_toggled", {
            "width": new_width,
            "collapsed": toggled.get()
        })

    # Returns the copyright APC and adjusts the position based on the toggled state
    @output
    @render.ui
    def copyright_company_URL_id():
        margin_left = "55px" if toggled.get() else "185px"
        return ui.div(
            ui.HTML("&#169;"),
            ui.span(datetime.now().year),
            ui.span(" "),
            ui.a("APC", href="https://approcess.com/", target="_blank"),
            style=f"margin-left: {margin_left};"
        )
    
    ##### Feedback Submission Logic START (Editable or Remove, as needed) #####
    email_string = reactive.value(None)

    @output
    @render.ui
    def submit_feedback_ui_id():
        return ui.a(
            ui.input_action_button(id="submit_feedback_id", label="Submit Feedback", icon=icon_svg("envelope"), width="100%", class_="Green-Button", style="margin-top: 125px;"),
            href=email_string()
        )

    @reactive.effect
    def _():
        req(input.feedback_type_id() and input.feedback_webapp_name_id() and input.feedback_details_id())
        feedback_type = input.feedback_type_id().replace(" ", "%20")
        WebApp_name = input.feedback_webapp_name_id().replace(" ", "%20")
        feedback_details = input.feedback_details_id().replace(" ", "%20")
        email_string.set(f"mailto:shinyapps@approcess.com?subject={feedback_type}:%20%20{WebApp_name}&body={feedback_details}")
##### Feedback Submission Logic END #####

##### Version Tracking START (Do Not Edit) #####
    @output
    @render.text
    def Version_track_id():
        return Web_app_version
##### Version Tracking END #####
  
##### Server Definition END #####

# Set up the app with the ui, server, and point it in the direction of the www folder
www_dir = Path(__file__).parent / "www"
#app = App(app_ui, server, static_assets={"/www": resource_dir})
app = App(app_ui, server, static_assets=www_dir)
# Run the app and launch it in the browser
#app.run(launch_browser=True)
# if __name__ == "__main__":
#     import asyncio
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(app.serve())