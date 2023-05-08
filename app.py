import gradio as gr
import argparse
import configparser
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time
import uuid
import shutil
from tqdm import tqdm
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

import warnings
warnings.filterwarnings("ignore")

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    id = video_input.split("/")[-1].strip(".mp4")
    
    frames = []
    user_name = time.time()
    operation_log = [("",""),("","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "input" : videos[id]["input"],
        "output" : videos[id]["output"],
        "scale_percent" : scale_percent,
        }
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_state["origin_images"][0], \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True, value=operation_log)

def run_example(example):
    return video_input

    
# get the select frame from gradio slider
def select_template(ideo_state, interactive_state):

    # images = video_state[1]
    video_state["select_frame_number"] = 0

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
        # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    operation_log = [("",""), ("","Normal")]

    return video_state["painted_images"][0], video_state, interactive_state, operation_log

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), ("","Normal")]
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),("","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("","Normal")]
    return select_frame, operation_log

# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("","Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    resized_masks, resized_painted_masks = [], []
    for masks, painted_masks in zip(video_state["masks"], video_state["painted_images"]):
        width = int(masks.shape[1] * 100 / video_state["scale_percent"])
        height = int(masks.shape[0] * 100 / video_state["scale_percent"])
        dim = (width, height)
        
        # resize image
        resized_mask = cv2.resize(masks, dim, interpolation = cv2.INTER_AREA)
        resized_painted_mask = cv2.resize(painted_masks, dim, interpolation = cv2.INTER_AREA)
        
        resized_masks.append(resized_mask)
        resized_painted_masks.append(resized_painted_mask)
        
    video_state["masks"] = resized_masks
    video_state["painted_images"] = resized_painted_masks
        
    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video
    interactive_state["inference_times"] += 1
    
    shutil.copy(video_output, '%s/vis.mp4'%(video_state["output"]))
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                           interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                           interactive_state["positive_click_times"],
                                                                                                                                        interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    if interactive_state["mask_save"]:
        if not os.path.exists(video_state["output"]):
            os.makedirs(video_state["output"])

        filenames = sorted(os.listdir(video_state["input"]))
        for filename, mask, painted_image in zip(filenames, video_state["masks"], video_state["painted_images"]):
            i = int(filename.split(".")[0])

            np.save(os.path.join(video_state["output"], '{:05d}.npy'.format(i)), mask)
            cv2.imwrite(os.path.join(video_state["output"], '{:05d}.jpg'.format(i)), cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB))
        # save_mask(video_state["masks"], video_state["video_name"])
    #### shanggao code for mask save
    return video_output, video_state, interactive_state, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # print(output_path)
    # for frame in frames:
    #     video.write(frame)
    
    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def isImageFile(str):
    import re 
    # Regex to check valid image file extension.
    regex = "([^\\s]+(\\.(?i)(jpe?g|png|gif|bmp))$)"
    
    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty
    # return false
    if (str == None):
        return False

    # Return if the string
    # matched the ReGex
    if(re.search(p, str)):
        return True
    else:
        return False
    
# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth", 
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


folder ="./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
# args.port = 12212
# args.device = "cuda:1"
# args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)


input_dir, output_dir, uuids = [], [], []
config = configparser.RawConfigParser()
config.read("database/configs/%s.config" % args.config)
for vidid in range(len(config.sections()) - 1):
    img_path = config.get("data_%d" % vidid, "img_path")
    input_dir.append(img_path)
    output_dir.append(img_path.replace("JPEGImages", "Annotations"))

print(input_dir, output_dir)
assert len(input_dir) == len(output_dir), "Config Error"

shutil.rmtree("./tmp")
os.mkdir("./tmp")

scale_percent = args.scale_percent # percent of original size

for path in tqdm(input_dir):
    frames = []
    id =  str(uuid.uuid4())
    uuids.append(id)
    for file in sorted(os.listdir(path)):
        if isImageFile(file):
            frame = cv2.imread(path + "/" + file)
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    
    frames = torch.from_numpy(np.asarray(frames))
    generate_video_from_frames(frames, "./tmp/{}.mp4".format(id))

videos = {}
for id, input, output in zip(uuids, input_dir, output_dir):
    videos[id] = {"input" : input,
                  "output" : output}
        
with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )        
    
    def exit_gradio():
        os._exit(0)

    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row():
                video_input = gr.Video()


            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Load Video", interactive=True, variant="primary") 

                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True,
                                visible=False)
                            Add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False)
                            remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False) 
                            clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False).style(height=160)
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False).style(height=360)
      
                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(autosize=True, visible=False).style(height=360)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Tracking", visible=False)
                        exit = gr.Button(value="Exit", visible=True)

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[video_state, template_frame,
                 point_prompt, clear_button_click, Add_mask_button, template_frame,
                 tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status]
    )   

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    exit.click(fn=exit_gradio, inputs=[], outputs=None)
    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )
    
    # clear input
    video_input.clear(
        lambda: (
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": 0,
        "resize_ratio": 1
        },
        [[],[]],
        None,
        None,
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False)
                        
        ),
        [],
        [ 
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button,point_prompt, clear_button_click, 
            Add_mask_button, template_frame, tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status],
    )
    # set example
    gr.Markdown("##  Videos")
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "./tmp/", test_sample) for test_sample in os.listdir("./tmp")],
        fn=run_example,
        inputs=[
            video_input
        ],
        outputs=[video_input],
        # cache_examples=True,
    ) 

    iface.queue(concurrency_count=1)
    iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0", share=True)
    # iface.launch(debug=True, enable_queue=True)
    iface.close()
