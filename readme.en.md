# oceaneye-ai
This is a guide on how to train an artificial intelligence model for recognizing fish species. We will use Python and the Ultralytics library for training, and create the dataset using the LabelStudio program (open-source).

**Annotation**

- You can annotate photos using the LabelStudio program (open-source).
- First, create your account.
- Then, create a project, import your photos, select the annotation type, and add the classes into which you want to categorize the images.
- After that, you can start annotating.
- Once done, download the dataset in the YOLOv8 format.

**Why YOLOv8?**

- It is very user-friendly.
- It can be easily installed using `pip install ultralytics`.
- It has built-in commands for training, evaluation, and even detection from videos or images.
- It supports GPU if you have the correct CUDA driver and PyTorch with GPU support installed.

**Once we have the dataset, we can start training. A computer with an NVIDIA GPU is recommended, as it speeds up the process.**

**1. Setting Up the Environment (Windows)**
1. **Update drivers and install CUDA (optional)**
    - Skip this if you do not have an NVIDIA GPU.
    - For an NVIDIA RTX 4060, having the latest driver from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx) should be sufficient.
    - The official installation of CUDA Toolkit from [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit) is not mandatory if you install PyTorch with GPU support, but it can be helpful.
    - To verify if CUDA and the drivers are working, use the `nvidia-smi` command in the terminal.

2. **Install Python (3.8+, preferably 3.9 or 3.10)**
    - It is recommended to create a virtual environment using `venv`.
    ```
    cd c:\
    mkdir my_projects
    cd my_projects
    c:\Python310\python.exe -m venv c:\my_projects\oceaneye-ai
    c:\my_projects\oceaneye-ai\Scripts\activate
    ```

3. **Install PyTorch with CUDA support (only if you have an NVIDIA GPU)**
    - Generate the appropriate installation command for CUDA-supported PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/). Example (may vary by version):
    - ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
    - Verify if PyTorch detects your GPU:
    ```
    python
    import torch
    print(torch.cuda.is_available())  # Should return True
    ```

4. **Install Ultralytics (YOLOv8)**
    - ```pip install ultralytics```

**2. Data Preparation (Dataset)**

Assuming it is a **detection** dataset (i.e., you have bounding boxes for the fish), create your dataset using [LabelStudio](https://labelstud.io/).

**Dataset in YOLO format**

YOLOv8 expects the following files; Label Studio will export them, but you will need to create a `data.yaml` file and split the dataset into `train` and `val` sets with an 80:20 or 90:10 ratio.

- **images/** (directory) – images in JPEG/PNG format.
- **labels/** (directory) – text files containing bounding box labels corresponding to images.
- **data.yaml** (configuration file) – defines the data locations and class names (fish species).

Example structure:
```
fish_dataset/
├── images
│   ├── train
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val
│       ├── img101.jpg
│       ├── ...
├── labels
│   ├── train
│   │   ├── img001.txt
│   │   ├── ...
│   └── val
│       ├── img101.txt
│       ├── ...
└── data.yaml
```

**Each .txt file** corresponding to an image follows this format (YOLO format):
```
0 0.5 0.5 0.2 0.3
1 0.4 0.6 0.1 0.1
```
The first number is the **class ID** (e.g., `0 = Swordfish, 1 = Minnow, …`), and the next four numbers are **x_center, y_center, width, height** in normalized coordinates (0–1).

**The `data.yaml` file** looks like this:
```
train: path/to/fish_dataset/images/train
val: path/to/fish_dataset/images/val
names:
  0: Swordfish
  1: Minnow
  2: Other fish
```
(Adjust according to your dataset classes and names.)

If you already have a dataset in another format (e.g., COCO JSON, Pascal VOC XML, etc.), you can use conversion tools or directly YOLOv8, which supports some conversions.

**3. Running Training**

**Basic training with YOLOv8** can be done from the command line or Python.

**From the command line**:
```
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**From a Python script** (e.g., `train.py`):
```
from ultralytics import YOLO
# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")
# Start training
model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    project="fish_detect",
    name="exp1"
)
```

After training, the best model is saved in `fish_detect/exp1/best.pt`.

**Model evaluation**:
```
yolo detect val model=fish_detect/exp1/best.pt data=path/to/data.yaml
```

Or in Python:
```
model = YOLO("fish_detect/exp1/best.pt")
metrics = model.val()
print(metrics)
```

**Detection on a new video**:
```
yolo detect predict model=fish_detect/exp1/best.pt source=fish_aquarium.mp4
```

**4. Practical Tips**

- Ensure a large and balanced dataset.
- YOLOv8 applies useful augmentations automatically.
- Monitor training logs and loss graphs.
- Choose the right model size for your needs.

**5. Summary**
1. Create a virtual environment.
2. Install PyTorch with CUDA support and Ultralytics.
3. Prepare the dataset in YOLO format.
4. Train the model:
   ```yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640```
5. Use the trained model for new images or videos:
   ```yolo detect predict model=best.pt source=video.mp4```
6. Optionally fine-tune parameters for better performance.

---
This README provides step-by-step instructions while maintaining original formatting for GitHub Markdown.

