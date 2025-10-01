# Brain Tumor Classification (Deep Learning)

A simple convolutional neural network to classify brain MRI images as Tumor or No Tumor, plus a small Streamlit app for interactive inference.

## Features
- Binary classifier trained on MRI images in `datasets/yes` and `datasets/no`
- Keras/TensorFlow model training script (`mainTrain.py`)
- Quick single-image test script (`mainTest.py`)
- Streamlit app (`app.py`) for drag-and-drop prediction

## Repository Structure
- `mainTrain.py`: Train a small CNN on images in `datasets/`
- `mainTest.py`: Load a saved model and run inference on a single image
- `app.py`: Streamlit UI that loads a `.h5` model and predicts on uploaded images
- `datasets/`: Expected dataset layout
  - `yes/`: MRI images with tumor
  - `no/`: MRI images without tumor
  - `Br35H-Mask-RCNN/`: Extra dataset resources (not required for this CNN)
- `templates/`, `static/`, `uploads/`: Ancillary folders not required for CLI training
- `*.h5`: Saved Keras models (provided or produced by training)

## Requirements
- Python 3.9–3.11 recommended
- OS: Windows 10/11 (paths in scripts use Windows style)

Python packages:
- tensorflow (2.x)
- numpy
- opencv-python
- pillow
- scikit-learn
- streamlit

You can install via pip:
```bash
pip install tensorflow numpy opencv-python pillow scikit-learn streamlit
```

If you use a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # if you create one, otherwise use the list above
```

## Dataset Layout
Expected directory structure under the project root:
```
BrainTumor Classification DL/
  datasets/
    yes/   # tumor images (jpg/png)
    no/    # non-tumor images (jpg/png)
```
Place your images into `yes/` and `no/` accordingly. The scripts resize to 64×64 and normalize to [0,1].

## Training
The training script reads images from `datasets/no` and `datasets/yes`, splits into train/test, trains a small CNN, and saves a `.h5` model.

Run:
```bash
python mainTrain.py
```
Notes:
- `mainTrain.py` currently sets `IMAGE_DIRECTORY` to a Windows absolute path:
  ```python
  IMAGE_DIRECTORY = 'E:\\PROJECTS\\btc\\BrainTumor Classification DL\\datasets'
  ```
  If your project path is different, update this string or change to a relative path like:
  ```python
  IMAGE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'datasets')
  ```
- After training, the script saves a model file (e.g. `BrainTumor20EpochsCategorical10.h5`) in the project root.

## Quick Test (CLI)
`mainTest.py` loads a saved model and runs prediction on a single image.

Edit the image path and model filename if needed:
```python
model = load_model('BrainTumor20EpochsCategorical10.h5')
image = cv2.imread(r'path\\to\\your\\image.jpg')
```
Run:
```bash
python mainTest.py
```
The script prints the predicted class index (0 = No Tumor, 1 = Tumor).

## Streamlit App
The app loads a model and serves a simple web UI to upload an MRI image and get a prediction.

By default `app.py` loads:
```python
model = tf.keras.models.load_model('BrainTumor10EpochsCategorical.h5')
```
If you trained a different model name (e.g., `BrainTumor20EpochsCategorical10.h5`), update the filename in `app.py`.

Run the app:
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (typically `http://localhost:8501`). Upload a `.jpg`, `.jpeg`, or `.png` image. The app resizes to 64×64 and outputs the predicted class and confidence.

## Common Issues & Troubleshooting
- TensorFlow install issues on Windows: try installing `tensorflow` in a fresh venv. For CPU-only setups, `pip install tensorflow==2.12.*` often works well with Python 3.10.
- OpenCV read errors: ensure the image path is correct and the file exists; prefer raw strings `r'C:\\path\\to\\file.jpg'` for Windows paths.
- Shape/Channels mismatch: the app converts RGBA to RGB; ensure inputs are standard 3-channel images.
- Model file not found: confirm the `.h5` model exists in the project root and the filename in `app.py`/`mainTest.py` matches.
- Absolute paths: replace hard-coded absolute paths with relative paths to make the project portable.

## Notes
- Image size is fixed at 64×64; for better accuracy, consider larger inputs and a deeper model.
- This repository contains extra dataset folders used by other experiments; the minimal requirement for this CNN is only `datasets/yes` and `datasets/no`.
- The `.h5` files included may differ from what the training script outputs; align filenames across `mainTrain.py`, `mainTest.py`, and `app.py`.

## License
Specify your preferred license (e.g., MIT) here.
