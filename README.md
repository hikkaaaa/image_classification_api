The structure:

image-classifier/
├── ml_experiments/ # train your model here
│ ├── data/ # raw images (usually gitignored)
│ ├── notebooks/ # jupyter notebooks for EDA
│ └── train.py # script to train and save model.pth
│
├── app/ # source code for API
│ ├── **init**.py #treat this folder as a package
│ ├── main.py # FastAPI entry point & endpoints
│ ├── model.py # PyTorch model class definition (Must match training!)
│ └── utils.py # image preprocessing/transforms
│
├── static/ # frontend UI
│ ├── index.html
│ ├── script.js
│ └── styles.css
│
├── models/ # Where you move the trained model
│ └── classifier.pth # The saved weights file
│
├── Dockerfile # instructions to build the container
├── requirements.txt # Python dependencies
├── .gitignore # ignore virtualenv, **pycache**, and data
└── README.md

ml_experiments/ vs. app/:
You keep training scripts separate from server code. Once you train a model in ml_experiments/, you manually copy the resulting .pth file into the models/ folder.

app/model.py:
Neural Network class (e.g., class SimpleCNN(nn.Module)) in a file that can be imported by both train.py and main.py. This ensures the architecture is identical during training and inference.

This is the shared neural network architecture. We will use a classic 3-layer CNN designed for 32x32 images (like CIFAR-10).

CIFAR-10 has 10 classes: 
1. Airplane
2. Automobile
3. Bird
4. Cat 
5. Deer
6. Dog
7. Frog 
8. Horse
9. Ship
10. Truck
