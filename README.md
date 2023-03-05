# Interactive Clustering

Simple web interface to interact with various clustering algorithms and display their results.

**Note:** The backend uses the port 5000 and the frontend uses the port 3000.


## The interface

![Example interface](example_interface.png "Example interface")


## Installation

1) Clone the repository in the [Git Bash](https://git-scm.com/downloads) with `git clone https://github.com/ColinTr/InteractiveClustering`

2) Setup the [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) virtual environment for the backend:
```bash
#Create the virtual environment:
python -m venv backend/venv
```
```bash
# Activate the virtual environment...
# ...for Windows:
backend\venv\Scripts\activate
# ...for Linux/Mac:
source backend/venv/bin/activate
```
```bash
# Install PyTorch with CUDA:
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
# Check if torch supports GPU (you need CUDA 11 installed):
python -c "import torch; print(torch.cuda.is_available())"
# Install the rest of the requirements:
python -m pip install -r requirements.txt
```

3) Start the backend:
```bash
python backend/server.py
```

4) Install [Node.js](https://nodejs.org/en/) for the frontend website.

5) Build the website and run it with:

```bash
cd frontend
npm install
npm start
```

At this point, the backend python server is running, and the frontend website should be accessible on http://localhost:3000/
