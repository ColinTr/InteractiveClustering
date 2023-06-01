# Interactive Clustering

Simple web interface to interact with various clustering algorithms and display their results.

**/!\ Datasets must be placed in ./datasets/... and cannot be loaded from other folders.**

## The interface

![Example interface](example_interface.png "Example interface")


## Demonstration video for ECML 2023

<a href="http://www.youtube.com/watch?feature=player_embedded&v=W7ru8NHPj-8&cc_load_policy=1" target="_blank">
 <img src="youtube_thumbnail.png" alt="Watch the video" border="10" />
</a>


## Installation

**/!\ Warning:**
Installation wont work behind the **proxy** and on the **intranet**.
It is advised to deactivate the proxy and connect to a wifi with full internet access.


1) Clone the repository in the [Git Bash](https://git-scm.com/downloads) with `git clone https://github.com/ColinTr/InteractiveClustering`

2) Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) and open it **in administrator**:
```bash
cd backend
# Create the virtual environment with conda
conda env create --file environment.yml --prefix ./icvenv
# Activate the virtual environment
conda activate .\icvenv
# Check if torch supports GPU (you need CUDA 11 installed)
python -c "import torch; print(torch.cuda.is_available())"
```

3) Start the backend:
```bash
python server.py
```

4) Install [Node.js](https://nodejs.org/en/).

5) Open cmd **in administrator**:

```bash
cd frontend
npm install
npm start
```

At this point, the backend python server is running, and the frontend website should be accessible on http://localhost:3000/

**Note:** The backend uses the port 5000 and the frontend uses the port 3000.
