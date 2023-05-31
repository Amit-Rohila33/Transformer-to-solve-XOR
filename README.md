# XOR Parity Transformer

This project implements and trains a Transformer model to solve the XOR parity problem. The XOR parity problem involves predicting the XOR result of binary sequences.

## Getting Started

### Prerequisites

- Python 3.7 or above
- pip package manager

### Installation

1. Clone the repository:

- git clone https://github.com/your-username/xor-parity-transformer.git


2. Create a virtual environment (optional but recommended):

- python3 -m venv env


3. Activate the virtual environment:
- For Windows:
  ```
  env\Scripts\activate
  ```
- For macOS/Linux:
  ```
  source env/bin/activate
  ```

4. Install the required dependencies:

- pip install -r requirements.txt


### Usage

1. Generate the datasets:

   - python generate_ds.py


2. Train the Transformer model:

- python trainer.py --batch_size 64 --epochs 10 --lr 0.001


3. Perform hyperparameter grid search (optional):

- python hp.py



4. Explore the results in the IPython notebook:
- Open the `XOR_Parity_Training.ipynb` notebook using Jupyter Notebook or JupyterLab.
- Run the notebook cells to train the model and visualize the training process.

### Project Structure

- `generate_ds.py`: Generates the XOR parity datasets.
- `trainer.py`: Trains the Transformer model on the XOR parity problem.
- `transformer_model.py`: Defines the Transformer model architecture.
- `dataset.py`: Defines the XORParityDataset class for loading the datasets.
- `hp.py`: Performs a grid search for hyperparameters.
- `XOR_Parity_Training.ipynb`: Jupyter notebook with training and exploration steps.

