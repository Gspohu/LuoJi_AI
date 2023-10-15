# LuoJi_AI (Neural Network Operations Classifier)
Small unpretentious script to test the different types of neurons on several types of problem
This repository contains a Python script that demonstrates the use of neural networks for various logical operations. The purpose is to explore and test different types of neural networks on various types of problems and classify them.

## Features:
- Implementation of a basic feed-forward neural network.
- Definition of various logical operations including simple ones like AND, OR, and complex ones.
- Functionality to train, save, and load neural networks for each logical operation.
- Interactive CLI menu to interrogate trained neural networks, test their accuracy, and visualize Principal Component Analysis (PCA).
  
## Requirements:
- Python 3.x
- `numpy`
- `matplotlib`
- `sklearn`
- `sympy`
- `pickle`
- `questionary`
- `os`

## Installation:

1. Clone the repository:

   ```
   git clone https://github.com/Gspohu/LuoJi_AI.git
   ```

2. Navigate to the repository's directory:

   ```
   cd LuoJi_AI
   ```

3. Install the required packages (WIP):

   ```
   pip install -r requirements.txt
   ```

## Usage:

1. Run the script:

   ```
   python LuoJi_AI.py
   ```

2. The script will start with an interactive menu. Follow the on-screen prompts to:
   - Interrogate a neural network for a specific logical operation
   - Test the accuracy of a neural network for one or more logical operations
   - Display the PCA of a neural network for a specific logical operation

## Description:

The script contains:
- A basic feed-forward neural network implementation with forward and backward propagation
- A dictionary of extended logical operations, from basic operations like AND, OR, etc., to more complex functions
- Functions to save and load trained neural networks
- An interactive CLI menu to guide users through different functionalities


## Known Issues:

- The prime number test is currently non-functional
- The data displayed by the PCA graph currently does not provide meaningful insights

## Future Work:

- Introduce dynamic selection of neuron types within the network
- Enhance the user interface for a more intuitive experience
- Incorporate new types of neurons
- Merge `logical_operations_extended` and `training_ranges` for better code structure
- Implement a test for modulo operations
- Convert all comments to English. Originally, the project was intended for personal use only. However, as it grew and became more interesting than anticipated


## Contributing:

Feel free to fork the repository, make changes, and submit pull requests. Any contribution, from fixing typos to major features, is highly appreciated!

## License:

GPLv3

## Contact:

If you have any questions, issues, or feedback, please open an issue on this repository
