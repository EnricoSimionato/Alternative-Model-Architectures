# Alternative-Model-Architectures

Alternative-Model-Architectures is a research-oriented project developed as part of a master’s thesis, focusing on implementing and evaluating novel compression techniques for large language models (LLMs).
The primary strategies explored include weight sharing and matrix factorization, aiming to reduce model size and computational requirements without significantly compromising performance.
The Alternative-Model-Architectures repository implements the compression techniques explored in the study. It provides the necessary classes and utilities to conduct experiments using the proposed methodologies, while also enabling their evaluation.
Beyond the experiments presented in the thesis, it also includes additional trials on deep architecture compression.

## Features
-	Weight Sharing Implementations: Techniques to reduce redundancy by sharing weights across different layers or components of the model.
-	Matrix Factorization Methods: Decomposing large weight matrices into products of smaller matrices to achieve compression.
-	Modular Architecture: A flexible codebase that allows easy experimentation with various model architectures and compression strategies.
-	Docker Support: Containerized environment for consistent and reproducible experiments.

## Directory Structure
```
Alternative-Model-Architectures/
├── .idea/                      # IDE configuration files
├── src/                        # Source code directory
│   ├── models/                 # Contains model architecture definitions
│   ├── compression/            # Compression techniques implementations
│   ├── utils/                  # Utility functions and helpers
│   └── main.py                 # Entry point for running experiments
├── .dockerignore               # Specifies files to ignore in Docker builds
├── .gitignore                  # Specifies files to ignore in Git
├── Dockerfile                  # Docker image configuration
├── docker_container.sh         # Shell script to build and run Docker container
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation
1.	Clone the repository
  ```bash
  git clone [https://github.com/EnricoSimionato/Redundancy-Hunter.git](https://github.com/EnricoSimionato/Alternative-Model-Architectures.git)
  cd Alternative-Model-Architectures
  ```

2.	Create a virtual environment (optional but recommended)

3.	Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. You can launch experiments or analysis types by using the experiment_launcher.py script. It reads configuration files and triggers the corresponding experiments.
  ```bash
  python src/ --config path/to/config.yaml
  ```
2. Alternatively, for direct execution:
  ```bash
  python src/
  ```
To customize the experiment, edit the YAML files under experiments/configurations/.

3. Optionally, you can use Docker and run using:

  ```bash
  # Build and run in container
  bash docker_container.sh
  ```

### Configuration

Experiment configurations are managed through JSON or YAML files, allowing you to define parameters such as learning rate, batch size, number of epochs, and specific compression settings. Sample configuration files can be found in the configs/ directory.

### Results

After running experiments, results including training logs, evaluation metrics, and model checkpoints will be saved in the results/ directory. You can analyze these results to assess the effectiveness of different compression techniques.

## Contributions

## References

