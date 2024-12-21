# Action Recognition Model with Jetson Inference

This repository contains an **Action Recognition Model** developed using NVIDIA's **Jetson Utilities** and **Jetson Inference** frameworks. The project runs in a Docker container on NVIDIA Jetson devices, leveraging their powerful GPUs for efficient and real-time action recognition.

---

## Features
- **Action Recognition**: Detects and classifies human actions in video streams.
- **Optimized for Jetson**: Utilizes Jetson's GPU acceleration for real-time inference.
- **TensorRT Optimizations**: Includes TensorRT-optimized model for faster inference.
- **Dockerized Workflow**: Runs in a Docker container, ensuring a consistent and portable development environment.
- **Custom Model Training**: Supports training on custom datasets.

---

## Prerequisites

### Hardware
- NVIDIA Jetson device (e.g., Jetson Orin Nano, Xavier, etc.)

### Software
- NVIDIA JetPack SDK installed on your Jetson device
- Docker with NVIDIA Container Toolkit
- Git

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone git@github.com:tharunj29/Action-Detection-Jetson-Orin-Nano.git
cd Action-Detection-Jetson-Orin-Nano
```

### 2. Build and Run the Docker Container
#### Build the Docker Image:
```bash
docker build -t action-recognition .
```

#### Run the Docker Container:
```bash
docker run --rm -it --runtime=nvidia -v $(pwd):/jetson-inference action-recognition
```

### 3. Train or Test the Model
#### Train the Model:
To train the model on your custom dataset, place the dataset in the `data/` folder and run:
```bash
python train.py --dataset data/custom_dataset --epochs 50
```

#### Test the Model:
Run inference on a sample video:
```bash
python test.py --input videos/sample.mp4 --output results/output.mp4
```

---

## Folder Structure
```
.
├── build/                # Compiled binaries (included in the repository)
├── data/                 # Dataset folder
├── docker/               # Dockerfile and scripts for container setup
├── models/               # Trained models and checkpoints
├── utils/                # Utility scripts for preprocessing and analysis
├── videos/               # Sample input videos
├── train.py              # Training script
├── test.py               # Testing and inference script
├── action_net.py         # Action recognition model without TensorRT optimizations
├── action_net_zO.py      # Action recognition model with TensorRT optimizations
└── README.md             # Project documentation
```

---

## Usage

### Real-Time Action Recognition

### TensorRT Optimized Inference
To run inference with TensorRT optimizations:
```bash
python action_net_zO.py --input videos/sample.mp4 --output results/output_trt.mp4
```

## Standard Inference
To run inference without TensorRT optimizations:
```bash
python action_net.py --input videos/sample.mp4 --output results/output_standard.mp4
```

## Evaluate Model Performance
Run evaluation metrics on your test dataset:
```bash
python evaluate.py --dataset data/test_dataset
```

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add a new feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **NVIDIA Jetson Inference Framework**: For providing an optimized platform for AI and computer vision tasks.
- **Jetson Utilities**: For easy integration with Jetson devices.
- Open-source contributors and the broader AI community.

---

## Contact
If you have any questions or issues, please open an issue or contact:
- **Author**: Tharun Kumar Jayaprakash, Rashmi Chelliah
- **Email**: tj2557@columbia.edu, rc3605@columbia.edu

