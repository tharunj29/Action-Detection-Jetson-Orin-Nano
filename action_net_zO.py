import jetson.utils
import onnxruntime as ort
import numpy as np
import psutil
import time
from collections import deque

# Paths to the model and labels
model_path = "./models/action/Action-ResNet18/resnet-18-kinetics-moments.onnx"
labels_path = "./models/action/Action-ResNet18/labels.txt"

# Load class labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create an inference session with ONNX Runtime
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


# Get input and output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Model requirements:
sequence_length = 16
input_height = 112
input_width = 112

# Normalization parameters
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(cuda_img):
    # Create an empty CUDA image for the resized output
    resized_img = jetson.utils.cudaAllocMapped(width=input_width, height=input_>
    # Resize the image
    jetson.utils.cudaResize(cuda_img, resized_img)
    # Convert to numpy array
    img = jetson.utils.cudaToNumpy(resized_img)
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img[:,:,:3] - mean) / std  # Only use RGB channels
    # Transpose to CHW format
    img = np.transpose(img, (2, 0, 1))  # Shape: (3, 112, 112)
    return img

# Buffer to store the last N frames
frame_buffer = deque(maxlen=sequence_length)

# Initialize camera and display
camera = jetson.utils.gstCamera(1280, 720, "csi://0")
display = jetson.utils.glDisplay()

#print("Press 'q' to quit the live feed.")

# Variables for performance measurement
total_inference_time = 0.0
inference_count = 0
frame_count = 0
cpu_usage = []
memory_usage = []
fps_list = []

# Initialize font for overlay text
font = jetson.utils.cudaFont()

while display.IsOpen():
    # Start time for FPS calculation
    loop_start_time = time.time()

    # Monitor system resources before processing
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB

    # Capture frame from Jetson camera
    img, width, height = camera.CaptureRGBA()
    
    frame_count += 1
    
    preprocessed_frame = preprocess_frame(img)
    frame_buffer.append(preprocessed_frame)


    if len(frame_buffer) == sequence_length:
        # Stack frames into the required input shape for inference
        input_blob = np.stack(frame_buffer, axis=0)  # Shape: (16, 3, 112, 112)
        input_blob = np.expand_dims(input_blob, axis=0)  # Shape: (1, 16, 3, 11>
        input_blob = np.transpose(input_blob, (0, 2, 1, 3, 4))  # Shape: (1, 3,>

        # Measure inference time
        inference_start_time = time.time()
        outputs = session.run([output_name], {input_name: input_blob})[0]
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        # Accumulate inference time and count
        total_inference_time += inference_time
        inference_count += 1

        # Get the top class prediction
        class_idx = np.argmax(outputs, axis=1)[0]
        confidence = outputs[0, class_idx]
        class_label = labels[class_idx] if class_idx < len(labels) else "unknow>

        # Display the prediction on the frame
        text = f"Action: {class_label} ({confidence:.2f})"
        font.OverlayText(img, width, height, text, 5, 5, font.White, font.Gray4>

        # Display inference time on the frame
        inf_time_text = f"Inference Time: {inference_time * 1000:.2f} ms"
        font.OverlayText(img, width, height, inf_time_text, 5, 35, font.White, >

    # Monitor system resources after processing
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().used / (1024 ** 2)   # Convert to MB
    
    cpu_usage.append(cpu_after)
    memory_usage.append(memory_after - memory_before)

    # Calculate FPS 
    loop_end_time = time.time()
    fps = 1.0 / (loop_end_time - loop_start_time)
    
    fps_list.append(fps)

    fps_text = f"FPS: {fps:.2f}"
    font.OverlayText(img, width, height, fps_text, 5, 65, font.White, font.Gray>

    # Show the frame
    display.RenderOnce(img, width, height)
    display.SetTitle("Action Recognition")

    # Check for quit
    if not display.IsOpen():
        break

# Calculate average metrics 
if inference_count > 0:
    avg_inference_time = (total_inference_time / inference_count) * 1000 
    avg_fps = sum(fps_list) / len(fps_list) 
    avg_cpu = sum(cpu_usage) / len(cpu_usage) 
    avg_memory = sum(memory_usage) / len(memory_usage) 
   
    print(f"Average Inference Time: {avg_inference_time:.2f} ms") 
    print(f"Average FPS: {avg_fps:.2f}") 
    print(f"Average CPU Usage: {avg_cpu:.2f}%") 
    print(f"Average Memory Usage: {avg_memory:.2f} MB")

# Clean up 
camera.Close()
display.Close()

