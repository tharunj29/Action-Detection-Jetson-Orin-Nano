import jetson.inference
import jetson.utils
import time
import psutil
import numpy as np

# Load the action recognition network
action = jetson.inference.actionNet("resnet18-kinetics")

# Create the camera and display
camera = jetson.utils.gstCamera(1920, 1080, "csi://0")
display = jetson.utils.glDisplay()

# Set up frame rate limiting
frame_interval = 1.0 / 30.0  # 30 FPS
last_frame_time = time.time()

# Performance tracking variables
total_inference_time = 0.0
inference_count = 0
frame_count = 0
cpu_usage = []
memory_usage = []
fps_list = []

try:
    while display.IsOpen():
        # Start time for FPS and performance tracking
        loop_start_time = time.time()

        # Monitor system resources before processing
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().used / (1024 ** 2)  # Convert t>

        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            last_frame_time = current_time
            frame_count += 1

            # Capture the image
            img, width, height = camera.CaptureRGBA()

            # Performance tracking
            action_start_time = time.time()

            # Perform action classification directly on the frame
            top_action = action.Classify(img)

            # Extract class ID and get the description
            class_id = top_action[0]
            action_desc = action.GetClassDesc(class_id)

            # Print the inference result to the console
            #print(f"Frame {frame_count}: Action Detected - {action_desc}")

            # Overlay action description in the center of the frame
            font = jetson.utils.cudaFont()
            font.OverlayText(img, width, height, 
                             f"Action: {action_desc}", 
                             width // 2, height // 10, font.White, font.Gray40)

            # Calculate performance metrics
            action_end_time = time.time()
            action_inference_time = action_end_time - action_start_time

            # Update metrics
            total_inference_time += action_inference_time
            inference_count += 1
            fps = 1.0 / (action_inference_time if action_inference_time > 0 els>
            fps_list.append(fps)
            cpu_usage.append(psutil.cpu_percent(interval=None))
            memory_usage.append(psutil.virtual_memory().used / (1024 ** 2))

            # Render the image to display
            display.RenderOnce(img, width, height)

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

finally:
    # Release resources properly
    camera.Close()

