import torch
import cv2

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set model to eval mode
model.eval()

# Load input image
image_path = r'C:\SK\Python\int_task\sample.jpg'  
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Run inference
results = model(image)

# Extract detections
detections = results.pandas().xyxy[0]  

# Check for detections
if detections.empty:
    print("No objects detected in the image.")
else:
    # Annotate image with bounding boxes and labels
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # Put label
        text = f"{label} {confidence:.2f}"
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save output
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)
    print(f"Detection completed. Annotated image saved as '{output_path}'")
