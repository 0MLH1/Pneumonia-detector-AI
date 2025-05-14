from inference import get_model
import supervision as sv
import cv2

# define the image url to use for inference
image_file = "IM-0119-0001.jpeg"
image = cv2.imread(image_file)

# Add your API key here
api_key = "z9NqQxVyuOmHDt0GxxUN"

# Include the api_key parameter
model = get_model(model_id="chest-pneomonia/1", api_key=api_key)

results = model.infer(image)
# Print only the prediction class and confidence
prediction = results[0].predictions[0]
print(f"Prediction: {prediction.class_name} with confidence: {prediction.confidence}")
