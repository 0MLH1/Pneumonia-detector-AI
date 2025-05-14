from flask import Flask, render_template, request
from inference import get_model
import cv2

app = Flask(__name__)

# Add your API key here
api_key = "z9NqQxVyuOmHDt0GxxUN"
model = get_model(model_id="chest-pneomonia/1", api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the image file from the form
        file = request.files['image']
        # Save the image temporarily
        image_path = "uploaded_image.jpg"
        file.save(image_path)

        # Load the image and run inference
        image = cv2.imread(image_path)
        results = model.infer(image)
        
         # Get the prediction and confidence
        prediction = results[0].predictions[0]
        predicted_class = prediction.class_name
        confidence = prediction.confidence

        return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
