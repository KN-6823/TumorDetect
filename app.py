from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os

app = Flask(__name__)

model_path = 'model/model.tflite'  

# Load the TFLite model
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  
    print("TFLite model loaded successfully.")
except Exception as e:
    print("Error loading TFLite model:", e)
    interpreter = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if interpreter is None:
            print("Error: Model not loaded.")
            return jsonify({'error': 'Model not loaded'}), 500

        if 'image' not in request.files:
            print("Error: No image found in request.")
            return jsonify({'error': 'No image file in request'}), 400

        img_data = request.files['image'].read()
        print("Image data received, size:", len(img_data))

        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            print("Error: Image could not be decoded.")
            return jsonify({'error': 'Invalid image format'}), 400

        print("Image successfully decoded. Shape:", img_array.shape)

        # Preprocess the image
        img_array = cv2.resize(img_array, (150, 150))  
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0) 

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the prediction result (probability)
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"Prediction result: {prediction}")

        # Calculate the probability
        probability = float(prediction[0])
        print(f"Probability of Tumor Detection: {probability:.2f}")

        # Determine the result based on the probability
        result = 'Tumor Detected' if probability > 0.5 else 'No Tumor Detected'

        # Return both the result and the probability
        return jsonify({
            'result': result,
            'probability': f'{probability * 100:.2f}%'  # return as percentage
        })

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/')
def index():
    return render_template('index.html', name='World')

@app.route('/<name>')
def home(name):
    return render_template('index.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)














# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# from tflite_runtime.interpreter import Interpreter
# import os

# app = Flask(__name__)

# model_path = 'model/model.tflite'

# # Load the model
# try:
#     model = Interpreter(model_path)
#     model.allocate_tensors()
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Error loading model:", e)
#     # Log the error and handle it gracefully
#     model = None

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if model is None:
#             return jsonify({'error': 'Model not loaded'}), 500

#         # return jsonify({'result': "Predicting..."})

#         # Receive image data from the request
#         img_data = request.files['image'].read()
        
#         # Process the image data
#         img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
#         img_array = cv2.resize(img_array, (150, 150))
#         img_array = img_array / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         # Check if FLOAT32 or 64:
#         if img_array.dtype != np.float32:
#             img_array = img_array.astype(np.float32)
#         # Set the tensor (model input)
#         input_details = model.get_input_details()
#         model.set_tensor(input_details[0]['index'], img_array)
        
#         # Run the model
#         model.invoke()
        
#         # Get the tensor (model output)
#         output_details = model.get_output_details()
#         prediction = model.get_tensor(output_details[0]['index'])
        
#         result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
        
#         # Return prediction result
#         return jsonify({'result': result})
#     except Exception as e:
#         print("Error during prediction:", e)
#         return jsonify({'error': 'An error occurred during prediction'}), 500

# @app.route('/')
# def index():
#     return render_template('index.html', name='World')

# @app.route('/<name>')
# def home(name):
#     return render_template('index.html', name=name)

# if __name__ == '__main__':
#     app.run(debug=True)
