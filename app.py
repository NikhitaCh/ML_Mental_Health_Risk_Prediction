from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# We Load trained ML model
with open('mental_health_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        usage_level = int(request.form['usage_level'])
        anxiety = int(request.form['anxiety'])
        low_self_esteem = int(request.form['low_self_esteem'])
        depression = int(request.form['depression'])
        social_isolation = int(request.form['social_isolation'])
        social_connection = int(request.form['social_connection'])
        positive_self_image = int(request.form['positive_self_image'])

        # Convert to numpy array
        features = np.array([[
            usage_level,
            anxiety,
            low_self_esteem,
            depression,
            social_isolation,
            social_connection,
            positive_self_image
        ]])

        # Predict risk level
        prediction = model.predict(features)[0]
        result = prediction

        # Color coding
        if result == 'Low Risk':
            result_class = 'low-risk'
        elif result == 'Medium Risk':
            result_class = 'medium-risk'
        else:
            result_class = 'high-risk'

        return render_template(
            'index.html',
            prediction=result,
            result_class=result_class
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction=f'Error: {str(e)}',
            result_class='high-risk'
        )


# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
