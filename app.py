import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Futuristic HTML Template with Glassmorphic Design
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Predictor | AI Cost Estimation</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #050814;
            --card-bg: rgba(16, 23, 42, 0.75);
            --neon-cyan: #00f2fe;
            --neon-blue: #4facfe;
            --neon-purple: #9b51e0;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-glow: rgba(0, 242, 254, 0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(79, 172, 254, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(155, 81, 224, 0.1) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 600px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: 20px;
            box-shadow: 0 0 40px rgba(0, 242, 254, 0.15), inset 0 0 15px rgba(255, 255, 255, 0.05);
            padding: 40px;
            position: relative;
            overflow: hidden;
        }

        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: 2px;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
            letter-spacing: 0.5px;
        }

        .grid-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 1px;
            color: var(--neon-cyan);
            text-transform: uppercase;
        }

        input, select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(5, 8, 20, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--neon-cyan);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.3);
            background: rgba(5, 8, 20, 0.9);
        }

        select option {
            background: #0f172a;
            color: var(--text-main);
        }

        button {
            grid-column: span 2;
            margin-top: 10px;
            padding: 14px;
            width: 100%;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            color: #050814;
            font-family: 'Orbitron', sans-serif;
            font-weight: 800;
            font-size: 1rem;
            letter-spacing: 1.5px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0, 242, 254, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 242, 254, 0.6);
            filter: brightness(1.1);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 242, 254, 0.05);
            border: 1px dashed var(--neon-cyan);
            border-radius: 12px;
            text-align: center;
            animation: pulseGlow 2s infinite alternate;
        }

        .result-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        .result-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: var(--neon-cyan);
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 10px rgba(0, 242, 254, 0.1); }
            100% { box-shadow: 0 0 25px rgba(0, 242, 254, 0.3); }
        }

        @media (max-width: 600px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NEURAL PREDICTOR</h1>
            <p class="subtitle">AI-Powered Linear Regression Interface</p>
        </header>

        <form method="POST" class="grid-form">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required min="1" max="120" placeholder="e.g. 30" value="{{ request.form.get('age', '') }}">
            </div>

            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex" required>
                    <option value="" disabled selected>Select Sex</option>
                    <option value="1" {% if request.form.get('sex') == '1' %}selected{% endif %}>Male (1)</option>
                    <option value="0" {% if request.form.get('sex') == '0' %}selected{% endif %}>Female (0)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="bmi">BMI</label>
                <input type="number" step="0.1" id="bmi" name="bmi" required placeholder="e.g. 24.5" value="{{ request.form.get('bmi', '') }}">
            </div>

            <div class="form-group">
                <label for="children">Children</label>
                <input type="number" id="children" name="children" required min="0" max="10" placeholder="e.g. 2" value="{{ request.form.get('children', '') }}">
            </div>

            <div class="form-group">
                <label for="smoker">Smoker</label>
                <select id="smoker" name="smoker" required>
                    <option value="" disabled selected>Smoking Status</option>
                    <option value="1" {% if request.form.get('smoker') == '1' %}selected{% endif %}>Yes (1)</option>
                    <option value="0" {% if request.form.get('smoker') == '0' %}selected{% endif %}>No (0)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="region">Region Code</label>
                <select id="region" name="region" required>
                    <option value="" disabled selected>Select Region</option>
                    <option value="0" {% if request.form.get('region') == '0' %}selected{% endif %}>Southwest (0)</option>
                    <option value="1" {% if request.form.get('region') == '1' %}selected{% endif %}>Southeast (1)</option>
                    <option value="2" {% if request.form.get('region') == '2' %}selected{% endif %}>Northwest (2)</option>
                    <option value="3" {% if request.form.get('region') == '3' %}selected{% endif %}>Northeast (3)</option>
                </select>
            </div>

            <button type="submit">EXECUTE PREDICTION</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <div class="result-title">ESTIMATED VALUE RESULT</div>
            <div class="result-value">${{ "%.2f"|format(prediction) }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features matching the model's feature order: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['bmi']),
                float(request.form['children']),
                float(request.form['smoker']),
                float(request.form['region'])
            ]
            
            if model:
                # Reshape array for single prediction
                final_features = np.array([features])
                prediction = model.predict(final_features)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            
    return render_template_string(TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
