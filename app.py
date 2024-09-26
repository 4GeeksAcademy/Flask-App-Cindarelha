from flask import Flask, request, render_template
import joblib

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load("svm_spam_detector.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Ruta principal para mostrar el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para procesar el formulario
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener la URL del formulario
        url = request.form['url']
        
        # Transformar la URL usando el vectorizador
        url_vectorized = vectorizer.transform([url])
        
        # Realizar la predicción
        prediction = model.predict(url_vectorized)
        
        # Determinar el resultado
        if prediction[0] == 1:
            result = "La URL es Spam!"
        else:
            result = "La URL no es Spam."
        
        # Renderizar la plantilla con el resultado
        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)