from flask import Flask, request, render_template
from analizador import AnalizadorCultivos

app = Flask(__name__)

# Cargamos el analizador una sola vez de forma global al iniciar
print("=== INICIANDO CONFIGURACIÓN ===")
mi_analizador = AnalizadorCultivos('modelo_plantas.keras')
print("=== SERVIDOR LISTO ===")

@app.route('/', methods=['GET', 'POST'])
def inicio():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        if 'imagen' not in request.files:
            return render_template('index.html', error="No se seleccionó ninguna imagen.")
        
        archivo = request.files['imagen']
        if archivo.filename == '':
            return render_template('index.html', error="El archivo está vacío.")
        
        try:
            # Llamamos al método directamente en el mismo hilo de ejecución, sin subprocesos
            datos_resultado = mi_analizador.analizar_planta(archivo)
            return render_template('index.html', resultado=datos_resultado)
            
        except Exception as e:
            return render_template('index.html', error=f"Ocurrió un problema: {str(e)}")

if __name__ == '__main__':
    # Ejecutamos Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)