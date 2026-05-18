import tensorflow as tf
import numpy as np
from PIL import Image
import io
import builtins  

# Parche absoluto para que la capa Lambda no busque a ciegas
builtins.tf = tf

# Forzar a TensorFlow a no pelear por la tarjeta gráfica
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

def parche_forma_lambda(self, input_shape):
    return input_shape
tf.keras.layers.Lambda.compute_output_shape = parche_forma_lambda

class AnalizadorCultivos:
    
    def __init__(self, ruta_modelo):
        print("Cargando modelo de IA...")
        # Cargamos el modelo normal y corriente
        self.modelo = tf.keras.models.load_model(ruta_modelo, safe_mode=False, compile=False)
        
        self.clases = {
            0: 'Enfermedad, Raiz roja de la caña', 1: 'azucar_rust', 2: 'azucar_sana', 
            3: 'azucar_yellow', 4: 'platano_cordana', 5: 'platano_panama', 
            6: 'platano_sana', 7: 'platano_sigatoka'
        }
        
        self.tratamientos = {
            'Enfermedad, Raiz roja de la caña': 'Tratamiento sugerido: Eliminar plantas severamente infectadas, mejorar el drenaje del suelo y aplicar fungicidas.',
            'azucar_rust': 'Tratamiento sugerido: Usar fungicidas específicos para la roya y asegurar una buena circulación de aire.',
            'azucar_sana': '¡Excelente! La planta de caña está sana. Mantén las buenas prácticas de riego.',
            'azucar_yellow': 'Tratamiento sugerido: Controlar los pulgones vectores del virus y desinfectar herramientas de corte.',
            'platano_cordana': 'Tratamiento sugerido: Remover las hojas más afectadas y aplicar fungicidas a base de cobre.',
            'platano_panama': 'Alerta (Mal de Panamá): Enfermedad muy grave. Aislar la zona, destruir plantas infectadas y desinfectar herramientas.',
            'platano_sana': '¡Excelente! La planta de plátano está sana. Mantén el monitoreo preventivo.',
            'platano_sigatoka': 'Tratamiento sugerido: Realizar poda sanitaria (deshoje) y aplicar fungicidas sistémicos y protectantes.'
        }

    def analizar_planta(self, archivo_imagen):
        # Procesamos la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read()))
        imagen = imagen.resize((224, 224)) 
        arreglo_imagen = tf.keras.preprocessing.image.img_to_array(imagen)
        arreglo_imagen = np.expand_dims(arreglo_imagen, axis=0)
        
        # Hacemos la predicción directamente
        predicciones = self.modelo.predict(arreglo_imagen)
        indice = int(np.argmax(predicciones))
        probabilidad = float(np.max(predicciones))
        
        nombre_enfermedad = self.clases.get(indice, "Desconocida")
        tratamiento_sugerido = self.tratamientos.get(nombre_enfermedad, "Consultar a un especialista.")
        
        return {
            'enfermedad': nombre_enfermedad,
            'probabilidad': round(probabilidad * 100, 2),
            'tratamiento': tratamiento_sugerido
        }