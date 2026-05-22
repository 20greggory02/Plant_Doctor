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
        
        # SUPER-DICCIONARIO: Frases amigables y recomendaciones dobles
        self.info_clases = {
            0: {
                'nombre': 'La hoja presenta una enfermedad, es Raíz Roja de la caña.',
                'principal': 'Aplicar fungicidas sistémicos específicos y eliminar las plantas severamente infectadas para evitar propagación.',
                'alternativa': 'Mejorar urgentemente el drenaje del suelo y realizar surcos para que el agua no se estanque en las raíces.'
            },
            1: {
                'nombre': 'La hoja presenta una enfermedad, es Roya en la caña de azúcar.',
                'principal': 'Usar fungicidas específicos para la roya (triazoles) aplicados por un profesional agrícola.',
                'alternativa': 'Realizar un deshoje de las partes más afectadas para asegurar una buena circulación de aire y reducir la humedad.'
            },
            2: {
                'nombre': 'La hoja corresponde a una Caña de Azúcar completamente sana. ¡Excelente!',
                'principal': 'Mantener las buenas prácticas de riego y el cronograma habitual de fertilización.',
                'alternativa': 'Asegurar limpieza constante de malezas para evitar que compitan por los nutrientes.'
            },
            3: {
                'nombre': 'La hoja presenta una anomalía, es Síndrome de la Hoja Amarilla (Yellow Leaf).',
                'principal': 'Desinfectar rigurosamente todas las herramientas de corte y usar semillas certificadas libres de virus para el próximo ciclo.',
                'alternativa': 'Controlar los pulgones (vectores del virus) usando trampas cromáticas o insecticidas de contacto económicos.'
            },
            4: {
                'nombre': 'La hoja presenta una enfermedad, es Mancha de Cordana en el plátano.',
                'principal': 'Aplicar fungicidas a base de cobre o mancozeb siguiendo un calendario de rotación para evitar resistencia.',
                'alternativa': 'Remover y destruir (quemar o enterrar) las hojas más afectadas para detener la fuente de infección sin gastar en químicos.'
            },
            5: {
                'nombre': '¡Alerta! La planta presenta Mal de Panamá (Fusariosis), una enfermedad muy grave.',
                'principal': 'Aplicar estricta cuarentena. Aislar la zona, destruir plantas infectadas de raíz y evitar el tránsito de maquinaria en esa área.',
                'alternativa': 'Desinfectar botas y herramientas con cloro; no sembrar plátano en ese suelo por mucho tiempo y usar variedades resistentes.'
            },
            6: {
                'nombre': 'La hoja corresponde a una planta de Plátano completamente sana. ¡Buen trabajo!',
                'principal': 'Mantener el monitoreo preventivo y el plan de nutrición con potasio, que es esencial para el plátano.',
                'alternativa': 'Realizar labores culturales rutinarias como el control de malezas y deshoje de hojas secas naturales.'
            },
            7: {
                'nombre': 'La hoja presenta una enfermedad, es Sigatoka Negra en el plátano.',
                'principal': 'Aplicar fungicida sistémico específico alternando con protectores, guiado por un agrónomo.',
                'alternativa': 'Realizar un deshoje sanitario inmediato (cortar las partes enfermas) para reducir las esporas radicalmente de forma gratuita.'
            }
        }

    def analizar_planta(self, archivo_imagen):
        # Procesamos la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read()))
        imagen = imagen.resize((224, 224)) 
        arreglo_imagen = tf.keras.preprocessing.image.img_to_array(imagen)
        arreglo_imagen = np.expand_dims(arreglo_imagen, axis=0)
        
        # Hacemos la predicción
        predicciones = self.modelo.predict(arreglo_imagen)
        indice = int(np.argmax(predicciones))
        probabilidad = float(np.max(predicciones))
        
        # Extraemos la información del diccionario
        info = self.info_clases.get(indice, {
            'nombre': 'Enfermedad no reconocida.',
            'principal': 'Consultar con un experto agrónomo.',
            'alternativa': 'Aislar la planta preventivamente.'
        })
        
        # Enviamos las variables exactas que tu HTML está esperando ahora
        return {
            'enfermedad': info['nombre'],
            'probabilidad': round(probabilidad * 100, 2),
            'tratamiento_principal': info['principal'],
            'tratamiento_alternativo': info['alternativa']
        }