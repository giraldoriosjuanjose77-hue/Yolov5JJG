import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning(f"Intentando método alternativo de carga...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar versiones compatibles:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta.
        """)
        return None


# --- Interfaz principal ---
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes capturadas con tu cámara o cargadas desde tu dispositivo.
""")

# --- Cargar el modelo ---
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# --- Configuración de parámetros ---
if model:
    st.sidebar.title("Parámetros de configuración")

    st.sidebar.subheader("📷 Fuente de imagen")
    input_option = st.sidebar.radio("Selecciona cómo ingresar la imagen:", ["📸 Cámara", "🖼️ Subir imagen"])

    st.sidebar.subheader("🎛️ Parámetros del modelo")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    st.sidebar.subheader("⚙️ Opciones avanzadas")
    try:
        model.agnostic = st.sidebar.checkbox('NMS class-agnostic', False)
        model.multi_label = st.sidebar.checkbox('Múltiples etiquetas por caja', False)
        model.max_det = st.sidebar.number_input('Detecciones máximas', 10, 2000, 1000, 10)
    except:
        st.sidebar.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    # --- Captura o carga de imagen ---
    st.markdown("---")
    st.subheader("📸 Captura o carga de imagen")

    if input_option == "📸 Cámara":
        img_input = st.camera_input("Toma una foto")
    else:
        img_input = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    # --- Procesamiento ---
    if img_input:
        bytes_data = img_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        # --- Mostrar resultados ---
        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Imagen con detecciones")
                results.render()
                st.image(results.ims[0], channels='BGR', use_container_width=True)

            with col2:
                st.subheader("📊 Objetos detectados")

                label_names = model.names
                category_count = {}
                for category in categories:
                    idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[idx] = category_count.get(idx, 0) + 1

                data = []
                for idx, count in category_count.items():
                    label = label_names[idx]
                    confidence = scores[categories == idx].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza promedio": f"{confidence:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index('Categoría')['Cantidad'])
                else:
                    st.info("No se detectaron objetos con los parámetros actuales.")
                    st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
        except Exception as e:
            st.error(f"Error al procesar resultados: {str(e)}")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# --- Pie de página ---
st.markdown("---")
st.caption("""
**Acerca de la aplicación:** Esta app usa YOLOv5 para detección de objetos en tiempo real.  
Desarrollada con Streamlit y PyTorch.
""")
