import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from PIL import Image
import os
import streamlit.components.v1 as components

# Try to import TensorFlow/Keras with fallback
try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model
        MODEL_AVAILABLE = True
    except ImportError:
        MODEL_AVAILABLE = False
        st.error("⚠️ TensorFlow/Keras not available. Disease detection disabled.")

# --- High Contrast & Readable Styles ---
st.markdown("""
    <style>
        body, .stApp { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .publisher {
            font-family: 'Montserrat', sans-serif;
            color: #ffffff;
            font-weight: 900;
            letter-spacing: 4px;
            text-align: center;
            font-size: 2.5rem;
            text-shadow: 3px 3px 8px rgba(0,0,0,0.4);
            margin-bottom: 10px;
        }
        .branded-title {
            font-family: 'Montserrat', sans-serif;
            color: #ffffff;
            font-weight: 900;
            letter-spacing: 2px;
            text-align: center;
            font-size: 2.8rem;
            text-shadow: 2px 2px 12px rgba(0,0,0,0.5);
        }
        .subtitle {
            color: #f0f0f0;
            font-size: 1.4rem;
            text-align: center;
            font-weight: 600;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
        }
        .data-block {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            padding: 20px;
            margin-bottom: 28px;
        }
        .welcome {
            font-size: 1.25rem; 
            color: #2d3748;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 14px 22px;
            margin-bottom: 18px;
            font-weight: 600;
        }
        .big-sub { 
            color: #5a67d8; 
            font-weight: 700; 
            margin-bottom: 12px; 
            font-size: 1.4rem;
            text-align: center;
        }
        .remedy-highlight { 
            color: #22543d; 
            background: #c6f6d5; 
            padding: 10px 16px; 
            border-radius: 10px; 
            display: inline-block;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .detected-block {
            color: #742a2a; 
            background: #fed7d7;
            border-radius: 10px; 
            font-size: 1.2rem;
            padding: 10px 16px; 
            margin: 12px 0;
            display: inline-block;
            font-weight: 700;
        }
        .centered { text-align: center; }
        .footer-brand {
            color: #ffffff;
            font-size: 1.3rem;
            font-weight: 800;
            text-align: center;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='publisher'>Made by GROUP-20</div>", unsafe_allow_html=True)

st.markdown(
    "<h1 class='branded-title'>🌿 Smart Farm Monitoring Dashboard 🌿</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle' style='font-size:1.15rem; margin-top:8px; margin-bottom:20px;'>"
    "Monitor your farm, analyze leaf health, and get instant remedies. Published by <b>DEVRAJ GIRI</b>."
    "</div>", unsafe_allow_html=True)

st.markdown("<div class='welcome'>Welcome, <b>GROUP-20</b>! Upload a leaf image and environment readings below:</div>", unsafe_allow_html=True)

# Supabase config with error handling
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://qgsthykitdzbxopqlolk.supabase.co")
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFnc3RoeWtpdGR6YnhvcHFsb2xrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIwMTQxNDAsImV4cCI6MjA3NzU5MDE0MH0.P_GLW_mTE1bAjlABIiZdC7YY7dD0zCxcf0H9pdistQs")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

# Load disease info with error handling
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
except FileNotFoundError:
    st.warning("⚠️ disease_info.json not found. Using default remedies.")
    disease_info = {
        "Healthy": {"remedy": "Your plant is healthy. Keep providing adequate water and sunlight."},
        "Powdery Mildew": {"remedy": "Use a fungicide spray and ensure proper air circulation."},
        "Leaf Spot": {"remedy": "Remove infected leaves and avoid overhead watering."}
    }

# Load model with error handling
model = None
class_names = ["Healthy", "Powdery Mildew", "Leaf Spot"]

if MODEL_AVAILABLE:
    try:
        model_path = "model/plant_disease_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.warning(f"⚠️ Model file not found at {model_path}. Disease detection will be simulated.")
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")

# Sensor Data UI
c1, c2, c3 = st.columns(3)
with c1:
    temperature = st.slider("🌡 Temperature (°C)", 0, 100, 25)
with c2:
    humidity = st.slider("💧 Humidity (%)", 0, 100, 45)
with c3:
    moisture = st.slider("🌱 Soil Moisture (%)", 0, 100, 55)

uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded by DEVRAJ GIRI', width=340)

if st.button("🔎 Analyze & Submit Data", type="primary"):
    try:
        disease, prediction_score, remedy = None, None, None
        
        if uploaded_file:
            if model is not None:
                # Real prediction
                image = Image.open(uploaded_file).resize((128, 128))
                img_array = np.array(image).reshape((1, 128, 128, 3)) / 255.0
                prediction = model.predict(img_array)
                disease = class_names[np.argmax(prediction)]
                prediction_score = float(np.max(prediction))
            else:
                # Simulated prediction (for demo without model)
                import random
                disease = random.choice(class_names)
                prediction_score = random.uniform(0.7, 0.95)
                st.info("ℹ️ Using simulated prediction (model not available)")
            
            remedy = disease_info.get(disease, {}).get("remedy", "No remedy information available.")
            
            st.markdown(
                f"<div class='detected-block centered'>🍃 Detected Disease: <b>{disease}</b> (Confidence: {prediction_score:.2f})</div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='remedy-highlight centered'><b>💊 Remedy:</b> {remedy}</div>",
                unsafe_allow_html=True)
            
            # Falling leaves animation
            components.html("""
                <div id="leaf-container"></div>
                <style>
                    #leaf-container {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        pointer-events: none;
                        z-index: 9999;
                    }
                    .leaf {
                        position: absolute;
                        font-size: 40px;
                        animation: fall linear forwards;
                    }
                    @keyframes fall {
                        to {
                            transform: translateY(100vh) rotate(360deg);
                            opacity: 0;
                        }
                    }
                </style>
                <script>
                    const container = document.getElementById('leaf-container');
                    const leafEmojis = ['🍃', '🌿', '🍀', '🌱'];
                    for (let i = 0; i < 30; i++) {
                        const leaf = document.createElement('div');
                        leaf.className = 'leaf';
                        leaf.textContent = leafEmojis[Math.floor(Math.random() * leafEmojis.length)];
                        leaf.style.left = Math.random() * 100 + '%';
                        leaf.style.animationDuration = (Math.random() * 3 + 2) + 's';
                        leaf.style.animationDelay = (Math.random() * 2) + 's';
                        container.appendChild(leaf);
                    }
                    setTimeout(() => container.innerHTML = '', 6000);
                </script>
            """, height=0)
        
        # Save to Supabase
        if supabase:
            record = {
                "temperature": temperature,
                "humidity": humidity,
                "moisture": moisture,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "disease": disease,
                "confidence": prediction_score,
                "publisher": "DEVRAJ GIRI"
            }
            response = supabase.table("sensor_data").insert(record).execute()
            if response.data:
                st.success("✅ Data uploaded for GROUP-20!", icon="🌱")
            else:
                st.error(f"❌ Upload failed for publisher: DEVRAJ GIRI.")
        else:
            st.warning("⚠️ Supabase not connected. Data not saved.")
            
    except Exception as e:
        st.error(f"⚠️ Error (DEVRAJ GIRI): {e}")

st.markdown("<div class='data-block'><div class='big-sub'>📊 Live Monitoring Charts </div></div>", unsafe_allow_html=True)

if supabase:
    try:
        data = supabase.table("sensor_data").select("*").order("id", desc=True).limit(100).execute()
        if data.data:
            df = pd.DataFrame(data.data)
            st.markdown("<div class='data-block'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1.15rem; color:#2d3748; margin-bottom:10px; text-align:center; font-weight:600;'><b>📋 Data Table</b></div>", unsafe_allow_html=True)
            st.dataframe(df, height=300)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='data-block'>", unsafe_allow_html=True)
            st.markdown("<div class='big-sub'>📈 Temperature, Humidity, Moisture Trends</div>", unsafe_allow_html=True)
            st.line_chart(df.set_index("id")[["temperature", "humidity", "moisture"]])
            st.markdown("</div>", unsafe_allow_html=True)

            if "disease" in df.columns:
                st.markdown("<div class='data-block'>", unsafe_allow_html=True)
                st.markdown("<div class='big-sub'>📊 Disease Distribution</div>", unsafe_allow_html=True)
                st.bar_chart(df["disease"].value_counts())
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No data uploaded yet. Publisher: DEVRAJ GIRI.")
    except Exception as e:
        st.error(f"⚠️ Could not fetch data for DEVRAJ GIRI: {e}")
else:
    st.warning("⚠️ Supabase not connected. Cannot display live data.")

# Footer brand
st.markdown("""
    <hr style='border:1px solid rgba(255,255,255,0.3); margin-top:40px;'/>
    <div class='footer-brand' style='margin-top:20px; margin-bottom:20px;'>
        Made with 💙 by <b>GROUP-20</b> — Smart Farm Solutions
        <br><span style='font-size:1rem;'>All rights reserved DEVRAJ GIRI</span>
    </div>
""", unsafe_allow_html=True)

st.caption("You may manually refresh the page to update data. Publisher: DEVRAJ GIRI.")
