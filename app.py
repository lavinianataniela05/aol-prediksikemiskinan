import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kemiskinan Indonesia",
    page_icon="📊",
    layout="wide"
)

def load_model():
    """Load model dan preprocessing artifacts"""
    try:
        model_artifacts = joblib.load('poverty_prediction_model.pkl')
        return model_artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_poverty_rate(input_data, model_artifacts):
    """Fungsi untuk memprediksi persentase kemiskinan"""
    try:
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        le_provinsi = model_artifacts['label_encoder_provinsi']
        le_kabupaten = model_artifacts['label_encoder_kabupaten']
        features = model_artifacts['features']
        
        input_df = pd.DataFrame([input_data])
        
        # Encode provinsi dan kabupaten
        if 'Provinsi' in input_df.columns:
            input_df['Provinsi_encoded'] = le_provinsi.transform(input_df['Provinsi'])
        if 'Kabupaten/Kota' in input_df.columns:
            input_df['Kabupaten_encoded'] = le_kabupaten.transform(input_df['Kabupaten/Kota'])
        
        # Urutkan kolom sesuai dengan features
        input_processed = input_df[features]
        
        # Scale features
        input_scaled = scaler.transform(input_processed)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return prediction
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        return None

def main():
    # Header
    st.title("🏠 Prediksi Persentase Kemiskinan Indonesia")
    
    # Load model
    model_artifacts = load_model()
    if model_artifacts is None:
        st.stop()
    
    # Input data
    st.sidebar.header("Input Data")
    
    provinsi_options = ['ACEH', 'BALI', 'BANTEN', 'JAWA BARAT', 'JAWA TENGAH', 'JAWA TIMUR', 'DKI JAKARTA']
    
    kabupaten_options = {
        'ACEH': ['Aceh Barat', 'Aceh Besar', 'Kota Banda Aceh'],
        'JAWA BARAT': ['Bandung', 'Bogor', 'Kota Bandung'],
        'JAWA TENGAH': ['Semarang', 'Surakarta', 'Magelang'],
        'JAWA TIMUR': ['Surabaya', 'Malang', 'Kediri']
    }
    
    provinsi = st.sidebar.selectbox("Provinsi", provinsi_options)
    
    if provinsi in kabupaten_options:
        kabupaten_kota = st.sidebar.selectbox("Kabupaten/Kota", kabupaten_options[provinsi])
    else:
        kabupaten_kota = st.sidebar.text_input("Kabupaten/Kota")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        rata_sekolah = st.slider("Rata Sekolah (tahun)", 0.0, 15.0, 8.0, 0.1)
        pengeluaran = st.number_input("Pengeluaran per Kapita", 5000, 30000, 12000)
        umur_hidup = st.slider("Umur Harapan Hidup", 50.0, 80.0, 70.0, 0.1)
        ipm = st.slider("IPM", 50.0, 90.0, 70.0, 0.1)
    
    with col2:
        pengangguran = st.slider("Pengangguran (%)", 0.0, 15.0, 5.0, 0.1)
        partisipasi_kerja = st.slider("Partisipasi Kerja (%)", 50.0, 90.0, 70.0, 0.1)
        sanitasi = st.slider("Akses Sanitasi (%)", 0.0, 100.0, 80.0, 0.1)
        air_minum = st.slider("Akses Air Minum (%)", 0.0, 100.0, 85.0, 0.1)
    
    if st.sidebar.button("🚀 Prediksi Kemiskinan"):
        # Prepare input data
        input_data = {
            'Rata-rata Lama Sekolah': rata_sekolah,
            'Pengeluaran per Kapita': pengeluaran,
            'Umur Harapan Hidup (UHH)': umur_hidup,
            'Indeks Pembangunan Manusia (IPM)': ipm,
            'Tingkat Pengangguran Terbuka (TPT) - Agustus': pengangguran,
            'Tingkat Partisipasi Angkatan Kerja (TPAK) - Agustus': partisipasi_kerja,
            'Rumah Tangga yang Memiliki Akses Terhadap Sanitasi Layak': sanitasi,
            'Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak': air_minum,
            'Provinsi': provinsi,
            'Kabupaten/Kota': kabupaten_kota
        }
        
        # Prediction
        with st.spinner('Sedang memprediksi...'):
            prediction = predict_poverty_rate(input_data, model_artifacts)
        
        if prediction is not None:
            # Display prediction
            st.success(f"### Hasil Prediksi: {prediction:.2f}%")
            
            # Status kemiskinan
            if prediction < 5:
                status = "🟢 Sangat Rendah"
            elif prediction < 10:
                status = "🟡 Rendah"
            elif prediction < 15:
                status = "🟠 Sedang"
            elif prediction < 20:
                status = "🔴 Tinggi"
            else:
                status = "💀 Sangat Tinggi"
            
            st.info(f"**Tingkat Kemiskinan:** {status}")
            
            # Visualisasi sederhana
            fig, ax = plt.subplots(figsize=(8, 3))
            
            levels = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
            ranges = ['<5%', '5-10%', '10-15%', '15-20%', '>20%']
            colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
            
            # Tentukan posisi prediksi
            for i, level in enumerate(levels):
                ax.barh(level, 1, color=colors[i], alpha=0.7)
                ax.text(0.5, i, ranges[i], ha='center', va='center', fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_title('Tingkat Kemiskinan')
            ax.axis('off')
            
            # st.pyplot(fig)
            
            # Info model
            st.subheader("Informasi Model")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Akurasi (R²)", f"{model_artifacts['performance']['Test_R2']:.3f}")
            with col2:
                st.metric("Error (RMSE)", f"{model_artifacts['performance']['Test_RMSE']:.2f}%")
            with col3:
                st.metric("Algoritma", model_artifacts['performance']['Algorithm'])
    
    else:
        st.info("👈 Silakan isi data di sidebar dan klik 'Prediksi Kemiskinan'")

if __name__ == "__main__":
    main()