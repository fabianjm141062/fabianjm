import pandas as pd
import streamlit as st

# Load the data from Excel files
def load_data():
    uji_relevansi_df = pd.read_excel('RISIKO.xlsx', sheet_name='Uji Relevansi')
    probabilitas_df = pd.read_excel('RISIKO.xlsx', sheet_name='Probabilitas')
    dampak_df = pd.read_excel('RISIKO.xlsx', sheet_name='Dampak')
    return uji_relevansi_df, probabilitas_df, dampak_df

# Define scoring for each scale
probabilitas_scores = {'SJ': 0, 'J': 1, 'C': 2, 'S': 3, 'SS': 4}
dampak_scores = {'SR': 0, 'R': 1, 'S': 2, 'T': 3, 'ST': 4}

# Evaluate relevancy
def evaluate_relevancy(df):
    df['Total'] = df['SETUJU'] + df['TIDAK']
    df['Percentage'] = (df['SETUJU'] / df['Total']) * 100
    df['Keterangan'] = df['Percentage'].apply(lambda x: '+' if x > 50 else '-')
    return df

# Calculate scale
def calculate_scale(df, scale_scores, scale_type):
    df['SI (%)'] = df.apply(lambda row:
                            (sum(scale_scores.get(cat, 0) * row.get(cat, 0) for cat in scale_scores) /
                             (4 * sum(row.get(cat, 0) for cat in scale_scores))) * 100, axis=1)
    df[f'Kategori {scale_type}'] = pd.cut(df['SI (%)'],
                            bins=[0, 12.5, 37.5, 62.5, 87.5, 100],
                            labels=['Sangat Jarang', 'Jarang', 'Cukup', 'Sering', 'Sangat Sering'] if scale_type == 'Probabilitas'
                            else ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
                            right=True)
    return df

# Calculate severity index
def calculate_severity_index(prob_df, impact_df):
    risk_data = pd.merge(prob_df[['Kode Risiko', 'SI (%)', 'Kategori Probabilitas']],
                        impact_df[['Kode Risiko', 'SI (%)', 'Kategori Dampak']],
                        on='Kode Risiko', suffixes=('_P', '_I'))
    
    scale_map_probabilitas = {'Sangat Jarang': 1, 'Jarang': 2, 'Cukup': 3, 'Sering': 4, 'Sangat Sering': 5}
    scale_map_dampak = {'Sangat Rendah': 1, 'Rendah': 2, 'Sedang': 3, 'Tinggi': 4, 'Sangat Tinggi': 5}

    risk_data['Skala_P'] = risk_data['Kategori Probabilitas'].map(scale_map_probabilitas).astype(float)
    risk_data['Skala_I'] = risk_data['Kategori Dampak'].map(scale_map_dampak).astype(float)

    risk_data['P x I'] = risk_data['Skala_P'] * risk_data['Skala_I']

    def categorize_risk(p_x_i):
        if p_x_i >= 15:
            return 'High Risk'
        elif 6 <= p_x_i < 15:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    risk_data['Kategori Risiko'] = risk_data['P x I'].apply(categorize_risk)

    return risk_data

# Streamlit app
def main():
    st.title("Analisis Risiko Infrastruktur")

    # Load data
    uji_relevansi_df, probabilitas_df, dampak_df = load_data()

    st.sidebar.header("Pengaturan")

    if st.sidebar.button("Update Data"):
        uji_relevansi_results = evaluate_relevancy(uji_relevansi_df)
        probabilitas_df = calculate_scale(probabilitas_df, probabilitas_scores, 'Probabilitas')
        dampak_df = calculate_scale(dampak_df, dampak_scores, 'Dampak')
        risk_data = calculate_severity_index(probabilitas_df, dampak_df)

        st.write("Hasil Uji Relevansi")
        st.dataframe(uji_relevansi_results)

        st.write("Hasil Analisis Risiko")
        st.dataframe(risk_data)

        # Save results
        with pd.ExcelWriter('hasil_analisis_risiko_dengan_keterangan.xlsx') as writer:
            uji_relevansi_results.to_excel(writer, sheet_name='Uji Relevansi Hasil', index=False)
            risk_data.to_excel(writer, sheet_name='Hasil Analisis', index=False)

        st.success("Hasil analisis risiko dan uji relevansi telah disimpan di 'hasil_analisis_risiko_dengan_keterangan.xlsx'.")

if _name_ == "_main_":
    main()
