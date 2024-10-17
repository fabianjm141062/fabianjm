import streamlit as st

# Fungsi untuk menghitung BMR berdasarkan umur, jenis kelamin, berat dan tinggi badan
def calculate_bmr(age, gender, weight, height):
    if gender == "male":
        # Rumus BMR untuk pria
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender == "female":
        # Rumus BMR untuk wanita
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return bmr

# Fungsi untuk menghitung kebutuhan kalori harian berdasarkan tingkat aktivitas
def calculate_daily_calories(bmr, activity_level):
    if activity_level == "sedentary":
        return bmr * 1.2
    elif activity_level == "lightly active":
        return bmr * 1.375
    elif activity_level == "moderately active":
        return bmr * 1.55
    elif activity_level == "very active":
        return bmr * 1.725
    else:
        return bmr * 1.9

# Fungsi untuk merekomendasikan asupan nutrisi
def nutrition_recommendation(daily_calories, medical_history):
    nutrition_plan = {}
    
    nutrition_plan['calories'] = daily_calories

    if 'diabetes' in medical_history:
        nutrition_plan['carbohydrates'] = f"{0.45 * daily_calories // 4} grams"
        nutrition_plan['proteins'] = f"{0.25 * daily_calories // 4} grams"
        nutrition_plan['fats'] = f"{0.30 * daily_calories // 9} grams"
        nutrition_plan['notes'] = (
            "Rekomendasi untuk penderita diabetes:\n"
            "- Karbohidrat kompleks: gandum utuh, oatmeal, dan kacang-kacangan.\n"
            "- Protein: ikan, ayam tanpa lemak, dan kacang-kacangan.\n"
            "- Lemak sehat: minyak zaitun, alpukat, dan kacang-kacangan."
        )
    elif 'hypertension' in medical_history:
        nutrition_plan['carbohydrates'] = f"{0.50 * daily_calories // 4} grams"
        nutrition_plan['proteins'] = f"{0.30 * daily_calories // 4} grams"
        nutrition_plan['fats'] = f"{0.20 * daily_calories // 9} grams"
        nutrition_plan['notes'] = (
            "Rekomendasi untuk penderita hipertensi:\n"
            "- Batasi asupan garam dan pilih makanan tinggi kalium: sayuran hijau, pisang, dan ubi.\n"
            "- Protein: ikan, ayam tanpa kulit, dan tahu/tempe.\n"
            "- Lemak sehat: minyak zaitun, kacang-kacangan."
        )
    else:
        nutrition_plan['carbohydrates'] = f"{0.50 * daily_calories // 4} grams"
        nutrition_plan['proteins'] = f"{0.30 * daily_calories // 4} grams"
        nutrition_plan['fats'] = f"{0.20 * daily_calories // 9} grams"
        nutrition_plan['notes'] = (
            "Rekomendasi umum:\n"
            "- Karbohidrat: gandum utuh, nasi merah, dan buah-buahan.\n"
            "- Protein: daging tanpa lemak, telur, kacang-kacangan.\n"
            "- Lemak sehat: minyak zaitun, kacang-kacangan, dan alpukat."
        )

    return nutrition_plan

# Fungsi untuk menghitung BMI
def calculate_bmi(weight, height):
    height_m = height / 100  # Konversi tinggi ke meter
    bmi = weight / (height_m ** 2)
    if bmi < 18.5:
        status = "Underweight"
    elif 18.5 <= bmi < 24.9:
        status = "Normal weight"
    elif 25.0 <= bmi < 29.9:
        status = "Overweight"
    else:
        status = "Obesity"
    return bmi, status

# Fungsi untuk rekomendasi vitamin dan mineral
def vitamin_mineral_recommendation(medical_history):
    if 'diabetes' in medical_history:
        return "Vitamin D, Magnesium, Chromium, Zinc"
    elif 'hypertension' in medical_history:
        return "Potassium, Magnesium, Vitamin D"
    else:
        return "Vitamin C, Vitamin E, Omega-3"

# Fungsi rekomendasi olahraga berdasarkan aktivitas dan riwayat kesehatan
def exercise_recommendation(activity_level, medical_history):
    if 'diabetes' in medical_history:
        exercise_plan = (
            "Untuk penderita diabetes:\n"
            "- Lakukan olahraga aerobik (jalan kaki, berenang, bersepeda) selama 30 menit, 5 hari per minggu.\n"
            "- Tambahkan latihan kekuatan (angkat beban atau resistance band) dua kali seminggu."
        )
    elif 'hypertension' in medical_history:
        exercise_plan = (
            "Untuk penderita hipertensi:\n"
            "- Lakukan latihan aerobik intensitas sedang seperti berjalan cepat atau berenang selama 30-45 menit, 5 hari per minggu.\n"
            "- Hindari latihan angkat beban yang terlalu berat."
        )
    elif activity_level == "sedentary":
        exercise_plan = (
            "Karena tingkat aktivitas rendah:\n"
            "- Mulailah dengan aktivitas ringan seperti jalan kaki selama 30 menit setiap hari.\n"
            "- Tambahkan aktivitas peregangan atau yoga untuk meningkatkan fleksibilitas."
        )
    elif activity_level == "lightly active":
        exercise_plan = (
            "Dengan aktivitas ringan:\n"
            "- Lakukan latihan aerobik moderat seperti lari ringan atau bersepeda selama 150 menit per minggu.\n"
            "- Tambahkan latihan kekuatan (angkat beban atau bodyweight training) dua kali seminggu."
        )
    elif activity_level == "moderately active":
        exercise_plan = (
            "Dengan aktivitas moderat:\n"
            "- Lakukan latihan intensitas tinggi seperti lari, bersepeda, atau HIIT selama 75 menit per minggu.\n"
            "- Kombinasikan dengan latihan kekuatan setidaknya dua kali per minggu."
        )
    else:  # Very active
        exercise_plan = (
            "Dengan tingkat aktivitas tinggi:\n"
            "- Lakukan latihan intensitas tinggi seperti HIIT, lari jarak jauh, atau bersepeda selama 150 menit per minggu.\n"
            "- Kombinasikan dengan latihan kekuatan setidaknya tiga kali seminggu."
        )
    return exercise_plan

# Streamlit UI
st.title("Kalkulator Kebutuhan Nutrisi, BMI, Vitamin, dan Olahraga")

# Input data pengguna
age = st.number_input("Masukkan umur:", min_value=0, max_value=120, step=1)
gender = st.selectbox("Masukkan jenis kelamin:", ["male", "female"])
weight = st.number_input("Masukkan berat badan (kg):", min_value=0.0, step=0.1)
height = st.number_input("Masukkan tinggi badan (cm):", min_value=0.0, step=0.1)
activity_level = st.selectbox("Masukkan tingkat aktivitas:", ["sedentary", "lightly active", "moderately active", "very active"])
medical_history = st.multiselect("Masukkan riwayat penyakit:", ["None", "diabetes", "hypertension"])

if "None" in medical_history:
    medical_history = []  # Kosongkan riwayat jika "None" dipilih

# Jika pengguna sudah melengkapi input, hitung BMR, BMI, nutrisi, vitamin, dan olahraga
if st.button("Hitung Kebutuhan Nutrisi"):
    # Hitung BMR
    bmr = calculate_bmr(age, gender, weight, height)

    # Hitung kebutuhan kalori harian
    daily_calories = calculate_daily_calories(bmr, activity_level)

    # Hitung BMI
    bmi, status = calculate_bmi(weight, height)

    # Buat rekomendasi nutrisi
    nutrition_plan = nutrition_recommendation(daily_calories, medical_history)

    # Buat rekomendasi vitamin dan mineral
    vitamins = vitamin_mineral_recommendation(medical_history)

    # Buat rekomendasi olahraga
    exercise_plan = exercise_recommendation(activity_level, medical_history)

    # Tampilkan hasil
    st.subheader("Rekomendasi Kebutuhan Nutrisi Harian")
    st.write(f"Kalori harian: {nutrition_plan['calories']:.2f} kkal")
    st.write(f"Karbohidrat: {nutrition_plan['carbohydrates']}")
    st.write(f"Protein: {nutrition_plan['proteins']}")
    st.write(f"Lemak: {nutrition_plan['fats']}")
    st.write(f"Catatan: {nutrition_plan['notes']}")

    st.subheader("Indeks Massa Tubuh (BMI)")
   
