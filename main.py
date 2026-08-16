
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. Veri Seti Oluşturma
# ==========================================
np.random.seed(42)
n_samples = 200

# Sentetik veriyi oluşturuyoruz
data = {
    'yas': np.random.randint(18, 70, n_samples),
    'gelir': np.random.randint(20000, 150000, n_samples),
    'abonelik_suresi': np.random.randint(1, 60, n_samples),
    'destek_talebi_sayisi': np.random.randint(0, 10, n_samples),
    'sehir': np.random.choice(['Ankara', 'Istanbul', 'Izmir', 'Bursa', 'Antalya'], n_samples),
    'uyelik_tipi': np.random.choice(['Aylik', 'Yillik', 'Premium'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # %70 kalır, %30 ayrılır
}

df = pd.DataFrame(data)

# Kasıtlı olarak birkaç eksik değer (NaN) ekleyelim ki eksik veri adımını uygulayabilelim
df.loc[10:15, 'gelir'] = np.nan
df.loc[20:22, 'sehir'] = np.nan

# ==========================================
# 2. Veri İnceleme (EDA)
# ==========================================
print("--- VERİ İNCELEME ---")
print("İlk 5 Satır:\n", df.head())
print("\nSatır ve Sütun Sayısı:", df.shape)
print("\nHedef Değişken (Churn) Dağılımı:\n", df['churn'].value_counts())

# ==========================================
# 3. Eksik Değer Kontrolü ve Doldurma
# ==========================================
print("\n--- EKSİK DEĞER KONTROLÜ ---")
print(df.isnull().sum())

# Sayısal değişken (gelir) için medyan, kategorik (şehir) için mod kullanalım
df['gelir'] = df['gelir'].fillna(df['gelir'].median())
df['sehir'] = df['sehir'].fillna(df['sehir'].mode()[0])
print("\nEksik değerler medyan ve mod yöntemleriyle dolduruldu.")

# ==========================================
# 4. Öznitelik Üretme (Feature Engineering)
# ==========================================
# Müşterinin hiç destek talebi açıp açmadığını belirten basit bir özellik (1 veya 0)
df['destek_talebi_var_mi'] = df['destek_talebi_sayisi'].apply(lambda x: 1 if x > 0 else 0)
print("\nYeni öznitelik üretildi: 'destek_talebi_var_mi'")

# ==========================================
# 5. Kategorik Değişkenleri Dönüştürme (One-Hot Encoding)
# ==========================================
df = pd.get_dummies(df, columns=['sehir', 'uyelik_tipi'], drop_first=True)
print("\nKategorik değişkenler One-Hot Encoding ile dönüştürüldü.")

# ==========================================
# 6. Train, Validation ve Test Kümelerine Ayırma
# ==========================================
# Önce Bağımlı (y) ve Bağımsız (X) değişkenleri ayıralım
X = df.drop('churn', axis=1)
y = df['churn']

# 1. Adım: Veriyi Train+Validation (%80) ve Test (%20) olarak ayır
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 2. Adım: Train+Validation verisini kendi içinde Train (%75) ve Validation (%25) olarak ayır
# Böylece toplam verinin -> %60 Train, %20 Validation, %20 Test olur
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\nVeri bölme işlemi tamamlandı:")
print(f"Train: {X_train.shape[0]} satır | Val: {X_val.shape[0]} satır | Test: {X_test.shape[0]} satır")

# ==========================================
# 7. Sayısal Değişkenleri Ölçekleme (Scaling)
# ==========================================
scaler = StandardScaler()
# Sadece sayısal kolonları ölçekleyelim, one-hot encode edilmiş kolonlara dokunmamak daha iyidir
num_cols = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ==========================================
# 8. Model Eğitimi ve Validation Karşılaştırması
# ==========================================
print("\n--- MODEL EĞİTİMİ VE VALIDATION SONUÇLARI ---")

# Modelleri tanımlama
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5) # Bonus model
}

best_model_name = ""
best_val_accuracy = 0
trained_models = {}

for name, model in models.items():
    # Eğit
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Validation üzerinde tahmin
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_name = name

print(f"\nSeçilen En İyi Model: {best_model_name} (Validation başarısına göre)")

# ==========================================
# 9. Test Verisi Üzerinde Değerlendirme
# ==========================================
print(f"\n--- {best_model_name.upper()} TEST SONUÇLARI ---")
best_model = trained_models[best_model_name]
test_preds = best_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
print(f"Accuracy : {accuracy_score(y_test, test_preds):.4f}")
print(f"Precision: {precision_score(y_test, test_preds, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_preds, zero_division=0):.4f}")
print(f"F1-Score : {f1_score(y_test, test_preds, zero_division=0):.4f}")

# ==========================================
# 10. Sonuç ve Yorum Çıktısı
# ==========================================
print("\n--- PROJE SONUCU VE YORUM ---")
print(f"""
Bu çalışmada üç farklı model denenmiş olup validation seti üzerindeki 
doğruluk oranlarına bakılarak {best_model_name} seçilmiştir. 
Veri setimizin küçük ve sentetik olarak rastgele oluşturulmuş olması sebebiyle 
modellerin hedef değişkeni öğrenme kapasitesi sınırlı kalmış olabilir. 
Gerçek dünya verilerinde, özellikle sınıflar arası dengesizlik durumunda (Churn=1 azınlıktaysa), 
Accuracy yerine Recall ve F1-Score metriklerinin daha yüksek olması hedeflenmelidir.
""")