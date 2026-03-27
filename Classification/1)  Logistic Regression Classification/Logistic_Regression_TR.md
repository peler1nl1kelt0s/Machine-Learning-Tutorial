# Lojistik Regresyon — Nedir ve Nasıl Çalışır?

## Lojistik Regresyon Nedir?

Lojistik regresyon, adında "regresyon" geçmesine rağmen bir **sınıflandırma** algoritmasıdır. Bir veri noktasının belirli bir sınıfa ait olma **olasılığını** tahmin eder ve bu olasılığı `(0, 1)` aralığında bir değer olarak çıktı verir.

Yaygın kullanım örnekleri:

- Tümörün **iyi huylu mu kötü huylu mu** olduğunu tahmin etme
- E-postanın **spam mi yoksa normal mi** olduğunu belirleme
- Bir işlemin **sahte mi gerçek mi** olduğunu sınıflandırma

---

## Neden Doğrusal Regresyon Kullanılmaz?

Doğrusal regresyon, `[0, 1]` aralığının dışında değerler üretebilir; bu da olasılık hesabı için anlamsızdır. Lojistik regresyon bu sorunu **sigmoid fonksiyonu** ile çözer.

---

## Sigmoid Fonksiyonu

Sigmoid fonksiyonu, herhangi bir gerçek sayıyı `(0, 1)` aralığına sıkıştırır:

```
σ(z) = 1 / (1 + e^(−z))
```

**Temel özellikleri:**

1. **Olasılıksal çıktı** — Sonuç her zaman `(0, 1)` arasındadır, doğrudan olasılık olarak yorumlanabilir.
2. **Türevlenebilir** — Gradyan inişi için gereklidir; türevi `σ(z) · (1 − σ(z))` şeklindedir.
3. **Monoton** — `z` arttıkça tahmin edilen olasılık da artar.

**Karar eşiği:**

- `σ(z) ≥ 0.5` → sınıf `1` olarak tahmin et
- `σ(z) < 0.5` → sınıf `0` olarak tahmin et

---

## Çalışma Mantığı (Adım Adım)

### 1. Parametrelerin Başlatılması

```python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)  # küçük, sıfır olmayan değer
    b = 0.0
    return w, b
```

- **Ağırlıklar (w)** küçük bir sabit değerle (`0.01`) başlatılır. Tüm ağırlıklar sıfır olsaydı, her nöron aynı çıktıyı ve aynı gradyanı üretirdi; model hiçbir şey öğrenemezdi (*simetri problemi*).
- **Bias (b)** sıfırdan başlar; simetri sorunu yalnızca ağırlıklar için geçerlidir.

---

### 2. İleri Yayılım (Forward Propagation)

İleri yayılımda tahmin ve maliyet hesaplanır:

```
z   = wᵀ · X + b          → doğrusal kombinasyon
ŷ   = σ(z)                → sigmoid uygula → olasılık tahmini
```

**İkili Çapraz Entropi Kayıp Fonksiyonu:**

```
L(y, ŷ) = −y · log(ŷ) − (1 − y) · log(1 − ŷ)
```

| Gerçek Etiket | Açıklama |
|---------------|----------|
| `y = 1` | Kayıp = `−log(ŷ)`. Model `ŷ = 1` tahmin ederse kayıp → 0; `ŷ → 0` ise kayıp → ∞ |
| `y = 0` | Kayıp = `−log(1 − ŷ)`. Simetrik mantık geçerlidir |

Bu fonksiyon, güvenle yapılan yanlış tahminleri çok ağır şekilde cezalandırır.

**Maliyet (Cost):** Tüm eğitim örnekleri üzerindeki ortalama kayıp:

```
Cost = (1/m) · Σ L(yᵢ, ŷᵢ)
```

| Terim | Tanım |
|-------|-------|
| **Kayıp (Loss)** | Tek bir eğitim örneği için hata |
| **Maliyet (Cost)** | Tüm eğitim örnekleri üzerindeki ortalama kayıp |

Eğitim sırasında **maliyet** minimize edilir.

---

### 3. Geri Yayılım (Backward Propagation)

Geri yayılımda ağırlıkların ve bias'ın maliyete göre gradyanları hesaplanır:

```
dz = ŷ − y
dw = (1/m) · X · dzᵀ
db = (1/m) · Σ dz
```

Bu gradyanlar, maliyetin hangi yönde ve ne kadar hızlı arttığını gösterir; bu yüzden parametreleri **ters** yönde güncelleriz.

---

### 4. Gradyan İnişi (Gradient Descent)

Ağırlıklar ve bias her iterasyonda güncellenir:

```
w ← w − α · dw
b ← b − α · db
```

- `α` (alfa) **öğrenme oranıdır** — her adımda ne kadar büyük bir adım attığımızı belirler.
- Güncelleme döngüsü, maliyet yakınsayana (önemli ölçüde azalmayı bırakana) kadar sabit sayıda iterasyon boyunca tekrar eder.

**Öğrenme oranının etkisi:**

| Durum | Sonuç |
|-------|-------|
| Çok büyük `α` | Maliyet ıraksayabilir (salınım, ıraksamaya yol açar) |
| Çok küçük `α` | Model çok yavaş öğrenir |
| Uygun `α` | Maliyet düzenli olarak azalır ve yakınsır |

---

## Tüm Akış — Özet

```
X (giriş özellikleri)
    ↓
z = wᵀX + b          [doğrusal kombinasyon]
    ↓
ŷ = σ(z)             [sigmoid → olasılık tahmini]
    ↓
L = −y·log(ŷ) − (1−y)·log(1−ŷ)   [kayıp hesaplama]
    ↓
dw, db               [gradyanları hesapla]
    ↓
w ← w − α·dw         [parametreleri güncelle]
b ← b − α·db
    ↓
[iterasyonları tekrar et → maliyet minimize edilir]
```

---

## Örnek: scikit-learn ile Lojistik Regresyon

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Veriyi ölçeklendir
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim / test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Modeli eğit
model = LogisticRegression(solver='saga', max_iter=1000)
model.fit(X_train, y_train)

# Test doğruluğu
accuracy = model.score(X_test, y_test)
print(f"Test Doğruluğu: {accuracy:.4f}")
```

---

## Normalleştirme Neden Önemlidir?

Lojistik regresyon gradyan inişi kullanır. Özellikler farklı ölçeklerdeyse (örn. `alan_ortalama` ~ 500 vs. `pürüzlülük_ortalama` ~ 0.1), büyük değerli özellikler ağırlık güncellemelerine hâkim olur ve eğitimi yavaşlatır veya dengesiz hale getirir. **Min-Max normalleştirme** her özelliği `[0, 1]` aralığına çeker:

```
x_norm = (x − x_min) / (x_max − x_min)
```

---

## Lojistik Regresyonun Güçlü ve Zayıf Yönleri

| Güçlü Yönler | Zayıf Yönler |
|--------------|--------------|
| Yorumlanabilir (ağırlıklar özellik önemini gösterir) | Yalnızca doğrusal karar sınırı |
| Büyük veri kümelerinde ölçeklenebilir | Karmaşık, doğrusal olmayan ilişkileri zayıf modelliyor |
| Olasılıksal çıktı üretir | Çok sayıda ilgisiz özelliğe karşı hassas |
| Eğitim ve tahmin hızlı | Çok sınıflı görevler için uzantı gerektirir |

---

## İlgili Dosyalar

- 📓 **[Logistic Regression.ipynb](./Logistic%20Regression.ipynb)** — NumPy ile sıfırdan uygulama ve scikit-learn karşılaştırması
- 📄 **[Classification.md](../Classification.md)** — KNN, SVM ve Naive Bayes dahil tüm sınıflandırma algoritmalarının İngilizce özeti
