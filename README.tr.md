# Makine Öğrenmesi

🇬🇧 [English](README.md)

Temel makine öğrenmesi algoritmalarını kapsayan, düzenli çalışma notları ve uygulamalı örneklerden oluşan bir koleksiyon. Her konu kendi klasöründedir; klasördeki markdown dosyası sezgiyi, arkasındaki matematiği (adım adım türetimler ve bu repodaki veri setlerinden üretilmiş grafiklerle) ve scikit-learn kod parçalarını anlatır.

---

## İçerik

### 📈 [Regresyon](./Regression/Regression.tr.md)
Girdi özniteliklerinden sürekli sayısal değerler tahmin etmek.

| # | Algoritma | Ana fikir |
|---|-----------|-----------|
| 1 | Basit Doğrusal Regresyon | Bir girdi ile bir çıktı arasına bir doğru uydur |
| 2 | Çoklu Doğrusal Regresyon | Doğrusal regresyonu birden çok girdi özniteliğine genişlet |
| 3 | Polinom Regresyon | Özniteliklerin kuvvetlerini ekleyerek eğri ilişkileri modelle |
| 4 | Karar Ağacı Regresyonu | Öznitelik uzayını tekrar tekrar böl; her yaprakta sabit bir değer tahmin et |
| 5 | Rastgele Orman Regresyonu | Daha kararlı tahminler için ortalaması alınan karar ağaçları topluluğu |
| 6 | Model Değerlendirme | R², MSE, SSR, SST — modelin veriye ne kadar iyi uyduğunu ölçmek |

### 🔷 [Sınıflandırma](./Classification/Classification.tr.md)
Bir veri noktasının hangi kategoriye ait olduğunu tahmin etmek.

| # | Algoritma | Ana fikir |
|---|-----------|-----------|
| 1 | Lojistik Regresyon | Sigmoid ile ikili sınıflandırma; sıfırdan ve sklearn ile uygulama |
| 2 | K-En Yakın Komşu (KNN) | En yakın K eğitim örneğinin çoğunluk oyuyla sınıflandır |
| 3 | Destek Vektör Makinesi (SVM) | İki sınıfı ayıran en büyük marjlı hiperdüzlemi bul |
| 4 | Naive Bayes | Bayes teoremine ve öznitelik bağımsızlığı varsayımına dayanan olasılıksal sınıflandırıcı |
| 5 | Karar Ağacı ile Sınıflandırma | Entropiyi (ya da Gini değerini) en çok azaltan öznitelik ve eşikte böl |
| 6 | Rastgele Orman ile Sınıflandırma | Bootstrap alt örneklemlerinde eğitilen rastgele ağaçlar çoğunluk oyuyla birleşir |
| 7 | Karmaşıklık Matrisi ve Metrikler | Doğruluk, precision, recall, F1, ROC/AUC — sınıflandırıcının hangi hataları yaptığını ölçmek |
| 8 | K-means Kümeleme | Etiketsiz noktaları en yakın merkeze ata; kareli uzaklık toplamını azalt |
| 9 | Hiyerarşik Kümeleme | Grupları linkage kuralıyla birleştir; dendrogramı kes |
| 10 | NLP: Duygu Sınıflandırma | Metni sparse BoW/TF-IDF özelliklerine dönüştür; modelleri CV ile karşılaştır |
| 11 | PCA (Temel Bileşen Analizi) | Yüksek boyutlu veriyi en büyük varyans yönlerine indirge ve görselleştir |
| 12 | Model Seçimi (K-Katlı CV ve Grid Search CV) | Hiperparametreleri Pipeline içinde sızıntısız çapraz doğrulamayla seç |
| 13 | Öneri Sistemleri | İçerik tabanlı, kosinüs işbirlikçi filtreleme ve SVD gizli faktör modelleri |

**[Classification: 10 çalıştırılmış notebook, kodlar, grafikler ve kurulum](Classification/README.tr.md).** Kümeleme bölümleri gözetimsiz öğrenmedir. Notebook’larda Türkçe ve İngilizce açıklamalar, kayıtlı gerçek çıktılar bulunur.

---

## Araçlar ve Kütüphaneler

- **Dil:** Python 3
- **Ana kütüphane:** [scikit-learn](https://scikit-learn.org/)
- **Yardımcılar:** NumPy, Matplotlib, Pandas

---

> Her algoritma klasöründe adım adım kod ve satır içi açıklamalar içeren bir Jupyter notebook bulunur; teori ise klasörün `.md` dosyasındadır. Classification notebook’larında Türkçe açıklamalar da bulunur.
