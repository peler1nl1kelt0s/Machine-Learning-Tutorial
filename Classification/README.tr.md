# Classification — çalıştırılmış notebook dizini

🇬🇧 [English](README.md)

Bütün 10 bölümün notebook’ları kod, açıklamalar, tablolar ve gerçek çalıştırma çıktılarıyla kaydedilmiştir. GitHub’da `.ipynb` dosyasını açarak sonuçları görebilirsiniz. Kodun tamamı ilgili notebook içindedir; ayrı bir `.py` dosyasını okumak gerekmez.

**Teori:** [English](Classification.md) · [Türkçe](Classification.tr.md). Notebook’larda İngilizce anlatımın yanında Türkçe açıklamalar bulunur; kod iki dilde tekrar edilmez.

| # | Çalıştırılmış notebook | İçerik | Kaydedilen sonuç |
|---|---|---|---|
| 1 | [Lojistik Regresyon](1%29%20%20Logistic%20Regression%20Classification/Logistic%20Regression.ipynb) | NumPy gradyanları, kararlı kayıp, sklearn, karmaşıklık matrisi | Test doğruluğu 97.37% |
| 2 | [K-En Yakın Komşu](2%29%20K-Nearest%20Neighbour%20%28KNN%29%20Classification/K-Nearest%20Neighbour%20%28KNN%29.ipynb) | Eğitimde ölçekleme, CV ile K seçimi, komşu oylamasını inceleme | Test doğruluğu 95.61% |
| 3 | [Destek Vektör Makinesi](3%29%20Support%20Vector%20Machine%20%28SVM%29%20Classification/Support%20Vector%20Machine%20%28SVM%29.ipynb) | Doğrusal/RBF çekirdek seçimi, destek vektörleri, marj grafiği | Test doğruluğu 97.37% |
| 4 | [Naive Bayes](4%29%20Naive%20Bayes%20Classification/Naive%20Bayes.ipynb) | Gaussian yoğunluklar, önseller, elle log posterior | Test doğruluğu 93.86% |
| 5 | [Karar Ağacı](5%29%20Decision%20Tree%20Classification/Decision%20Tree.ipynb) | CV ile derinlik/yaprak seçimi, entropi, kök bilgi kazancı | Test doğruluğu 92.98% |
| 6 | [Rastgele Orman](6%29%20Random%20Forest%20Classification/Random%20Forest.ipynb) | Bootstrap, OOB, ağaç olasılıklarının ortalaması, önemler | Test doğruluğu 96.49% |
| 7 | [Değerlendirme ve algoritma karşılaştırması](7%29%20Model%20Evaluation/Model%20Evaluation.ipynb) | Aynı katlarda altı model, OOF eşik, karmaşıklık/ROC/PR | CV ile seçilen SVM: test 98.25% |
| 8 | [K-means kümeleme](8%29%20K-Means%20Clustering/K-Means.ipynb) | Sıfırdan Lloyd, K-means++, dirsek, silhouette, sınırlar | WCSS 150,669.43; silhouette 0.7444 |
| 9 | [Hiyerarşik kümeleme](9%29%20Hierarchical%20Clustering/Hierarchical%20Clustering.ipynb) | Elle örnek, Ward SSE/yükseklik, dendrogram, linkage karşılaştırma | 3 küme; SciPy/sklearn ARI = 1 |
| 10 | [NLP: duygu sınıflandırma](10%29%20Natural%20Language%20Processing/NLP%20Sentiment%20Classification.ipynb) | Veri okuma, BoW, elle TF-IDF, CSR, altı pipeline, yorumlama | Test doğruluğu 83.89% |

## Kurulum ve yeniden çalıştırma

Python **3.10–3.12** kullanın; doğrulanan ortam Python 3.10’dur. Repo kökünde:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Classification/requirements.txt
python Classification/run_notebooks.py
```

Windows’ta etkinleştirme komutu: `.venv\Scripts\activate`.

Yalnız bir bölüm veya birkaç bölüm:

```bash
python Classification/run_notebooks.py --chapter 10
python Classification/run_notebooks.py --chapter 8 --chapter 9
```

Runner her notebook’u komutu çalıştıran Python ile **yeni bir kernel’de** açar, tüm hücreleri sırayla yürütür ve çıktıları aynı `.ipynb` içine kaydeder. Hatalı hücrede durur; o notebook’un önceki kayıtlı sürümünü değiştirmez. Çalıştırılan sonuçları [notebook_results.json](notebook_results.json) içinde de toplar. Yeniden çalıştırma notebook çıktılarında Git değişikliği oluşturabilir.

Etkileşimli bir notebook editöründe de aynı sanal ortamın kernel’ini seçip **Restart Kernel → Run All** kullanabilirsiniz. Notebook’lar birbirlerinin değişkenlerine bağlı değildir. Veri indirmek veya API anahtarı eklemek gerekmez.

## Ölçümleri nasıl okumalı?

- **1–7:** Yerel Wisconsin verisi; ortak stratified %80/%20 ayrımı, `random_state=42`. Ölçekleyici yalnız eğitim satırlarında, CV sırasında her fold’un eğitim kısmında fit edilir. Test kümesi hiperparametre seçmez. 2, 3 ve 5. bölümler CV ile ayar seçer; 7. bölüm altı sabit konfigürasyonu karşılaştırır. Bu yüzden 3 ve 7’deki SVM sonuçları aynı olmak zorunda değildir. CV’de daha iyi seçilen ayarın tek bir test ayrımında daha iyi çıkması garanti değildir.

- **8–9:** Aynı seed ile üretilen 3.000 etiketsiz sentetik nokta. WCSS ve silhouette geometrik ölçülerdir; accuracy değildir. 9. bölümdeki ARI, iki uygulamanın gruplarını karşılaştırır; gerçek etiket başarısı değildir.

- **10:** [UCI duygu verisi](datasets/nlp_sentiment/README.md); 3.000 ham kayıttan tekrarları ayıklayınca 2.979 kayıt, 2.383 eğitim ve 596 test örneği. En yüksek eğitim-CV macro-F1 ile seçilen word+character TF-IDF + logistic regression: **%83,89** test doğruluğu. Bu, önceki farklı veri/hedef için bildirilen %48’in yeniden ölçümü değildir.

Teori sayfalarındaki 1–7. bölümlerin eski grafik ve örnek tabloları farklı ayrımlar kullanabilir; bunlar kendi başlıklarındaki deneylere aittir. Güncel ortak ölçüm düzeninin sonuçları yukarıdaki notebook’larda bulunur. Grafiklerdeki hata çubukları, fold/karıştırma değişkenliğidir; nüfus için güven aralığı değildir.

## Diğer Python dosyaları

- [run_notebooks.py](run_notebooks.py): tüm notebook’ları veya seçilen bölümleri çalıştırır.
- [nlp_text_classification.py](nlp_text_classification.py): 10. bölümün duygu deneyini komut satırından tekrarlar.
- [generate_clustering_figures.py](generate_clustering_figures.py) ve [generate_nlp_figures.py](generate_nlp_figures.py): 8–10. bölümlerin Türkçe/İngilizce teori görsellerini `images/` altında üretir. NLP görsellerinden önce `nlp_text_classification.py` çalıştırılır.
