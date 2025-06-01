# Laporan Proyek Machine Learning Terapan | Husnul Khatimah

# link file readme.md di github : https://github.com/husnlkhatmh/ml-terapan2
 
## Domain Proyek : Rekomendasi Makanan

## Latar Belakang
Perkembangan teknologi informasi telah mendorong munculnya sistem rekomendasi sebagai solusi untuk membantu pengguna dalam pengambilan keputusan, termasuk dalam pemilihan makanan. Beragam pilihan makanan yang tersedia di era digital membuat pengguna sering kali mengalami kesulitan dalam menentukan pilihan yang sesuai dengan selera dan kebutuhan diet masing-masing.

Penelitian sebelumnya menunjukkan bahwa sistem rekomendasi mampu meningkatkan kualitas pengalaman pengguna dengan memberikan saran yang relevan dan personal. Ricci et al. (2015) menegaskan bahwa sistem rekomendasi efektif dalam mengurangi beban informasi dan meningkatkan kepuasan pengguna. Selain itu, Zhang et al. (2019) menyoroti pentingnya data interaksi pengguna dalam meningkatkan akurasi dan adaptasi rekomendasi seiring perubahan preferensi.

Dalam konteks ini, data yang lengkap mengenai karakteristik makanan dan preferensi pengguna menjadi dasar penting untuk mengembangkan sistem rekomendasi yang efektif dan tepat guna. Dengan memanfaatkan data tersebut, sistem dapat memberikan rekomendasi yang lebih personal dan sesuai kebutuhan individu.

Penelitian ini bertujuan untuk mengembangkan sistem rekomendasi makanan yang dapat membantu pengguna dalam memilih makanan yang sesuai dengan preferensi mereka berdasarkan data interaksi pengguna dan karakteristik makanan.

## Business Understanding
### Problem Statements
-	Bagaimana merancang sistem rekomendasi makanan yang mampu memanfaatkan informasi konten makanan sekaligus pola preferensi pengguna untuk memberikan rekomendasi yang lebih relevan dan personal?
- Bagaimana mengukur keefektifan sistem rekomendasi makanan yang dikembangkan?


### Goals
-	Mengembangkan sistem rekomendasi makanan yang mampu memberikan saran makanan secara personal dengan mempertimbangkan atribut makanan (seperti kategori, dan tipe vegetarian) serta perilaku pengguna (yaitu riwayat penilaian makanan).
-	Mengevaluasi performa sistem rekomendasi yang dikembangkan menggunakan metrik yang sesuai untuk menilai efektivitas dan kualitas hasil rekomendasi.

### Solution Statement
Solusi yang dikembangkan adalah sistem rekomendasi makanan dengan dua pendekatan yang diuji secara terpisah, yaitu content-based filtering, yang memanfaatkan informasi seperti kategori, dan tipe vegetarian, serta collaborative filtering yang menggunakan pola interaksi dan penilaian pengguna.

Kedua pendekatan akan dievaluasi menggunakan metrik Precision untuk mengukur relevansi rekomendasi, dan Mean Absolute Error (MAE) untuk menilai akurasi prediksi terhadap preferensi pengguna.

## Data Understanding
Link dataset : https://www.kaggle.com/code/gracehephzibahm/food-recommendation-system-easy-comprehensive/input 

1. Data Makanan
   
   Berisi 400 entri makanan dengan 5 kolom:
     - Food_ID (int64): ID unik setiap makanan
     - Name (object): Nama makanan
     - C_Type (object): Kategori makanan
     - Veg_Non (object): Keterangan apakah makanan vegetarian atau non-vegetarian
     - Describe (object): Deskripsi singkat mengenai makanan
     - Semua kolom tidak memiliki nilai kosong (non-null).
    
3. Data Rating Pengguna
   
   Berisi 512 entri interaksi pengguna dengan makanan melalui rating, dengan 3 kolom:
     - User_ID (float64): ID pengguna
     - Food_ID (float64): ID makanan yang dirating
     - Rating (float64): Nilai rating yang diberikan pengguna
     - Terdapat beberapa nilai null pada User_ID, Food_ID, dan Rating sebanyak 1 baris.

Dataset ini menyediakan informasi lengkap mengenai karakteristik makanan dan preferensi pengguna dalam bentuk rating, yang akan digunakan untuk membangun dan mengevaluasi sistem rekomendasi makanan.

**Exploratory Data Analysis**

Pada data rating, ditemukan satu baris yang mengandung nilai kosong (NaN). Selanjutnya, dilakukan pengecekan data duplikat pada kedua dataset, yaitu data makanan dan data rating, dan hasilnya tidak ditemukan duplikat sehingga tidak perlu dilakukan penghapusan.

Informasi statistik pada masing masing kolom pada data rating:

|index|User\_ID|Food\_ID|Rating|
|---|---|---|---|
|count|511\.0|511\.0|511\.0|
|mean|49\.06849315068493|125\.31115459882584|5\.438356164383562|
|std|28\.73921290350848|91\.29262932706344|2\.8662362487559436|
|min|1\.0|1\.0|1\.0|
|25%|25\.0|45\.5|3\.0|
|50%|49\.0|111\.0|5\.0|
|75%|72\.0|204\.0|8\.0|
|max|100\.0|309\.0|10\.0|

Rata-rata User_ID adalah sekitar 49 dengan rentang dari 1 sampai 100 pengguna. Food_ID berkisar dari 1 sampai 309 dengan rata-rata sekitar 125. Untuk Rating, nilai rata-ratanya sekitar 5.4 dengan rentang dari 1 hingga 10, dan nilai tengah (median) rating adalah 5, menunjukkan distribusi penilaian yang cukup merata dari rendah hingga tinggi.

```python
# Jumlah makanan unik: 400
# ttipe makanan:  ['Healthy Food' 'Snack' 'Dessert' 'Japanese' 'Indian' 'French' 'Mexican'
 'Italian' 'Chinese' 'Beverage' 'Thai' 'Korean' ' Korean' 'Vietnames'
 'Nepalese' 'Spanish']
# tipe vegetarian: ['veg' 'non-veg']
```

```python
# Jumlah Rating: 10
# TTipe Rating:  [ 4.  3.  5.  1.  8.  9. 10.  6.  2.  7. nan]
```
Melihat jumlah makanan unik, jenis makanan, dan tipe vegetarian dalam data makanan, serta jumlah dan variasi rating dalam data rating. Hasilnya menunjukkan ada 400 makanan unik dengan berbagai kategori, serta dua tipe vegetarian yaitu veg dan non-veg. Pada kolom kategori makanan (C_Type), ditemukan duplikasi penulisan kategori "Korean" yang muncul sebagai 'Korean' dan ' Korean' karena perbedaan spasi.  Selain itu, terdapat 10 nilai rating berbeda dari 1 hingga 10 dalam data rating, yang menunjukkan bahwa pengguna memanfaatkan seluruh skala penilaian yang tersedia. Nilai NaN pada kolom Rating yang akan dihapus.

**Analisi Distribusi Data**
notes : grafik **Jumlah Makanan berdasarkan Tipe** di notebook

![image](https://github.com/user-attachments/assets/450f491b-7034-4cb1-a359-bc8cecbb2bb9)

Grafik tersebut menunjukkan jumlah makanan berdasarkan tipe dalam dataset. Tipe makanan terbanyak adalah Indian dengan 88 makanan, diikuti oleh Healthy Food (58) dan Dessert (53). Sementara itu, beberapa kategori seperti Korean dan Spanish hanya memiliki 1 makanan, dengan duplikasi label Korean yang perlu dibersihkan..      

notes : grafik **Jumlah Makanan berdasarkan Tipe Vegetarian** di notebook

![image](https://github.com/user-attachments/assets/a8156924-5f85-4e18-872f-a5d22e8f07eb)

Grafik menunjukkan jumlah makanan berdasarkan tipe vegetarian, dengan 238 makanan veg dan 162 non-veg. Ini menunjukkan bahwa makanan vegetarian lebih dominan.

notes : grafik **Distribusi Rating** di notebook

![image](https://github.com/user-attachments/assets/305c01e2-8b34-47e1-a644-2f5a688ac42d)

notes : grafik **Boxplot Rating** di notebook

![image](https://github.com/user-attachments/assets/a0c41d70-8d7b-4343-a029-830121b902d7)

Grafik menunjukkan distribusi rating dari 1 hingga 10, dengan rating 3, 5, dan 10 memiliki jumlah terbanyak (63, 61, dan 61), sedangkan rating 8 paling sedikit (39). Distribusi rating tampak bervariasi dan tidak simetris. Berdasarkan grafik boxplot, tidak terdapat outlier, yang menunjukkan bahwa seluruh nilai rating berada dalam rentang yang wajar.

## Data Preparation
### **Content Based Filtering**
#### 1. Persiapan Data
Membersihkan nilai-nilai pada kolom ```'C_Type'``` dari spasi di awal atau akhir teks menggunakan fungsi ```.str.strip():```.  ```data = food_df.drop(columns=['Describe'])``` menghapus kolom 'Describe' karena tidak dibutuhkan. Kemudian, kolom ```'C_Type'``` dan ```'Veg_Non'``` digabung menjadi satu kolom baru ```'combined'```, yang berisi informasi gabungan jenis makanan dan apakah makanan tersebut vegetarian atau non-vegetarian. Ini akan digunakan sebagai dasar untuk menghitung kesamaan antar item.

#### 2. TF-IDF Vectorization
Baris ```vectorizer = TfidfVectorizer()``` membuat objek TF-IDF. Kemudian ```fit_transform(data['combined'])``` mengubah teks dalam kolom ```'combined'``` menjadi representasi numerik (matriks TF-IDF), di mana setiap kata diberi bobot berdasarkan seberapa penting kata tersebut dalam dataset. Ini membantu mengenali fitur unik tiap makanan.

#### 3. Ekstraksi Fitur dan Cosine Similarity
```feature_names = vectorizer.get_feature_names_out()``` menampilkan daftar semua kata unik yang digunakan sebagai fitur dalam TF-IDF.
```cosine_sim = cosine_similarity(tfidf_matrix)``` menghitung kemiripan antar item makanan berdasarkan sudut antar vektor TF-IDF. Nilai kemiripan ini digunakan untuk merekomendasikan makanan yang mirip satu sama lain.

### **Collaborative Filtering**
#### 1. Menghapus missing value & Tipe Data Disesuaikan
Menghapus missing value pada data rating di ketiga kolom. Mengubah kolom ```User_ID``` dan ```Food_ID``` bertipe int, agar bisa diproses lebih lanjut secara numerik, terutama untuk encoding dan pemodelan.

#### 2. Encoding ID User dan Food
- ```unique().tolist()``` mengambil daftar unik dari ```User_ID``` dan ```Food_ID```.
- Dictionary ```user_to_user_encoded``` dan ```food_to_food_encoded``` digunakan untuk mengubah ID asli menjadi angka mulai dari 0 (encoding).
- Dictionary ```user_encoded_to_user``` dan ```food_encoded_to_food``` disiapkan untuk membalikkan encoding (decoding).
- DataFrame kemudian di-update dengan ID yang telah diencoding agar dapat digunakan dalam model yang membutuhkan input numerik.

#### 4. Normalisasi Rating
Rating dibagi 10 agar berada pada skala 0–1 ```rating_df['Rating'] = rating_df['Rating'] / 10.0```, yang umum digunakan untuk model machine learning agar lebih stabil dalam proses training.

#### 5. Pengacakan Data (Shuffle)
Data diacak menggunakan sample(frac=1) untuk menghindari urutan yang mungkin bias dan reset_index(drop=True) untuk merapikan indeks setelah pengacakan.

#### 6. Split Data dan Persiapan Input-Output Model
Split Data dan Persiapan Input-Output Model
Data dibagi menjadi train (80%) dan validation (20%) menggunakan ```train_test_split```.
Input model berupa pasangan ```[User_ID, Food_ID]```, dan target (output) adalah nilai ```Rating```. Ini akan digunakan dalam training dan validasi model collaborative filtering.

## Modelling dan Recomendation

- **Content Based Filtering**

Sistem rekomendasi Content-Based Filtering bekerja dengan cara merekomendasikan makanan berdasarkan kemiripan atribut konten dari item yang sudah dipilih oleh pengguna. Contohnya, jika pengguna memilih menu "Rosemary Roasted Vegetables", sistem akan mencari dan menampilkan menu lain yang memiliki fitur serupa, seperti tipe makanan (C_Type) dan status vegetarian atau non-vegetarian (Veg_Non).

Content-Based Filtering unggul dalam memberikan rekomendasi yang personal karena fokus pada atribut item yang dipilih pengguna. Sistem ini tidak bergantung pada data interaksi pengguna lain, sehingga cocok untuk situasi dengan data pengguna yang terbatas. Selain itu, pendekatan ini relatif mudah diimplementasikan dan menghasilkan rekomendasi yang relevan berdasarkan karakteristik item secara langsung (Lops, de Gemmis, & Semeraro, 2011).

Namun, sistem ini memiliki keterbatasan, terutama bergantung pada kualitas dan kelengkapan fitur item yang tersedia. Jika fitur tidak representatif, hasil rekomendasi bisa kurang akurat. Selain itu, rekomendasi yang dihasilkan cenderung monoton karena hanya menampilkan item yang sangat mirip, sehingga variasi rekomendasi menjadi terbatas. Pendekatan ini juga menghadapi masalah cold start untuk item baru yang belum memiliki data fitur lengkap, sehingga sulit direkomendasikan (Pazzani & Billsus, 2007).


  ```python
  def get_recommendation_df(item_name, top_k=5):
  # Ambil data item yang dipilih pengguna, mengambil kolom 'Name', 'C_Type', dan 'Veg_Non'
      df_awal = data[data['Name'] == item_name][['Name', 'C_Type', 'Veg_Non']].copy().reset_index(drop=True)
    
      # Ambil skor similarity tanpa dirinya sendiri
      sim_scores = cm_df[item_name].drop(index=item_name)
      # top_k item dengan skor similarity tertinggi
      top_similar = sim_scores.sort_values(ascending=False).head(top_k)
    
      # Ambil data dan skor similarity
      df_rekomendasi = data.set_index('Name').loc[top_similar.index][['C_Type', 'Veg_Non']].copy()
      df_rekomendasi['Similarity'] = top_similar.values
      df_rekomendasi = df_rekomendasi.reset_index()
    
      return df_awal, df_rekomendasi
  ```
  - **Input**: Nama item (misal "Rosemary Roasted Vegetables") dan jumlah rekomendasi top_k.
  - **Output**:
    df_awal: Data informasi dari item yang dipilih.
    df_rekomendasi: Data top rekomendasi beserta kemiripan (Similarity) tertinggi.

  **Output Top N Rekomendasi**
  ```python
  df_awal, df_rekomendasi = get_recommendation_df("Rosemary Roasted Vegetables", top_k=5)

  # Tampilkan hasil
  print("Data Awal:")
  display(df_awal)

  print("\nRekomendasi:")
  display(df_rekomendasi)
  ```

  Data awal:
  index|Name|C\_Type|Veg\_Non|
  |---|---|---|---|
  |0|Rosemary Roasted Vegetables|Healthy Food|veg|

  Rekomendasi:
  |index|Name|C\_Type|Veg\_Non|Similarity|
  |---|---|---|---|---|
  |0|Quinoa Tabbouleh|Healthy Food|veg|1\.000000000000000|
  |1|summer squash salad|Healthy Food|veg|1\.000000000000000|
  |2|Shirazi Salad|Healthy Food|veg|1\.000000000000000|
  |3|spicy watermelon soup|Healthy Food|veg|1\.000000000000000|
  |4|carrot ginger soup|Healthy Food|veg|1\.000000000000000|

  
- **Colaborative Filtering**

Collaborative Filtering merupakan pendekatan sistem rekomendasi yang memanfaatkan data interaksi antara pengguna dan item, seperti rating atau klik, tanpa memerlukan informasi fitur eksplisit dari pengguna maupun item. 

Pendekatan ini memiliki kelebihan dalam menangkap hubungan laten antara pengguna dan item tanpa membutuhkan informasi konten secara eksplisit. Model ini fleksibel, cocok untuk data berskala besar, dan mampu belajar representasi yang efisien melalui embedding.

Namun, kelemahannya terletak pada masalah cold-start, di mana model kesulitan merekomendasikan item atau pengguna baru yang belum memiliki interaksi. Selain itu, model ini sangat bergantung pada jumlah data interaksi dan membutuhkan komputasi yang cukup besar saat pelatihan.

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_food, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.food_embedding = layers.Embedding(
            num_food, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.food_bias = layers.Embedding(num_food, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        food_vector = self.food_embedding(inputs[:, 1])
        food_bias = self.food_bias(inputs[:, 1])

        # Gunakan reduce_sum untuk dot product
        dot_user_food = tf.reduce_sum(user_vector * food_vector, axis=1, keepdims=True)

        # Tambahkan bias
        x = dot_user_food + user_bias + food_bias

        # Output sigmoid, dan squeeze agar bentuknya (batch)
        return tf.nn.sigmoid(tf.squeeze(x, axis=1))

model = RecommenderNet(num_users, num_food, 50)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

early_stopping = EarlyStopping(
    monitor='val_mean_absolute_error',        
    patience=5,                 
    restore_best_weights=True  
)

```
Pada kode diatas, Model deep-learning ```RecommenderNet``` membangun sistem rekomendasi Collaborative Filtering dengan memetakan ID pengguna dan makanan ke vektor laten melalui layer embedding. Skor relevansi dihitung melalui dot product antar vektor, ditambah bias (```user_bias``` dan ```food_bias```), lalu diaktifkan dengan fungsi sigmoid untuk menghasilkan output antara 0–1. Model ini menggunakan **Adam**, **Binary Crossentropy**, **L2 regularization** untuk meningkatkan akurasi dan mencegah overfitting, dan dan **EarlyStopping** untuk menghentikan pelatihan saat performa validasi tidak membaik selama 5 epoch.

**Pelatihan Model**
```python
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)
``` 

**Hasil Pelatihan Model**

Hasil Training Model (Epoch 1–16)

| Epoch | Loss (Train) | MAE (Train) | Loss (Val) | MAE (Val) |
| ----- | ------------ | ----------- | ---------- | --------- |
| 1     | 0.6914       | 0.2509      | 0.6895     | 0.2434    |
| 2     | 0.6817       | 0.2339      | 0.6890     | 0.2428    |
| 3     | 0.6707       | 0.2220      | 0.6886     | 0.2424    |
| 4     | 0.6596       | 0.2041      | 0.6883     | 0.2418    |
| 5     | 0.6491       | 0.1936      | 0.6879     | 0.2414    |
| 6     | 0.6368       | 0.1766      | 0.6877     | 0.2410    |
| 7     | 0.6225       | 0.1765      | 0.6874     | 0.2405    |
| 8     | 0.6081       | 0.1694      | 0.6872     | 0.2401    |
| 9     | 0.5941       | 0.1608      | 0.6871     | 0.2399    |
| 10    | 0.5854       | 0.1417      | 0.6870     | 0.2397    |
| 11    | 0.5850       | 0.1201      | 0.6870     | 0.2395    |
| 12    | 0.5652       | 0.1192      | 0.6871     | 0.2398    |
| 13    | 0.5637       | 0.1024      | 0.6871     | 0.2399    |
| 14    | 0.5598       | 0.0860      | 0.6871     | 0.2401    |
| 15    | 0.5493       | 0.0780      | 0.6873     | 0.2403    |
| 16    | 0.5484       | 0.0699      | 0.6874     | 0.2404    |


**Output Top N Rekomendasi**

Makanan yang sudah dirating user ID 2:
|index|Name|C\_Type|Veg\_Non|Rating|
|---|---|---|---|---|
|0|moong dal kiwi coconut soup|Indian|veg|8\.0|
|1|carrot ginger soup|Healthy Food|veg|8\.0|
|2|chicken nimbu dhaniya shorba|Beverage|non-veg|9\.0|
|3|christmas tree pizza|Italian|veg|1\.0|

Rekomendasi untuk User ID 2:
|index|Name|C\_Type|Veg\_Non|Predicted\_Rating|
|---|---|---|---|---|
|43|andhra pan fried pomfret|Indian|veg|6\.194409370422363|
|183|vegetable bruschetta|Italian|veg|6\.1571831703186035|
|202|banana and chia tea cake|Dessert|veg|6\.036716461181641|
|257|lamb korma|Indian|non-veg|6\.017825603485107|
|294|tandoori chicken|Indian|non-veg|5\.991272449493408|

## Evaluation
Evaluasi kinerja model dilakukan dengan menggunakan sejumlah metrik utama yaitu Precision dan mean absoulete error
 
  1. Precision (Content Based Filtering) : mengukur proporsi rekomendasi yang tepat dari keseluruhan item yang disarankan kepada pengguna. Precision pada top-N menunjukkan berapa banyak dari N rekomendasi teratas yang benar-benar relevan atau sesuai dengan preferensi pengguna.

$$
\text{Precision} = \frac{\text{Jumlah item rekomendasi yang relevan}}{\text{Total jumlah item yang direkomendasikan}}
$$

Data awal:
|index|Name|C\_Type|Veg\_Non|
|---|---|---|---|
|0|Rosemary Roasted Vegetables|Healthy Food|veg|
  
Hasil rekomendasi:

Rekomendasi:
|index|Name|C\_Type|Veg\_Non|Similarity|
|---|---|---|---|---|
|0|Quinoa Tabbouleh|Healthy Food|veg|1\.000000000000000|
|1|summer squash salad|Healthy Food|veg|1\.000000000000000|
|2|Shirazi Salad|Healthy Food|veg|1\.000000000000000|
|3|spicy watermelon soup|Healthy Food|veg|1\.000000000000000|
|4|carrot ginger soup|Healthy Food|veg|1\.000000000000000|
 
Semua 5 rekomendasi memenuhi kriteria tersebut → artinya semuanya relevan.

```math
\text{Precision@5} = \frac{5 \text{ item relevan}}{5 \text{ item yang direkomendasikan}} = 1.0 \text{ atau } 100\%
```

### Mean Absolute Error (MAE)

**Mean Absolute Error (MAE)** digunakan untuk mengukur seberapa besar rata-rata kesalahan antara nilai rating aktual dengan rating yang diprediksi oleh model. Semakin kecil nilai MAE, semakin baik kinerja model.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |
$$

Keterangan:

* $n$: jumlah total prediksi
* $\hat{r}_i$: rating yang diprediksi untuk item ke-i
* $r_i$: rating aktual (sebenarnya) untuk item ke-i

Hasil Metrik Evaluasi :
notes : grafik **Training and Validation Mean Absolute Error (MAE)** di notebook

![image](https://github.com/user-attachments/assets/667d5611-8284-4138-8500-00df081a056e)


Selama proses pelatihan model sebanyak 18 epoch, evaluasi menggunakan mean absolute error Selama 16 epoch pelatihan, performa model pada data latih menunjukkan peningkatan signifikan, ditandai dengan penurunan loss dari 0.6914 ke 0.5484 dan MAE dari 0.2509 ke 0.0699. Ini menunjukkan model mampu mempelajari pola dengan baik. Namun, pada data validasi, penurunan MAE hanya terjadi di awal (dari 0.2434 ke 0.2395 hingga epoch 11), lalu stagnan hingga 0.2404 di epoch 16. Nilai loss validasi pun tidak menunjukkan perbaikan berarti sejak epoch ke-10. Pola ini mengindikasikan overfitting, di mana model terlalu menyesuaikan diri dengan data latih dan kurang mampu melakukan generalisasi terhadap data baru.

## kesimpulan
Proyek ini berhasil mengembangkan dua pendekatan sistem rekomendasi makanan, yaitu Content-Based Filtering dan Collaborative Filtering, yang masing-masing memiliki kelebihan dan keterbatasan. Pendekatan Content-Based Filtering memanfaatkan atribut makanan seperti kategori dan tipe vegetarian untuk memberikan rekomendasi yang sangat relevan, terutama bagi pengguna baru. Hal ini terbukti dari nilai Precision yang mencapai 100%, menandakan rekomendasi yang tepat dan sesuai preferensi pengguna.

Collaborative Filtering menggunakan model deep learning berbasis embedding yang mampu mempelajari pola interaksi pengguna dengan makanan secara efektif, terlihat dari penurunan nilai Mean Absolute Error (MAE) pada data pelatihan. Namun, model ini mengalami overfitting, karena performanya pada data validasi stagnan dan kurang mampu memprediksi data baru dengan baik. Hal ini mungkin sebagian disebabkan oleh keterbatasan ukuran data yang tidak besar, sehingga model kurang mampu menangkap pola umum yang berlaku pada data yang lebih luas.

Evaluasi menunjukkan bahwa kedua pendekatan memiliki potensi dalam membangun sistem rekomendasi makanan yang efektif, namun perlu dilakukan pengembangan lebih lanjut untuk mengatasi overfitting dan meningkatkan generalisasi model.

## Referensi

Ricci, F., Rokach, L., Shapira, B., & Kantor, P. B. (2015). Recommender Systems Handbook (2nd ed.). Springer.

Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep Learning Based Recommender System: A Survey and New Perspectives. ACM Computing Surveys, 52(1), 1–38.

Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. Recommender Systems Handbook, 73-105.

Pazzani, M., & Billsus, D. (2007). Content-based Recommendation Systems. The Adaptive Web, 325-341
