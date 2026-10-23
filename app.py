import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Machine Learning - Kelompok 2",
    layout="wide",
)


# =========================================================
# JUDUL APLIKASI
# =========================================================

st.title("Machine Learning Experiment")

st.caption(
    "Implementasi Antarmuka Model Machine Learning — Tugas Proyek 3"
)

with st.expander("Informasi Kelompok"):
    st.write(
        """
        **Kelompok 2**

        - M Mahdi Rasyad
        - Yuyun Sari Dewi
        - Nabila Destriana
        - Laudia Anjelina
        - Mouzza Cinta Laura

        **Program Studi Manajemen Informatika**  
        **Fakultas Ilmu Komputer — Universitas Sriwijaya**
        """
    )


# =========================================================
# SIDEBAR UTAMA
# =========================================================

st.sidebar.title("Machine Learning")

menu = st.sidebar.radio(
    "Pilih Model",
    [
        "Supervised Learning",
        "Unsupervised Learning",
    ],
)


# =========================================================
# SUPERVISED LEARNING
# =========================================================

if menu == "Supervised Learning":

    st.header("Prediksi Harga Rumah")

    st.write(
        "Halaman ini digunakan untuk melakukan eksperimen "
        "prediksi harga rumah menggunakan **Linear Regression** "
        "dan **Decision Tree Regressor**."
    )

    st.divider()

    # -----------------------------------------------------
    # 1. UPLOAD DATASET
    # -----------------------------------------------------

    st.subheader("1. Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload Housing Prices Dataset",
        type=["csv", "xlsx"],
        key="supervised_dataset",
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("Dataset berhasil dimuat!")

            # -------------------------------------------------
            # 2. INFORMASI DATASET
            # -------------------------------------------------

            st.subheader("2. Informasi Dataset")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Jumlah Data",
                df.shape[0],
            )

            col2.metric(
                "Jumlah Kolom",
                df.shape[1],
            )

            col3.metric(
                "Missing Value",
                int(df.isnull().sum().sum()),
            )

            st.write("**Cuplikan Dataset:**")

            st.dataframe(
                df.head(10),
                use_container_width=True,
            )

            # -------------------------------------------------
            # 3. PENGATURAN MODEL
            # -------------------------------------------------

            st.subheader("3. Pengaturan Model")

            if "price" in df.columns:
                target_index = list(df.columns).index("price")
            else:
                target_index = 0

            target = st.selectbox(
                "Pilih kolom target",
                df.columns,
                index=target_index,
            )

            test_size = st.slider(
                "Test Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
            )

            st.caption(
                f"{100 - test_size}% data digunakan untuk training "
                f"dan {test_size}% untuk testing."
            )

            model_name = st.selectbox(
                "Pilih Algoritma",
                [
                    "Linear Regression",
                    "Decision Tree Regressor",
                ],
            )

            max_depth = None

            if model_name == "Decision Tree Regressor":

                max_depth = st.slider(
                    "Max Depth",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                )

            # -------------------------------------------------
            # MULAI TRAINING
            # -------------------------------------------------

            if st.button(
                "Mulai Training",
                type="primary",
                use_container_width=True,
                key="training_button",
            ):

                X = df.drop(columns=[target])
                y = df[target]

                if not pd.api.types.is_numeric_dtype(y):
                    st.error(
                        "Kolom target harus berupa data numerik "
                        "untuk model regresi."
                    )
                    st.stop()

                categorical_columns = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                numeric_columns = X.select_dtypes(
                    include=np.number
                ).columns.tolist()

                transformers = []

                if categorical_columns:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                            categorical_columns,
                        )
                    )

                if numeric_columns:
                    transformers.append(
                        (
                            "numeric",
                            "passthrough",
                            numeric_columns,
                        )
                    )

                if not transformers:
                    st.error(
                        "Tidak ditemukan fitur yang dapat digunakan "
                        "untuk proses training."
                    )
                    st.stop()

                preprocessor = ColumnTransformer(
                    transformers=transformers
                )

                if model_name == "Linear Regression":
                    model = LinearRegression()

                else:
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        random_state=42,
                    )

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size / 100,
                    random_state=42,
                )

                pipeline.fit(
                    X_train,
                    y_train,
                )

                y_pred = pipeline.predict(
                    X_test
                )

                st.success(
                    f"{model_name} berhasil dilatih!"
                )

                # -------------------------------------------------
                # 4. DATA TRAINING DAN TESTING
                # -------------------------------------------------

                st.subheader(
                    "4. Training Data & Testing Data"
                )

                tab_train, tab_test = st.tabs(
                    [
                        "Training Data",
                        "Testing Data",
                    ]
                )

                with tab_train:

                    training_data = X_train.copy()
                    training_data[target] = y_train

                    st.write(
                        f"Jumlah data training: "
                        f"**{len(training_data)} data**"
                    )

                    st.dataframe(
                        training_data.head(10),
                        use_container_width=True,
                    )

                with tab_test:

                    testing_data = X_test.copy()
                    testing_data[target] = y_test

                    st.write(
                        f"Jumlah data testing: "
                        f"**{len(testing_data)} data**"
                    )

                    st.dataframe(
                        testing_data.head(10),
                        use_container_width=True,
                    )

                # -------------------------------------------------
                # 5. EVALUASI MODEL
                # -------------------------------------------------

                mae = mean_absolute_error(
                    y_test,
                    y_pred,
                )

                mse = mean_squared_error(
                    y_test,
                    y_pred,
                )

                rmse = np.sqrt(mse)

                r2 = r2_score(
                    y_test,
                    y_pred,
                )

                st.subheader(
                    "5. Hasil Evaluasi Model"
                )

                metric1, metric2, metric3, metric4 = st.columns(4)

                metric1.metric(
                    "MAE",
                    f"{mae:,.2f}",
                )

                metric2.metric(
                    "MSE",
                    f"{mse:,.2f}",
                )

                metric3.metric(
                    "RMSE",
                    f"{rmse:,.2f}",
                )

                metric4.metric(
                    "R² Score",
                    f"{r2:.4f}",
                )

                # -------------------------------------------------
                # 6. TABEL AKTUAL VS PREDIKSI
                # -------------------------------------------------

                st.subheader(
                    "6. Aktual vs Prediksi"
                )

                hasil_prediksi = pd.DataFrame(
                    {
                        "Harga Aktual": y_test.values,
                        "Harga Prediksi": y_pred,
                        "Selisih": np.abs(
                            y_test.values - y_pred
                        ),
                    }
                )

                st.dataframe(
                    hasil_prediksi.head(20),
                    use_container_width=True,
                )

                csv_prediksi = hasil_prediksi.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Hasil Prediksi",
                    data=csv_prediksi,
                    file_name="hasil_prediksi_harga_rumah.csv",
                    mime="text/csv",
                )

                # -------------------------------------------------
                # 7. VISUALISASI
                # -------------------------------------------------

                st.subheader(
                    "7. Visualisasi Aktual vs Prediksi"
                )

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                ax.scatter(
                    y_test,
                    y_pred,
                    alpha=0.7,
                )

                minimum = min(
                    y_test.min(),
                    y_pred.min(),
                )

                maximum = max(
                    y_test.max(),
                    y_pred.max(),
                )

                ax.plot(
                    [minimum, maximum],
                    [minimum, maximum],
                    linestyle="--",
                )

                ax.set_xlabel(
                    "Harga Aktual"
                )

                ax.set_ylabel(
                    "Harga Prediksi"
                )

                ax.set_title(
                    f"Aktual vs Prediksi - {model_name}"
                )

                ax.grid(
                    alpha=0.2
                )

                st.pyplot(fig)

                plt.close(fig)

                # -------------------------------------------------
                # 8. INTERPRETASI HASIL
                # -------------------------------------------------

                st.subheader(
                    "8. Interpretasi Hasil"
                )

                if r2 >= 0.8:
                    kategori_r2 = "sangat baik"

                elif r2 >= 0.6:
                    kategori_r2 = "cukup baik"

                elif r2 >= 0.4:
                    kategori_r2 = "sedang"

                else:
                    kategori_r2 = "masih rendah"

                st.write(
                    f"""
                    Model **{model_name}** menghasilkan
                    **R² sebesar {r2:.4f}** dengan
                    **RMSE sebesar {rmse:,.2f}**.

                    Berdasarkan nilai R² tersebut, kemampuan model
                    dalam menjelaskan variasi harga rumah berada
                    pada kategori **{kategori_r2}**.

                    Semakin kecil nilai MAE dan RMSE menunjukkan
                    kesalahan prediksi yang semakin rendah,
                    sedangkan nilai R² yang semakin mendekati 1
                    menunjukkan kemampuan model yang semakin baik
                    dalam menjelaskan variasi harga rumah.
                    """
                )

                st.info(
                    """
                    **Interpretasi Metrik:**

                    • **MAE** menunjukkan rata-rata kesalahan
                    absolut prediksi.

                    • **MSE** menunjukkan rata-rata kuadrat
                    kesalahan prediksi.

                    • **RMSE** menunjukkan besar kesalahan prediksi
                    dalam satuan yang sama dengan harga.

                    • **R²** menunjukkan kemampuan model menjelaskan
                    variasi harga rumah. Nilai yang semakin mendekati
                    1 menunjukkan performa model yang semakin baik.
                    """
                )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat memproses dataset: {e}"
            )

    else:

        st.info(
            "Silakan upload Housing Prices Dataset "
            "untuk memulai proses Supervised Learning."
        )


# =========================================================
# UNSUPERVISED LEARNING
# =========================================================

elif menu == "Unsupervised Learning":

    st.header("Segmentasi Kota")

    st.write(
        "Halaman ini digunakan untuk mengelompokkan kota berdasarkan "
        "karakteristik demografi dan kualitas hidup menggunakan "
        "**K-Means** dan **Agglomerative Hierarchical Clustering**."
    )

    st.divider()

    # -----------------------------------------------------
    # 1. UPLOAD DATASET
    # -----------------------------------------------------

    st.subheader("1. Upload Dataset")

    uploaded_city = st.file_uploader(
        "Upload City Lifestyle Dataset",
        type=["csv", "xlsx"],
        key="unsupervised_dataset",
    )

    if uploaded_city is not None:

        try:

            if uploaded_city.name.lower().endswith(".csv"):
                city_df = pd.read_csv(uploaded_city)
            else:
                city_df = pd.read_excel(uploaded_city)

            st.success("Dataset berhasil dimuat!")

            # -------------------------------------------------
            # 2. INFORMASI DATASET
            # -------------------------------------------------

            st.subheader(
                "2. Informasi Dataset"
            )

            info1, info2, info3 = st.columns(3)

            info1.metric(
                "Jumlah Data",
                city_df.shape[0],
            )

            info2.metric(
                "Jumlah Kolom",
                city_df.shape[1],
            )

            info3.metric(
                "Missing Value",
                int(city_df.isnull().sum().sum()),
            )

            st.write(
                "**Cuplikan Dataset:**"
            )

            st.dataframe(
                city_df.head(10),
                use_container_width=True,
            )

            # -------------------------------------------------
            # 3. PENGATURAN CLUSTERING
            # -------------------------------------------------

            st.subheader(
                "3. Pengaturan Clustering"
            )

            numeric_columns_city = city_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            default_features = [
                "population_density",
                "avg_income",
                "internet_penetration",
                "avg_rent",
                "air_quality_index",
                "public_transport_score",
                "happiness_score",
                "green_space_ratio",
            ]

            available_default_features = [
                col
                for col in default_features
                if col in numeric_columns_city
            ]

            selected_features = st.multiselect(
                "Pilih fitur yang akan dianalisis",
                options=numeric_columns_city,
                default=available_default_features,
            )

            algorithm = st.selectbox(
                "Pilih Algoritma",
                [
                    "K-Means",
                    "Agglomerative Hierarchical Clustering",
                ],
            )

            jumlah_cluster = st.slider(
                "Jumlah Cluster (K)",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
            )

            st.caption(
                "Pada Proyek 2, K = 3 memberikan hasil "
                "Silhouette Score terbaik pada pengujian "
                "jumlah cluster."
            )

            # -------------------------------------------------
            # MULAI CLUSTERING
            # -------------------------------------------------

            if st.button(
                "Mulai Clustering",
                type="primary",
                use_container_width=True,
                key="clustering_button",
            ):

                if len(selected_features) < 2:

                    st.warning(
                        "Pilih minimal 2 fitur untuk "
                        "melakukan clustering."
                    )

                else:

                    X_cluster = city_df[
                        selected_features
                    ].copy()

                    if X_cluster.isnull().sum().sum() > 0:

                        st.error(
                            "Fitur yang dipilih memiliki missing value. "
                            "Silakan periksa dataset."
                        )

                        st.stop()

                    # -----------------------------------------
                    # STANDARDISASI
                    # -----------------------------------------

                    scaler = StandardScaler()

                    X_scaled = scaler.fit_transform(
                        X_cluster
                    )

                    # -----------------------------------------
                    # PEMODELAN
                    # -----------------------------------------

                    if algorithm == "K-Means":

                        clustering_model = KMeans(
                            n_clusters=jumlah_cluster,
                            random_state=42,
                            n_init=10,
                        )

                        cluster_labels = (
                            clustering_model.fit_predict(
                                X_scaled
                            )
                        )

                    else:

                        clustering_model = (
                            AgglomerativeClustering(
                                n_clusters=jumlah_cluster,
                                linkage="ward",
                            )
                        )

                        cluster_labels = (
                            clustering_model.fit_predict(
                                X_scaled
                            )
                        )

                    hasil_cluster = city_df.copy()

                    hasil_cluster[
                        "Cluster"
                    ] = cluster_labels

                    st.success(
                        f"Clustering menggunakan "
                        f"{algorithm} berhasil!"
                    )

                    # -----------------------------------------
                    # 4. EVALUASI
                    # -----------------------------------------

                    st.subheader(
                        "4. Hasil Evaluasi Clustering"
                    )

                    silhouette = silhouette_score(
                        X_scaled,
                        cluster_labels,
                    )

                    db_index = davies_bouldin_score(
                        X_scaled,
                        cluster_labels,
                    )

                    ch_index = calinski_harabasz_score(
                        X_scaled,
                        cluster_labels,
                    )

                    eval1, eval2, eval3 = st.columns(3)

                    eval1.metric(
                        "Silhouette Score",
                        f"{silhouette:.4f}",
                    )

                    eval2.metric(
                        "Davies-Bouldin Index",
                        f"{db_index:.4f}",
                    )

                    eval3.metric(
                        "Calinski-Harabasz Index",
                        f"{ch_index:.4f}",
                    )

                    if algorithm == "K-Means":

                        st.metric(
                            "Inertia / WCSS",
                            f"{clustering_model.inertia_:.4f}",
                        )

                    # -----------------------------------------
                    # 5. DISTRIBUSI CLUSTER
                    # -----------------------------------------

                    st.subheader(
                        "5. Distribusi Cluster"
                    )

                    cluster_counts = (
                        hasil_cluster["Cluster"]
                        .value_counts()
                        .sort_index()
                    )

                    cluster_table = pd.DataFrame(
                        {
                            "Cluster": cluster_counts.index,
                            "Jumlah Data": cluster_counts.values,
                        }
                    )

                    st.dataframe(
                        cluster_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        cluster_counts
                    )

                    # -----------------------------------------
                    # 6. HASIL CLUSTERING
                    # -----------------------------------------

                    st.subheader(
                        "6. Hasil Clustering"
                    )

                    st.dataframe(
                        hasil_cluster.head(30),
                        use_container_width=True,
                    )

                    csv_cluster = hasil_cluster.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Hasil Clustering",
                        data=csv_cluster,
                        file_name="hasil_clustering_city.csv",
                        mime="text/csv",
                    )

                    # -----------------------------------------
                    # 7. PCA
                    # -----------------------------------------

                    st.subheader(
                        "7. Visualisasi Cluster dengan PCA"
                    )

                    pca = PCA(
                        n_components=2
                    )

                    X_pca = pca.fit_transform(
                        X_scaled
                    )

                    fig, ax = plt.subplots(
                        figsize=(9, 6)
                    )

                    scatter = ax.scatter(
                        X_pca[:, 0],
                        X_pca[:, 1],
                        c=cluster_labels,
                        alpha=0.7,
                    )

                    ax.set_xlabel(
                        "Principal Component 1"
                    )

                    ax.set_ylabel(
                        "Principal Component 2"
                    )

                    ax.set_title(
                        f"Hasil Clustering - {algorithm}"
                    )

                    ax.grid(
                        alpha=0.2
                    )

                    legend = ax.legend(
                        *scatter.legend_elements(),
                        title="Cluster",
                    )

                    ax.add_artist(
                        legend
                    )

                    st.pyplot(
                        fig
                    )

                    plt.close(fig)

                    explained_variance = (
                        pca.explained_variance_ratio_.sum()
                        * 100
                    )

                    st.write(
                        f"Dua komponen PCA menjelaskan "
                        f"**{explained_variance:.2f}%** "
                        f"variasi data."
                    )

                    # -----------------------------------------
                    # 8. KARAKTERISTIK CLUSTER
                    # -----------------------------------------

                    st.subheader(
                        "8. Karakteristik Cluster"
                    )

                    profile = (
                        hasil_cluster
                        .groupby("Cluster")[
                            selected_features
                        ]
                        .mean()
                    )

                    st.dataframe(
                        profile,
                        use_container_width=True,
                    )

                    # -----------------------------------------
                    # 9. INTERPRETASI HASIL
                    # -----------------------------------------

                    st.subheader(
                        "9. Interpretasi Hasil"
                    )

                    st.write(
                        f"""
                        Proses clustering menggunakan
                        **{algorithm}** dengan
                        **{jumlah_cluster} cluster**
                        menghasilkan:

                        - Silhouette Score: **{silhouette:.4f}**
                        - Davies-Bouldin Index: **{db_index:.4f}**
                        - Calinski-Harabasz Index: **{ch_index:.4f}**

                        Silhouette Score yang semakin tinggi
                        menunjukkan pemisahan antar-cluster
                        yang semakin baik.

                        Davies-Bouldin Index yang semakin rendah
                        menunjukkan cluster yang semakin kompak
                        dan terpisah.

                        Calinski-Harabasz Index yang semakin tinggi
                        menunjukkan struktur cluster yang semakin
                        jelas.
                        """
                    )

                    if algorithm == "K-Means":

                        st.write(
                            f"Inertia/WCSS yang diperoleh adalah "
                            f"**{clustering_model.inertia_:.4f}**."
                        )

                    st.info(
                        """
                        **Interpretasi Metrik:**

                        • **Silhouette Score:** semakin tinggi,
                        pemisahan antar-cluster semakin baik.

                        • **Davies-Bouldin Index:** semakin rendah,
                        hasil clustering semakin baik.

                        • **Calinski-Harabasz Index:** semakin tinggi,
                        cluster semakin terpisah dengan baik.

                        • **Inertia/WCSS:** digunakan khusus pada
                        K-Means untuk melihat jarak data terhadap
                        centroid cluster.
                        """
                    )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat memproses dataset: {e}"
            )

    else:

        st.info(
            "Silakan upload City Lifestyle Dataset "
            "untuk memulai proses Unsupervised Learning."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Learning Experiment | "
    "Tugas Proyek 3 | Kelompok 2 | "
    "Manajemen Informatika — Universitas Sriwijaya"
)