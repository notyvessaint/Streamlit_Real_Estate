import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analiza Pietei Imobiliare",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navigare")
page = st.sidebar.radio(
    "Selecteaza pagina de analiza:",
    ["Acasa", "Panou de Piata", "Predictia Pretului", "Clustere de Investitii", "Analiza Statistica"]
)

@st.cache_data
def load_data():
    return pd.read_csv('Housing.csv')

@st.cache_data
def get_market_basics(df):
    bedroom_values = sorted(df['bedrooms'].unique())
    return {
        'price_min': int(df['price'].min()),
        'price_max': int(df['price'].max()),
        'price_q25': int(df['price'].quantile(0.25)),
        'price_q75': int(df['price'].quantile(0.75)),
        'bedroom_values': bedroom_values,
        'missing_values': df.isnull().sum()
    }

@st.cache_data
def preprocess_for_regression(df):
    df_prep = df.copy()
    categorical_cols = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'prefarea', 'furnishingstatus'
    ]
    category_maps = {}
    for col in categorical_cols:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[col] = le.fit_transform(df_prep[col])
            category_maps[col] = {label: idx for idx, label in enumerate(le.classes_)}
    return df_prep, category_maps

@st.cache_resource
def fit_regression_model(df_prep):
    return ols(
        'price ~ area + bedrooms + bathrooms + stories + mainroad + guestroom + basement + hotwaterheating + airconditioning + parking + prefarea + furnishingstatus',
        data=df_prep
    ).fit()

@st.cache_data
def run_clustering(df):
    clustering_features = ['price', 'area', 'bedrooms', 'bathrooms', 'parking']
    x_cluster = df[clustering_features].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_cluster)

    k_values = list(range(2, 8))
    inertias = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(x_scaled)
        inertias.append(kmeans.inertia_)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(x_scaled)

    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    prices_by_cluster = df_clustered.groupby('cluster')['price'].mean().sort_values()
    cluster_names = {
        prices_by_cluster.index[0]: "Segment Buget",
        prices_by_cluster.index[1]: "Segment Randament Ridicat",
        prices_by_cluster.index[2]: "Segment Lux"
    }
    df_clustered['segment'] = df_clustered['cluster'].map(cluster_names)

    cluster_summary = df_clustered.groupby('cluster').agg({
        'price': ['mean', 'min', 'max', 'count'],
        'area': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean',
        'parking': 'mean'
    }).round(2)

    return df_clustered, k_values, inertias, cluster_names, cluster_summary

@st.cache_data
def get_correlation_outputs(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    price_corr = correlation_matrix['price'].sort_values(ascending=False)
    corr_df = price_corr.reset_index()
    corr_df.columns = ['feature', 'correlation']
    return correlation_matrix, corr_df

@st.cache_data
def get_anova_table(df):
    model_anova = ols('price ~ C(furnishingstatus)', data=df).fit()
    return anova_lm(model_anova, typ=2)

df = load_data()
market_basics = get_market_basics(df)

if page == "Acasa":
    st.title("Analiza Pietei Imobiliare si Strategia de Investitii")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## Prezentare proiect
        Aceasta aplicatie analizeaza tendintele din piata imobiliara pentru a identifica
        zone cu potential ridicat de investitie si pentru a estima preturile proprietatilor.

        ### Obiective principale:
        - **Analiza pietei**: Intelegerea distributiilor de pret si a tendintelor pe zone
        - **Predictia pretului**: Estimarea valorii proprietatilor prin regresie multipla
        - **Clusterizare investitii**: Identificarea segmentelor (Lux, Buget, Randament ridicat)
        - **Perspective statistice**: Validarea relatiilor dintre caracteristici si pret
        """)

    with col2:
        st.info("""
        ### Informatii despre setul de date
        - **Proprietati totale**: {} inregistrari
        - **Interval de pret**: ₹{:,.0f} - ₹{:,.0f}
        - **Pret mediu**: ₹{:,.0f}
        - **Caracteristici**: {} variabile
        """.format(
            len(df),
            df['price'].min(),
            df['price'].max(),
            df['price'].mean(),
            len(df.columns)
        ))

    st.markdown("---")
    st.subheader("Previzualizare set de date")
    st.dataframe(df, width='stretch')

elif page == "Panou de Piata":
    st.title("Panou de piata")
    st.markdown("---")

    st.subheader("Filtreaza proprietatile")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_price = st.slider(
            "Pret minim (₹)",
            market_basics['price_min'],
            market_basics['price_max'],
            market_basics['price_q25']
        )

    with col2:
        max_price = st.slider(
            "Pret maxim (₹)",
            market_basics['price_min'],
            market_basics['price_max'],
            market_basics['price_q75']
        )

    with col3:
        bedrooms = st.multiselect(
            "Numar de dormitoare",
            market_basics['bedroom_values'],
            default=market_basics['bedroom_values']
        )

    missing_values = market_basics['missing_values']
    if missing_values.sum() > 0:
        st.warning(f"Valori lipsa detectate: {missing_values[missing_values > 0].to_dict()}")
    else:
        st.success("Nu exista valori lipsa in setul de date")

    filtered_df = df[
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['bedrooms'].isin(bedrooms))
    ]

    st.info(f"Sunt afisate {len(filtered_df)} din {len(df)} proprietati")
    st.subheader("Analiza distributiei preturilor")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            filtered_df,
            x='price',
            nbins=30,
            title='Distributia preturilor',
            labels={'price': 'Pret (₹)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, width='stretch')

    with col2:
        fig_box = px.box(
            filtered_df,
            x='bedrooms',
            y='price',
            title='Pret in functie de numarul de dormitoare',
            labels={'price': 'Pret (₹)', 'bedrooms': 'Dormitoare'},
            color='bedrooms'
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, width='stretch')

    st.subheader("Analiza caracteristicilor proprietatilor")

    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(
            filtered_df,
            x='area',
            y='price',
            color='bathrooms',
            size='bedrooms',
            title='Suprafata vs pret (colorat dupa bai)',
            labels={'area': 'Suprafata (sq ft)', 'price': 'Pret (₹)'},
            hover_data=['bedrooms', 'bathrooms', 'parking']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width='stretch')

    with col2:
        furnish_stats = filtered_df.groupby('furnishingstatus')['price'].agg(['mean', 'count'])
        fig_bar = px.bar(
            furnish_stats.reset_index(),
            x='furnishingstatus',
            y='mean',
            title='Pret mediu dupa gradul de mobilare',
            labels={'furnishingstatus': 'Mobilare', 'mean': 'Pret mediu (₹)'},
            color='mean',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, width='stretch')

    st.subheader("Rezumat statistic dupa caracteristici")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Statistici descriptive dupa dormitoare**")
        bedroom_stats = filtered_df.groupby('bedrooms').agg({
            'price': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(0)
        st.dataframe(bedroom_stats, width='stretch')

    with col2:
        st.markdown("**Statistici descriptive dupa acces la drumul principal**")
        mainroad_stats = filtered_df.groupby('mainroad').agg({
            'price': ['count', 'mean', 'median', 'std']
        }).round(0)
        st.dataframe(mainroad_stats, width='stretch')

elif page == "Predictia Pretului":
    st.title("Model de predictie a pretului")
    st.markdown("---")

    st.subheader("Pregatirea datelor")
    df_prep, category_maps = preprocess_for_regression(df)
    model = fit_regression_model(df_prep)
    st.success("Variabilele categorice au fost codificate")

    st.subheader("Model de regresie multipla")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("R-patrat", f"{model.rsquared:.4f}")

    with col2:
        st.metric("R2 ajustat", f"{model.rsquared_adj:.4f}")

    with col3:
        st.metric("Statistica F", f"{model.fvalue:.2f}")

    with col4:
        st.metric("Prob(F)", f"{model.f_pvalue:.2e}")

    with col5:
        st.metric("AIC", f"{model.aic:.0f}")

    st.subheader("Metrici de performanta ale modelului")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R-patrat", f"{model.rsquared:.4f}")

    with col2:
        st.metric("R2 ajustat", f"{model.rsquared_adj:.4f}")

    with col3:
        st.metric("Statistica F", f"{model.fvalue:.2f}")

    with col4:
        st.metric("AIC", f"{model.aic:.2f}")

    coef_df = pd.DataFrame({
        "Variabila": model.params.index,
        "Coeficient": model.params.values,
        "Eroare standard": model.bse.values,
        "Statistica t": model.tvalues.values,
        "Valoare p": model.pvalues.values
    })
    ci_df = model.conf_int(alpha=0.05).reset_index(drop=True)
    coef_df["Limita inferioara 95%"] = ci_df[0].values
    coef_df["Limita superioara 95%"] = ci_df[1].values
    coef_df["Semnificativ (p<0.05)"] = np.where(coef_df["Valoare p"] < 0.05, "Da", "Nu")
    coef_df["Impact"] = np.where(coef_df["Coeficient"] >= 0, "Pozitiv", "Negativ")

    st.subheader("Coeficientii modelului")
    st.dataframe(
        coef_df.style.format({
            "Coeficient": "{:,.2f}",
            "Eroare standard": "{:,.2f}",
            "Statistica t": "{:.2f}",
            "Valoare p": "{:.4f}",
            "Limita inferioara 95%": "{:,.2f}",
            "Limita superioara 95%": "{:,.2f}"
        }),
        width='stretch'
    )

    st.subheader("Detalii complete ale modelului")
    model_info_df = pd.DataFrame({
        "Indicator": [
            "Variabila dependenta",
            "Metoda",
            "Numar observatii",
            "Grade libertate model",
            "Grade libertate reziduale",
            "Tip covarianta",
            "Data rulare",
            "Ora rulare"
        ],
        "Valoare": [
            "price",
            "OLS",
            f"{int(model.nobs)}",
            f"{int(model.df_model)}",
            f"{int(model.df_resid)}",
            f"{model.cov_type}",
            pd.Timestamp.now().strftime("%d-%m-%Y"),
            pd.Timestamp.now().strftime("%H:%M:%S")
        ]
    })
    st.dataframe(model_info_df, width='stretch', hide_index=True)

    diagnoza_df = pd.DataFrame({
        "Indicator": [
            "Log-Likelihood",
            "AIC",
            "BIC",
            "Statistica F",
            "Prob(F-statistica)",
            "R-patrat",
            "R2 ajustat",
            "Numar conditie"
        ],
        "Valoare": [
            f"{model.llf:,.2f}",
            f"{model.aic:,.2f}",
            f"{model.bic:,.2f}",
            f"{model.fvalue:,.2f}",
            f"{model.f_pvalue:.2e}",
            f"{model.rsquared:.4f}",
            f"{model.rsquared_adj:.4f}",
            f"{model.condition_number:,.2f}"
        ]
    })

    st.subheader("Diagnoza modelului")
    st.dataframe(diagnoza_df, width='stretch', hide_index=True)

    with st.expander("Raport tehnic complet (formatat)"):
        st.code(model.summary().as_text(), language="text")

    st.subheader("Prezicerea pretului unei proprietati")
    col1, col2, col3 = st.columns(3)

    with col1:
        pred_area = st.number_input("Suprafata (sq ft)", min_value=1000, max_value=20000, value=7000)
        pred_bedrooms = st.number_input("Dormitoare", min_value=1, max_value=5, value=3)
        pred_bathrooms = st.number_input("Bai", min_value=1, max_value=5, value=2)

    with col2:
        pred_stories = st.number_input("Etaje", min_value=1, max_value=4, value=2)
        pred_parking = st.number_input("Locuri de parcare", min_value=0, max_value=3, value=1)
        pred_mainroad = st.selectbox("Acces la drumul principal", ["yes", "no"], format_func=lambda x: "da" if x == "yes" else "nu")

    with col3:
        pred_guestroom = st.selectbox("Camera de oaspeti", ["yes", "no"], format_func=lambda x: "da" if x == "yes" else "nu")
        pred_basement = st.selectbox("Subsol", ["yes", "no"], format_func=lambda x: "da" if x == "yes" else "nu")
        pred_ac = st.selectbox("Aer conditionat", ["yes", "no"], format_func=lambda x: "da" if x == "yes" else "nu")

    pred_furnish = st.selectbox(
        "Grad de mobilare",
        ["furnished", "semi-furnished", "unfurnished"],
        format_func=lambda x: {
            "furnished": "mobilata",
            "semi-furnished": "semi-mobilata",
            "unfurnished": "nemobilata"
        }[x]
    )

    pred_data = pd.DataFrame({
        'area': [pred_area],
        'bedrooms': [pred_bedrooms],
        'bathrooms': [pred_bathrooms],
        'stories': [pred_stories],
        'mainroad': [1 if pred_mainroad == 'yes' else 0],
        'guestroom': [1 if pred_guestroom == 'yes' else 0],
        'basement': [1 if pred_basement == 'yes' else 0],
        'hotwaterheating': [0],
        'airconditioning': [1 if pred_ac == 'yes' else 0],
        'parking': [pred_parking],
        'prefarea': [0],
        'furnishingstatus': [category_maps['furnishingstatus'][pred_furnish]]
    })

    prediction = model.predict(pred_data)[0]
    st.markdown("---")
    st.success(f"### Pret estimat: ₹{prediction:,.0f}")

    pred_summary = model.get_prediction(pred_data)
    pred_ci = pred_summary.conf_int(alpha=0.05)

    st.info(f"""
    **Interval de incredere 95%:**
    - Limita inferioara: ₹{pred_ci[0][0]:,.0f}
    - Limita superioara: ₹{pred_ci[0][1]:,.0f}
    """)

elif page == "Clustere de Investitii":
    st.title("Clusterizare pentru oportunitati de investitii")
    st.markdown("---")

    st.subheader("Analiza clusterelor K-Means")
    with st.spinner("Se pregatesc datele pentru clusterizare..."):
        df_clustered, k_values, inertias, cluster_names, cluster_summary = run_clustering(df)

    st.success("Caracteristicile au fost scalate cu StandardScaler")

    fig_elbow = px.line(
        x=k_values,
        y=inertias,
        title='Metoda cotului pentru K optim',
        labels={'x': 'Numar de clustere (K)', 'y': 'Inertie'},
        markers=True
    )
    st.plotly_chart(fig_elbow, width='stretch')

    st.subheader("Caracteristicile clusterelor")
    st.dataframe(cluster_summary, width='stretch')

    st.subheader("Segmente de investitie")

    for cluster_id, segment_name in cluster_names.items():
        segment_data = df_clustered[df_clustered['cluster'] == cluster_id]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(f"{segment_name}", f"{len(segment_data)} proprietati")
        with col2:
            st.metric("Pret mediu", f"₹{segment_data['price'].mean():,.0f}")
        with col3:
            st.metric("Suprafata medie", f"{segment_data['area'].mean():.0f} sq ft")
        with col4:
            st.metric("Dormitoare medii", f"{segment_data['bedrooms'].mean():.1f}")

    st.subheader("Vizualizarea clusterelor")
    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(
            df_clustered,
            x='area',
            y='price',
            color='segment',
            size='bedrooms',
            hover_data=['bedrooms', 'bathrooms', 'parking'],
            title='Clustere de investitii: suprafata vs pret',
            labels={'area': 'Suprafata (sq ft)', 'price': 'Pret (₹)'},
            color_discrete_map={
                "Segment Buget": "#00CC96",
                "Segment Randament Ridicat": "#FFD700",
                "Segment Lux": "#EF553B"
            }
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width='stretch')

    with col2:
        cluster_counts = df_clustered['segment'].value_counts()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title='Distributia proprietatilor pe segmente',
            color_discrete_map={
                "Segment Buget": "#00CC96",
                "Segment Randament Ridicat": "#FFD700",
                "Segment Lux": "#EF553B"
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, width='stretch')

elif page == "Analiza Statistica":
    st.title("Analiza statistica si perspective")
    st.markdown("---")

    st.subheader("Demonstratie de acces la date")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Utilizare .loc[] - acces pe etichete**")
        sample_idx = st.number_input("Selecteaza indexul randului", 0, len(df) - 1, 0)
        selected_row = df.loc[sample_idx]
        st.dataframe(selected_row.to_frame().astype(str), width='stretch')

    with col2:
        st.write("**Utilizare .iloc[] - acces pe pozitie**")
        position = st.number_input("Selecteaza pozitia", 0, len(df) - 1, 0)
        iloc_row = df.iloc[position]
        st.dataframe(iloc_row.to_frame().astype(str), width='stretch')

    st.subheader("Analiza corelatiilor")
    correlation_matrix, corr_df = get_correlation_outputs(df)

    fig_corr = px.imshow(
        correlation_matrix,
        title='Matricea de corelatie a caracteristicilor',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, width='stretch')

    st.subheader("Caracteristici cel mai puternic corelate cu pretul")

    fig_corr_bar = px.bar(
        corr_df,
        x='correlation',
        y='feature',
        title='Corelatia cu pretul',
        labels={'correlation': 'Coeficient de corelatie', 'feature': 'Caracteristica'},
        orientation='h',
        color='correlation',
        color_continuous_scale='RdBu_r'
    )
    fig_corr_bar.update_layout(height=400)
    st.plotly_chart(fig_corr_bar, width='stretch')

    st.subheader("ANOVA: Impactul gradului de mobilare asupra pretului")

    furnished = df[df['furnishingstatus'] == 'furnished']['price']
    semifurnished = df[df['furnishingstatus'] == 'semi-furnished']['price']
    unfurnished = df[df['furnishingstatus'] == 'unfurnished']['price']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mobilata (medie)", f"₹{furnished.mean():,.0f}")
    with col2:
        st.metric("Semi-mobilata (medie)", f"₹{semifurnished.mean():,.0f}")
    with col3:
        st.metric("Nemobilata (medie)", f"₹{unfurnished.mean():,.0f}")

    anova_table = get_anova_table(df)
    st.write("**Rezultate ANOVA:**")
    st.dataframe(anova_table, width='stretch')

    st.subheader("Perspective cheie si recomandari")

    st.markdown(f"""
    ### Perspective de piata:
    1. **Interval de pret**: Proprietatile variaza intre ₹{df['price'].min():,.0f} si ₹{df['price'].max():,.0f}
    2. **Pret mediu**: ₹{df['price'].mean():,.0f}
    3. **Numarul cel mai frecvent de dormitoare**: {df['bedrooms'].mode()[0]} dormitoare
    4. **Impactul accesului la drumul principal**: Proprietatile de pe drumul principal au in medie ₹{df[df['mainroad']=='yes']['price'].mean():,.0f}

    ### Recomandari de investitie:
    - **Segment Buget**: Potrivit pentru investitori la inceput de drum, cost de intrare mai redus
    - **Segment Randament Ridicat**: Profil echilibrat intre risc si randament
    - **Segment Lux**: Proprietati premium cu potential mai mare de apreciere

    ### Importanta caracteristicilor:
    - **Suprafata** este cel mai puternic predictor al pretului (corelatie: {correlation_matrix.loc['area', 'price']:.3f})
    - **Dormitoarele** au o corelatie puternica cu pretul (corelatie: {correlation_matrix.loc['bedrooms', 'price']:.3f})
    - **Accesul la drumul principal** creste valoarea proprietatii cu aproximativ {((df[df['mainroad']=='yes']['price'].mean() / df[df['mainroad']=='no']['price'].mean() - 1) * 100):.1f}%
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
Aplicatie de analiza a pietei imobiliare | Sursa date: Housing Prices Dataset | Construita cu Streamlit
</div>
""", unsafe_allow_html=True)
