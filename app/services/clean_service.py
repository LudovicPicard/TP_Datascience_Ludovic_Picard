import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import uuid
from app.dataset_manager import DATASETS, CLEANERS


def cast_to_str(x):
    return x.astype(str)

def fit_cleaner(dataset_id: str, impute_strategy="mean", outlier_strategy="clip", categorical_strategy="one_hot"):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise ValueError(f"Dataset {dataset_id} inconnu")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy))
    ])

    # Categorical pipeline
    if categorical_strategy == "one_hot":
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cast", FunctionTransformer(cast_to_str, validate=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    else:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cast", FunctionTransformer(cast_to_str, validate=False)),
            ("encoder", OrdinalEncoder())
        ])

    # Full pipeline
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Fit
    preprocessor.fit(df)

    cleaner_id = str(uuid.uuid4())
    CLEANERS[cleaner_id] = {
        "pipeline": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "outlier_strategy": outlier_strategy
    }

    # Rapport qualité avant
    report = {
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "outliers": {col: int(((df[col] - df[col].mean()).abs() > 3*df[col].std()).sum()) for col in numeric_cols if df[col].dtype != "object"},
        "types": {col: str(df[col].dtype) for col in df.columns}
    }

    return cleaner_id, report

def transform_dataset(cleaner_id: str, dataset_id: str):
    cleaner = CLEANERS.get(cleaner_id)
    df = DATASETS.get(dataset_id)
    if cleaner is None or df is None:
        raise ValueError("Cleaner ou dataset inconnu")

    preprocessor = cleaner["pipeline"]
    numeric_cols = cleaner["numeric_cols"]
    categorical_cols = cleaner["categorical_cols"]
    outlier_strategy = cleaner["outlier_strategy"]

    # Copie du dataframe
    df_clean = df.copy()

    # Numeric outliers
    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].dtype != "object":
            if outlier_strategy == "clip":
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean[col] = df_clean[col].clip(lower=mean-3*std, upper=mean+3*std)
            elif outlier_strategy == "remove":
                df_clean = df_clean[(df_clean[col] - df_clean[col].mean()).abs() <= 3*df_clean[col].std()]

    # Duplicates
    duplicates_removed = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()

    # Transformation pipeline
    transformed_array = preprocessor.transform(df_clean)
    transformed_df = pd.DataFrame(transformed_array, columns=[
        *numeric_cols,
        *(
            preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
            if hasattr(preprocessor.named_transformers_["cat"]["encoder"], "get_feature_names_out") else categorical_cols
        )
    ])

    # Remplacer NaN/inf avant retour JSON
    transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan)

    # Sauvegarder le nouveau dataset
    new_dataset_id = str(uuid.uuid4())
    DATASETS[new_dataset_id] = transformed_df

    return {
        "dataset_id": new_dataset_id,
        "data": transformed_df.head(20).replace({np.nan: None}).to_dict(orient="records"),
        "duplicates_removed": int(duplicates_removed),
        "imputed": int(df_clean.isna().sum().sum())
    }

def get_report(dataset_id: str):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise ValueError(f"Dataset {dataset_id} inconnu")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    report = {
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "outliers": {col: int(((df[col] - df[col].mean()).abs() > 3*df[col].std()).sum()) for col in numeric_cols if df[col].dtype != "object"},
        "types": {col: str(df[col].dtype) for col in df.columns}
    }

    return report
