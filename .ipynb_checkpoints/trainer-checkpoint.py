import mlrun
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlrun.frameworks.sklearn import apply_mlrun

def train(
    dataset: mlrun.DataItem,
    label_column: str = 'target',
    n_estimators: int = 100,
    max_depth: int = None,
    max_features: str = 'sqrt',  # or 'log2' for better speed/performance balance
    model_name: str = "breast_cancer_rf"
):
    # Load dataset
    df = dataset.as_df()
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define model with optimal CPU settings
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,  # use all available CPU cores
        random_state=42
    )

    # Track model with MLRun
    apply_mlrun(model=model, model_name=model_name, x_test=X_test, y_test=y_test)

    # Train model
    model.fit(X_train, y_train)