import mlrun
from kfp import dsl

@dsl.pipeline(name="breast-cancer-demo", description="MLRun pipeline for breast cancer classification")
def pipeline(model_name="breast_cancer_classifier"):

    # Load breast cancer dataset
    ingest = mlrun.run_function(
        "load-breast-cancer-data",  
        name="load-breast-cancer-data",
        params={"format": "csv"},
        outputs=["dataset", "label_column"],
    )

    # Train model with hyperparameter tuning
    train = mlrun.run_function(
        "trainer",
        name="train-model",
        inputs={"dataset": ingest.outputs["dataset"]},
        params={"label_column": ingest.outputs["label_column"]},
        hyperparams={
            "n_estimators": [10, 100, 200],
            "max_depth": [2, 5, 10]
        },
        selector="max.accuracy",
        outputs=["model"],
    )

    # Deploy model for real-time inference
    deploy = mlrun.deploy_function(
        "serving",
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model"],
                "class_name": "ClassifierModel"
            }
        ],
        mock=False
    )