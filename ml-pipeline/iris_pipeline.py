from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model
from kfp import compiler

# -------------------------
# COMPONENT 1: LOAD & SPLIT DATA
# -------------------------
@component(
    base_image="python:3.11",
    packages_to_install=["--default-timeout=120", "pandas", "scikit-learn"]
)
def load_data(
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset]
):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)
    print("Train and Test datasets saved successfully.")


# -------------------------
# COMPONENT 2: TRAIN MODEL
# -------------------------
@component(
    base_image="python:3.11",
    packages_to_install=["--default-timeout=120", "pandas", "scikit-learn"]
)
def train_model(
    train_dataset: Input[Dataset],
    output_model: Output[Model]
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    df = pd.read_csv(train_dataset.path)
    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with open(output_model.path, "wb") as f:
        pickle.dump(model, f)
    print("Model trained successfully.")


# -------------------------
# COMPONENT 3: EVALUATE MODEL
# -------------------------
@component(
    base_image="python:3.11",
    packages_to_install=["--default-timeout=120", "pandas", "scikit-learn"]
)
def evaluate_model(
    test_dataset: Input[Dataset],
    input_model: Input[Model]
):
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(test_dataset.path)
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    with open(input_model.path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")


# -------------------------
# PIPELINE ORCHESTRATION
# -------------------------
@dsl.pipeline(
    name="iris-ml-pipeline",
    description="A clean, linear pipeline that splits data before processing."
)
def iris_pipeline():
    load_data_task = load_data()
    
    train_model_task = train_model(
        train_dataset=load_data_task.outputs['train_dataset']
    )
    
    evaluate_model_task = evaluate_model(
        test_dataset=load_data_task.outputs['test_dataset'],
        input_model=train_model_task.outputs['output_model']
    )


# -------------------------
# COMPILATION
# -------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_pipeline.yaml"
    )
    print("Pipeline compiled successfully to iris_pipeline.yaml")
