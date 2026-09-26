from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()

    def run(self):
        raw_path = self.data_ingestion.initiate()
        train_path, val_path, test_path = self.data_transformation.initiate(raw_path)
        return train_path, val_path, test_path


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_path, val_path, test_path = train_pipeline.run()
    model_trainer = ModelTrainer()
    val_metrics = model_trainer.initiate(train_path, val_path)
    print(val_metrics)
