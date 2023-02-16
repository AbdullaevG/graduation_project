from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .train_model_params import ModelParams, EmbeddingModelParams


class TrainingPipelineParams:
    """Structure for pipeline parameters"""
    modelparams: ModelParams
    embeddingmodelparams: EmbeddingModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)

def read_training_pipeline_params(path: str):
    """Read config for model training"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

