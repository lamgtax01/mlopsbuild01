# DataPrep
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor

def get_dataprep_processor(
    processing_instance_type,
    processing_instance_count,
    role,
    base_job_prefix="CustomerChurn"
) -> SKLearnProcessor:
    
    sm_sess = sagemaker.session.Session()
    
    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-CustomerChurn-preprocess",  # choose any name
        sagemaker_session=sm_sess,
        role=role,
    )
    return sklearn_processor
