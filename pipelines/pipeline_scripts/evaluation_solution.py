# Evaluation
import sagemaker

import param

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

def get_evaluation_processor(docker_image_name, role) -> ScriptProcessor:
    
    sm_sess = sagemaker.session.Session()

    # Processing step for evaluation
    processor = ScriptProcessor(
        image_uri=docker_image_name,
        command=["python3"],
        instance_type=param.processing_instance_type,
        instance_count=param.processing_instance_count,
        base_job_name="CustomerChurn/eval-script",
        sagemaker_session=sm_sess,
        role=role,
    )
    
    return processor
