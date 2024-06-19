import os
import json
from time import strftime, gmtime

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from .dataprep_solution import get_dataprep_processor
from .modeling_solution import get_modeling_estimator
from .evaluation_solution import get_evaluation_processor

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_my_solutions_vars():
    vars_path = os.path.join(".", "pipelines", "pipeline_scripts", "my_pipeline-vars.json")

    with open(vars_path, "rb") as f:
        my_vars = json.loads(f.read())
        
    return my_vars

def get_pipeline(region,
                 role=None,
                 default_bucket=None,
                 model_package_group_name="MLOpsCustomerChurnPackageGroup",  # Choose any name
                 pipeline_name="MLOpsFinalChurnMLPipeline",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
                 base_job_prefix="CustomerChurn",  # Choose any name
                ) -> Pipeline:
    
    # Get config vars
    my_vars = get_my_solutions_vars()
    bucket = my_vars["bucket"]
    prefix = my_vars["prefix"]
    region = my_vars["region"]
    docker_image_name = my_vars["docker_image_name"]
    s3uri_raw = my_vars["s3uri_raw"]
    s3_dataprep_code_uri = my_vars["s3_dataprep_code_uri"]
    s3_modeling_code_uri = my_vars["s3_modeling_code_uri"]
    train_script_name = my_vars["train_script_name"]
    s3_evaluation_code_uri = my_vars["s3_evaluation_code_uri"]
    role = my_vars["role"]

    sagemaker_session = sagemaker.session.Session()

    # Parameters for data preparation step
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=s3uri_raw # S3 URI where we stored the raw data
    )
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )

    # Add an input parameter to define the training instance type
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )


    # Cache for 30 minutes
    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")

    sklearn_processor = get_dataprep_processor(processing_instance_type, processing_instance_count, role)

    # Processing step for feature engineering
    step_process = ProcessingStep(
        name="CustomerChurnProcess",  # choose any name
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=s3_dataprep_code_uri,
        job_arguments=["--input-data", input_data],
        cache_config=cache_config
    )


    xgb_train = get_modeling_estimator(bucket,
                                       prefix,
                                       s3_modeling_code_uri, 
                                       docker_image_name,
                                       role,
                                       entry_point_script = train_script_name)


    step_train = TrainingStep(
        name="CustomerChurnTrain",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                        s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                            "train"
                        ].S3Output.S3Uri,
                        content_type="text/csv"
                     ),
            "validation": TrainingInput(
                        s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                            "validation"
                        ].S3Output.S3Uri,
                        content_type="text/csv"
                     )
        },
        cache_config=cache_config
    )     


    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    script_eval = get_evaluation_processor(docker_image_name, role)

    # Processing step for evaluation
    step_eval = ProcessingStep(
            name="CustomerChurnEval",
            processor=script_eval,
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation", source="/opt/ml/processing/evaluation"
                ),
            ],
            code=s3_evaluation_code_uri,
            property_files=[evaluation_report],
            cache_config=cache_config
    )


    # Model metrics that will be associated with RegisterModel step
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    #model_package_group_name="CustomerChurnPackageGroup"

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="CustomerChurnRegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )


    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.8,  # You can change the threshold here
    )
    step_cond = ConditionStep(
        name="CustomerChurnAccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )



    # Experiment configs
    create_date = lambda: strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    experiment_name=f"pipeline-customer-churn-prediction-xgboost-{create_date()}"
    trial_name=f"pipeline-framework-trial-{create_date()}"

    pipeline_experiment_config = PipelineExperimentConfig(
        experiment_name = experiment_name,
        trial_name = trial_name
    )


    pipeline = Pipeline(
            name=pipeline_name,
            parameters=[
                input_data,
                processing_instance_type,
                processing_instance_count,
                training_instance_type,
                model_approval_status,
            ],
            steps=[step_process, step_train, step_eval, step_cond],
            sagemaker_session=sagemaker_session,
        )
    
    return pipeline
