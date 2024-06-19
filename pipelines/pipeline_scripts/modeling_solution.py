# Modeling
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput

from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig

import param

def get_modeling_estimator(bucket,
                           prefix,
                           s3_modeling_code_uri,
                           docker_image_name,
                           role,
                           entry_point_script = 'xgboost_customer_churn.py') -> sagemaker.estimator.Estimator:
    
    sm_sess = sagemaker.session.Session()

    # Input configs
    hyperparams = {"sagemaker_program": entry_point_script,
                   "sagemaker_submit_directory": s3_modeling_code_uri,
                   "max_depth": 5,
                   "subsample": 0.8,
                   "num_round": 600,
                   "eta": 0.2,
                   "gamma": 4,
                   "min_child_weight": 6,
                   "objective": 'binary:logistic',
                   "verbosity": 0
                  }

    # Debugger configs
    debug_rules = [
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        Rule.sagemaker(rule_configs.overtraining()),
        Rule.sagemaker(rule_configs.overfit())
    ]

    # Estimator configs
    xgb = sagemaker.estimator.Estimator(image_uri=docker_image_name,
                                        role=role,
                                        hyperparameters=hyperparams,
                                        instance_count=param.training_instance_count, 
                                        instance_type=param.training_instance_type,
                                        output_path=f's3://{bucket}/{prefix}/output',
                                        base_job_name='pipeline-xgboost-customer-churn',
                                        sagemaker_session=sm_sess,
                                        rules=debug_rules)
    
    return xgb
