import logging
import azure.functions as func
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Experiment, Datastore
from azureml.exceptions import ProjectSystemException
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import PyTorch
import shutil
import os
import json
import time


def main(req: func.HttpRequest) -> (func.HttpResponse):
    logging.info('Python HTTP trigger function processed a request.')

    # For now this can be a POST where we have <base url>/api/HttpTrigger?start=<any string>
    image_url = req.params.get('start')
    logging.info(type(image_url))

    # Use service principal secrets to create authentication vehicle and 
    # define workspace object
    try:    
        svc_pr = ServicePrincipalAuthentication(
            tenant_id=os.getenv('TENANT_ID', ''),
            service_principal_id=os.getenv('APP_ID', ''),
            service_principal_password=os.getenv('PRINCIPAL_PASSWORD', ''))

        ws = Workspace(subscription_id=os.getenv('AZURE_SUB', ''),
                    resource_group=os.getenv('RESOURCE_GROUP', ''),
                    workspace_name=os.getenv('WORKSPACE_NAME',''),
                    auth=svc_pr)
        print("Found workspace {} at location {} using Azure CLI \
            authentication".format(ws.name, ws.location))
    # Usually because authentication didn't work
    except ProjectSystemException as err:
        print('Authentication did not work.')
        return json.dumps('ProjectSystemException')
    # Need to create the workspace
    except Exception as err:
        ws = Workspace.create(name=os.getenv('WORKSPACE_NAME', ''),
                    subscription_id=os.getenv('AZURE_SUB', ''), 
                    resource_group=os.getenv('RESOURCE_GROUP', ''),
                    create_resource_group=True,
                    location='westus', # Or other supported Azure region   
                    auth=svc_pr)
        print("Created workspace {} at location {}".format(ws.name, ws.location))

       

    # choose a name for your cluster - under 16 characters
    cluster_name = "gpuforpytorch"

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except ComputeTargetException:
        print('Creating a new compute target...')
        # AML Compute config - if max_nodes are set, it becomes persistent storage that scales
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                            min_nodes=0,
                                                            max_nodes=2)
        # create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current cluster. 
    # print(compute_target.get_status().serialize())

    # # Create a project directory and copy training script to ii
    project_folder = os.path.join(os.getcwd(), 'HttpTrigger', 'project')
    # os.makedirs(project_folder, exist_ok=True)
    # shutil.copy(os.path.join(os.getcwd(), 'HttpTrigger', 'pytorch_train.py'), project_folder)

    # Create an experiment
    experiment_name = 'fish-no-fish'
    experiment = Experiment(ws, name=experiment_name)

    # Use an AML Data Store for training data
    ds = Datastore.register_azure_blob_container(workspace=ws, 
        datastore_name='funcdefaultdatastore', 
        container_name=os.getenv('STORAGE_CONTAINER_NAME_TRAINDATA', ''),
        account_name=os.getenv('STORAGE_ACCOUNT_NAME', ''), 
        account_key=os.getenv('STORAGE_ACCOUNT_KEY', ''),
        create_if_not_exists=True)

    # Use an AML Data Store to save models back up to
    ds_models = Datastore.register_azure_blob_container(workspace=ws, 
        datastore_name='modelsdatastorage', 
        container_name=os.getenv('STORAGE_CONTAINER_NAME_MODELS', ''),
        account_name=os.getenv('STORAGE_ACCOUNT_NAME', ''), 
        account_key=os.getenv('STORAGE_ACCOUNT_KEY', ''),
        create_if_not_exists=True)

    # Set up for training ("trans" flag means - use transfer learning and 
    # this should download a model on compute)
    # Using /tmp to store model and info due to the fact that
    # creating new folders and files on the Azure Function host
    # will trigger the function to restart.
    script_params = {
        '--data_dir': ds.as_mount(),
        '--num_epochs': 30,
        '--learning_rate': 0.01,
        '--output_dir': '/tmp/outputs',
        '--trans': 'True'
    }

    # Instantiate PyTorch estimator with upload of final model to
    # a specified blob storage container (this can be anything)
    estimator = PyTorch(source_directory=project_folder, 
                        script_params=script_params,
                        compute_target=compute_target,
                        entry_script='pytorch_train.py',
                        use_gpu=True,
                        inputs=[ds_models.as_upload(path_on_compute='./outputs/model_finetuned.pth')])

    run = experiment.submit(estimator)
    print(run.get_details())
    
    # # The following would certainly be blocking, but that's ok for debugging
    # while run.get_status() not in ['Completed', 'Failed']: # For example purposes only, not exhaustive
    #    print('Run {} not in terminal state'.format(run.id))
    #    time.sleep(10)

    return json.dumps(run.get_status())
