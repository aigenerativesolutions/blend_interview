"""
Cloud Function to trigger model retraining on GCP Compute Engine
Professional MLOps orchestration function
"""
import json
import logging
from datetime import datetime
from google.cloud import compute_v1
from google.cloud import storage
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT')
ZONE = os.environ.get('COMPUTE_ZONE', 'us-central1-a')
BUCKET_NAME = f"blend-mlops-models-{PROJECT_ID}"
MACHINE_TYPE = f"zones/{ZONE}/machineTypes/n1-standard-4"
IMAGE_FAMILY = "ubuntu-2204-lts"
IMAGE_PROJECT = "ubuntu-os-cloud"

def create_training_instance(request):
    """
    HTTP Cloud Function to create a Compute Engine VM for training
    Triggered by API call or webhook
    """
    
    try:
        # Enable CORS for all requests
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600'
            }
            return ('', 204, headers)
        
        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        # Parse request
        request_json = request.get_json(silent=True)
        request_args = request.args
        
        # Generate unique instance name
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        instance_name = f"mlops-training-{timestamp}"
        
        logger.info(f"🚀 Creating training instance: {instance_name}")
        
        # Initialize Compute Engine client
        compute_client = compute_v1.InstancesClient()
        
        # Define startup script from Cloud Storage
        startup_script_url = f"gs://{BUCKET_NAME}/scripts/training-vm-startup.sh"
        
        # Instance configuration
        instance_config = {
            "name": instance_name,
            "machine_type": MACHINE_TYPE,
            "boot_disk": {
                "initialize_params": {
                    "source_image": f"projects/{IMAGE_PROJECT}/global/images/family/{IMAGE_FAMILY}",
                    "disk_size_gb": "100",
                    "disk_type": f"projects/{PROJECT_ID}/zones/{ZONE}/diskTypes/pd-ssd"
                },
                "auto_delete": True
            },
            "network_interfaces": [
                {
                    "network": "global/networks/default",
                    "access_configs": [
                        {
                            "type": "ONE_TO_ONE_NAT",
                            "name": "External NAT"
                        }
                    ]
                }
            ],
            "service_accounts": [
                {
                    "email": "default",
                    "scopes": [
                        "https://www.googleapis.com/auth/cloud-platform"
                    ]
                }
            ],
            "metadata": {
                "items": [
                    {
                        "key": "startup-script-url",
                        "value": startup_script_url
                    },
                    {
                        "key": "shutdown-script",
                        "value": "echo 'Training VM shutting down'"
                    }
                ]
            },
            "labels": {
                "purpose": "mlops-training",
                "created-by": "cloud-function",
                "project": "blend-mlops"
            },
            "scheduling": {
                "preemptible": True  # Cost optimization for training workloads
            }
        }
        
        # Create the instance
        operation = compute_client.insert(
            project=PROJECT_ID,
            zone=ZONE,
            instance_resource=instance_config
        )
        
        logger.info(f"✅ Instance creation initiated: {operation.name}")
        
        # Store training job metadata
        training_metadata = {
            "job_id": f"training-{timestamp}",
            "instance_name": instance_name,
            "zone": ZONE,
            "machine_type": MACHINE_TYPE,
            "created_at": datetime.utcnow().isoformat(),
            "status": "creating",
            "operation_name": operation.name
        }
        
        # Upload metadata to Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"training-jobs/{timestamp}/metadata.json")
        blob.upload_from_string(json.dumps(training_metadata, indent=2))
        
        return ({
            "status": "success",
            "message": f"Training instance {instance_name} creation initiated",
            "job_id": f"training-{timestamp}",
            "instance_name": instance_name,
            "zone": ZONE,
            "operation": operation.name,
            "estimated_duration": "15-30 minutes",
            "monitoring": f"gcloud compute instances describe {instance_name} --zone={ZONE}"
        }, 200, headers)
        
    except Exception as e:
        logger.error(f"❌ Error creating training instance: {str(e)}")
        return ({
            "status": "error",
            "message": f"Failed to create training instance: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }, 500, {'Access-Control-Allow-Origin': '*'})

def get_training_status(request):
    """
    HTTP Cloud Function to check training job status
    """
    try:
        request_json = request.get_json(silent=True)
        job_id = request_json.get('job_id') if request_json else request.args.get('job_id')
        
        if not job_id:
            return {"error": "job_id parameter required"}, 400
        
        # Get job metadata from Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Extract timestamp from job_id
        timestamp = job_id.replace('training-', '')
        blob = bucket.blob(f"training-jobs/{timestamp}/metadata.json")
        
        if not blob.exists():
            return {"error": "Training job not found"}, 404
        
        metadata = json.loads(blob.download_as_text())
        
        # Check instance status
        compute_client = compute_v1.InstancesClient()
        try:
            instance = compute_client.get(
                project=PROJECT_ID,
                zone=ZONE,
                instance=metadata['instance_name']
            )
            instance_status = instance.status
        except:
            instance_status = "TERMINATED"
        
        # Check if models are uploaded (training completed)
        latest_model_blob = bucket.blob("models/latest/model_metadata.json")
        training_completed = latest_model_blob.exists()
        
        return {
            "job_id": job_id,
            "instance_name": metadata['instance_name'],
            "instance_status": instance_status,
            "training_completed": training_completed,
            "created_at": metadata['created_at'],
            "metadata": metadata
        }, 200
        
    except Exception as e:
        logger.error(f"❌ Error checking training status: {str(e)}")
        return {"error": str(e)}, 500