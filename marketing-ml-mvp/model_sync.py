"""
Model Synchronization Service
Downloads and manages models from GCP Cloud Storage for local usage
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import hashlib
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSyncManager:
    """
    Professional Model Synchronization Manager
    Handles model downloads, versioning, and caching from GCP Cloud Storage
    """
    
    def __init__(self, 
                 project_id: str = None,
                 bucket_name: str = None,
                 local_models_dir: str = "models",
                 cache_duration_hours: int = 1):
        """
        Initialize Model Sync Manager
        
        Args:
            project_id: GCP project ID
            bucket_name: Cloud Storage bucket name
            local_models_dir: Local directory for model storage
            cache_duration_hours: Hours before checking for updates
        """
        # Configuration
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME') or f"blend-mlops-models-{self.project_id}"
        self.local_models_dir = Path(local_models_dir)
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        # Paths
        self.local_models_dir.mkdir(exist_ok=True)
        self.cache_info_file = self.local_models_dir / "sync_cache.json"
        self.current_model_dir = self.local_models_dir / "current"
        self.current_model_dir.mkdir(exist_ok=True)
        
        # Initialize GCS client
        self.gcs_client = self._initialize_gcs_client()
        
        logger.info(f"🔄 ModelSync initialized - Project: {self.project_id}, Bucket: {self.bucket_name}")
    
    def _initialize_gcs_client(self):
        """Initialize Google Cloud Storage client with proper authentication"""
        try:
            from google.cloud import storage
            
            # Try different authentication methods
            auth_methods = [
                ("Service Account Key", lambda: os.getenv('GOOGLE_APPLICATION_CREDENTIALS')),
                ("Default Credentials", lambda: storage.Client()),
            ]
            
            for method_name, auth_func in auth_methods:
                try:
                    if method_name == "Service Account Key":
                        cred_path = auth_func()
                        if cred_path and os.path.exists(cred_path):
                            client = storage.Client.from_service_account_json(cred_path)
                            logger.info(f"✅ GCS authenticated via {method_name}")
                            return client
                    else:
                        client = auth_func()
                        # Test the connection
                        list(client.list_buckets(max_results=1))
                        logger.info(f"✅ GCS authenticated via {method_name}")
                        return client
                except Exception as e:
                    logger.warning(f"⚠️ {method_name} failed: {e}")
                    continue
            
            logger.error("❌ Could not authenticate with Google Cloud Storage")
            return None
            
        except ImportError:
            logger.warning("⚠️ google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            return None
    
    def _load_cache_info(self) -> Dict[str, Any]:
        """Load cache information from local file"""
        if self.cache_info_file.exists():
            try:
                with open(self.cache_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cache info: {e}")
        
        return {}
    
    def _save_cache_info(self, cache_info: Dict[str, Any]):
        """Save cache information to local file"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save cache info: {e}")
    
    def _get_remote_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Get model metadata from Cloud Storage"""
        if not self.gcs_client:
            logger.warning("⚠️ GCS client not available")
            return None
        
        try:
            bucket = self.gcs_client.bucket(self.bucket_name)
            
            # Check if latest model metadata exists
            metadata_blob = bucket.blob("models/latest/model_metadata.json")
            
            if not metadata_blob.exists():
                logger.warning("⚠️ No model metadata found in Cloud Storage")
                return None
            
            # Download and parse metadata
            metadata_content = metadata_blob.download_as_text()
            metadata = json.loads(metadata_content)
            
            logger.info(f"📊 Remote model version: {metadata.get('version', 'unknown')}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get remote model metadata: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def _download_model_files(self, remote_metadata: Dict[str, Any]) -> bool:
        """Download model files from Cloud Storage"""
        if not self.gcs_client:
            return False
        
        try:
            bucket = self.gcs_client.bucket(self.bucket_name)
            
            # Files to download
            model_files = [
                "models/latest/final_model.pkl",
                "models/latest/xgb_base_model.json", 
                "models/latest/temperature_calibrator.pkl",
                "models/latest/model_metadata.json",
                "models/latest/pipeline_summary.json"
            ]
            
            # Create temporary download directory
            temp_dir = self.local_models_dir / "temp_download"
            temp_dir.mkdir(exist_ok=True)
            
            logger.info("📥 Downloading model files...")
            
            downloaded_files = []
            for file_path in model_files:
                try:
                    blob = bucket.blob(file_path)
                    if blob.exists():
                        local_file = temp_dir / Path(file_path).name
                        blob.download_to_filename(local_file)
                        downloaded_files.append(local_file)
                        logger.info(f"✅ Downloaded: {Path(file_path).name}")
                    else:
                        logger.warning(f"⚠️ File not found in GCS: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {file_path}: {e}")
            
            if not downloaded_files:
                logger.error("❌ No model files downloaded")
                shutil.rmtree(temp_dir)
                return False
            
            # Move files to current model directory
            for file_path in downloaded_files:
                target_path = self.current_model_dir / file_path.name
                shutil.move(str(file_path), str(target_path))
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Model files downloaded to: {self.current_model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model files: {e}")
            return False
    
    def check_for_updates(self, force: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if model updates are available
        
        Args:
            force: Force check regardless of cache duration
            
        Returns:
            Tuple of (update_available, update_info)
        """
        logger.info("🔍 Checking for model updates...")
        
        # Load cache info
        cache_info = self._load_cache_info()
        last_check = cache_info.get('last_check')
        
        # Check if we need to update based on cache duration
        if not force and last_check:
            try:
                last_check_time = datetime.fromisoformat(last_check)
                if datetime.now() - last_check_time < self.cache_duration:
                    logger.info(f"⏱️ Using cached info (checked {last_check})")
                    return False, cache_info
            except Exception:
                pass
        
        # Get remote model metadata
        remote_metadata = self._get_remote_model_metadata()
        
        if not remote_metadata:
            logger.warning("⚠️ Cannot check for updates - no remote metadata")
            return False, {}
        
        # Compare versions
        local_version = cache_info.get('local_version')
        remote_version = remote_metadata.get('version')
        
        update_info = {
            'last_check': datetime.now().isoformat(),
            'remote_version': remote_version,
            'local_version': local_version,
            'remote_metadata': remote_metadata
        }
        
        if local_version != remote_version:
            logger.info(f"🆕 Model update available: {local_version} → {remote_version}")
            return True, update_info
        else:
            logger.info(f"✅ Model is up to date: {local_version}")
            # Update cache with current check time
            cache_info.update(update_info)
            self._save_cache_info(cache_info)
            return False, update_info
    
    def sync_model(self, force: bool = False) -> Dict[str, Any]:
        """
        Synchronize model from Cloud Storage
        
        Args:
            force: Force download regardless of version
            
        Returns:
            Dictionary with sync results
        """
        logger.info("🔄 Starting model synchronization...")
        
        start_time = datetime.now()
        
        try:
            # Check for updates
            update_available, update_info = self.check_for_updates(force)
            
            if not update_available and not force:
                return {
                    'success': True,
                    'action': 'no_update_needed',
                    'message': 'Model is already up to date',
                    'local_version': update_info.get('local_version'),
                    'remote_version': update_info.get('remote_version'),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            # Download model files
            remote_metadata = update_info.get('remote_metadata', {})
            download_success = self._download_model_files(remote_metadata)
            
            if not download_success:
                return {
                    'success': False,
                    'action': 'download_failed',
                    'message': 'Failed to download model files',
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            # Update cache info
            cache_info = {
                'last_sync': datetime.now().isoformat(),
                'last_check': datetime.now().isoformat(),
                'local_version': remote_metadata.get('version'),
                'remote_version': remote_metadata.get('version'),
                'sync_source': 'cloud_storage',
                'files_downloaded': True,
                'remote_metadata': remote_metadata
            }
            
            self._save_cache_info(cache_info)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Model sync completed successfully in {duration:.1f}s")
            
            return {
                'success': True,
                'action': 'model_updated',
                'message': f'Model updated to version {remote_metadata.get("version")}',
                'local_version': remote_metadata.get('version'),
                'remote_version': remote_metadata.get('version'),
                'duration': duration,
                'files_path': str(self.current_model_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Model sync failed: {e}")
            return {
                'success': False,
                'action': 'sync_failed',
                'message': f'Sync failed: {str(e)}',
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    def get_local_model_info(self) -> Dict[str, Any]:
        """Get information about the local model"""
        cache_info = self._load_cache_info()
        
        # Check if model files exist
        model_files = [
            'final_model.pkl',
            'model_metadata.json',
            'temperature_calibrator.pkl'
        ]
        
        existing_files = []
        for file_name in model_files:
            file_path = self.current_model_dir / file_name
            if file_path.exists():
                existing_files.append({
                    'name': file_name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {
            'local_version': cache_info.get('local_version'),
            'last_sync': cache_info.get('last_sync'),
            'last_check': cache_info.get('last_check'),
            'models_directory': str(self.current_model_dir),
            'existing_files': existing_files,
            'total_files': len(existing_files),
            'cache_info': cache_info
        }
    
    def auto_sync_on_startup(self) -> Dict[str, Any]:
        """Automatically sync model on application startup"""
        logger.info("🚀 Auto-sync on startup...")
        
        local_info = self.get_local_model_info()
        
        # If no local model, force sync
        if not local_info['existing_files']:
            logger.info("📥 No local model found - forcing initial sync")
            return self.sync_model(force=True)
        
        # Otherwise, check for updates
        return self.sync_model(force=False)

# Convenience functions for easy integration

def create_model_sync_manager() -> ModelSyncManager:
    """Create a configured ModelSyncManager instance"""
    return ModelSyncManager()

def quick_model_sync(force: bool = False) -> Dict[str, Any]:
    """Quick model synchronization"""
    sync_manager = create_model_sync_manager()
    return sync_manager.sync_model(force=force)

def get_model_status() -> Dict[str, Any]:
    """Get current model status"""
    sync_manager = create_model_sync_manager()
    return sync_manager.get_local_model_info()

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Synchronization Manager')
    parser.add_argument('--sync', action='store_true', help='Sync model from Cloud Storage')
    parser.add_argument('--force', action='store_true', help='Force sync regardless of cache')
    parser.add_argument('--status', action='store_true', help='Show local model status')
    parser.add_argument('--check', action='store_true', help='Check for updates')
    
    args = parser.parse_args()
    
    sync_manager = create_model_sync_manager()
    
    if args.status:
        info = sync_manager.get_local_model_info()
        print("📊 Local Model Status:")
        print(json.dumps(info, indent=2, default=str))
    
    elif args.check:
        update_available, update_info = sync_manager.check_for_updates()
        print(f"🔍 Update Available: {update_available}")
        print(json.dumps(update_info, indent=2, default=str))
    
    elif args.sync:
        result = sync_manager.sync_model(force=args.force)
        print("🔄 Sync Result:")
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print("🔄 Model Sync Manager")
        print("Usage:")
        print("  python model_sync.py --status   # Show local model status")
        print("  python model_sync.py --check    # Check for updates")
        print("  python model_sync.py --sync     # Sync model") 
        print("  python model_sync.py --sync --force  # Force sync")