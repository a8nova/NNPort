"""Job manager for tracking porting jobs"""
import uuid
from typing import Dict, Any, Optional
from datetime import datetime


class JobManager:
    """Manages porting job lifecycle and state"""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, metadata: Dict[str, Any]) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "state": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "logs": [],
            "error": None
        }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job state"""
        if job_id not in self._jobs:
            return False
        
        self._jobs[job_id].update(updates)
        self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        return True
    
    def list_jobs(self) -> list:
        """List all jobs"""
        return list(self._jobs.values())
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


