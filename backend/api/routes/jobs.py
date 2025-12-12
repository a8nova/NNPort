"""Job-related API routes"""
from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter(prefix="/jobs")


# Note: The job_manager is initialized in main.py
# These routes will access it through dependency injection or module-level reference

@router.get("/")
async def list_jobs():
    """List all jobs"""
    # This would typically use dependency injection
    # For now, return empty list as placeholder
    return {"jobs": []}


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Get job details by ID"""
    # Placeholder - actual implementation would query job_manager
    raise HTTPException(status_code=404, detail="Job not found")


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    # Placeholder - actual implementation would use job_manager
    raise HTTPException(status_code=404, detail="Job not found")


