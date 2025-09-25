"""
Building footprint API endpoints
CRUD operations for building footprint data
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from app.core.database import get_db
from app.models.building_footprint import BuildingFootprint
from app.models.user import User
from app.api.deps import get_current_user, get_premium_user
from app.schemas.building_schemas import (
    BuildingFootprintResponse, BuildingFootprintFilter,
    BuildingFootprintUpdate, BuildingStatistics
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[BuildingFootprintResponse])
async def get_buildings(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    state_name: Optional[str] = Query(None, description="Filter by state name"),
    confidence_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score"),
    area_min: Optional[float] = Query(None, ge=0.0, description="Minimum building area"),
    area_max: Optional[float] = Query(None, ge=0.0, description="Maximum building area"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get building footprints with filtering and pagination
    
    Returns a list of building footprints based on the specified filters.
    """
    try:
        query = db.query(BuildingFootprint)
        
        # Apply filters
        if state_name:
            query = query.filter(BuildingFootprint.state_name == state_name)
        
        if confidence_min is not None:
            query = query.filter(BuildingFootprint.confidence_score >= confidence_min)
        
        if area_min is not None:
            query = query.filter(BuildingFootprint.area >= area_min)
        
        if area_max is not None:
            query = query.filter(BuildingFootprint.area <= area_max)
        
        # For non-admin users, only show their own buildings
        if current_user.role.value != "admin":
            query = query.filter(BuildingFootprint.user_id == current_user.id)
        
        # Order by creation date (newest first)
        query = query.order_by(desc(BuildingFootprint.created_at))
        
        # Apply pagination
        buildings = query.offset(skip).limit(limit).all()
        
        logger.info(f"✅ Retrieved {len(buildings)} buildings for user {current_user.username}")
        
        return buildings
        
    except Exception as e:
        logger.error(f"❌ Failed to get buildings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve buildings")

@router.get("/{building_id}", response_model=BuildingFootprintResponse)
async def get_building(
    building_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific building footprint by ID
    """
    try:
        building = db.query(BuildingFootprint).filter(
            BuildingFootprint.id == building_id
        ).first()
        
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Check ownership for non-admin users
        if current_user.role.value != "admin" and building.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return building
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get building {building_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve building")

@router.put("/{building_id}", response_model=BuildingFootprintResponse)
async def update_building(
    building_id: int,
    building_update: BuildingFootprintUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a building footprint
    
    Allows users to update their building footprint data.
    """
    try:
        building = db.query(BuildingFootprint).filter(
            BuildingFootprint.id == building_id
        ).first()
        
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Check ownership for non-admin users
        if current_user.role.value != "admin" and building.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        for field, value in building_update.dict(exclude_unset=True).items():
            if hasattr(building, field):
                setattr(building, field, value)
        
        building.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(building)
        
        logger.info(f"✅ Building {building_id} updated by {current_user.username}")
        
        return building
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update building {building_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update building")

@router.delete("/{building_id}")
async def delete_building(
    building_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a building footprint
    
    Allows users to delete their building footprint data.
    """
    try:
        building = db.query(BuildingFootprint).filter(
            BuildingFootprint.id == building_id
        ).first()
        
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Check ownership for non-admin users
        if current_user.role.value != "admin" and building.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        db.delete(building)
        db.commit()
        
        logger.info(f"✅ Building {building_id} deleted by {current_user.username}")
        
        return {"status": "success", "message": "Building deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete building {building_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete building")

@router.get("/by-state/{state_name}", response_model=List[BuildingFootprintResponse])
async def get_buildings_by_state(
    state_name: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_premium_user)
):
    """
    Get all buildings in a specific state
    
    Premium feature: Access to state-wide building data.
    """
    try:
        query = db.query(BuildingFootprint).filter(
            BuildingFootprint.state_name == state_name
        )
        
        # For non-admin users, only show their own buildings
        if current_user.role.value != "admin":
            query = query.filter(BuildingFootprint.user_id == current_user.id)
        
        buildings = query.order_by(desc(BuildingFootprint.created_at)).offset(skip).limit(limit).all()
        
        logger.info(f"✅ Retrieved {len(buildings)} buildings for state {state_name}")
        
        return buildings
        
    except Exception as e:
        logger.error(f"❌ Failed to get buildings for state {state_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve state buildings")

@router.get("/statistics/overview")
async def get_building_statistics(
    state_name: Optional[str] = Query(None, description="Filter statistics by state"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get building footprint statistics and analytics
    
    Returns comprehensive statistics about extracted building footprints.
    """
    try:
        from sqlalchemy import func
        
        query = db.query(BuildingFootprint)
        
        # Apply state filter if provided
        if state_name:
            query = query.filter(BuildingFootprint.state_name == state_name)
        
        # For non-admin users, only show their own buildings
        if current_user.role.value != "admin":
            query = query.filter(BuildingFootprint.user_id == current_user.id)
        
        # Calculate statistics
        total_buildings = query.count()
        
        if total_buildings == 0:
            return {
                "total_buildings": 0,
                "average_area": 0,
                "average_confidence": 0,
                "state_distribution": {},
                "extraction_methods": {},
                "quality_metrics": {}
            }
        
        # Average area and confidence
        avg_area = db.query(func.avg(BuildingFootprint.area)).filter(
            query.whereclause
        ).scalar() or 0
        
        avg_confidence = db.query(func.avg(BuildingFootprint.confidence_score)).filter(
            query.whereclause
        ).scalar() or 0
        
        # State distribution
        state_dist = db.query(
            BuildingFootprint.state_name,
            func.count(BuildingFootprint.id).label('count')
        ).filter(
            query.whereclause
        ).group_by(BuildingFootprint.state_name).all()
        
        state_distribution = {state: count for state, count in state_dist if state}
        
        # Extraction methods
        method_dist = db.query(
            BuildingFootprint.extraction_method,
            func.count(BuildingFootprint.id).label('count')
        ).filter(
            query.whereclause
        ).group_by(BuildingFootprint.extraction_method).all()
        
        extraction_methods = {method: count for method, count in method_dist if method}
        
        # Quality metrics
        high_quality_count = query.filter(BuildingFootprint.quality_score >= 0.8).count()
        regularized_count = query.filter(BuildingFootprint.regularized == True).count()
        
        statistics = {
            "total_buildings": total_buildings,
            "average_area": float(avg_area),
            "average_confidence": float(avg_confidence),
            "state_distribution": state_distribution,
            "extraction_methods": extraction_methods,
            "quality_metrics": {
                "high_quality_buildings": high_quality_count,
                "high_quality_percentage": (high_quality_count / total_buildings) * 100,
                "regularized_buildings": regularized_count,
                "regularized_percentage": (regularized_count / total_buildings) * 100
            }
        }
        
        logger.info(f"✅ Generated building statistics for user {current_user.username}")
        
        return statistics
        
    except Exception as e:
        logger.error(f"❌ Failed to generate building statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate statistics")

@router.post("/export")
async def export_buildings(
    state_name: Optional[str] = None,
    format: str = Query("geojson", regex="^(geojson|shapefile|csv)$"),
    confidence_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_premium_user)
):
    """
    Export building footprints in various formats
    
    Premium feature: Export building data in GeoJSON, Shapefile, or CSV format.
    """
    try:
        query = db.query(BuildingFootprint)
        
        # Apply filters
        if state_name:
            query = query.filter(BuildingFootprint.state_name == state_name)
        
        if confidence_min is not None:
            query = query.filter(BuildingFootprint.confidence_score >= confidence_min)
        
        # For non-admin users, only export their own buildings
        if current_user.role.value != "admin":
            query = query.filter(BuildingFootprint.user_id == current_user.id)
        
        buildings = query.all()
        
        if not buildings:
            raise HTTPException(status_code=404, detail="No buildings found for export")
        
        # Create export task (this would be a Celery task in production)
        from app.tasks.data_export import export_buildings_task
        
        task_result = export_buildings_task.delay(
            building_ids=[b.id for b in buildings],
            export_format=format,
            user_id=current_user.id
        )
        
        logger.info(f"✅ Export task created for {len(buildings)} buildings")
        
        return {
            "status": "export_started",
            "task_id": task_result.id,
            "building_count": len(buildings),
            "format": format,
            "message": "Export task has been queued. Check task status for completion."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create export task: {e}")
        raise HTTPException(status_code=500, detail="Failed to create export task")