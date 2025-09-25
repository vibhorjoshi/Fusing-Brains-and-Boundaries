"""
ML processing tasks for Celery
Background task processing for building extraction and state processing
"""

from celery import current_task
from sqlalchemy.orm import sessionmaker
import logging
from typing import Optional, Dict, Any, List
import traceback
from datetime import datetime
import os
import numpy as np

from app.core.celery_app import celery_app
from app.core.database import engine
from app.models.processing_job import ProcessingJob, JobStatus
from app.models.building_footprint import BuildingFootprint
from app.services.s3_service import s3_service
from app.core.config import settings

logger = logging.getLogger(__name__)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """Get database session for tasks"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise

@celery_app.task(bind=True, name="extract_buildings_from_image")
def extract_buildings_from_image(
    self,
    job_id: int,
    confidence_threshold: float = 0.5,
    apply_regularization: bool = True
):
    """
    Extract building footprints from satellite imagery using Mask R-CNN
    
    Args:
        job_id: Processing job ID
        confidence_threshold: Confidence threshold for detections
        apply_regularization: Whether to apply geometric regularization
    """
    db = None
    try:
        db = get_db_session()
        
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"🚀 Starting building extraction job {job_id}")
        
        # Load ML model (mock implementation - replace with actual model)
        self.update_state(
            state='PROGRESS',
            meta={'current_step': 'Loading ML model', 'progress': 10}
        )
        
        # Simulate model loading
        import time
        time.sleep(2)
        
        # Process image (mock implementation)
        self.update_state(
            state='PROGRESS', 
            meta={'current_step': 'Processing image', 'progress': 30}
        )
        
        # Mock building extraction results
        mock_buildings = [
            {
                'confidence': 0.85,
                'polygon': [[0, 0], [100, 0], [100, 100], [0, 100]],
                'area': 10000.0,
                'quality_score': 0.9
            },
            {
                'confidence': 0.75,
                'polygon': [[200, 200], [300, 200], [300, 300], [200, 300]],
                'area': 10000.0,
                'quality_score': 0.8
            },
            {
                'confidence': 0.92,
                'polygon': [[400, 100], [500, 100], [500, 200], [400, 200]],
                'area': 10000.0,
                'quality_score': 0.95
            }
        ]
        
        # Filter by confidence threshold
        filtered_buildings = [
            b for b in mock_buildings 
            if b['confidence'] >= confidence_threshold
        ]
        
        self.update_state(
            state='PROGRESS',
            meta={'current_step': 'Applying regularization', 'progress': 60}
        )
        
        # Apply regularization if enabled
        if apply_regularization:
            time.sleep(1)  # Mock regularization processing
            logger.info(f"✅ Applied regularization to {len(filtered_buildings)} buildings")
        
        # Save results to database
        self.update_state(
            state='PROGRESS',
            meta={'current_step': 'Saving results', 'progress': 80}
        )
        
        saved_buildings = []
        for building_data in filtered_buildings:
            building = BuildingFootprint(
                job_id=job_id,
                user_id=job.user_id,
                geometry_type="Polygon",
                coordinates=building_data['polygon'],
                area=building_data['area'],
                confidence_score=building_data['confidence'],
                quality_score=building_data['quality_score'],
                extraction_method="mask_rcnn",
                regularized=apply_regularization
            )
            
            db.add(building)
            saved_buildings.append(building)
        
        db.commit()
        
        # Update job completion
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress_percentage = 100.0
        job.output_results = {
            'total_buildings': len(saved_buildings),
            'confidence_threshold': confidence_threshold,
            'regularization_applied': apply_regularization,
            'average_confidence': sum(b['confidence'] for b in filtered_buildings) / len(filtered_buildings) if filtered_buildings else 0,
            'processing_time_seconds': (datetime.utcnow() - job.started_at).total_seconds()
        }
        
        db.commit()
        
        logger.info(f"✅ Building extraction completed: {len(saved_buildings)} buildings extracted")
        
        return {
            'status': 'completed',
            'buildings_extracted': len(saved_buildings),
            'job_id': job_id,
            'processing_time': job.output_results['processing_time_seconds']
        }
        
    except Exception as e:
        error_msg = f"Building extraction failed: {str(e)}"
        logger.error(f"❌ {error_msg}\n{traceback.format_exc()}")
        
        if db:
            try:
                job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = error_msg
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception as db_error:
                logger.error(f"❌ Failed to update job status: {db_error}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg}
        )
        
        raise Exception(error_msg)
        
    finally:
        if db:
            db.close()

@celery_app.task(bind=True, name="process_state_data")
def process_state_data(
    self,
    job_id: int,
    state_name: str,
    layer_type: str = "avg"
):
    """
    Process building footprint data for a complete US state
    
    Args:
        job_id: Processing job ID
        state_name: US state name
        layer_type: Type of raster layer to process
    """
    db = None
    try:
        db = get_db_session()
        
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"🚀 Starting state processing job {job_id} for {state_name}")
        
        # Mock state processing - in production, this would:
        # 1. Load state raster data from S3
        # 2. Tile the large raster into manageable chunks
        # 3. Process each tile with the ML model
        # 4. Aggregate results and save to database
        
        self.update_state(
            state='PROGRESS',
            meta={'current_step': f'Loading {state_name} data', 'progress': 10}
        )
        
        import time
        time.sleep(3)  # Mock data loading
        
        # Mock tile processing
        total_tiles = 100  # Mock number of tiles for the state
        processed_tiles = 0
        total_buildings = 0
        
        for tile_idx in range(total_tiles):
            # Update progress
            processed_tiles += 1
            progress = 10 + (processed_tiles / total_tiles) * 80
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'current_step': f'Processing tile {tile_idx + 1}/{total_tiles}',
                    'progress': progress,
                    'tiles_processed': processed_tiles,
                    'total_tiles': total_tiles
                }
            )
            
            # Mock tile processing (in production, run ML model on tile)
            time.sleep(0.1)  # Mock processing time
            
            # Mock building detection results for this tile
            buildings_in_tile = np.random.randint(0, 10)  # 0-10 buildings per tile
            total_buildings += buildings_in_tile
            
            # Save mock buildings to database (sample)
            if buildings_in_tile > 0 and tile_idx < 5:  # Only save first 5 tiles for demo
                for i in range(buildings_in_tile):
                    building = BuildingFootprint(
                        job_id=job_id,
                        user_id=job.user_id,
                        geometry_type="Polygon",
                        coordinates=[[
                            [tile_idx * 100 + i * 10, tile_idx * 100 + i * 10],
                            [tile_idx * 100 + i * 10 + 50, tile_idx * 100 + i * 10],
                            [tile_idx * 100 + i * 10 + 50, tile_idx * 100 + i * 10 + 50],
                            [tile_idx * 100 + i * 10, tile_idx * 100 + i * 10 + 50]
                        ]],
                        area=2500.0,  # 50x50 building
                        confidence_score=0.7 + np.random.random() * 0.3,
                        quality_score=0.8 + np.random.random() * 0.2,
                        extraction_method="mask_rcnn_state",
                        state_name=state_name,
                        layer_type=layer_type
                    )
                    
                    db.add(building)
            
            # Commit every 10 tiles
            if tile_idx % 10 == 0:
                db.commit()
        
        # Final save
        self.update_state(
            state='PROGRESS',
            meta={'current_step': 'Finalizing results', 'progress': 95}
        )
        
        db.commit()
        
        # Update job completion
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress_percentage = 100.0
        job.output_results = {
            'state_name': state_name,
            'layer_type': layer_type,
            'total_tiles_processed': total_tiles,
            'total_buildings_detected': total_buildings,
            'processing_time_seconds': (datetime.utcnow() - job.started_at).total_seconds()
        }
        
        db.commit()
        
        logger.info(f"✅ State processing completed for {state_name}: {total_buildings} buildings detected")
        
        return {
            'status': 'completed',
            'state_name': state_name,
            'total_buildings': total_buildings,
            'tiles_processed': total_tiles,
            'processing_time': job.output_results['processing_time_seconds']
        }
        
    except Exception as e:
        error_msg = f"State processing failed for {state_name}: {str(e)}"
        logger.error(f"❌ {error_msg}\n{traceback.format_exc()}")
        
        if db:
            try:
                job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = error_msg
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception as db_error:
                logger.error(f"❌ Failed to update job status: {db_error}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg}
        )
        
        raise Exception(error_msg)
        
    finally:
        if db:
            db.close()