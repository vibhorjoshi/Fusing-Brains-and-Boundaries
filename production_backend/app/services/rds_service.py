"""
AWS RDS Service for database operations and management
Production-ready PostgreSQL RDS integration
"""

import boto3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import psycopg2
from sqlalchemy import create_engine, text
import json

from app.core.config import settings
from app.core.logging import log_error, log_performance

logger = logging.getLogger(__name__)

class RDSService:
    """AWS RDS service for database management"""
    
    def __init__(self):
        """Initialize RDS service with AWS configuration"""
        try:
            # Create RDS client
            aws_config = settings.get_aws_config()
            self.rds_client = boto3.client('rds', **aws_config)
            
            # Database configuration
            self.db_instance_identifier = settings.AWS_RDS_ENDPOINT
            self.region = settings.AWS_REGION
            
            logger.info(f"✅ RDS Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RDS service: {e}")
            raise
    
    def get_db_instance_info(self, instance_identifier: str = None) -> Dict[str, Any]:
        """
        Get RDS instance information and status
        
        Args:
            instance_identifier: RDS instance identifier
            
        Returns:
            dict: Instance information
        """
        try:
            instance_id = instance_identifier or self.db_instance_identifier
            
            if not instance_id:
                return {"error": "No RDS instance identifier configured"}
            
            response = self.rds_client.describe_db_instances(
                DBInstanceIdentifier=instance_id
            )
            
            db_instance = response['DBInstances'][0]
            
            return {
                "instance_identifier": db_instance['DBInstanceIdentifier'],
                "engine": db_instance['Engine'],
                "engine_version": db_instance['EngineVersion'],
                "instance_class": db_instance['DBInstanceClass'],
                "status": db_instance['DBInstanceStatus'],
                "endpoint": {
                    "address": db_instance['Endpoint']['Address'],
                    "port": db_instance['Endpoint']['Port']
                },
                "allocated_storage_gb": db_instance['AllocatedStorage'],
                "storage_type": db_instance['StorageType'],
                "storage_encrypted": db_instance['StorageEncrypted'],
                "multi_az": db_instance['MultiAZ'],
                "publicly_accessible": db_instance['PubliclyAccessible'],
                "vpc_security_groups": [
                    {
                        "group_id": sg['VpcSecurityGroupId'],
                        "status": sg['Status']
                    }
                    for sg in db_instance['VpcSecurityGroups']
                ],
                "availability_zone": db_instance['AvailabilityZone'],
                "preferred_backup_window": db_instance['PreferredBackupWindow'],
                "backup_retention_period": db_instance['BackupRetentionPeriod'],
                "preferred_maintenance_window": db_instance['PreferredMaintenanceWindow'],
                "latest_restorable_time": db_instance['LatestRestorableTime'].isoformat(),
                "auto_minor_version_upgrade": db_instance['AutoMinorVersionUpgrade'],
                "deletion_protection": db_instance['DeletionProtection']
            }
            
        except ClientError as e:
            log_error(e, "RDS describe instance", instance=instance_identifier)
            return {"error": str(e)}
        except Exception as e:
            log_error(e, "RDS instance info", instance=instance_identifier)
            return {"error": str(e)}
    
    def get_performance_metrics(self, 
                               instance_identifier: str = None,
                               start_time: datetime = None,
                               end_time: datetime = None) -> Dict[str, Any]:
        """
        Get RDS performance metrics from CloudWatch
        
        Args:
            instance_identifier: RDS instance identifier
            start_time: Start time for metrics (default: 1 hour ago)
            end_time: End time for metrics (default: now)
            
        Returns:
            dict: Performance metrics
        """
        try:
            instance_id = instance_identifier or self.db_instance_identifier
            
            if not instance_id:
                return {"error": "No RDS instance identifier configured"}
            
            # Default time range: last hour
            end_time = end_time or datetime.utcnow()
            start_time = start_time or (end_time - timedelta(hours=1))
            
            # Create CloudWatch client
            aws_config = settings.get_aws_config()
            cloudwatch = boto3.client('cloudwatch', **aws_config)
            
            # Define metrics to retrieve
            metrics = [
                'CPUUtilization',
                'DatabaseConnections', 
                'FreeableMemory',
                'FreeStorageSpace',
                'ReadIOPS',
                'WriteIOPS',
                'ReadLatency',
                'WriteLatency',
                'ReadThroughput',
                'WriteThroughput'
            ]
            
            performance_data = {}
            
            for metric_name in metrics:
                try:
                    response = cloudwatch.get_metric_statistics(
                        Namespace='AWS/RDS',
                        MetricName=metric_name,
                        Dimensions=[
                            {
                                'Name': 'DBInstanceIdentifier',
                                'Value': instance_id
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=300,  # 5-minute intervals
                        Statistics=['Average', 'Maximum']
                    )
                    
                    datapoints = response.get('Datapoints', [])
                    if datapoints:
                        # Sort by timestamp
                        datapoints.sort(key=lambda x: x['Timestamp'])
                        
                        latest = datapoints[-1]
                        performance_data[metric_name] = {
                            'current_average': latest['Average'],
                            'current_maximum': latest['Maximum'],
                            'timestamp': latest['Timestamp'].isoformat(),
                            'unit': latest['Unit']
                        }
                        
                        # Calculate trends
                        if len(datapoints) > 1:
                            previous = datapoints[-2]
                            trend = ((latest['Average'] - previous['Average']) / previous['Average']) * 100
                            performance_data[metric_name]['trend_percent'] = round(trend, 2)
                    
                except Exception as e:
                    logger.warning(f"Failed to get metric {metric_name}: {e}")
                    performance_data[metric_name] = {"error": str(e)}
            
            return {
                "instance_identifier": instance_id,
                "metrics_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                },
                "performance_metrics": performance_data
            }
            
        except Exception as e:
            log_error(e, "RDS performance metrics", instance=instance_identifier)
            return {"error": str(e)}
    
    def create_db_snapshot(self, 
                          instance_identifier: str = None,
                          snapshot_identifier: str = None) -> Dict[str, Any]:
        """
        Create manual database snapshot
        
        Args:
            instance_identifier: RDS instance identifier
            snapshot_identifier: Snapshot identifier (auto-generated if not provided)
            
        Returns:
            dict: Snapshot information
        """
        try:
            instance_id = instance_identifier or self.db_instance_identifier
            
            if not instance_id:
                return {"error": "No RDS instance identifier configured"}
            
            # Generate snapshot identifier if not provided
            if not snapshot_identifier:
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                snapshot_identifier = f"{instance_id}-manual-snapshot-{timestamp}"
            
            response = self.rds_client.create_db_snapshot(
                DBSnapshotIdentifier=snapshot_identifier,
                DBInstanceIdentifier=instance_id,
                Tags=[
                    {
                        'Key': 'CreatedBy',
                        'Value': 'BuildingFootprintAI'
                    },
                    {
                        'Key': 'Purpose',
                        'Value': 'Manual backup'
                    },
                    {
                        'Key': 'CreatedAt',
                        'Value': datetime.utcnow().isoformat()
                    }
                ]
            )
            
            snapshot = response['DBSnapshot']
            
            logger.info(f"✅ DB snapshot created: {snapshot_identifier}")
            
            return {
                "snapshot_identifier": snapshot['DBSnapshotIdentifier'],
                "instance_identifier": snapshot['DBInstanceIdentifier'],
                "status": snapshot['Status'],
                "snapshot_create_time": snapshot['SnapshotCreateTime'].isoformat(),
                "engine": snapshot['Engine'],
                "allocated_storage_gb": snapshot['AllocatedStorage'],
                "port": snapshot['Port']
            }
            
        except ClientError as e:
            log_error(e, "RDS create snapshot", instance=instance_identifier)
            return {"error": str(e)}
    
    def list_db_snapshots(self, 
                         instance_identifier: str = None,
                         max_records: int = 20) -> List[Dict[str, Any]]:
        """
        List database snapshots
        
        Args:
            instance_identifier: RDS instance identifier
            max_records: Maximum number of snapshots to return
            
        Returns:
            list: List of snapshots
        """
        try:
            instance_id = instance_identifier or self.db_instance_identifier
            
            describe_args = {
                'MaxRecords': max_records,
                'SnapshotType': 'manual'  # Only manual snapshots
            }
            
            if instance_id:
                describe_args['DBInstanceIdentifier'] = instance_id
            
            response = self.rds_client.describe_db_snapshots(**describe_args)
            
            snapshots = []
            for snapshot in response.get('DBSnapshots', []):
                snapshots.append({
                    "snapshot_identifier": snapshot['DBSnapshotIdentifier'],
                    "instance_identifier": snapshot['DBInstanceIdentifier'],
                    "status": snapshot['Status'],
                    "snapshot_create_time": snapshot['SnapshotCreateTime'].isoformat(),
                    "engine": snapshot['Engine'],
                    "allocated_storage_gb": snapshot['AllocatedStorage'],
                    "encrypted": snapshot['Encrypted'],
                    "percent_progress": snapshot['PercentProgress']
                })
            
            return snapshots
            
        except ClientError as e:
            log_error(e, "RDS list snapshots", instance=instance_identifier)
            return []
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """
        Test database connectivity and performance
        
        Returns:
            dict: Connectivity test results
        """
        try:
            start_time = datetime.utcnow()
            
            # Create test connection
            engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT 1 as test_value"))
                test_value = result.fetchone()[0]
                
                # Test database version
                version_result = conn.execute(text("SELECT version()"))
                db_version = version_result.fetchone()[0]
                
                # Test current time
                time_result = conn.execute(text("SELECT NOW()"))
                db_time = time_result.fetchone()[0]
                
                # Test table count
                table_count_result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_count = table_count_result.fetchone()[0]
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            log_performance("rds_connectivity_test", duration)
            
            return {
                "connectivity": "success",
                "test_value": test_value,
                "database_version": db_version.split()[1] if db_version else "unknown",
                "database_time": db_time.isoformat() if db_time else None,
                "table_count": table_count,
                "response_time_ms": duration * 1000,
                "database_url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "configured"
            }
            
        except Exception as e:
            log_error(e, "RDS connectivity test")
            return {
                "connectivity": "failed",
                "error": str(e),
                "database_url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "configured"
            }
    
    def get_database_size_info(self) -> Dict[str, Any]:
        """
        Get database size and storage information
        
        Returns:
            dict: Database size information
        """
        try:
            engine = create_engine(settings.DATABASE_URL)
            
            with engine.connect() as conn:
                # Get database size
                db_size_result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
                           pg_database_size(current_database()) as db_size_bytes
                """))
                db_size_row = db_size_result.fetchone()
                
                # Get table sizes
                table_sizes_result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """))
                
                table_sizes = []
                for row in table_sizes_result:
                    table_sizes.append({
                        "table_name": row[1],
                        "size_human": row[2],
                        "size_bytes": row[3]
                    })
                
                # Get connection info
                connections_result = conn.execute(text("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                """))
                connections_row = connections_result.fetchone()
            
            return {
                "database_size": {
                    "size_human": db_size_row[0],
                    "size_bytes": db_size_row[1],
                    "size_mb": db_size_row[1] / (1024 * 1024)
                },
                "table_sizes": table_sizes,
                "connections": {
                    "total": connections_row[0],
                    "active": connections_row[1],
                    "idle": connections_row[2]
                }
            }
            
        except Exception as e:
            log_error(e, "RDS database size info")
            return {"error": str(e)}

# Global RDS service instance
rds_service = RDSService()