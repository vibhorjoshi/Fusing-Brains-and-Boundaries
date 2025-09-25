"""
AWS S3 Service for file storage and management
Production-ready S3 integration with presigned URLs and lifecycle management
"""

import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
import mimetypes
import hashlib
from pathlib import Path
import io

from app.core.config import settings
from app.models.file_storage import FileStorage, FileType, StorageProvider
from app.core.logging import log_error, log_performance

logger = logging.getLogger(__name__)

class S3Service:
    """AWS S3 service for file operations"""
    
    def __init__(self):
        """Initialize S3 service with AWS configuration"""
        try:
            # Create S3 client with configuration
            aws_config = settings.get_aws_config()
            self.s3_client = boto3.client('s3', **aws_config)
            self.s3_resource = boto3.resource('s3', **aws_config)
            
            # Configuration
            self.bucket_name = settings.AWS_S3_BUCKET
            self.region = settings.AWS_REGION
            self.prefix = settings.AWS_S3_PREFIX
            
            logger.info(f"✅ S3 Service initialized for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 service: {e}")
            raise
    
    async def upload_file(self, 
                         file_data: BinaryIO, 
                         filename: str,
                         file_type: FileType,
                         user_id: int,
                         metadata: Dict[str, Any] = None) -> FileStorage:
        """
        Upload file to S3 and create database record
        
        Args:
            file_data: File content as binary stream
            filename: Original filename
            file_type: Type of file being uploaded
            user_id: ID of user uploading the file
            metadata: Additional file metadata
            
        Returns:
            FileStorage: Database record for uploaded file
        """
        try:
            start_time = datetime.utcnow()
            
            # Read file content
            file_content = file_data.read()
            file_size = len(file_content)
            
            # Generate file hashes
            md5_hash = hashlib.md5(file_content).hexdigest()
            sha256_hash = hashlib.sha256(file_content).hexdigest()
            
            # Generate S3 key
            timestamp = datetime.utcnow().strftime("%Y/%m/%d")
            s3_key = f"{self.prefix}{timestamp}/{user_id}/{md5_hash}_{filename}"
            
            # Prepare metadata for S3
            s3_metadata = {
                "user_id": str(user_id),
                "file_type": file_type.value,
                "original_filename": filename,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "md5_hash": md5_hash,
                "sha256_hash": sha256_hash
            }
            
            # Add custom metadata if provided
            if metadata:
                for key, value in metadata.items():
                    s3_metadata[f"custom_{key}"] = str(value)
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or 'application/octet-stream'
            
            # Upload to S3
            upload_args = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_content,
                'ContentType': content_type,
                'Metadata': s3_metadata,
                'ServerSideEncryption': 'AES256'
            }
            
            # Add lifecycle tags
            upload_args['Tagging'] = f"file_type={file_type.value}&user_id={user_id}"
            
            self.s3_client.put_object(**upload_args)
            
            # Create database record
            file_record = FileStorage(
                filename=f"{md5_hash}_{filename}",
                original_filename=filename,
                file_extension=Path(filename).suffix.lower(),
                mime_type=content_type,
                file_size_bytes=file_size,
                file_hash_md5=md5_hash,
                file_hash_sha256=sha256_hash,
                file_type=file_type,
                storage_provider=StorageProvider.AWS_S3,
                storage_path=s3_key,
                storage_bucket=self.bucket_name,
                storage_region=self.region,
                user_id=user_id,
                metadata=metadata
            )
            
            # Log performance
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_performance("s3_upload", duration, file_size_mb=file_size/(1024*1024))
            
            logger.info(f"✅ File uploaded to S3: {filename} ({file_size} bytes)")
            return file_record
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            log_error(e, "S3 upload", file=filename, error_code=error_code)
            raise Exception(f"S3 upload failed: {error_code}")
        except Exception as e:
            log_error(e, "File upload", file=filename)
            raise
    
    async def download_file(self, file_record: FileStorage) -> bytes:
        """
        Download file from S3
        
        Args:
            file_record: Database record of file to download
            
        Returns:
            bytes: File content
        """
        try:
            start_time = datetime.utcnow()
            
            response = self.s3_client.get_object(
                Bucket=file_record.storage_bucket,
                Key=file_record.storage_path
            )
            
            file_content = response['Body'].read()
            
            # Update last accessed timestamp
            file_record.last_accessed = datetime.utcnow()
            
            # Log performance
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_performance("s3_download", duration, file_size_mb=len(file_content)/(1024*1024))
            
            logger.info(f"✅ File downloaded from S3: {file_record.filename}")
            return file_content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            log_error(e, "S3 download", file=file_record.filename, error_code=error_code)
            raise Exception(f"S3 download failed: {error_code}")
        except Exception as e:
            log_error(e, "File download", file=file_record.filename)
            raise
    
    def generate_presigned_url(self, 
                              file_record: FileStorage, 
                              expires_in: int = 3600,
                              http_method: str = 'GET') -> str:
        """
        Generate presigned URL for direct S3 access
        
        Args:
            file_record: Database record of file
            expires_in: URL expiration time in seconds (default 1 hour)
            http_method: HTTP method (GET for download, PUT for upload)
            
        Returns:
            str: Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object' if http_method == 'GET' else 'put_object',
                Params={
                    'Bucket': file_record.storage_bucket,
                    'Key': file_record.storage_path
                },
                ExpiresIn=expires_in
            )
            
            logger.info(f"✅ Generated presigned URL for: {file_record.filename}")
            return url
            
        except ClientError as e:
            log_error(e, "Presigned URL generation", file=file_record.filename)
            raise Exception(f"Failed to generate presigned URL: {e}")
    
    def generate_upload_presigned_url(self, 
                                    s3_key: str,
                                    content_type: str = None,
                                    expires_in: int = 3600) -> Dict[str, Any]:
        """
        Generate presigned URL for direct file upload to S3
        
        Args:
            s3_key: S3 object key
            content_type: MIME type of file
            expires_in: URL expiration time in seconds
            
        Returns:
            dict: Presigned post data including URL and form fields
        """
        try:
            conditions = []
            fields = {}
            
            if content_type:
                conditions.append({"Content-Type": content_type})
                fields["Content-Type"] = content_type
            
            # Add file size limits
            conditions.append(["content-length-range", 1, settings.MAX_FILE_SIZE_MB * 1024 * 1024])
            
            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=expires_in
            )
            
            logger.info(f"✅ Generated upload presigned URL for: {s3_key}")
            return response
            
        except ClientError as e:
            log_error(e, "Upload presigned URL generation", s3_key=s3_key)
            raise Exception(f"Failed to generate upload presigned URL: {e}")
    
    async def delete_file(self, file_record: FileStorage) -> bool:
        """
        Delete file from S3
        
        Args:
            file_record: Database record of file to delete
            
        Returns:
            bool: True if successful
        """
        try:
            self.s3_client.delete_object(
                Bucket=file_record.storage_bucket,
                Key=file_record.storage_path
            )
            
            logger.info(f"✅ File deleted from S3: {file_record.filename}")
            return True
            
        except ClientError as e:
            log_error(e, "S3 file deletion", file=file_record.filename)
            return False
    
    async def copy_file(self, 
                       source_file: FileStorage, 
                       destination_key: str) -> bool:
        """
        Copy file within S3
        
        Args:
            source_file: Source file record
            destination_key: Destination S3 key
            
        Returns:
            bool: True if successful
        """
        try:
            copy_source = {
                'Bucket': source_file.storage_bucket,
                'Key': source_file.storage_path
            }
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=destination_key
            )
            
            logger.info(f"✅ File copied in S3: {source_file.filename} -> {destination_key}")
            return True
            
        except ClientError as e:
            log_error(e, "S3 file copy", source=source_file.filename, destination=destination_key)
            return False
    
    def list_objects(self, 
                    prefix: str = None, 
                    max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket
        
        Args:
            prefix: Object key prefix filter
            max_keys: Maximum number of objects to return
            
        Returns:
            list: List of object metadata
        """
        try:
            list_args = {
                'Bucket': self.bucket_name,
                'MaxKeys': max_keys
            }
            
            if prefix:
                list_args['Prefix'] = prefix
            
            response = self.s3_client.list_objects_v2(**list_args)
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                })
            
            return objects
            
        except ClientError as e:
            log_error(e, "S3 list objects", prefix=prefix)
            return []
    
    def get_bucket_info(self) -> Dict[str, Any]:
        """
        Get S3 bucket information and statistics
        
        Returns:
            dict: Bucket information
        """
        try:
            # Get bucket location
            location_response = self.s3_client.get_bucket_location(
                Bucket=self.bucket_name
            )
            
            # Get bucket size (approximate)
            cloudwatch = boto3.client('cloudwatch', **settings.get_aws_config())
            
            size_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': self.bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=datetime.utcnow() - timedelta(days=1),
                EndTime=datetime.utcnow(),
                Period=86400,
                Statistics=['Average']
            )
            
            bucket_size = 0
            if size_response['Datapoints']:
                bucket_size = size_response['Datapoints'][-1]['Average']
            
            return {
                "bucket_name": self.bucket_name,
                "region": location_response.get('LocationConstraint') or 'us-east-1',
                "size_bytes": int(bucket_size),
                "size_gb": bucket_size / (1024**3),
                "accessible": True
            }
            
        except Exception as e:
            log_error(e, "S3 bucket info retrieval")
            return {
                "bucket_name": self.bucket_name,
                "region": self.region,
                "accessible": False,
                "error": str(e)
            }

# Global S3 service instance
s3_service = S3Service()