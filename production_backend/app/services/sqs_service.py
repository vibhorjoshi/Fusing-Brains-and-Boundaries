"""
AWS SQS Service for task queue management
Production-ready message queue for ML processing jobs
"""

import boto3
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import uuid

from app.core.config import settings
from app.models.processing_job import ProcessingJob, JobStatus, JobType
from app.core.logging import log_error, log_performance

logger = logging.getLogger(__name__)

class SQSService:
    """AWS SQS service for task queue management"""
    
    def __init__(self):
        """Initialize SQS service with AWS configuration"""
        try:
            # Create SQS client
            aws_config = settings.get_aws_config()
            self.sqs_client = boto3.client('sqs', **aws_config)
            
            # Queue configuration
            self.queue_url = settings.AWS_SQS_QUEUE_URL
            self.dlq_url = settings.AWS_SQS_DLQ_URL
            self.region = settings.AWS_REGION
            
            logger.info(f"✅ SQS Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQS service: {e}")
            raise
    
    async def send_job_message(self, 
                              processing_job: ProcessingJob, 
                              delay_seconds: int = 0,
                              priority: int = 0) -> str:
        """
        Send processing job to SQS queue
        
        Args:
            processing_job: Job to queue for processing
            delay_seconds: Delay before message becomes available
            priority: Message priority (0-9, higher = more priority)
            
        Returns:
            str: SQS Message ID
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare message body
            message_body = {
                "job_id": processing_job.id,
                "job_uuid": processing_job.uuid,
                "job_type": processing_job.job_type.value,
                "user_id": processing_job.user_id,
                "priority": priority,
                "created_at": processing_job.created_at.isoformat(),
                "input_parameters": processing_job.input_parameters,
                "ml_model_version": processing_job.ml_model_version,
                "confidence_threshold": processing_job.confidence_threshold,
                "apply_regularization": processing_job.apply_regularization,
                "batch_size": processing_job.batch_size,
                "geographic_bounds": processing_job.geographic_bounds,
                "target_states": processing_job.target_states
            }
            
            # Message attributes for filtering and routing
            message_attributes = {
                'job_type': {
                    'StringValue': processing_job.job_type.value,
                    'DataType': 'String'
                },
                'priority': {
                    'StringValue': str(priority),
                    'DataType': 'Number'
                },
                'user_id': {
                    'StringValue': str(processing_job.user_id),
                    'DataType': 'Number'
                }
            }
            
            # Add geographic attributes if available
            if processing_job.target_states:
                message_attributes['states'] = {
                    'StringValue': ','.join(processing_job.target_states),
                    'DataType': 'String'
                }
            
            # Send message to SQS
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body),
                DelaySeconds=delay_seconds,
                MessageAttributes=message_attributes,
                MessageGroupId=f"job_type_{processing_job.job_type.value}",  # For FIFO queues
                MessageDeduplicationId=f"job_{processing_job.uuid}_{int(datetime.utcnow().timestamp())}"
            )
            
            message_id = response['MessageId']
            
            # Update job with SQS message ID
            processing_job.aws_job_id = message_id
            processing_job.status = JobStatus.QUEUED
            processing_job.queued_at = datetime.utcnow()
            
            # Log performance
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_performance("sqs_send_message", duration, job_type=processing_job.job_type.value)
            
            logger.info(f"✅ Job queued in SQS: {processing_job.uuid} (Message ID: {message_id})")
            return message_id
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            log_error(e, "SQS send message", job_id=processing_job.id, error_code=error_code)
            raise Exception(f"Failed to queue job: {error_code}")
        except Exception as e:
            log_error(e, "Job queueing", job_id=processing_job.id)
            raise
    
    async def receive_messages(self, 
                              max_messages: int = 10,
                              wait_time_seconds: int = 20,
                              visibility_timeout: int = 300) -> List[Dict[str, Any]]:
        """
        Receive messages from SQS queue
        
        Args:
            max_messages: Maximum number of messages to receive
            wait_time_seconds: Long polling wait time
            visibility_timeout: How long message is invisible to other consumers
            
        Returns:
            list: List of SQS messages
        """
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                AttributeNames=['All'],
                MessageAttributeNames=['All'],
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=visibility_timeout
            )
            
            messages = response.get('Messages', [])
            
            # Parse message bodies
            parsed_messages = []
            for message in messages:
                try:
                    body = json.loads(message['Body'])
                    parsed_message = {
                        'message_id': message['MessageId'],
                        'receipt_handle': message['ReceiptHandle'],
                        'body': body,
                        'attributes': message.get('Attributes', {}),
                        'message_attributes': message.get('MessageAttributes', {})
                    }
                    parsed_messages.append(parsed_message)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message body: {message['MessageId']}")
            
            logger.info(f"✅ Received {len(parsed_messages)} messages from SQS")
            return parsed_messages
            
        except ClientError as e:
            log_error(e, "SQS receive messages")
            return []
    
    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete processed message from queue
        
        Args:
            receipt_handle: SQS receipt handle for message
            
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info("✅ Message deleted from SQS")
            return True
            
        except ClientError as e:
            log_error(e, "SQS delete message", receipt_handle=receipt_handle)
            return False
    
    async def change_message_visibility(self, 
                                      receipt_handle: str,
                                      visibility_timeout: int) -> bool:
        """
        Change message visibility timeout (extend processing time)
        
        Args:
            receipt_handle: SQS receipt handle
            visibility_timeout: New visibility timeout in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            
            logger.info(f"✅ Message visibility changed to {visibility_timeout}s")
            return True
            
        except ClientError as e:
            log_error(e, "SQS change visibility", receipt_handle=receipt_handle)
            return False
    
    async def send_batch_messages(self, jobs: List[ProcessingJob]) -> List[str]:
        """
        Send multiple jobs to queue in batch
        
        Args:
            jobs: List of processing jobs to queue
            
        Returns:
            list: List of message IDs
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare batch entries (max 10 per batch)
            entries = []
            message_ids = []
            
            for i, job in enumerate(jobs[:10]):  # SQS batch limit is 10
                message_body = {
                    "job_id": job.id,
                    "job_uuid": job.uuid,
                    "job_type": job.job_type.value,
                    "user_id": job.user_id,
                    "created_at": job.created_at.isoformat(),
                    "input_parameters": job.input_parameters
                }
                
                entries.append({
                    'Id': str(i),
                    'MessageBody': json.dumps(message_body),
                    'MessageAttributes': {
                        'job_type': {
                            'StringValue': job.job_type.value,
                            'DataType': 'String'
                        }
                    },
                    'MessageGroupId': f"batch_{job.job_type.value}",
                    'MessageDeduplicationId': f"batch_{job.uuid}_{int(datetime.utcnow().timestamp())}"
                })
            
            # Send batch
            response = self.sqs_client.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
            
            # Process successful messages
            for success in response.get('Successful', []):
                message_ids.append(success['MessageId'])
            
            # Log failed messages
            for failed in response.get('Failed', []):
                logger.error(f"Failed to send message {failed['Id']}: {failed['Message']}")
            
            # Log performance
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_performance("sqs_batch_send", duration, message_count=len(message_ids))
            
            logger.info(f"✅ Batch sent {len(message_ids)} messages to SQS")
            return message_ids
            
        except ClientError as e:
            log_error(e, "SQS batch send", job_count=len(jobs))
            return []
    
    def get_queue_attributes(self) -> Dict[str, Any]:
        """
        Get queue statistics and attributes
        
        Returns:
            dict: Queue attributes
        """
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            attributes = response.get('Attributes', {})
            
            return {
                "queue_url": self.queue_url,
                "approximate_number_of_messages": int(attributes.get('ApproximateNumberOfMessages', 0)),
                "approximate_number_of_messages_not_visible": int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                "approximate_number_of_messages_delayed": int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                "created_timestamp": attributes.get('CreatedTimestamp'),
                "last_modified_timestamp": attributes.get('LastModifiedTimestamp'),
                "visibility_timeout_seconds": int(attributes.get('VisibilityTimeout', 30)),
                "message_retention_period": int(attributes.get('MessageRetentionPeriod', 345600)),
                "receive_message_wait_time_seconds": int(attributes.get('ReceiveMessageWaitTimeSeconds', 0))
            }
            
        except ClientError as e:
            log_error(e, "SQS get queue attributes")
            return {"error": str(e)}
    
    async def purge_queue(self) -> bool:
        """
        Purge all messages from queue (use with caution!)
        
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            logger.warning("⚠️ SQS queue purged - all messages deleted")
            return True
            
        except ClientError as e:
            log_error(e, "SQS purge queue")
            return False

# Global SQS service instance
sqs_service = SQSService()