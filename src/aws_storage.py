"""
AWS S3 Storage Module
----------------------
Uploads all local pipeline outputs to S3.
Downloads processed data back for ML/Tableau.
"""

import os
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(
    '/home/deepak/stock_pipeline/config/.env'
)

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_BUCKET     = os.getenv('AWS_BUCKET')
AWS_REGION     = os.getenv('AWS_REGION',
                            'ap-south-1')
BASE           = '/home/deepak/stock_pipeline'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 Client
# ---------------------------------------------------------------------------
def get_s3_client():
    """Create and return authenticated S3 client."""
    try:
        client = boto3.client(
            's3',
            aws_access_key_id     = AWS_ACCESS_KEY,
            aws_secret_access_key = AWS_SECRET_KEY,
            region_name           = AWS_REGION
        )
        # Verify credentials
        client.list_buckets()
        log.info("S3 client authenticated successfully")
        return client
    except NoCredentialsError:
        log.error("AWS credentials not found")
        raise
    except ClientError as e:
        log.error(f"AWS authentication failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Upload Single File
# ---------------------------------------------------------------------------
def upload_file(s3, local_path, s3_key):
    """Upload a single file to S3."""
    try:
        s3.upload_file(
            local_path,
            AWS_BUCKET,
            s3_key
        )
        log.info(f"Uploaded: {s3_key}")
        return True
    except ClientError as e:
        log.error(f"Upload failed {s3_key}: {e}")
        return False


# ---------------------------------------------------------------------------
# Upload Folder
# ---------------------------------------------------------------------------
def upload_folder(s3, local_folder, s3_prefix):
    """
    Recursively upload all files in a local
    folder to S3 under the given prefix.
    """
    folder = Path(local_folder)
    if not folder.exists():
        log.warning(
            f"Folder not found: {local_folder}"
        )
        return 0

    uploaded = 0
    failed   = 0

    for file_path in folder.rglob('*'):
        if file_path.is_file():
            # Build S3 key
            relative = file_path.relative_to(
                folder
            )
            s3_key = f"{s3_prefix}/{relative}" \
                     .replace('\\', '/')

            success = upload_file(
                s3,
                str(file_path),
                s3_key
            )
            if success:
                uploaded += 1
            else:
                failed += 1

    log.info(
        f"Folder upload complete — "
        f"uploaded: {uploaded}, failed: {failed}"
    )
    return uploaded


# ---------------------------------------------------------------------------
# Download File
# ---------------------------------------------------------------------------
def download_file(s3, s3_key, local_path):
    """Download a single file from S3."""
    try:
        os.makedirs(
            os.path.dirname(local_path),
            exist_ok=True
        )
        s3.download_file(
            AWS_BUCKET, s3_key, local_path
        )
        log.info(f"Downloaded: {s3_key}")
        return True
    except ClientError as e:
        log.error(
            f"Download failed {s3_key}: {e}"
        )
        return False


# ---------------------------------------------------------------------------
# List S3 Files
# ---------------------------------------------------------------------------
def list_s3_files(s3, prefix=''):
    """List all files in the S3 bucket."""
    try:
        response = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=prefix
        )
        files = [
            obj['Key']
            for obj in response.get(
                'Contents', []
            )
        ]
        return files
    except ClientError as e:
        log.error(f"List failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Upload All Pipeline Data
# ---------------------------------------------------------------------------
def upload_all_data(s3):
    """Upload all pipeline outputs to S3."""
    log.info("Starting full pipeline upload to S3")
    print("\nUploading to S3 ...")
    print("-" * 50)

    # Define what to upload
    upload_map = {
        f'{BASE}/data/raw':       'raw',
        f'{BASE}/data/processed': 'processed',
        f'{BASE}/data/viz':       'viz',
        f'{BASE}/data/reports':   'reports',
        f'{BASE}/models':         'models',
    }

    total = 0
    for local, prefix in upload_map.items():
        print(f"  Uploading {prefix}/ ...")
        count = upload_folder(s3, local, prefix)
        print(f"    {count} files uploaded")
        total += count

    print("-" * 50)
    print(f"  Total files uploaded: {total}")
    return total


# ---------------------------------------------------------------------------
# Verify Upload
# ---------------------------------------------------------------------------
def verify_upload(s3):
    """List all S3 files and print summary."""
    print("\nVerifying S3 contents ...")
    print("-" * 50)

    prefixes = [
        'raw', 'processed',
        'viz', 'reports', 'models'
    ]

    grand_total = 0
    for prefix in prefixes:
        files = list_s3_files(s3, prefix)
        print(f"  {prefix:<12}: {len(files)} files")
        grand_total += len(files)

    print("-" * 50)
    print(f"  Total in S3: {grand_total} files")
    print(
        f"  Bucket: s3://{AWS_BUCKET}/"
    )
    return grand_total


# ---------------------------------------------------------------------------
# Daily Sync (for Airflow)
# ---------------------------------------------------------------------------
def daily_sync():
    """
    Upload only today's outputs to S3.
    Called by Airflow DAG after processing.
    """
    log.info("Starting daily S3 sync ...")

    s3 = get_s3_client()

    # Only sync viz + reports daily
    # (raw + processed already uploaded)
    sync_folders = {
        f'{BASE}/data/viz':     'viz',
        f'{BASE}/data/reports': 'reports',
    }

    for local, prefix in sync_folders.items():
        upload_folder(s3, local, prefix)

    log.info("Daily S3 sync complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("AWS S3 Storage Upload")
    print("=" * 50)

    # Step 1: Authenticate
    print("\nAuthenticating with AWS ...")
    s3 = get_s3_client()
    print("  Authentication successful")

    # Step 2: Upload everything
    upload_all_data(s3)

    # Step 3: Verify
    verify_upload(s3)

    print("\n" + "=" * 50)
    print("Output")
    print("-" * 50)
    print(
        f"  Bucket : s3://{AWS_BUCKET}/"
    )
    print(
        f"  Region : {AWS_REGION}"
    )
    print(
        "  Console: https://s3.console.aws.amazon.com"
        f"/s3/buckets/{AWS_BUCKET}"
    )
    print("=" * 50)