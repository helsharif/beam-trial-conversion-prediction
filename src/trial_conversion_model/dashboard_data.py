"""Read-only S3 artifact access and dashboard transformations."""

from dataclasses import dataclass
from datetime import date
from io import BytesIO
import json
import os
from pathlib import Path
import re

import boto3
from botocore.config import Config
from dotenv import dotenv_values
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def settings():
    """Local .env may point at an external .env; process environment wins."""
    local = dotenv_values(ROOT / '.env')
    path = os.environ.get('DASHBOARD_ENV_FILE') or local.get('DASHBOARD_ENV_FILE')
    external = {}
    if path:
        if not Path(path).is_file():
            raise ValueError('DASHBOARD_ENV_FILE does not point to an existing file.')
        external = dotenv_values(path)
    return {**external, **local, **os.environ}


def connect():
    values = settings()
    bucket = values.get('S3_BUCKET')
    if not bucket:
        raise ValueError('Set S3_BUCKET and AWS credentials in .env, or set DASHBOARD_ENV_FILE to your S3 .env path.')
    session = boto3.Session(
        aws_access_key_id=values.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=values.get('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=values.get('AWS_SESSION_TOKEN'),
        region_name=values.get('AWS_REGION') or values.get('AWS_DEFAULT_REGION', 'us-east-2'),
        profile_name=values.get('AWS_PROFILE'),
    )
    return bucket, session.client('s3', config=Config(
        connect_timeout=5, read_timeout=20, retries={'max_attempts': 2},
    ))


@dataclass(frozen=True)
class Artifact:
    key: str
    day: date
    modified: str
    etag: str


def list_artifacts(client, bucket, prefix, suffix):
    result = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            match = re.search(r'(\d{4}-\d{2}-\d{2})', obj['Key'])
            if not match or not obj['Key'].endswith(suffix) or not obj['Size']:
                continue
            try:
                day = date.fromisoformat(match[1])
            except ValueError:
                continue
            result.append(Artifact(obj['Key'], day, obj['LastModified'].isoformat(), obj.get('ETag', '')))
    return sorted(result, key=lambda a: (a.day, a.modified, a.key))


def read_bytes(client, bucket, artifact):
    response = client.get_object(Bucket=bucket, Key=artifact.key)
    try:
        return response['Body'].read()
    finally:
        response['Body'].close()


def parse_scores(raw):
    frame = pd.read_csv(BytesIO(raw), dtype={'trial_id': 'string', 'user_id': 'string'})
    if 'conversion_probability' not in frame and 'conv_prob' in frame:
        frame = frame.rename(columns={'conv_prob': 'conversion_probability'})
    if 'conversion_probability' not in frame:
        raise ValueError('Prediction CSV is missing conversion_probability.')
    probability = pd.to_numeric(frame['conversion_probability'], errors='coerce')
    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError('Prediction probabilities must be numbers between 0 and 1.')
    frame['conversion_probability'] = probability
    frame['cancellation_risk'] = 1 - probability
    frame = frame.sort_values('conversion_probability', kind='stable').reset_index(drop=True)
    frame['rank'] = range(1, len(frame) + 1)
    first = [c for c in ['rank', 'trial_id', 'user_id', 'conversion_probability', 'cancellation_risk'] if c in frame]
    return frame[first + [c for c in frame if c not in first]]


def parse_verdict(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get('drift')) is not bool:
        raise ValueError('Drift JSON must contain a boolean drift verdict.')
    share = value.get('drifted_share')
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ValueError('Drift JSON must contain drifted_share between 0 and 1.')
    return value


def choose_verdicts(records):
    """Prefer measured reports over explicitly backfilled reports on the same day."""
    chosen = {}
    for artifact, verdict in sorted(records, key=lambda pair: (
        pair[0].day, pair[1].get('source') != 'backfill', pair[0].modified, pair[0].key,
    )):
        chosen[artifact.day] = (artifact, verdict)
    return chosen
