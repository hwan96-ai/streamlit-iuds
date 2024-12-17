import boto3
from datetime import datetime
import pytz

def get_aws_session():
    temp_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
    sts_client = temp_session.client('sts')
    response = sts_client.assume_role(
    RoleArn=RoleArn,
    RoleSessionName=RoleSessionName,
    DurationSeconds=43200, #12시간
)
    
    session = boto3.Session(
        aws_access_key_id=response['Credentials']['AccessKeyId'],
        aws_secret_access_key=response['Credentials']['SecretAccessKey'],
        aws_session_token=response['Credentials']['SessionToken'],
    )
    
    return session

def get_bedrock_client(session=None):
    if session is None:
        session = get_aws_session()
    
    return session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )