import boto3
from datetime import datetime
import pytz
import os

def get_aws_session():
    # 환경 변수에서 인증 정보 가져오기
    aws_access_key_id = os.environ.get('aws_access_key_id')
    aws_secret_access_key = os.environ.get('aws_secret_access_key')
    role_arn = os.environ.get('role_arn')
    role_session_name = os.environ.get('role_session_name')
    
    # 필수 인증 정보 확인
    if not all([aws_access_key_id, aws_secret_access_key, role_arn, role_session_name]):
        raise ValueError("AWS 인증 정보가 환경 변수에 설정되지 않았습니다.")

    temp_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    sts_client = temp_session.client('sts')
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=role_session_name,
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
