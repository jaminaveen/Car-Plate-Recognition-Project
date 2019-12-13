"""
Code that goes along with the Airflow located at:
http://airflow.readthedocs.org/en/latest/tutorial.html
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Following are defaults which can be overridden later on
default_args = {
    'owner': 'Shiqi_dai',
    'depends_on_past': False,
    'start_date': datetime(2019, 12, 12),
    'email': ['dai.shi@husky.neu.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('Car_Plate_Recognition', default_args=default_args, schedule_interval=None)

# t1, t2, t3 and t4- 2 more tasks to be added S3 downloads and csv merging

t0 = BashOperator(
    task_id='part0',
    bash_command='pip install --user nltk',
    dag=dag)
t1 = BashOperator(
    task_id='part01',
    bash_command='pip install --user boto3',
    dag=dag)
t2 = BashOperator(
    task_id='part02',
    bash_command='pip install --user xlrd',
    dag=dag)

t3 = BashOperator(
    task_id='part1',
    bash_command='python /usr/local/airflow/dags/preprocessing.py',
    dag=dag)

t4 = BashOperator(
    task_id='part2',
    bash_command='python /usr/local/airflow/dags/clusteringweightage.py',
    dag=dag)

t0 >> t1 >> t2 >> t3 >> t4