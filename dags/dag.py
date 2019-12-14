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

dag = DAG('CNN-Training-dag', default_args=default_args, schedule_interval=None)

# t1, t2, t3 and t4- 2 more tasks to be added S3 downloads and csv merging

'''
t0 = BashOperator(
    task_id='part0',
    bash_command='pip install --user -r /usr/local/airflow/requirements.txt',
    dag=dag)
'''

t0 = BashOperator(
    task_id='start_training',
    bash_command='echo "training begins"',
    dag=dag)
	
t1 = BashOperator(
    task_id='prepareData',
    bash_command='python /usr/local/airflow/dags/PrepareCNNData.py',
    dag=dag)

t2 = BashOperator(
    task_id='trainCNNModel',
    bash_command='python /usr/local/airflow/dags/TrainCNN.py',
    dag=dag)
	
t3 = BashOperator(
    task_id='prepareCRNNData',
    bash_command='python /usr/local/airflow/dags/PrepareCRNNData.py',
    dag=dag)

t4 = BashOperator(
    task_id='trainCRNNModel',
    bash_command='python /usr/local/airflow/dags/TrainCRNN.py',
    dag=dag)
	
t5 = BashOperator(
    task_id='StartInference',
    bash_command='echo "Inference begins"',
    dag=dag)
	
t6 = BashOperator(
    task_id='LicensePlateDetectionUsingYoloV3',
    bash_command='python /usr/local/airflow/dags/Detection.py',
    dag=dag)

t7 = BashOperator(
    task_id='PlateSegmentation',
    bash_command='python /usr/local/airflow/dags/segmentation.py',
    dag=dag)
	
t8 = BashOperator(
    task_id='CharPredictions',
    bash_command='python /usr/local/airflow/dags/CharPredictor.py',
    dag=dag)
	
t9 = BashOperator(
    task_id='MTurkValidation',
    bash_command='python /usr/local/airflow/dags/MTurk-segmentation_validation.py',
    dag=dag)

t10 = BashOperator(
    task_id='DataCollection',
    bash_command='echo "Collecting data from platesmania.com"',
    dag=dag)

	
t11 = BashOperator(
    task_id='ScrapeCarAndLicensePlateData',
    bash_command='python /usr/local/airflow/dags/Scraper.py',
    dag=dag)

## Data Collection pipeline

t10 >> t11 >>  t6

## Training pipelines
t0 >> t1 >> t2
t0 >> t3 >> t4

## Inference Pipeline
t5 >> t6 >> t7 >> t8 >> t9


