import os
import json
import requests
import time
from datetime import timedelta

# logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def start_cluster(
    host: str, 
    cluster_id: str, 
    token: str,
) -> None:
    '''
    start cluster if it's in a terminated state
    '''
    
    # get cluster state
    res = requests.get(
                url=f"{os.environ['DATABRICKS_HOST']}/api/2.0/clusters/get",
                json={"cluster_id": os.environ['CLUSTER_ID']},
                headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
            )
            
    state = res.json()['state']
    log.info(f'cluster current state: {state}')

    # try to start cluster if it's TERMINATED
    # if state not in ['RUNNING', 'PENDING', 'TERMINATING']:
    if state == 'TERMINATED':
        try:
            res = requests.post(
                        url=f"{os.environ['DATABRICKS_HOST']}/api/2.0/clusters/start",
                        json={"cluster_id": os.environ["CLUSTER_ID"]},
                        headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
                    )
        except:
            res.raise_for_status()


def wait_for_cluster(
    host: str, 
    cluster_id: str, 
    token: str, 
    n_trys: int=60, 
    sleep_seconds: int=30
) -> None:
    '''
    function to wait for cluster to be in a running state
    before continuing
    '''

    i=0
    t0 = time.time()
    while i < n_trys:
        
        res = requests.get(
            url=f"{host}/api/2.0/clusters/get",
            json={"cluster_id": cluster_id},
            headers={"Authorization": f"Bearer {token}"}
        )
        state = res.json()['state']

        t1 = time.time()
        log.info(f'state try count: {i} - state: {state} - elapsed time: {timedelta(seconds=t1-t0)}')
        {timedelta(seconds=t1-t0)}
        i+=1
        
        if state in ['RUNNING', 'RESIZING']:
            break

        time.sleep(sleep_seconds)

    # TODO: TimeoutError raise error if we reach end of while loop
    # https://docs.python.org/3/library/exceptions.html#TimeoutError


