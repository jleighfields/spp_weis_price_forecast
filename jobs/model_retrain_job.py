# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin
from dotenv import load_dotenv
import time
load_dotenv()  # take environment variables from .env.

print("starting Studio...")
s = Studio(name="data-collection", teamspace="spp-weis", user="jleighfields-yst2q")
s.start(machine=Machine.T4)
for i in range(30):
    print(f'status: {s.status} - count: {i}')
    if s.status == 'Running':
        break

    time.sleep(30)


try:
    s.run("cd ~/spp_weis_price_forecast && python scripts/model_retrain.py")
except Exception as e:
        print("command failed with error: ", e)

print("Stopping Studio")
s.stop()
