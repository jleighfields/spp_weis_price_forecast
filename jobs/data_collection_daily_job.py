# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin
from dotenv import load_dotenv
from datetime import datetime,timezone

now_utc = datetime.now(timezone.utc)

load_dotenv()  # take environment variables from .env.

# Start the studio
s = Studio(name="data-collection", teamspace="spp-weis", user="jleighfields-yst2q")
print("starting Studio...")
s.start()

# Install plugin if not installed (in this case, it is already installed)
s.install_plugin("jobs")

jobs_plugin = s.installed_plugins["jobs"]

# --------------------------
# Example 1: Submit a job
# --------------------------

command = '''
cd ~/spp_weis_price_forecast
python scripts/data_collection/data_collection_daily.py
'''

jobs_plugin.run(command, name=f"lmp_daily_{now_utc}", machine=Machine.CPU)


print("Stopping Studio")
s.stop()
    