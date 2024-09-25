# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin

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
jobs_plugin.run("python my_dummy_file", name="my_first_job", machine=Machine.A10G)

# --------------------------
# Example 2: Hyperparameter sweep
# --------------------------
learning_rates = ['0.01', '0.02', '0.03']
for lr in learning_rates:
    job_cmd = f'python my_dummy_file --lr {lr}'
    jobs_plugin.run(job_cmd, name="my-sweep-1", machine=Machine.A10G)

# --------------------------
# Example 3: Benchmark model on different GPUs
# --------------------------
machines = [Machine.A10G, Machine.A100, Machine.V100]
for machine in machines:
    job_cmd = f'python my_dummy_file'
    jobs_plugin.run(job_cmd, name="my-benchmark-1", machine=machine)

print("Stopping Studio")
s.stop()
    