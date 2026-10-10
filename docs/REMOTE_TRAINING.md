# Remote training worker

The Car Scan API remains the source of truth for uploaded datasets, scripts, and
training results. When a remote worker is selected, the API bundles only the
selected model, its required scripts/weights, and the selected dataset, sends
that temporary bundle to the worker over SSH, and runs it in a resource-limited
Docker container. Logs stream back to the UI; `runs/` is copied back to the
central server, and the worker's temporary directory/container is removed.

## Worker configuration

Set `CAR_SCAN_TRAIN_WORKERS` on the Car Scan API host to a JSON array. Example
for the tested CPU worker:

```json
[{"id":"nginx01","name":"nginx01 · CPU (8 cores)","host":"10.0.200.12","user":"admindev","dockerImage":"car-scan-web:local","device":"cpu","cpus":4,"memory":"6g"}]
```

The worker must have SSH and Docker available, and the configured image must
already exist on it (or be pullable). No remote service port or SMB share is
needed. The worker is CPU-only; the provided limits reserve at most 4 cores and
6 GiB RAM for one training container. Lower those values if the worker runs
other important workloads.

Use a dedicated SSH key with access limited to the worker account, and pin the
worker host key in a `known_hosts` file. Do not use a password in environment
variables or source control. For Docker Compose production, set
`CAR_SCAN_TRAIN_SSH_KEY_FILE` and `CAR_SCAN_TRAIN_SSH_KNOWN_HOSTS_FILE` to the
host paths of those files, then start with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d --build
```

On Linux hosts, restrict the private-key file to the service administrator
(for example, mode `600`). The matching private key and pinned host key must be
readable inside the application container at the paths in the overlay. The
app's `.env` should contain `CAR_SCAN_TRAIN_WORKERS`; it must not contain the
private key itself. `docker-compose.yml` forwards that setting to the API.

For deployments that run Python directly rather than in Docker, set
`CAR_SCAN_TRAIN_SSH_KEY` and `CAR_SCAN_TRAIN_SSH_KNOWN_HOSTS` to readable local
file paths and ensure OpenSSH `ssh` and `scp` are installed.

## Data and security boundaries

- The web app performs dataset upload, validation, split, and delete on the
  central server. The UI's worker choice only affects starting a training run.
- The worker receives a transient job bundle under `/tmp/onevision-training/`;
  training runs with Docker networking disabled and CPU/memory limits.
- The app imports only the worker's `runs/` output. It rejects links and paths
  outside `runs/`, and removes the job folder/container after completion.
- If the worker disconnects during result retrieval, check the UI log and
  remove only that run's UUID directory under `/tmp/onevision-training/` after
  confirming the associated Docker container has stopped.
