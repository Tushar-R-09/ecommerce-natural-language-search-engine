# 🔍 Natural Language & Semantic Search Engine

A high-performance semantic search engine built for e-commerce product discovery using dense vector embeddings and **Qdrant Vector Database**. It extracts, cleans, and encodes rich product metadata from the Amazon Reviews 2023 dataset into dense vector representations for natural language query retrieval.

---

## 🚀 Quick Start: Qdrant Vector Database Setup

To run Qdrant vector database in Docker for the first time:

1. **Create the Docker network:**
```bash
docker network create semantic-search
```

2. **Run the Qdrant container:**
```bash
docker run -d \
  --name qdrant \
  --network semantic-search \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

- **REST API:** `http://localhost:6333`
- **gRPC API:** `http://localhost:6334`
- **Qdrant Web UI Dashboard:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

To verify the container is running:
```bash
docker ps
```

To stop or restart the container later:
```bash
docker stop qdrant
docker start qdrant
```

---

## 🛠️ Tech Stack & Architecture

- **Embedding Model:** `BAAI/bge-base-en-v1.5` via `sentence-transformers`
- **Vector Database:** [Qdrant](https://qdrant.tech/)
- **Dataset:** `McAuley-Lab/Amazon-Reviews-2023` (`raw_meta_Electronics`) via Hugging Face `datasets`
- **Data Processing:** `pandas`, `numpy`, `pyarrow` (Parquet batch storage)
- **Environment & Package Manager:** `uv` / Python 3.12+

---

## 📂 Project Structure

```text
natural_language_search_engine/
├── app/                                 # Streamlit search application
│   └── streamlit_app.py
├── brand_vocab/                         # Brand vocabulary data
├── data/                                # Generated embedding batches (Parquet)
├── k8s/                                 # Production Kubernetes manifests & Kind config
│   ├── configmap.yaml                   # Centralized application environment variables
│   ├── indexing-job.yaml                # Batch vector indexing Job
│   ├── kind-config.yaml                 # Local Kind cluster configuration
│   ├── network-policy.yaml              # Zero-Trust database isolation policy
│   ├── qdrant-pvc.yaml                  # Qdrant PersistentVolumeClaim
│   ├── qdrant-service.yaml              # Qdrant ClusterIP & Headless services
│   ├── qdrant-statefulset.yaml          # Qdrant database StatefulSet
│   ├── streamlit-deployment.yaml        # Streamlit multi-replica Deployment
│   ├── streamlit-hpa.yaml               # HorizontalPodAutoscaler (2-5 replicas)
│   ├── streamlit-pdb.yaml               # PodDisruptionBudget (minAvailable: 1)
│   └── streamlit-service.yaml           # Streamlit internal ClusterIP service
├── notebooks/                           # Exploratory data analysis & experiments
│   ├── eda.ipynb                        # Dataset exploration & metadata schema analysis
│   ├── embedding_model.ipynb            # Model exploration & similarity evaluation
│   ├── exploring_embedding_files.ipynb  # Inspecting generated parquet batches
│   └── product_embeding_generation.ipynb
├── scripts/
│   ├── 01_embedding_generation.py       # Batch embedding generation script
│   ├── 02_create_quadrant_index.py      # Qdrant collection creation & indexing
│   ├── 03_generate_brand_vocab.py       # Brand vocabulary generation
│   ├── 04_generate_evaluation_data.py   # Synthetic test dataset generator
│   ├── 05_evaluation.py                 # Retrieval & reranking benchmark
│   └── download_models.py               # Pre-caches ML models for Docker build
├── src/
│   ├── embedding_generator/             # Dense vector generation logic
│   │   └── embedding_generator.py
│   ├── preprocessing/                   # Text formatting & metadata extraction
│   │   └── preprocessing.py
│   └── utils/                           # Central configuration settings
│       └── config.py
├── Dockerfile                           # Docker configuration for search app
├── pyproject.toml                       # Project metadata & dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Tushar-R-09/simple-python-app.git
cd natural_language_search_engine
```

### 2. Create and activate a virtual environment

Using `uv`:
```bash
uv venv
source .venv/bin/activate
```

Or using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -e .
# or: pip install -e .
```

---

## 🏃 Execution Pipeline

### Step 1: Start Qdrant
Ensure the Qdrant container is up and running:
```bash
docker start qdrant
```
*(If running for the first time, see [Quick Start](#-quick-start-qdrant-vector-database-setup) to create the network and container).*

### Step 2: Generate Product Embeddings
Load dataset samples, preprocess text descriptions, and generate normalized vector embeddings saved as Parquet batches in `data/`:
```bash
python scripts/01_embedding_generation.py
```

### Step 3: Index Vectors into Qdrant
Create the Qdrant collection and upload product vectors and metadata payloads:
```bash
python scripts/02_create_quadrant_index.py
```

### Step 4: Build Search Application Docker Image
Build the hardened Docker image for the search application:
```bash
docker build -t natural-language-search:v1.0.0 -t natural-language-search:latest .
```

Verify that the container runs as the non-root `appuser` (UID 1000):
```bash
docker run --rm natural-language-search:latest whoami
```

Verify that the ML models are pre-cached for offline resilience:
```bash
docker run --rm natural-language-search:latest ls -lh /app/.cache/huggingface/hub
```

Verify the image contents and scripts:
```bash
docker run --rm natural-language-search:latest ls -l /app/scripts
```

### Step 5: Run Search Application Container
Launch the Streamlit search application container attached to the `semantic-search` network:
```bash
docker run -d \
  --name semantic-search \
  --network semantic-search \
  -p 8501:8501 \
  -e QDRANT_HOST=qdrant \
  -e QDRANT_PORT=6333 \
  natural-language-search:latest
```

- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)

To stop or restart the application container later:
```bash
docker stop semantic-search
docker start semantic-search
```

---

## ☸️ Kubernetes Cluster Setup (Kind)

Follow these steps sequentially to deploy the entire search engine architecture onto a local Kubernetes (Kind) cluster. Every step includes the exact command, what it does behind the scenes, and how to verify it.

---

### Step 1: Create the Kind Cluster with Host Data Mount
Create a local Kubernetes cluster named `semantic-search` using the custom multi-node Kind configuration in [`k8s/kind-config.yaml`](k8s/kind-config.yaml).
```bash
kind create cluster --name semantic-search --config k8s/kind-config.yaml
```
- **What this does:**
  - Provisions a local Docker-in-Docker Kubernetes control-plane node.
  - Mounts your local dataset directory (`$(pwd)/data`) directly into the Kind control-plane container at `/mnt/data`, allowing batch indexing jobs to access Parquet embeddings without uploading them to cloud storage.

---

### Step 2: Verify Cluster Health and Node Mount
Ensure the Kubernetes control plane is healthy and verify the local host dataset mount inside the Kind node:
```bash
# Verify cluster control-plane connectivity
kubectl cluster-info

# Verify node is in Ready state
kubectl get nodes

# Confirm dataset files are visible inside the Kind node
docker exec semantic-search-control-plane ls -lah /mnt/data
```
- **What this does:**
  - `kubectl cluster-info`: Checks that the Kubernetes API server and CoreDNS are operational.
  - `kubectl get nodes`: Confirms that the `semantic-search-control-plane` node is active and ready to schedule pods.
  - `docker exec ...`: Verifies that the Parquet batch files from your host machine are accessible at `/mnt/data` inside the node.

---

### Step 3: Build the Application Docker Image
Build the hardened, non-root Docker container image for the search application:
```bash
docker build -t natural-language-search:v1.0.0 -t natural-language-search:latest .
```
- **What this does:**
  - Packages the Streamlit application, retrieval modules, preprocessing pipeline, and dependencies using `uv`.
  - Runs `scripts/download_models.py` during build time to pre-cache transformer models (`BAAI/bge-base-en-v1.5` and `ms-marco-MiniLM-L-6-v2`) inside `/app/.cache/huggingface`, ensuring pods start instantly without downloading weights from Hugging Face at runtime.
  - Configures non-root user `appuser` (UID 1000) for container security.

---

### Step 4: Verify the Built Image Security & Offline Cache
Verify non-root execution and cached models before sideloading to Kubernetes:
```bash
# Verify non-root execution (UID 1000)
docker run --rm natural-language-search:latest whoami

# Verify pre-cached model weights
docker run --rm natural-language-search:latest ls -lh /app/.cache/huggingface/hub
```
- **What this does:**
  - Confirms the container enforces least-privilege security (`appuser`).
  - Confirms that model snapshots exist in the image filesystem so pods can run offline and survive air-gapped environments.

---

### Step 5: Sideload Docker Image into the Kind Cluster
Transfer the newly built local Docker image into the Kind cluster node:
```bash
kind load docker-image natural-language-search:latest --name semantic-search
```
- **What this does:**
  - Copies the local Docker image directly into the Kind node's internal `containerd` container runtime.
  - Eliminates the need to push images to an external registry like Docker Hub or GitHub Packages (`imagePullPolicy: IfNotPresent` will use this cached image).

---

### Step 6: Verify Image inside Kind Container Runtime
Confirm that the Kind node's `containerd` runtime successfully registered the image:
```bash
docker exec -it semantic-search-control-plane crictl images | grep natural-language-search
```
- **What this does:**
  - Queries `crictl` (Container Runtime Interface CLI) inside the Kind node to guarantee that `natural-language-search:latest` is present and ready for Kubernetes pod scheduling.

---

### Step 7: Deploy Central Application ConfigMap
Apply the centralized environment variables configuration defined in [`k8s/configmap.yaml`](k8s/configmap.yaml):
```bash
kubectl apply -f k8s/configmap.yaml
```
- **What this does:**
  - Creates the `search-engine-config` ConfigMap containing environment variables: `QDRANT_HOST=qdrant`, `QDRANT_PORT=6333`, `COLLECTION_NAME=semantic_search`, `EMBEDDING_MODEL_NAME`, and `RERANKER_MODEL_NAME`.
- **Verification:**
  ```bash
  kubectl get configmap search-engine-config -o yaml
  ```

---

### Step 8: Provision Qdrant PersistentVolumeClaim (PVC)
Deploy the persistent disk volume claim defined in [`k8s/qdrant-pvc.yaml`](k8s/qdrant-pvc.yaml):
```bash
kubectl apply -f k8s/qdrant-pvc.yaml
```
- **What this does:**
  - Requests a 10Gi PersistentVolume via the default `standard` StorageClass in Kind (host-path provisioner).
  - Guarantees that vector embeddings and collection indexes persist across Qdrant pod restarts or node upgrades.
- **Verification:**
  ```bash
  kubectl get pvc qdrant-storage
  ```

---

### Step 9: Deploy Qdrant Networking Services (ClusterIP & Headless)
Deploy the internal Kubernetes services defined in [`k8s/qdrant-service.yaml`](k8s/qdrant-service.yaml):
```bash
kubectl apply -f k8s/qdrant-service.yaml
```
- **What this does:**
  - Creates the `qdrant` ClusterIP service routing traffic to port 6333 (HTTP REST API) and port 6334 (gRPC).
  - Creates the `qdrant-headless` headless service (`clusterIP: None`) for stable network identities required by the Qdrant StatefulSet.
- **Verification:**
  ```bash
  kubectl get svc -l app=qdrant
  ```

---

### Step 10: Deploy Qdrant Vector Database (StatefulSet)
Launch the Qdrant database StatefulSet defined in [`k8s/qdrant-statefulset.yaml`](k8s/qdrant-statefulset.yaml):
```bash
kubectl apply -f k8s/qdrant-statefulset.yaml
```
- **What this does:**
  - Starts the single-replica Qdrant database pod (`qdrant-0`).
  - Mounts the `qdrant-storage` PVC to `/qdrant/storage`.
  - Configures container resource limits (CPU requests `200m`, limits `1000m`; Memory requests `512Mi`, limits `2Gi`) and liveness/readiness probes targeting `/healthz` and `/readyz`.

---

### Step 11: Wait for Qdrant Pod to be Ready
Ensure the Qdrant database is fully initialized before attempting to index vectors or connect the web app:
```bash
# Monitor StatefulSet rollout until completed
kubectl rollout status statefulset/qdrant

# Verify pod status is 1/1 Running
kubectl get pods -l app=qdrant
```
- **What this does:**
  - Blocks until the readiness probe passes (`1/1 Running`), confirming Qdrant is accepting connections on port 6333.

---

### Step 12: Run the Batch Vector Indexing Job
Populate Qdrant with your dataset embeddings by running the Kubernetes Job defined in [`k8s/indexing-job.yaml`](k8s/indexing-job.yaml):
```bash
kubectl apply -f k8s/indexing-job.yaml
```
- **What this does:**
  - Spawns a dedicated batch indexing pod (`qdrant-indexing-*`) with `restartPolicy: Never`.
  - Mounts `/mnt/data` from the Kind node to `/app/data` inside the pod.
  - Executes `python scripts/02_create_quadrant_index.py` using `uv`, which creates the Qdrant collection and inserts all Parquet vector batches into `http://qdrant:6333`.

---

### Step 13: Monitor Indexing Job Progress & Logs
Stream the indexing logs in real-time until all batches are uploaded:
```bash
# Stream indexing logs
kubectl logs -f -l app=qdrant-indexing

# Verify job status shows 1/1 Completions
kubectl get job qdrant-indexing
```
- **What this does:**
  - Follows the output as batches are converted and inserted into the `semantic_search` collection.
  - When finished, you should see `✅ Done!` and the Job status will display `COMPLETIONS: 1/1`.

---

### Step 14: Deploy the Streamlit Application Service
Expose the Streamlit web application internally using the service in [`k8s/streamlit-service.yaml`](k8s/streamlit-service.yaml):
```bash
kubectl apply -f k8s/streamlit-service.yaml
```
- **What this does:**
  - Creates the `streamlit` ClusterIP service routing traffic on port 8501 to pods matching `app: streamlit`.
- **Verification:**
  ```bash
  kubectl get svc streamlit
  ```

---

### Step 15: Deploy the Streamlit Application Pods
Deploy the multi-replica Streamlit frontend defined in [`k8s/streamlit-deployment.yaml`](k8s/streamlit-deployment.yaml):
```bash
kubectl apply -f k8s/streamlit-deployment.yaml
```
- **What this does:**
  - Deploys 2 replicas of the search UI container.
  - Enforces `RollingUpdate` strategy (`maxSurge: 1`, `maxUnavailable: 0`) for zero-downtime updates.
  - Injects environment variables from `search-engine-config` ConfigMap.
  - Enforces non-root security context (UID 1000) and configures liveness and readiness probes targeting `/_stcore/health`.

---

### Step 16: Verify Application Rollout & Database Connectivity
Confirm that both Streamlit replicas are running and test connectivity to Qdrant:
```bash
# Wait for deployment rollout to finish
kubectl rollout status deployment/streamlit

# Verify all pods are in Running state
kubectl get pods -l app=streamlit

# Test DNS and HTTP communication from inside a Streamlit pod to Qdrant
STREAMLIT_POD=$(kubectl get pod -l app=streamlit -o jsonpath="{.items[0].metadata.name}")
kubectl exec -it "$STREAMLIT_POD" -- python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/collections').read().decode())"
```
- **What this does:**
  - Confirms the deployment reaches `2/2` ready pods.
  - Tests internal cluster DNS resolution (`qdrant:6333`) and queries Qdrant directly from within the Streamlit pod to verify collection status.

---

### Step 17: Deploy High Availability Policies (PDB & HPA)
Apply the PodDisruptionBudget and HorizontalPodAutoscaler manifests for production reliability:
```bash
# 1. Apply PodDisruptionBudget (guarantees >= 1 replica available during node maintenance)
kubectl apply -f k8s/streamlit-pdb.yaml

# 2. Apply HorizontalPodAutoscaler (auto-scales 2 to 5 replicas based on CPU/Memory)
kubectl apply -f k8s/streamlit-hpa.yaml
```
- **What this does:**
  - `streamlit-pdb.yaml`: Ensures that voluntarily evicting pods during cluster upgrades will never drop active replicas below 1.
  - `streamlit-hpa.yaml`: Configures the Kubernetes Horizontal Pod Autoscaler to dynamically scale replicas between 2 and 5 based on 70% CPU or 80% memory utilization.
- **Verification:**
  ```bash
  kubectl get pdb streamlit-pdb
  kubectl get hpa streamlit-hpa
  ```

---

### Step 18: Deploy Zero-Trust Network Isolation Policy
Enforce network traffic restrictions using the policy defined in [`k8s/network-policy.yaml`](k8s/network-policy.yaml):
```bash
kubectl apply -f k8s/network-policy.yaml
```
- **What this does:**
  - Implements a Zero-Trust security perimeter around the Qdrant vector database.
  - Blocks all inbound traffic to Qdrant ports 6333 and 6334 *except* from pods labeled `app: streamlit` and `app: qdrant-indexing`.
- **Verification:**
  ```bash
  kubectl get networkpolicy qdrant-network-policy
  ```

---

### Step 19: Access the Streamlit Search Web UI Locally
Port-forward the Streamlit service from inside the Kubernetes cluster to your local machine:
```bash
kubectl port-forward service/streamlit 8501:8501
```
- **What this does:**
  - Binds Kubernetes ClusterIP port `8501` to `localhost:8501` on your host machine.
- **Open your browser and navigate to:**
  👉 **[http://localhost:8501](http://localhost:8501)**

Test searches such as:
- *"wireless noise cancelling headphones with deep bass"*
- *"mechanical gaming keyboard with rgb"*

---

### Cluster Maintenance & Teardown Commands

- **Check all resources in the cluster:**
  ```bash
  kubectl get all
  ```

- **Re-run the Indexing Job:**
  Before running the indexing job again, delete the completed job:
  ```bash
  kubectl delete job qdrant-indexing
  kubectl apply -f k8s/indexing-job.yaml
  ```

- **Restart Streamlit pods (Rolling restart):**
  ```bash
  kubectl rollout restart deployment/streamlit
  ```

- **Delete the entire Kind cluster and cleanup:**
  ```bash
  kind delete cluster --name semantic-search
  ```

---

## ⚙️ Configuration

Key configurations can be adjusted in [`src/utils/config.py`](src/utils/config.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `dataset_id` | `McAuley-Lab/Amazon-Reviews-2023` | Hugging Face Dataset identifier |
| `dataset_name` | `raw_meta_Electronics` | Dataset subset / category |
| `sample_size` | `100,000` | Number of product records sampled |
| `model_name` | `BAAI/bge-base-en-v1.5` | Dense embedding transformer model |
| `batch_size` | `32` | Batch size for embedding encoding |
| `device` | `cuda` / `mps` / `cpu` | Auto-detected compute hardware |
