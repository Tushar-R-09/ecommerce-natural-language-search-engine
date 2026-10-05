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
│   ├── qdrant-backup-cronjob.yaml       # Automated daily snapshot backup CronJob
│   ├── qdrant-pvc.yaml                  # Qdrant PersistentVolumeClaim
│   ├── qdrant-service.yaml              # Qdrant ClusterIP & Headless services
│   ├── qdrant-statefulset.yaml          # Qdrant database StatefulSet
│   ├── streamlit-deployment.yaml        # Streamlit multi-replica Deployment
│   ├── streamlit-hpa.yaml               # HorizontalPodAutoscaler (2-5 replicas)
│   ├── streamlit-ingress.yaml           # Ingress with WebSocket & TLS/SSL support
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

1. **Create the Kind cluster:**
```bash
kind create cluster --name semantic-search --config k8s/kind-config.yaml
```

2. **Verify cluster, nodes, and host mount:**
```bash
kubectl cluster-info
kubectl get nodes

# Verify host data mount inside Kind control-plane node
docker exec semantic-search-control-plane ls -lah /mnt/data
```

3. **Build and load local Docker image into Kind:**
```bash
# Build the application image with version tag and latest tag
docker build -t natural-language-search:v1.0.0 -t natural-language-search:latest .

# Verify non-root user and pre-cached ML models
docker run --rm natural-language-search:latest whoami
docker run --rm natural-language-search:latest ls -lh /app/.cache/huggingface/hub

# Verify image contents
docker run --rm natural-language-search:latest ls -l /app/scripts

# Load image into Kind cluster
kind load docker-image natural-language-search:latest --name semantic-search
```

Verify the image is loaded inside the Kind node:
```bash
docker exec -it semantic-search-control-plane crictl images | grep natural-language-search
```

4. **Deploy Application ConfigMap:**
```bash
kubectl apply -f k8s/configmap.yaml
```

To verify the ConfigMap:
```bash
kubectl get configmap search-engine-config -o yaml
```

5. **Deploy Qdrant Storage (PersistentVolumeClaim):**
```bash
kubectl apply -f k8s/qdrant-pvc.yaml
```

To verify the PVC and StorageClass:
```bash
kubectl get pvc
kubectl get storageclass
```

6. **Deploy Qdrant Services (ClusterIP & Headless):**
```bash
kubectl apply -f k8s/qdrant-service.yaml
```

To verify the services:
```bash
kubectl get svc
```

7. **Deploy Qdrant Database (StatefulSet):**
```bash
kubectl apply -f k8s/qdrant-statefulset.yaml
```

To verify Qdrant is running:
```bash
kubectl get statefulset
kubectl get pods -l app=qdrant
```

8. **Deploy Automated Snapshot Backup (CronJob):**
```bash
kubectl apply -f k8s/qdrant-backup-cronjob.yaml
```

To verify the CronJob or trigger a manual snapshot test:
```bash
kubectl get cronjob
kubectl create job --from=cronjob/qdrant-snapshot-backup manual-backup-test
kubectl logs job/manual-backup-test
```

9. **Deploy the Streamlit application:**
```bash
kubectl apply -f k8s/streamlit-deployment.yaml
```

To verify the deployment and pod status:
```bash
kubectl get deployments
kubectl get pods
```

To inspect pod details or view logs:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

To test DNS resolution and connectivity to the Qdrant service from inside the Streamlit pod:
```bash
# Verify DNS resolution
kubectl exec -it <streamlit-pod-name> -- python -c "import socket; print(socket.gethostbyname('qdrant'))"

# Verify HTTP API connectivity to Qdrant
kubectl exec -it <streamlit-pod-name> -- \
  python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/collections').read().decode())"
```

10. **Deploy High Availability Policies (PDB & HPA):**
```bash
# Apply PodDisruptionBudget (guarantees >= 1 replica during maintenance)
kubectl apply -f k8s/streamlit-pdb.yaml

# Apply HorizontalPodAutoscaler (auto-scales 2 to 5 replicas)
kubectl apply -f k8s/streamlit-hpa.yaml
```

To verify high availability status:
```bash
kubectl get pdb
kubectl get hpa
```

11. **Expose the Streamlit Service (ClusterIP):**
```bash
kubectl apply -f k8s/streamlit-service.yaml
```

To verify the service:
```bash
kubectl get svc streamlit
```

12. **Deploy Ingress and Zero-Trust NetworkPolicy:**
```bash
# Apply production Ingress with WebSocket and TLS/SSL annotations
kubectl apply -f k8s/streamlit-ingress.yaml

# Apply NetworkPolicy isolating Qdrant database access to authorized pods
kubectl apply -f k8s/network-policy.yaml
```

To verify Ingress and NetworkPolicy:
```bash
kubectl get ingress
kubectl get networkpolicy
```

13. **Run Qdrant Indexing Job:**
Execute the batch indexing job to index dataset embeddings into Qdrant inside the cluster:
```bash
kubectl apply -f k8s/indexing-job.yaml
```

To monitor the job and view logs:
```bash
kubectl get jobs
kubectl logs -f -l app=qdrant-indexing
```

To delete or reset the indexing job (e.g., before re-running):
```bash
kubectl delete job qdrant-indexing
```

14. **Access the Streamlit Web UI:**
Port-forward the service to access the application locally:
```bash
kubectl port-forward service/streamlit 8501:8501
```

- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)

To delete the cluster when needed:
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
