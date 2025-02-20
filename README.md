## Debug Sagemaker locally

This is a short recipe for replicating the horrendous Sagemaker training environment locally, containing:
- Creating a simple Docker image
- Recreating in appropriate paths to mimic SM endpoints
- Pulling in prebuilt AWS DLCs 
- Miscelaneous hints/notes on how SM works

Notes:
- This example mostly is concerned with replicating the environment. Therefore, it does not offer a virtual split of a single GPU into multiple instances, or multi node setup.

### 1. Build a simple docker image.
```
cd container
docker build -t sagemaker-local-training .
```

### 2. Running the docker

#### A. Run training script directly.

```Docker
sudo docker run --rm \
 -v $(pwd)/opt/ml/input/data/training:/opt/ml/input/data/training \
 -v $(pwd)/opt/ml/input/config:/opt/ml/input/config \
 -v $(pwd)/opt/ml/output:/opt/ml/model \
 -v $(pwd)/opt/ml/code:/opt/ml/code \
 sagemaker-local-training train.py
```

#### B. Enter into the image (terminal).

```
sudo chown -R $(id -u):$(id -g) ./opt/ml
sudo docker run --user $(id -u):$(id -g) --rm -it \
 -v $(pwd)/opt/ml/input/data/training:/opt/ml/input/data/training \
 -v $(pwd)/opt/ml/input/config:/opt/ml/input/config \
 -v $(pwd)/opt/ml/model:/opt/ml/model \
 -v $(pwd)/opt/ml/code:/opt/ml/code \
 --entrypoint /bin/bash \
 sagemaker-local-training

```


### 3. Downloading and testing AWS DLCs.

First:
- Make sure to have AWS configured (run `aws configure` from the terminal or via a saved config file at `~/.aws/config`)
- Make sure to have awscli version 2+ [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). `sudo apt` installs a version <2.

By default, above instructions install awscli to `/usr/bin/local`. Consider adding below to the path for calling AWS.
```
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

- Then pull the image from ECR (sample image with Pytorch and Huggingface Transformers libraries):
```
aws ecr get-login-password --region us-west-2 | sudo docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
sudo docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04
```

- Run:

```Docker
sudo chown -R $(id -u):$(id -g) ./opt/ml
sudo docker run --user $(id -u):$(id -g) --rm -it \
 -v $(pwd)/opt/ml/input/data/training:/opt/ml/input/data/training \
 -v $(pwd)/opt/ml/input/config:/opt/ml/input/config \
 -v $(pwd)/opt/ml/model:/opt/ml/model \
 -v $(pwd)/opt/ml/code:/opt/ml/code \
 --entrypoint /bin/bash \
 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04
```

### 4. Directory structure.

```
.
├── container/                     # Docker-related files
│   └── Dockerfile                 # Builds the reusable Docker image
├── opt/                           # Mirrors SageMaker directory structure
│   └── ml/
│       ├── input/
│       │   ├── data/
│       │   │   └── training/
│       │   │       └── train.csv   # Training data
│       │   └── config/
│       │       └── hyperparameters.json  # Hyperparameters
│       └── model/                 # Model artifacts output (after training)
├── code/                          # Training code (mounted at runtime)
│   └── train.py                   # Training script
└── output/                        # Optional: For capturing logs or additional outputs

```

#### A. Sagemaker directory structure.

Amazon SageMaker provides a structured directory layout when running training and inference jobs. These directories are mounted in the `/opt/ml/` directory inside the container.

---

##### **1. `/opt/ml/input/`**  
Contains input data, configuration, and code passed to the container.

- **`/opt/ml/input/config/`**: Configuration files for the job.  
  - **`hyperparameters.json`**: Hyperparameters passed to the training job.  
  - **`inputdataconfig.json`**: Input channel information (for multiple datasets).  
  - **`resourceconfig.json`**: Resource configurations like hosts and current host.  
  - **`trainingjobconfig.json`** *(training jobs only)*: Detailed job configuration.  

- **`/opt/ml/input/data/`**: Data channels.  
  - **`/opt/ml/input/data/<channel_name>/`**: Contains dataset files for the channel.  
  Example: `/opt/ml/input/data/train/`, `/opt/ml/input/data/validation/`  

- **`/opt/ml/input/code/`**: User-provided training/inference scripts.

---

#### **2. `/opt/ml/model/`**  
Output directory for the trained model artifacts.  
- At the end of training, SageMaker packages the contents and uploads them to the specified S3 output location.

---

#### **3. `/opt/ml/output/`**  
Contains output generated during training.  
- **`/opt/ml/output/failure`**: Error information if the job fails.  
- **`/opt/ml/output/data/`**: Output files (e.g., metrics for training jobs).  

---

#### **4. `/opt/ml/code/`**  
A copy of the user script directory for execution inside the container.  
- This is where your entry-point script (e.g., `train.py`) runs.

---

#### **5. `/opt/ml/checkpoints/`** *(optional)*  
Directory for saving intermediate checkpoints during training.  
- Useful for resuming training in case of interruptions.

---

#### **6. `/opt/ml/processing/`** *(processing jobs only)*  
Used in SageMaker Processing jobs.  
- **`/opt/ml/processing/input/`**: Input data.  
- **`/opt/ml/processing/output/`**: Output data.  
- **`/opt/ml/processing/code/`**: Code used for processing.  

---

### **Quick Summary Table**

| Directory                        | Purpose                                     |
|-----------------------------------|---------------------------------------------|
| `/opt/ml/input/config/`          | Job configurations (hyperparameters, etc.)  |
| `/opt/ml/input/data/<channel>/`  | Input datasets                              |
| `/opt/ml/input/code/`            | User-supplied scripts                       |
| `/opt/ml/code/`                  | Execution directory of user script         |
| `/opt/ml/model/`                 | Saved model artifacts                      |
| `/opt/ml/output/data/`           | Training job outputs (e.g., metrics)        |
| `/opt/ml/output/failure`         | Failure logs (if job fails)                 |
| `/opt/ml/checkpoints/`           | Intermediate checkpoints                   |
| `/opt/ml/processing/`            | Processing job input/output/code           |

---



#### B. Channels.

Channels (e.g., `opt/ml/data/train/`) are not fixed. You define the directory when you set up the job. 

```python
import sagemaker
from sagemaker.estimator import Estimator

# Define the estimator
estimator = Estimator(
    image_uri="your-training-image",
    role="your-sagemaker-role",
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/output"
)

# Define data channels
channels = {
    "train": "s3://your-bucket/train-data/",
    "validation": "s3://your-bucket/validation-data/"
}

# Pass channels into the `fit()`:
estimator.fit(inputs=channels)
```


However, some built-in features Sagemaker has may require specific names (e.g., `/data/train/`). 

Example: Amazon SageMaker Processing for processing jobs for data processing steps.
```python
from sagemaker.processing import Processor, [...]
[...]
processor.run(
    inputs=[ProcessingInput(source="s3://your-bucket/input-data", destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination="s3://your-bucket/output-data")]
)
```
