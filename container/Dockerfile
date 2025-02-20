FROM python:3.8-slim

# Install necessary dependencies
RUN pip install pandas scikit-learn

# Create directories mimicking SageMaker paths
RUN mkdir -p /opt/ml/input/data/training
RUN mkdir -p /opt/ml/input/config
RUN mkdir -p /opt/ml/model

# Set working directory (code will be mounted here)
WORKDIR /opt/ml/code

# Default entrypoint
ENTRYPOINT ["python"]
