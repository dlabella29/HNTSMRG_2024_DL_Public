# Use the PyTorch base image with CUDA 12.0
FROM --platform=linux/amd64 pytorch/pytorch

WORKDIR /

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Copy the application files and change ownership to the created user
COPY --chown=user:user requirements.txt /
# Upgrade pip and setuptools

RUN pip install --no-cache-dir --no-color -r /requirements.txt

ENV PYTHONUNBUFFERED=1
ENV AUTOpred="/HN_workdir/segresnet_0/prediction_testing/"
ENV AUTOscripts="/"

# Create a user and group for running the container securely

# Create directories for outputs, set ownership to user, and provide full permissions
RUN mkdir -p $AUTOpred \
    && chown -R user:user $AUTOpred \
    && chmod -R 777 $AUTOpred


COPY --chown=user:user . $AUTOscripts
COPY --chown=user:user inference.py /

# Switch to the created user
USER user

# Define the entry point command to run the inference script
ENTRYPOINT ["python", "inference.py"]

