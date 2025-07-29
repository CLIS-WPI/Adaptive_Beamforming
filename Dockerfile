# Start from the official NVIDIA TensorFlow image.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# Set timezone to avoid interactive prompts during package installation
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Prevent interactive prompts from apt-get
ENV DEBIAN_FRONTEND=noninteractive

# --- Install Python Packages ---
# Install only the necessary Python packages for this specific simulation.
# Sionna for the communication simulation, Matplotlib for plotting, and Scikit-learn for t-SNE.
RUN python3 -m pip install --no-cache-dir --upgrade \
    'sionna' \
    'matplotlib' \
    'scikit-learn'\
    'tqdm'

# --- Copy Project Files ---
# Copy the main simulation script into the workspace
COPY main.py ./

# --- Default Command ---
# Set the main script as the default command to run when the container starts.
# To get an interactive shell instead, you can run:
# docker run -it --gpus all <your_image_name> /bin/bash
CMD ["python3", "main.py"]