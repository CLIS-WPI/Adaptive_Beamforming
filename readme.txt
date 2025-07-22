docker build --no-cache -t adaptive_beamforming .

docker run --rm -it --gpus all -v "$(pwd):/workspace" adaptive_beamforming /bin/bash
