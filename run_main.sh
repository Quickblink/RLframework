docker build -t imrsnnrl docker_build && nvidia-docker run -v $(pwd):/home/developer --shm-size=1g --net host --ipc host --rm -t imrsnnrl python3 main.py $1
