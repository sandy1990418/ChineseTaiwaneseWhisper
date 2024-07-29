sudo systemctl restart docker
docker build --no-cache -t asr-service .
docker container run -d -it -d --name asr-container --gpus all -p 8000:8000 --env-file app/.env  asr-service  /bin/bash
docker exec -it 3699a2000783 bash


docker rmi 4a5bbc3f2008
docker rm 4a5bbc3f20
docker ps -a 
docker build -t asr-service .
docker run -d --name asr-container --gpus all -p 8000:8000 --env-file .env asr-service
docker exec -it 536549e721d0 bash