docker build -t chinese-taiwanese-whisper:latest -f app/Dockerfile .
docker container run -d -it --privileged=true --name whisperdocker -p 8000:8000 --gpus all  chinese-taiwanese-whisper:latest 