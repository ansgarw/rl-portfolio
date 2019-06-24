# rl-portfolio

The project uses a Docker image to ensure compatibility. Everyone [install Docker](https://docs.docker.com)!
To download our Docker image, run
```
docker pull ansgarw/rl-portfolio:latest
```
Then run the Docker image locally with
```
docker run -p 8888:8888 -v `pwd`/rl-portfolio:/home/jovyan ansgarw/rl-portfolio:latest
```
where `pwd` is the directory that contains the github repository (i.e., the folder `rl-portfolio`).

When introducing new requirements (Python packages), add them to `requirements.txt`. 
Then update the docker image by running 
```
docker build -t ansgarw/rl-portfolio:latest rl-portfolio
docker push ansgarw/privacydocker
```
(at the moment only Ansgar has permission to push, let's look into changing this)
