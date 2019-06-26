# rl-portfolio

The project uses a Docker image to ensure compatibility. 

Everyone [install Docker](https://docs.docker.com)!

For now, we'll use a pre-built image which has Jupyter notebooks and 
RL-relevant packages installed. 
See: https://github.com/jaimeps/docker-rl-gym

To install it, run
```
docker pull jaimeps/rl-gym
```
This will take a while to run, but only once.

Then to run the image, run
```
docker run -p 8888:8888 -v `pwd`/rl-portfolio:/home/jovyan jaimeps/rl-gym
```
and follow the instructrions, 
where `pwd` is the directory into which you've cloned our repository.
