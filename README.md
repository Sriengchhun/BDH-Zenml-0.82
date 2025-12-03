# $${\color{lightgreen}Sriengchhun}$$
## <u>Download docker from zenml github </u> ✅
```python 
## Clone zenml github
git clone https://github.com/zenml-io/zenml.git
cd /zenml/docker   (for mine: cd /home/chhun/2024/June/zenml/docker)
## Build Images
docker build -t bdh_zenml_server -f base.Dockerfile .
## Run docker (for test the docker image)
docker run -d -p 8080:80 bdh_zenml_server
## Remove all dangling images
docker image prune -y
```

## <u>Push docker image to docker hub </u> ✅

```python
$ docker login
$ docker tag <local_image_name> <dockerhub_username>/<repository_name>:<tag> 
## e.g: docker tag bdh_zenml_server:latest sriengchhun/bdh:0.1
$ docker images
$ docker push <dockerhub_username>/<repository_name>:<tag>
## e.g: docker push sriengchhun/bdh:0.1
```
> [!NOTE]
> my docker hub: https://hub.docker.com/layers/sriengchhun/bdh/0.4/images/sha256-22487d25eca9c909d1ced23b5296b1d7a367a3cdbc321f457aec049ea4bcfd38?context=repo

> [!TIP]
> <ins>*Pull image*:</ins> docker pull sriengchhun/bdh:0.4 


---

# 💘 <u>Add own Zenml docker image from docker hub</u> 💘
## <u>*Docker-compose*</u>

```
## Build the container
$ docker-compose up --build
## If you want to down the container
$ docker-compose down
```

## <u>*Run script file* </u>
*Config Container after using docker-compose up —build using script file*
- Install library
```
pip install -r requirements.txt 
```
- Set up environment
```
source ./run_all_config_script.sh
```
## <u>Test train and deploy model</u>

```
cd /app/Full_hyper_infienon && ./model_run.sh
```


## <u>API swagger</u> 🚩
![alt text](image-3.png)


*Readme.md trick: https://stackoverflow.com/questions/11509830/how-to-add-color-to-githubs-readme-md-file*