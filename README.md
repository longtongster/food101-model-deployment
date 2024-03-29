# food101-model-deployment
Deploy effnet b2 model that was fine-tuned on food 101 on gradio

## Adding a large file

In order to track large files it advised to install git lfs. The link below shows how to track a large file and commit it.
These steps have been followed to add the PyTorch model artifact. 

https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage

## install dependencies

### method 1 - using a requirements.txt

When starting an ubuntu EC2 image one first has to perform some housekeeping:

`sudo apt update -y`

followed by:

`supo apt upgrade -y`

after this your machine should be up to date. 

Now we can install pip3 that is required for the installation of python packages

`sudo apt install python3-pip -y`

The last thing to install is git for large files (not sure if this is needed)

`sudo apt install git-lfs`

Now we can install the necessary packages. First install install PyTorch for the cpu:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

Then install gradio

`pip3 install gradio`

now clone this repository on your local machine or VM:

`git clone https://github.com/longtongster/food101-model-deployment.git`

When you use a cloud provider (e.g. AWS EC2) make sure that port 8080 is open to the world (or your IP). 

Then cd into the repo and and enter:

`python3 app.py`

Now the app should be up and running at `<EC2 public IP>:8080`

### method 2 - using a bootstrap script on AWS EC2

Using a bootstrap script we can automate all the above manual steps.

```
#!/bin/bash
sudo apt update -y
supo apt upgrade -y
sudo apt install python3-pip -y
sudo apt install git-lfs -y
pip3 install gradio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/longtongster/food101-model-deployment.git
cd food101-model-deployment
python3 app.py
```

The logs of the userdata installation can be checked in the following files:
`/var/log/cloud-init.log`
and
`/var/log/cloud-init-output.log`

The gradio app is running on port 8080. Please make sure that it is open. Open a webbrowser and type

`<EC2 public IP>:8080`

### method 3 - containerize the app

First start an AWS EC2 instance with a slightly shorter bootstrap script

```
#!/bin/bash
sudo apt update -y
supo apt upgrade -y
sudo apt install python3-pip -y
sudo apt install git-lfs -y
sudo apt install git-lfs -y
sudo apt-get install docker-compose -y
sudo usermod -aG docker ubuntu
```
Once the image is up and running docker and git will be available. Clone the repo

`git clone https://github.com/longtongster/food101-model-deployment.git`

create a dockerfile

`vim Dockerfile`

and insert the following in the file

```
FROM python
WORKDIR /usr/src/app
RUN pip install gradio
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY ./food101-model-deployment ./
CMD [ "python", "./app.py" ]
```

build the image 

`sudo docker image build -t model-app .`

start a container from this image

`sudo docker container run -d -p 80:8080 model-app`

Port 80 of the host is mapped to 8080 on the container. Open a webbrowser and enter the ip ardress of the instance

`<EC2 public IP>`



