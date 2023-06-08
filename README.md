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
`supo apt upgrade -y`

after this your machine should be up to date. 

Now we can install pip3 that is required for the installation of python packages

`sudo apt install python3-pip -y`

The last thing to install is git for large files (not sure if this is needed)

`sudo apt install git-lfs`

now clone this repo on your VM.

Now we can install the necessary packages. First install install PyTorch for the cpu:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

Then install gradio

`pip3 install gradio`



### method 2 - using a bootstrap script on AWS EC2

TODO
