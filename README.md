# MICCAI2019
UBC Biomedical Engineering Team repo for MICCAI 2019

## Developer Installation

You will need [Python](https://www.python.org/downloads/) and [Pip](https://www.makeuseof.com/tag/install-pip-for-python/) installed. We will be using Python 3.7 for the project.
We use [pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management. This is optional, you can install all the libraries in Pipfile manually on your computer as well, but that might lead to compatibility issues when running the same scripts on other computers if the versions installed are different.

```bash
git clone https://github.com/MatthewMong/MICCAI2019.git
cd MICCAI2019/
pip install pipenv
pipenv install --dev
```
This will create a virtual environment will all the dependencies set up.

To enter the environment:

```bash
pipenv shell
```

Now you can run any command as you would normally would on your computer in the Virtualenv.

## Docker Alternative

Most people will be able to use the above code to get set up and running. If you prefer to use [Docker](https://docs.docker.com/install/), run the following commands after Docker is installed (Make sure you meet all the requirements for Docker.).

```bash
git clone https://github.com/MatthewMong/MICCAI2019.git
cd MICCAI2019/
docker pull nielsborie/ml-docker
docker run --name ML-env -p 8887:8888 nielsborie/ml-docker
```

## No Virtualenv

If you run into problems with both, you can install all libraries locally. 

```bash
git clone https://github.com/MatthewMong/MICCAI2019.git
cd MICCAI2019/
pip install -r requirements.txt
```

