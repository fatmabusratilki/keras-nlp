# Keras-NLP Project

This project was developed to perform various natural language processing (NLP) tasks using Keras-NLP. It has been adapted from Keras' official website to run in any environment.

## Downloading and Preparing Datasets

You can download the required datasets for the project from the links below. Due to their large size, they are not included in this repository:

- **WikiText-103**: [Download](https://developer.ibm.com/exchanges/data/all/wikitext-103/)
- **SST-2**: [Download](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)

### Preparing the Datasets

1. Extract the downloaded datasets.
2. Place the extracted files into a folder named **datasets** in the project directory.

## Running with Docker

You can follow the steps below to run the project using Docker.

### 1. Installing Docker

First, make sure Docker is installed on your system. If not, you can download it from the link below:

- [Download Docker](https://www.docker.com/)

### 2. Preparing the Project Directory

- **For Linux Users**:
  - Open a terminal and navigate to the project directory.
- **For Windows Users**:
  - Press `Win + R` to open the **Run** window.
  - Type `cmd` and press **Enter**.
  - In the command prompt, navigate to the project directory.

### 3. Building and Running the Docker Image

Run the following commands to build the Docker image and start the project:

```sh
# Build the Docker image
docker build -t keras-nlp-app .

# Run the Docker container
docker run --rm keras-nlp-app
```

Alternatively, you can pull the pre-built Docker image directly with the following command:

```sh
docker pull busratilki/keras-nlp-app:latest
```

## Contributing

If you would like to contribute to the project, feel free to submit a pull request (PR) or report any issues.

## License

This project is licensed under the [MIT License](LICENSE).

---

If you encounter any issues, please do not hesitate to contact us!

