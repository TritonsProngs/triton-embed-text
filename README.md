# triton-embed-text
A Triton Inference Server model repository for embedding text

Text embedding models convert text into dense numerical vectors, capturing semantic
meaning in a high-dimensional space. These vector representations enable machines to
process and understand textual data more effectively, facilitating various natural
language processing tasks.

* Document clustering and classification
    * Allows for making smaller downstream models that need less labeled data
* Semantic search and information retrieval
    * When paired with corresponding image embedding, enables searching for images by writing the alt-text description.
* Question/Answering Systems

The [embed_text](docs/embed_text.md) deployment is the main interface that should be
used by most clients. Currently supported models accessible within embed_text:

* [Multilingual E5 Text](docs/multilingual_e5_large.md) (default)
  Trained specifically to support multilingual retrieval capabilities, cross-lingual
  similarity search, and multilingual document classification.
* [SigLIP Text](docs/siglip_text.md)
  Use in conjunction with SigLIP Vision to perform zero-shot learning or semantic
  searching of images with textual descriptions.
* [E5 Large v2](docs/e5_large_v2.md)
  English only text embedding model. Useful for tasks that require English text embeddings.
  
## Running Tasks
Running tasks is orchestrated by using [Taskfile.dev](https://taskfile.dev/)

# Taskfile Instructions

This document provides instructions on how to run tasks defined in the `Taskfile.yml`.  

Create a task.env at the root of project to define enviroment overrides. 

## Tasks Overview

The `Taskfile.yml` includes the following tasks:

- `triton-start`
- `triton-stop`
- `model-import`
- `build-execution-env-all`
- `build-*-env` (with options: `embed_text`, `multilingual_e5_large`, `siglip_text`)

## Task Descriptions

### `triton-start`

Starts the Triton server.

```sh
task triton-start
```

### `triton-stop`

Stops the Triton server.

```sh
task triton-stop
```

### `model-import`

Import model files from huggingface

```sh
task model-import
```

### `task build-execution-env-all`

Builds all the conda pack environments used by Triton

```sh
task build-execution-env-all
```

### `task build-*-env`

Builds specific conda pack environments used by Triton

```sh
#Example 
task build-siglip_text-env
```

### `Complete Order`

Example of running multiple tasks to stage items needed to run Triton Server

```sh
task build-execution-env-all
task model-import
task triton-start
# Tail logs of running container
docker logs -f $(docker ps -q --filter "name=triton-inference-server")
```
