# SigLIP Text
This deployment hosts the [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
text model. You provide it your text and it returns the embedding vector (d=1152) that
can be used for zero/few-shot image classification. This is a Triton Inference Server
[ensemble](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)
of the [siglip_text_tokenizer](../model_repository/siglip_text_tokenizer) and the
[siglip_text_model](../model_repository/siglip_text_model) deployment models.

**Note** that the maximum input token size of 64 tokens is much smaller than most
language models. You should think of the text you want embedded as the caption/alt-text
for an image since that was the kind of data used to train this model.

Dynamic batching is enabled for this deployment, so clients simply send in a single
string of text to be embedded.

This is a lower level of abstraction, most clients likely should be using
[embed_text](embed_text.md) deployment.

Optional Request Parameters:
* `truncation`: str | bool, optional, default=False
  Passed along to Huggingface tokenizer class. See [tokenizer docs](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__) for more details. Mostly likely
  value will be `True` which truncates the provided text to the maximum length of
  tokens supported by the model. Otherwise, if the provided text is too many tokens,
  an error will be returned. Acceptable values for current tokenizer class are:
    * True or 'longest_first'
    * 'only_first'
    * 'only_second'
    * False or 'do_not_truncate'

## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 1] because we have dynamic batching and the first axis is
   the batch size and the second axis is the number of text strings to send (in this
   case always 1).
2. "datatype": "BYTES" because the input text is a string [don't ask why its not
   'STRING']

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"
caption = (
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon"
)

inference_request = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [caption],
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/siglip_text/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "siglip_text",
    "model_version": "1",
    "parameters": {
        "sequence_id": 0,
        "sequence_start": False,
        "sequence_end": False,
    },
    "outputs": [
        {
            "name": "EMBEDDING",
            "datatype": "FP32",
            "shape": [1, 1152],
            "data": [
                1.30078125,
                0.61572265,
                ...,
            ]
        }
    ]
}
"""

embedding = np.array(
    model_response["outputs"][0]["data"],
    dtype=np.float32
)
```

### Sending Many Requests
If you want to send a lot of text requests to be embedded, it's important that you send
each request in a multithreaded way to achieve optimal throughput.

NOTE: You will encounter a "OSError Too many open files" if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increase this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them once
they are completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"

input_texts = [
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, input_text in enumerate(input_texts):
        inference_request = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [input_text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/siglip_text/infer",
            json=inference_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response_json = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        if "error" not in response_json:
            data = response_json["outputs"][0]["data"]
            embedding = np.array(data, dtype=np.float32)
            embeddings[futures[future]] = embedding
        else:
            print(f"Error getting data from response: {response_json['error']}")

print(embeddings)
```

## Performance Analysis
There is some data in [data/siglip_text](../data/siglip_text/input_ids.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m siglip_text \
    -v \
    --input-data data/embed_text/imagenet_categories.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 1379.04 infer/sec. Avg latency: 43459 usec (std 6852 usec). 
  * Pass [2] throughput: 1407.58 infer/sec. Avg latency: 42621 usec (std 6420 usec). 
  * Pass [3] throughput: 1399.49 infer/sec. Avg latency: 42864 usec (std 6488 usec). 
  * Client: 
    * Request count: 100588
    * Throughput: 1395.38 infer/sec
    * Avg client overhead: 0.07%
    * Avg latency: 42978 usec (standard deviation 6597 usec)
    * p50 latency: 44836 usec
    * p90 latency: 47023 usec
    * p95 latency: 48051 usec
    * p99 latency: 51490 usec
    * Avg HTTP time: 42972 usec (send 35 usec + response wait 42937 usec + receive 0 usec)
  * Server: 
    * Inference count: 100588
    * Execution count: 100588
    * Successful request count: 100588
    * Avg request latency: 44343 usec (overhead 0 usec + queue 19049 usec + compute 25450 usec)

  * Composing models: 
  * siglip_text_model, version: 1
      * Inference count: 100588
      * Execution count: 3169
      * Successful request count: 100588
      * Avg request latency: 39918 usec (overhead 11 usec + queue 17777 usec + compute input 150 usec + compute infer 21978 usec + compute output 1 usec)

  * siglip_text_tokenize, version: 1
      * Inference count: 100648
      * Execution count: 11440
      * Successful request count: 100648
      * Avg request latency: 4600 usec (overhead 8 usec + queue 1272 usec + compute input 87 usec + compute infer 3232 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 1395.38 infer/sec, latency 42978 usec


## Validation
To validate that the model is performing as expected, we use some data from
[ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge).
The training data was nicely organized into subdirectories with each subdirectory
named after the Synset category and with each file name in a give subdirectory also
containing the {synset}_{file_id}.JPEG.

Working with images from the training data set, I put 10 images for each of the 1,000
categories into `train/{synset}` directory on my local machine. An additional
20 images for each of the 1,000 categories were placed into `valid/{synset}`.

```
train/
  - n01440764/
    - n01440764_3198.JPEG
    - n01440764_3199.JPEG
    - ...
  - n01443537/
    - n01443537_428.JPEG
    - ...
  - ...
```

In addition to the subset of images, I also downloaded the LOC_synset_mapping.txt. This
contains the synset category label and a description of the category. This data will be
used for performing the zero-shot accuracy validation. Here is the first
few lines:

| Label | Text Description |
| :----: | :-----------|
| n01440764 | tench, Tinca tinca |
| n01443537 | goldfish, Carassius auratus |
| n01484850 | great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias |
| n01491361 | tiger shark, Galeocerdo cuvieri |
| n01494475 | hammerhead, hammerhead shark |

### Zero-Shot KNN Classifier
The [SigLIP paper](https://arxiv.org/abs/2303.15343) uses zero-shot to measure the
quality of their embedding model. For zero-shot, you use a text description of the
category and embed that using the `siglip_text` deployment. This text embedding of the
category is what is used to fit a KNN Classifier. Taking each validation image
embedding, get the 10 nearest neighbors (where a neighbor is a text embedding of a
category), and use the neighbors' corresponding category label to predict the
classification of the validation image. We calculate both the top-1 and top-5 accuracy
where top-k means the classifier was correct if the true label appears among the top k
predicted category labels.

The SigLIP paper claims an ImageNet accuracy of 83.2% on the validation data of
ImageNet. The paper notes some tweak to the prompts and a few other details to
improve peformance. The results below show comparable accuracy.

### Results

|           | Top-1 Accuracy | Top-5 Accuracy | Prompt Template |
|:---------:| :------------: | :------------: | :-------------- |
| Zero-shot | 0.8196         | 0.9632         | A photo of a {text}. |

### Code
The code is available in
[model_repository/siglip_text/validate.py](../model_repository/siglip_text/validate.py)