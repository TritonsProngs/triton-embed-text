#  Multilingual E5 Text Embeddings
This deployment hosts the [Multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large)
text embedding model. You provide it your text and it returns the embedding vector
(d=1024) that can be used for information retrieval or building downstream classifiers.
This is a Triton Inference Server
[ensemble](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)
of the [multilingual_e5_large_tokenizer](../model_repository/multilingual_e5_large_tokenizer)
(CPU) and the [multilingual_e5_large_model](../model_repository/multilingual_e5_large_model)
(GPU) deployment models.

**Note**
  * Maximum input token size is 512 tokens which is the expected size. Text longer
    than this will throw an error.
  * As specified in the Huggingface model card. You need to prefix your text with
    "query: " or "passage: " before tokenizing the text. Note the space after the colon
      * Use "query: " & "passage: " correspondingly for asymmetric tasks such as
        passage retrieval in open QA or ad-hoc information retrieval
      * Use "query:" prefix for symmetric tasks such as semantic similarity, bitext
        mining, paraphrase retrieval
      * Use "query: " prefix if you want to use embeddings as features, such as linear
        probing classification or clustering

Dynamic batching is enabled for this deployment, so clients simply send in each request
separately.

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
   the batch size and the second axis the number of text strings to send (in this
   case always 1).
2. "datatype": "BYTES" because the input text is a string [don't ask why its not
   'STRING']

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)

inference_request = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/multilingual_e5_large/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "multilingual_e5_large",
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
            "shape": [1, 1024],
            "data": [
                0.00972,
                -0.0810,
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
Typically the default ulimit is 1024 on most system. Either increace this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"
input_texts = [
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape.",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(input_texts):
        inference_request = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/multilingual_e5_large/infer",
            json=inference_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response_json = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        else:
            if "error" not in response_json:
                data = response_json["outputs"][0]["data"]
                embedding = np.array(data, dtype=np.float32)
                embeddings[futures[future]] = embedding
            else:
                print(f"{futures[future]} threw {response_json['error']}")

print(embeddings)
```

## Performance Analysis
There is some data in [data/embed_text](../data/embed_text/multilingual_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m multilingual_e5_large \
    -v \
    --input-data data/embed_text/multilingual_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 254.764 infer/sec. Avg latency: 234461 usec (std 36027 usec). 
  * Pass [2] throughput: 254.9 infer/sec. Avg latency: 235338 usec (std 34028 usec). 
  * Pass [3] throughput: 255.511 infer/sec. Avg latency: 234226 usec (std 34697 usec). 
  * Client: 
    * Request count: 18369
    * Throughput: 255.058 infer/sec
    * Avg client overhead: 0.01%
    * Avg latency: 234675 usec (standard deviation 14687 usec)
    * p50 latency: 245945 usec
    * p90 latency: 249144 usec
    * p95 latency: 250274 usec
    * p99 latency: 258071 usec
    * Avg HTTP time: 234669 usec (send 73 usec + response wait 234596 usec + receive 0 usec)
  * Server: 
    * Inference count: 18369
    * Execution count: 18369
    * Successful request count: 18369
    * Avg request latency: 236117 usec (overhead 0 usec + queue 109536 usec + compute 126845 usec)

  * Composing models: 
  * multilingual_e5_large_model, version: 1
    * Inference count: 18369
    * Execution count: 586
    * Successful request count: 18369
    * Avg request latency: 230277 usec (overhead 12 usec + queue 107915 usec + compute input 178 usec + compute infer 122170 usec + compute output 1 usec)

  * multilingual_e5_large_tokenize, version: 1
    * Inference count: 18429
    * Execution count: 1937
    * Successful request count: 18429
    * Avg request latency: 6124 usec (overhead 8 usec + queue 1621 usec + compute input 97 usec + compute infer 4397 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 255.058 infer/sec, latency 234675 usec

## Validation
To validate that the model is performing as expected, we calculate the performance on the
[BUCC Bitext Mining dataset](https://huggingface.co/datasets/mteb/bucc-bitext-mining)
and compare the performance against results published in the
[Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672).

This dataset consists of 4 separate dataset with pairs of sentences in [zh-en, fr-en,
de-en, and ru-en]. Table 5 in the paper reports that the Multilingual E5 large model
achieved **98.6** on this benchmark. Unfortunately the paper doesn't give any details
as to how they did the evaluation. In particular, the BUCC Biitext Mining dataset is
supposed to consist of non-parallel sentences with only about 2-3% of the sentences
having a corresponding translated sentence in the other language. However, the
Huggingface test data has aligned sentences. This may make the task much too easy, but
we will proceed in the absence of more information.

For each language pair dataset, we query with one side and calculate the top-1 accuracy
of finding the corresponding pair in the other language. We calculate a weighted
average across the four sets of language pairs to get a single number. We use
approximate nearest neighbors to perform the search of the 4 nearest neighbors based
upon the cosine distance. We then perform two separate reranking methods before
choosing the top nearest neighbor from this candidate list.  The first is just the
cosine distance itself. The second is based upon a margin scoring approach that is
referenced in the technical approach. This approach is outlined in
[Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/abs/1811.01136).

### Results

| Language Pairs | Margin Accuracy | Cosine Accuracy | # of Records |
| :------------: | :-------------: | :-------------: | :----------: |
| zh-en | 99.53 | 99.21 | 1,899 |
| fr-en | 99.12 | 98.72 | 9,086 |
| de-en | 99.63 | 99.54 | 9,580 |
| ru-en | 97.91 | 97.71 | 14,435|
| **Mean** | **98.79** | **98.55** | |

These match well with the reported 98.6 in the technical report.

### Code
The code is available in [model_repository/multilingual_e5_large/validate.py](../model_repository/multilingual_e5_large/validate.py)