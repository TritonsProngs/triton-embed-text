#  E5 Large V2 Text Embeddings
This deployment hosts the [E5-large-v2](https://huggingface.co/intfloat/e5-large-v2)
text embedding model. You provide it your text and it returns the embedding vector
(d=1024) that can be used for information retrieval or building downstream classifiers.
This is a Triton Inference Server
[ensemble](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)
of the [e5_large_v2_tokenizer](../model_repository/e5_large_v2_tokenizer)
(CPU) and the [e5_large_v2_model](../model_repository/e5_large_v2_model)
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
    url=f"{base_url}/e5_large_v2/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "e5_large_v2",
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
                0.03348,
                -0.0828,
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
            url=f"{base_url}/e5_large_v2/infer",
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
There is some data in [data/embed_text](../data/embed_text/e5_large_v2_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m e5_large_v2 \
    -v \
    --input-data data/embed_text/e5_large_v2_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 265.562 infer/sec. Avg latency: 225136 usec (std 30013 usec). 
  * Pass [2] throughput: 266.444 infer/sec. Avg latency: 224411 usec (std 28739 usec). 
  * Pass [3] throughput: 268.515 infer/sec. Avg latency: 223900 usec (std 29727 usec). 
  * Client: 
    * Request count: 19219
    * Throughput: 266.84 infer/sec
    * Avg client overhead: 0.02%
    * Avg latency: 224480 usec (standard deviation 29501 usec)
    * p50 latency: 231017 usec
    * p90 latency: 236006 usec
    * p95 latency: 239420 usec
    * p99 latency: 257344 usec
    * Avg HTTP time: 224474 usec (send 64 usec + response wait 224410 usec + receive 0 usec)
  * Server: 
    * Inference count: 19219
    * Execution count: 19219
    * Successful request count: 19219
    * Avg request latency: 225841 usec (overhead 0 usec + queue 105416 usec + compute 120705 usec)

  * Composing models: 
  * e5_large_v2_model, version: 1
      * Inference count: 19219
      * Execution count: 620
      * Successful request count: 19219
      * Avg request latency: 219309 usec (overhead 13 usec + queue 103635 usec + compute input 191 usec + compute infer 115468 usec + compute output 1 usec)

  * e5_large_v2_tokenize, version: 1
      * Inference count: 19279
      * Execution count: 1931
      * Successful request count: 19279
      * Avg request latency: 6833 usec (overhead 8 usec + queue 1781 usec + compute input 101 usec + compute infer 4941 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 266.84 infer/sec, latency 224480 usec

## Validation
To validate that the model is performing as expected, we calculate the performance on
a subset of the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb) and
compare them against the results published in the
[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
and also described in their [github repo](https://github.com/microsoft/unilm/tree/master/e5)

We compare the performance against Table 17 in the paper for EmotionClassification,
TweetSentimentExtractionClassification, SprintDuplicateQuestions, TwitterSemEval2015,
STS15, and STSBenchmark.

### Results
| Task | Score | Reported in paper |
| --- | --- | --- |
|EmotionClassification | 0.503| 0.435 |
|TweetSentimentExtractionClassification | 0.624| 0.525 |
|SprintDuplicateQuestions | 0.949| 0.920 |
|TwitterSemEval2015 | 0.769| 0.647 |
|STS15 | 0.898| 0.758 |
|STSBenchmark | 0.877| 0.709 |

Every one of these scores is higher than those reported in the original E5 paper, so
it appears that the v2 model is an improvement.

### Code
The code is available in [model_repository/e5_large_v2/validate.py](../model_repository/e5_large_v2/validate.py)