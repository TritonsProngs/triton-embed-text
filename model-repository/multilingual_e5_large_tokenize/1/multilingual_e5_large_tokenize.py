import json
import numpy as np
from transformers import AutoTokenizer

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for
    multilingual-e5-large tokenization.
    """

    def initialize(self, args):
        """
        Initialize Tokenizer.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.model_config = model_config = json.loads(args["model_config"])

        # Specify the default truncate value. Can be overridden in request parameter
        default_truncation = model_config["parameters"]["default_truncation"][
            "string_value"
        ]
        if default_truncation in ["False", "false"]:
            self.default_truncation = False
        elif default_truncation in ["True", "true"]:
            self.default_truncation = True
        elif default_truncation in ["longest_first", "only_first", "only_second", "do_not_truncate"]:
            self.default_truncation = default_truncation
        else:
            raise ValueError(
                f"{default_truncation=:} does not match "
                "transformers.tokenizer.__call__ options"
            )


        self.tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-large", local_files_only=True
        )

    def process_request(self, request):
        """
        Process the input text request and get tokenized text.

        Parameters
        ----------
        request : pb_utils.InferenceRequest
            Inference request containing the input text.

        Returns
        -------
        np.ndarray
            Tokenized text, shape = (1, 512), dtype=np.int64
        """
        try:
            input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
        except Exception as exc:
            raise ValueError(f"Failed on getting input tensor from request: {exc}")

        # Handle any request parameters
        request_params = json.loads(request.parameters())
        truncation = request_params.get("truncation", self.default_truncation)

        try:
            input_text = [
                b.decode("utf-8") for b in input_text_tt.as_numpy().reshape(-1)
            ]
        except Exception as exc:
            raise ValueError(f"Failed on converting numpy to str request data: {exc}")

        # Make sure that input text is not empty
        if len(input_text[0]) == 0:
            raise ValueError(
                "You appeared to have submitted an empty string. Received: "
                + f"'{input_text[0]}'"
            )

        try:
            inputs = self.tokenizer(
                text=input_text,
                padding="max_length",
                truncation=truncation,
                return_tensors="pt",
            )
            input_ids_np = inputs["input_ids"].numpy()
            attention_mask_np = inputs["attention_mask"].numpy()
            n_tokens = input_ids_np.shape[-1]

            # Safety Checks
            # Could set truncation=True, but that seems dangerously silent for
            # something that could severely impact performance
            if n_tokens > 512:
                raise ValueError(
                    f"Processing {input_text} has {n_tokens} tokens which "
                    "exceeds max of 512. You could try setting request parameter "
                    "`truncation` to 'True' if you want to just use the first 512 "
                    "tokens"
                )
            for text in input_text:
                if not (text.startswith("query: ") or text.startswith("passage: ")):
                    raise ValueError(
                        f"'{text}' must start with 'query: ' or"
                        + f"'passage: ' prefix when using "
                        + f"multilingual_e5_large_tokenize. Note "
                        + f"the space after the colon."
                    )
        except Exception as exc:
            raise ValueError(
                f"Failed on multilingual_e5_large_tokenize(text=input_text): {exc}"
            )

        # Shape = [batch_size, 512], where batch_size should be 1
        return input_ids_np, attention_mask_np

    def execute(self, requests: list) -> list:
        """
        Execute a batch of tokenization requests on provided texts.

        Output Shape after tokenization = (1, 512), dtype=np.int64

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing text to be tokenized.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with tokenization results or error messages
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(
            f"multilingual_e5_large_tokenize.execute received {batch_size} requests"
        )
        responses = [None] * batch_size
        for batch_id, request in enumerate(requests):
            try:
                input_ids_np, attention_mask_np = self.process_request(request)
                input_ids_tt = pb_utils.Tensor("INPUT_IDS", input_ids_np)
                attention_mask_tt = pb_utils.Tensor("ATTENTION_MASK", attention_mask_np)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "INPUT_IDS", np.zeros((1, 512), dtype=np.int64)
                        ),
                        pb_utils.Tensor(
                            "ATTENTION_MASK", np.zeros((1, 512), dtype=np.int64)
                        ),
                    ],
                    error=pb_utils.TritonError(f"{exc}"),
                )
                responses[batch_id] = response
            else:
                response = pb_utils.InferenceResponse(
                    output_tensors=[input_ids_tt, attention_mask_tt]
                )
                responses[batch_id] = response

        return responses
