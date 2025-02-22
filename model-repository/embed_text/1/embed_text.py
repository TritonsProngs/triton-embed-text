import asyncio
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for text embedding
    models. Currently Multilingual-E5-Large and SigLIP are supported.
    """

    def initialize(self, args):
        """
        Initialize and load configuration parameters.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.model_config = model_config = json.loads(args["model_config"])

        self.embed_models = set(["multilingual_e5_large", "siglip_text", "e5_large_v2"])
        # Specify the default embedding model. Can be overriden in request parameter
        self.default_embed_model = model_config["parameters"]["default_embed_model"][
            "string_value"
        ]
        # Specify the default truncation value. Can be overridden in request parameter
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

    async def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided texts.

        Option Request Parameters
        -------------------------
        embed_model : str
            Specify which embedding model to use.
            If None, default_embed_model is used.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing text to be embedded.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with embedding results or error messages
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"embed_text.execute received {batch_size} requests")
        responses = [None] * batch_size
        inference_response_awaits = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            # Handle any request parameters
            request_params = json.loads(request.parameters())
            embed_model = request_params.get("embed_model", self.default_embed_model)
            truncation = request_params.get("truncation", self.default_truncation)

            if embed_model not in self.embed_models:
                responses[batch_id] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{embed_model=:} not in {self.embed_models}"
                    )
                )
                continue

            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"Failed on getting input tensor from request: {exc}"
                    )
                )
                responses[batch_id] = response
                continue
            else:
                infer_model_request = pb_utils.InferenceRequest(
                    model_name=embed_model,
                    parameters={"truncation": truncation},
                    requested_output_names=["EMBEDDING"],
                    inputs=[input_text_tt],
                )
                inference_response_awaits.append(infer_model_request.async_exec())
                valid_requests.append(batch_id)

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for model_response, batch_id in zip(inference_responses, valid_requests):
            if model_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    "Error embedding the text: " + f"{model_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
            else:
                embedding_tt = pb_utils.get_output_tensor_by_name(
                    model_response, "EMBEDDING"
                )
                response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
                responses[batch_id] = response

        return responses
