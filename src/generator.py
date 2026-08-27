class Generator:
    def __init__(self, model_name:str, device:str):
        from transformers import (AutoTokenizer, AutoModelForCausalLM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def generate(self, prompt:str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the "
                    "provided context. "
                    "If the answer is not present in the "
                    "context, say you don't know."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        # Handle both Tensor and BatchEncoding outputs
        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
        else:
            input_ids = inputs

        input_length = input_ids.shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                input_ids = input_ids,
                max_new_tokens=200
            )
        generated_tokens = output[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer