class Generator:
    def __init__(self, model_name:str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    def generate(self, prompt:str)-> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs,
            max_new_tokens=200
        )
        generated_tokens = output[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer