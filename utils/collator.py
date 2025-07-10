from transformers import DataCollatorWithPadding

class SmartCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "token_type_ids" in batch:
            batch["token_type_ids"].zero_()
        return batch
