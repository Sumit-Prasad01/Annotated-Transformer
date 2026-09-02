import os
import argparse
import torch
from utils.helper import read_yaml
from utils.logger import logger
from utils.custom_exception import CustomException
from src.data import Vocab
from src.models import make_model
from src.inference import Translator


def predict(text: str, config_path: str = "config/config.yaml", checkpoint_path: str = None):
    try:
        config = read_yaml(config_path)
        output_dir = config.get("output_dir", "outputs")
        vocab_path = config.get("vocab_path", os.path.join(output_dir, "vocab.pt"))

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please train the model first.")

        vocab_data = torch.load(vocab_path)
        vocab_src = Vocab(vocab_data["src_stoi"], vocab_data["src_itos"])
        vocab_tgt = Vocab(vocab_data["tgt_stoi"], vocab_data["tgt_itos"])

        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_dir, f"{config.get('checkpoint_prefix', 'model_')}best.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please train the model first.")

        model = make_model(
            src_vocab=len(vocab_src),
            tgt_vocab=len(vocab_tgt),
            N=config.get("num_layers", 6),
            d_model=config.get("d_model", 512),
            d_ff=config.get("d_ff", 2048),
            h=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
        )

        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        translator = Translator(
            model=model,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            max_len=config.get("max_decode_len", 72),
        )

        translation = translator.translate(text)
        print(f"\nSource (DE): {text}")
        print(f"Translation (EN): {translation}\n")
        return translation

    except Exception as e:
        logger.error("Error executing translation inference.")
        raise CustomException("Failed to execute translation inference", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Translation Inference with Transformer Model")
    parser.add_argument("--text", type=str, required=True, help="Input German sentence to translate")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    predict(args.text, args.config, args.checkpoint)
