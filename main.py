import argparse
import vertexai
from vertexai.preview.language_models import TextGenerationModel

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("character", help="Bible character")
parser.add_argument("prompt", help="What do want to ask them")


def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    print(f"Response from Model: {response.text}")


def main():
    """Main function."""
    args = parser.parse_args()
    config = vars(args)
    print(config)
    predict_large_language_model_sample("alex-gen-ai",
                                        "text-bison@001", 0.2, 256, 0.8, 40,
                                        '''You are to play the role of %s in the Bible. Please give Bible references that validate your answers. Keep your responses less than 500 words.

input: %s
output:
''' % (config["character"], config["prompt"]),
        "us-central1")


if __name__ == "__main__":
    main()
