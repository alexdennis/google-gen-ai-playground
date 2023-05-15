import argparse
import vertexai
from vertexai.preview.language_models import TextGenerationModel

parser = argparse.ArgumentParser(description="Universal Book Chatbot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("book", help="Book name")
parser.add_argument("character", help="Book character")
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
    return response.text


def main():
    """Main function."""
    args = parser.parse_args()
    config = vars(args)
    # print(config)
    content = '''You are a chatbot designed to give users the ability to talk to characters in any book. Play the role of %s in the %s.  Don't break out of character. Please give chapter and verse numbers that validate your answers where possible. Respond in less than 500 words and avoid repetition.

input: %s
output:
''' % (config["character"], config["book"], config["prompt"])
    # print(content)
    text = predict_large_language_model_sample("alex-gen-ai",
                                               "text-bison@001", 0.5, 256, 0.8, 40,
                                               content,
                                               "us-central1")
    print(config["character"], ": ", text)


if __name__ == "__main__":
    main()
