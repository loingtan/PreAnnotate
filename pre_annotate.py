import os
import json
import argparse
from tool import VietnamesePOSTagger
import glob
from tqdm import tqdm
import re
import warnings
import logging
import uuid

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def find_word_indices(sentence_text, word, search_start):
    """Find the start and end index of a word within text, preserving punctuation positions"""
    start_index = sentence_text.find(word, search_start)
    if start_index != -1:
        end_index = start_index + len(word)

     
        prev_char = sentence_text[start_index - 1] if start_index > 0 else ' '
        next_char = sentence_text[end_index] if end_index < len(
            sentence_text) else ' '

        last_char = word[-1] if word else ''
        if last_char in '.,!?:;"\'()[]{}':
            end_index -= 1  

        if (prev_char.isspace() or prev_char in '.,!?:;"\'()[]{}') and \
           (next_char.isspace() or next_char in '.,!?:;"\'()[]{}' or next_char == ''):
            return start_index, end_index

        #
        return find_word_indices(sentence_text, word, start_index + 1)
    return None, None


def clean_sentence(sentence):
    """Remove <s> and </s> tags from a sentence while preserving internal content exactly"""

    if sentence.startswith("<s>") and sentence.endswith("</s>"):
        return sentence[3:-4].strip()
    return sentence.strip()


def split_sentences(text):
    """
    Split text into sentences considering both punctuation and blank lines
    """

    paragraphs = re.split(r'\n\s*\n', text)

    sentences = []
    for paragraph in paragraphs:
        paragraph = clean_sentence(paragraph)
        paragraph_sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences.extend([s.strip() for s in paragraph_sentences if s.strip()])
    cleaned_sentences = sentences
    return [s for s in cleaned_sentences if s]  # Remove any empty sentences


def process_file(file_path, tagger):
    """Process a single file and return Label Studio compatible annotations"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

    # Split text into sentences
    sentences = []
    for line in text.splitlines():
        if line.strip():
            cleaned = clean_sentence(line)
            if cleaned:
                sentences.append(cleaned)

    result_objects = []
    for sentence in sentences:
        if not sentence.strip():
            continue

        try:
            data_obj = {
                "data": {
                    "text": sentence
                }
            }

            # Process with POS tagger
            tagged_tokens = tagger.predict_sentence(sentence)

            # Build annotations
            annotations = []
            current_offset = 0

            for word, tag in tagged_tokens:
                # Find word in sentence
                word_start, word_end = find_word_indices(
                    sentence, word, current_offset)

                if word_start is not None:
                    annotations.append({
                        "id": str(uuid.uuid4())[:8],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": word_start,
                            "end": word_end,
                            "text": word,
                            "labels": [tag]
                        }
                    })
                    current_offset = word_end

            if annotations:
                data_obj["predictions"] = [{
                    "model_version": "vietnamese_pos_tagger",
                    "score": 0.8,
                    "result": annotations
                }]
                result_objects.append(data_obj)

        except Exception as e:
            logging.error(
                f"Error processing sentence: {sentence[:30]}... - {str(e)}")

    return result_objects


def process_directory(input_dir, output_dir, model_path):
    """Process all text files in a directory and generate Label Studio compatible JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # Load the POS tagger model
    try:
        logging.info(f"Loading model from {model_path}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tagger = VietnamesePOSTagger().load_model(model_path)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Find all text files
    text_files = glob.glob(os.path.join(input_dir, "*.txt"))
    logging.info(f"Found {len(text_files)} text files to process")

    if not text_files:
        logging.warning(f"No .txt files found in directory: {input_dir}")
        return

    # Process each file
    for file_path in tqdm(text_files, desc="Processing files"):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")

        try:
            # Process file and get Label Studio format data
            result_objects = process_file(file_path, tagger)

            if result_objects:
                # Write to JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_objects, f, ensure_ascii=False, indent=2)
                logging.info(
                    f"Processed {filename} with {len(result_objects)} sentences")
            else:
                logging.warning(
                    f"No valid annotations generated for {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-annotate Vietnamese text files with POS tags for Label Studio')
    parser.add_argument('--input', required=True,
                        help='Input directory containing .txt files')
    parser.add_argument('--output', required=True,
                        help='Output directory for Label Studio JSON files')
    parser.add_argument('--model', required=True,
                        help='Path to the trained POS tagger model (.pkl)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.input):
        logging.error(f"Input directory not found: {args.input}")
        return
    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return

    process_directory(args.input, args.output, args.model)
    logging.info("Pre-annotation completed successfully")


if __name__ == "__main__":
    main()
