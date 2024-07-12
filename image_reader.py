#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PDF READER]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the PDF READER.
# It converts a PDF into LlamaIndex nodes
# using UnstructuredIO.
#####################################################
## TODO Board:

#####################################################
## IMPORTS
# import os
# import re
# import regex
# from typing import Any, List, IO, Optional

# # Unstructured Element Types
# from unstructured.documents import elements, email_elements

# # Image to Text models
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM

import onnxruntime_genai as og
import argparse
import time

model = og.Model('C:/Users/jdwh0/.cache/huggingface/hub/models--microsoft--Phi-3-vision-128k-instruct-onnx-cuda/snapshots/77717364052f6b70ae645b65ca6efd0da486d371/cuda-int4-rtn-block-32')

def main(args):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length
    if 'max_length' not in search_options:
        search_options['max_length'] = 2048

    chat_template = '<|user|>/n{input} <|end|>\n<|assistant|>'

    # Keep asking for input prompts in a loop
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if args.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")
