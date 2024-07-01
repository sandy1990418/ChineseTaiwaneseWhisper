# Chinese/Taiwanese Whisper ASR Project

This project implements an Automatic Speech Recognition (ASR) system for Chinese (Traditional) and Taiwanese using the Whisper model. It supports both full fine-tuning and PEFT (Parameter-Efficient Fine-Tuning) methods, as well as streaming inference.

## Features

- Fine-tuning of Whisper models on Chinese/Taiwanese data
- Support for PEFT methods (e.g., LoRA) for efficient fine-tuning
- Batch and streaming inference
- Gradio web interface for easy interaction with the model
- Optimized for T4 GPUs

## Project Structure

```
chinese_taiwanese_whisper_asr/
│
├── src/
│   ├── config/
│   │   └── train_config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── data_collator.py
│   ├── models/
│   │   └── whisper_model.py
│   ├── trainers/
│   │   └── whisper_trainer.py
│   └── inference/
│       └── flexible_inference.py
├── scripts/
│   ├── train.py
│   └── gradio_interface.py
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_inference.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sandy1990418/ChineseTaiwaneseWhisper.git
   cd ChineseTaiwaneseWhisper
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run:

```
python scripts/train.py --model_name_or_path "openai/whisper-small" \
                        --dataset_name "mozilla-foundation/common_voice_11_0" \
                        --language "zh-TW" \
                        --output_dir "./whisper-finetuned-zh-tw" \
                        --num_train_epochs 3 \
                        --per_device_train_batch_size 8 \
                        --learning_rate 5e-5 \
                        --fp16
```

For PEFT fine-tuning (e.g., using LoRA), add the `--use_peft` flag:

```
python scripts/train.py --model_name_or_path "openai/whisper-small" \
                        --dataset_name "mozilla-foundation/common_voice_11_0" \
                        --language "zh-TW" \
                        --output_dir "./whisper-peft-finetuned-zh-tw" \
                        --num_train_epochs 3 \
                        --per_device_train_batch_size 16 \
                        --learning_rate 1e-4 \
                        --fp16 \
                        --use_peft \
                        --peft_method "lora"
```

### Inference

To run the Gradio interface for interactive inference:

```
python scripts/gradio_interface.py
```

This will start a web server, and you'll see a URL in the console (usually `http://127.0.0.1:7860`). Open this URL in your web browser to access the Gradio interface.

## Customization

- To use a different dataset, modify the `dataset_name` parameter in the training script.
- To change the PEFT method, adjust the `peft_method` parameter and corresponding configurations in `src/config/train_config.py`.
- For inference optimizations, you can modify the `ChineseTaiwaneseASRInference` class in `src/inference/flexible_inference.py`.

## Testing

This project uses pytest for testing. To run the tests:

1. Ensure you have pytest installed:
   ```
   pip install pytest
   ```

2. Run the tests:
   ```
   pytest tests/
   ```

This will discover and run all the tests in the `tests/` directory.

To run tests with more detailed output, use:
```
pytest -v tests/
```

For test coverage information, install pytest-cov and run:
```
pip install pytest-cov
pytest --cov=src tests/
```

This will show you the test coverage for the `src/` directory.


## Performance Optimization

If you encounter memory issues on your T4 GPU, try the following:

1. Reduce the batch size (`--per_device_train_batch_size`)
2. Use a smaller Whisper model (e.g., "openai/whisper-tiny")
3. Increase gradient accumulation steps (`--gradient_accumulation_steps`)
4. Enable mixed precision training (`--fp16`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- Mozilla Common Voice for the dataset