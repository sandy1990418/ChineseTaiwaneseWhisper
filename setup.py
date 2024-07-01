from setuptools import setup, find_packages

setup(
    name="chinese_taiwanese_whisper_asr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.25.0",
        "datasets>=2.6.1",
        "librosa>=0.9.2",
        "numpy>=1.21.0",
        "tqdm>=4.62.3",
        "peft>=0.3.0",
        "faster-whisper>=0.5.0",
        "gradio>=3.20.0",
        "sentencepiece>=0.1.96",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Chinese/Taiwanese ASR system using Whisper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chinese-taiwanese-whisper-asr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)