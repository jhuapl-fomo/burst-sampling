# Burst Sampling: Generating and Quantifying Improbable Text

Code related to [To Burst or Not to Burst: Generating and Quantifying Improbable Text](https://arxiv.org/pdf/2401.15476)

**Authors**:
- Kuleen Sasse
- Ted Staley
- Efsun Kayi
- Samuel Barham

## Background

### Burst Sampling -- A Summary
Burst Sampling is a novel text generation technique that aims to enhance human-likeness in machine-generated text by dynamically adjusting sampling strategies. Unlike traditional approaches such as top-k and nucleus sampling, Burst Sampling injects controlled bursts of unlikely tokens at strategic points in the generation process. This method creates more human-like variations in text while maintaining overall coherence and readability.

### Recoverability -- A Summary
Recoverability is a metric introduced to quantify how well a given sampling method can reproduce a piece of text. The core idea is to measure the ability of a model to reconstruct a specific sequence under different sampling strategies. The paper also presents evidence that suggests higher recoverability indicates that the text distribution aligns closely with human-authored content, providing a useful signal for evaluating the quality of generated text. 

### Setup

Run 
```bash
conda env create -f environment.yml
```
to get the environment called `steerable-generation`

Activate the environment with 
```bash
conda activate steerable-generation
```

You **MUST** download with `en_core_web_sm`
```bash
python -m spacy download en_core_web_sm
```

### Data Setup
1. You must create a [kaggle](https://www.kaggle.com/) account to download the arxiv dataset.
2. After doing that, login and go to settings and go API and create new token 
3. Copy that json file to the computer at ~/.kaggle/kaggle.json
```
5. Run this command in a folder with the conda environment activated
```bash 
kaggle datasets download -d Cornell-University/arxiv
```
6. Unzip and discard the .zip file downloaded

### Running the whole process
```
python -m src --config config.toml --mode <MODE>
```

## Citation

If you use this code, please cite the associated paper:

```
@article{sasse2024burst,
  title={To Burst or Not to Burst: Generating and Quantifying Improbable Text},
  author={Sasse, Kuleen and Barham, Samuel and Kayi, Efsun Sarioglu and Staley, Edward W},
  journal={arXiv preprint arXiv:2401.15476},
  year={2024}
}
```

## Acknowledgments

Copyright (c) 2025 The Johns Hopkins University Applied Physics Laboratory LLC.

## License

This software is released under the MIT license.
