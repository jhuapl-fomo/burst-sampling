# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Any, Dict
import numpy as np
import re
import torch
import random
from tqdm import tqdm

from src.detector.metric_based.log_likelihood import LogLikelihoodMetric
from src.detector.thresholding_model import ThresholdingModel
from src.detector.detector_model import DetectorModel


class DetectGPTConfig:
    def __init__(
        self,
        random_fills,
        random_fills_tokens,
        pct_words_masked,
        buffer_size,
        mask_top_p,
        chunk_size,
        n_perturbation_rounds,
        perturbation_mode="z",
        n_perturbations=10,
        span_length=10,
    ):
        self.random_fills = random_fills
        self.random_fills_tokens = random_fills_tokens

        self.pct_words_masked = pct_words_masked

        # either d or z
        self.perturbation_mode = perturbation_mode

        self.n_perturbations = n_perturbations
        self.span_length = span_length

        self.buffer_size = buffer_size

        self.mask_top_p = mask_top_p

        self.chunk_size = chunk_size

        self.n_perturbation_rounds = n_perturbation_rounds


class DetectGPT(DetectorModel):
    def __init__(
        self,
        base_model,
        base_tokenizer,
        base_device,
        mask_model,
        mask_tokenizer,
        mask_device,
        detectgpt_config,
    ) -> None:
        super().__init__()

        self.clf_model = ThresholdingModel()
        self.log_likelihood = LogLikelihoodMetric(
            base_model, base_tokenizer, base_device
        )

        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.base_device = base_device

        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.mask_device = mask_device

        self.pattern = re.compile(r"<extra_id_\d+>")

        self.fill_dict = None
        self.n_positions = 512

        self.detectgpt_config = detectgpt_config

    def train_model(self, x_train, y_train, x_test, y_test) -> Dict[str, Any]:
        # get mask filling model (for DetectGPT only)
        if self.detectgpt_config.random_fills:
            self.fill_dict = set()

            for texts in x_train + x_test:
                for text in texts:
                    self.fill_dict.update(text.split())

            self.fill_dict = sorted(list(self.fill_dict))

        if not self.detectgpt_config.random_fills:
            try:
                self.n_positions = self.mask_model.config.n_positions
            except AttributeError:
                self.n_positions = 512

        perturbation_results = self.get_perturbation_results(
            x_train, y_train, x_test, y_test
        )

        res = self.run_perturbation_experiment(perturbation_results)

        return res

    def get_perturbation_results(self, x_train, y_train, x_test, y_test):
        p_train_text = self.perturb_texts(
            [x for x in x_train for _ in range(self.detectgpt_config.n_perturbations)],
            ceil_pct=False,
        )

        p_test_text = self.perturb_texts(
            [x for x in x_test for _ in range(self.detectgpt_config.n_perturbations)],
            ceil_pct=False,
        )

        for _ in range(self.detectgpt_config.n_perturbation_rounds - 1):
            try:
                p_train_text, p_test_text = self.perturb_texts(
                    p_train_text, ceil_pct=False
                ), self.perturb_texts(p_test_text, ceil_pct=False)
            except AssertionError:
                break

        train = []
        test = []

        for idx in range(len(x_train)):
            train.append(
                {
                    "text": x_train[idx],
                    "label": y_train[idx],
                    "perturbed_text": p_train_text[
                        idx
                        * self.detectgpt_config.n_perturbations : (idx + 1)
                        * self.detectgpt_config.n_perturbations
                    ],
                }
            )
        for idx in range(len(x_test)):
            test.append(
                {
                    "text": x_test[idx],
                    "label": y_test[idx],
                    "perturbed_text": p_test_text[
                        idx
                        * self.detectgpt_config.n_perturbations : (idx + 1)
                        * self.detectgpt_config.n_perturbations
                    ],
                }
            )

        for res in tqdm(train, desc="Computing log likelihoods"):
            p_ll = [self.log_likelihood.get_score(x) for x in res["perturbed_text"]]

            res["ll"] = self.log_likelihood.get_score(res["text"])
            res["all_perturbed_ll"] = p_ll
            res["perturbed_ll_mean"] = np.mean(p_ll)
            res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1

        for res in tqdm(test, desc="Computing log likelihoods"):
            p_ll = [self.log_likelihood.get_score(x) for x in res["perturbed_text"]]

            res["ll"] = self.log_likelihood.get_score(res["text"])
            res["all_perturbed_ll"] = p_ll
            res["perturbed_ll_mean"] = np.mean(p_ll)
            res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1
        results = {"train": train, "test": test}

        return results

    def perturb_texts_(self, texts, ceil_pct=False):
        if not self.detectgpt_config.random_fills:
            masked_texts = [self.tokenize_and_mask(x, ceil_pct) for x in texts]

            raw_fills = self.replace_masks(masked_texts)

            extracted_fills = self.extract_fills(raw_fills)
            perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            while "" in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
                print(idxs)
                for idx, x in enumerate(perturbed_texts):
                    if idx in idxs:
                        print(x)
                for idx, x in enumerate(texts):
                    if idx in idxs:
                        print(x)
                masked_texts = [
                    self.tokenize_and_mask(x, ceil_pct)
                    for idx, x in enumerate(texts)
                    if idx in idxs
                ]

                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = self.apply_extracted_fills(
                    masked_texts, extracted_fills
                )
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            if self.detectgpt_config.random_fills_tokens:
                # tokenize base_tokenizer
                tokens = self.base_tokenizer(
                    texts, return_tensors="pt", padding=True
                ).to(self.base)
                valid_tokens = tokens.input_ids != self.base_tokenizer.pad_token_id
                replace_pct = self.detectgpt_config.pct_words_masked * (
                    self.detectgpt_config.span_length
                    / (
                        self.detectgpt_config.span_length
                        + 2 * self.detectgpt_config.buffer_size
                    )
                )

                # replace replace_pct of input_ids with random tokens
                random_mask = (
                    torch.rand(tokens.input_ids.shape, device=self.base_device)
                    < replace_pct
                )
                random_mask &= valid_tokens
                random_tokens = torch.randint(
                    0,
                    self.base_tokenizer.vocab_size,
                    (random_mask.sum(),),
                    device=self.base_device,
                )
                # while any of the random tokens are special tokens, replace them with random non-special tokens
                while any(
                    self.base_tokenizer.decode(x)
                    in self.base_tokenizer.all_special_tokens
                    for x in random_tokens
                ):
                    random_tokens = torch.randint(
                        0,
                        self.base_tokenizer.vocab_size,
                        (random_mask.sum(),),
                        device=self.base_device,
                    )
                tokens.input_ids[random_mask] = random_tokens
                perturbed_texts = self.base_tokenizer.batch_decode(
                    tokens.input_ids, skip_special_tokens=True
                )
            else:
                masked_texts = [self.tokenize_and_mask(x, ceil_pct) for x in texts]
                perturbed_texts = masked_texts
                # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
                for idx, text in enumerate(perturbed_texts):
                    filled_text = text
                    for fill_idx in range(self.count_masks([text])[0]):
                        fill = random.sample(
                            self.fill_dict, self.detectgpt_config.span_length
                        )
                        filled_text = filled_text.replace(
                            f"<extra_id_{fill_idx}>", " ".join(fill)
                        )
                    perturbed_texts[idx] = filled_text

        return perturbed_texts

    def tokenize_and_mask(self, text, ceil_pct=False):
        tokens = text.split(" ")
        if len(tokens) > 512:
            tokens = tokens[:512]
        mask_string = "<<<mask>>>"

        n_spans = (
            self.detectgpt_config.pct_words_masked
            * len(tokens)
            / (
                self.detectgpt_config.span_length
                + self.detectgpt_config.buffer_size * 2
            )
        )
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(
                0, len(tokens) - self.detectgpt_config.span_length
            )
            end = start + self.detectgpt_config.span_length
            search_start = max(0, start - self.detectgpt_config.buffer_size)
            search_end = min(len(tokens), end + self.detectgpt_config.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = " ".join(tokens)
        return text

    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(
            self.mask_device
        )

        outputs = self.mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=self.detectgpt_config.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )

        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def count_masks(self, texts):
        return [
            len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts
        ]

    def perturb_texts(self, texts, ceil_pct=False):
        outputs = []
        for i in tqdm(
            range(0, len(texts), self.detectgpt_config.chunk_size),
            desc="Applying perturbations",
        ):
            outputs.extend(
                self.perturb_texts_(
                    texts[i : i + self.detectgpt_config.chunk_size], ceil_pct=ceil_pct
                )
            )
        return outputs

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(" ") for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(
            zip(tokens, extracted_fills, n_expected)
        ):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def run_perturbation_experiment(self, results):
        # Train
        train_predictions = []
        for res in results["train"]:
            if self.detectgpt_config.perturbation_mode == "d":
                train_predictions.append(res["ll"] - res["perturbed_ll_mean"])
            elif self.detectgpt_config.perturbation_mode == "z":
                if res["perturbed_ll_std"] == 0:
                    res["perturbed_ll_std"] = 1

                train_predictions.append(
                    (res["ll"] - res["perturbed_ll_mean"]) / res["perturbed_ll_std"]
                )

        # Test
        test_predictions = []
        for res in results["test"]:
            if self.detectgpt_config.perturbation_mode == "d":
                test_predictions.append(res["ll"] - res["perturbed_ll_mean"])
            elif self.detectgpt_config.perturbation_mode == "z":
                if res["perturbed_ll_std"] == 0:
                    res["perturbed_ll_std"] = 1

                test_predictions.append(
                    (res["ll"] - res["perturbed_ll_mean"]) / res["perturbed_ll_std"]
                )

        x_train = train_predictions
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = [_["label"] for _ in results["train"]]

        x_test = test_predictions
        x_test = np.expand_dims(x_test, axis=-1)
        y_test = [_["label"] for _ in results["test"]]

        result_dict = self.clf_model.train_model(x_train, y_train, x_test, y_test, "")

        return result_dict
