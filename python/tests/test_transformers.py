import inspect
import json
import os
import shutil

import numpy as np
import pytest
import test_utils

import ctranslate2


@pytest.fixture
def clear_transformers_cache():
    """Clears the Transformers model cache after each test when running in a CI."""
    yield
    clear_transformers_cache_in_ci()


def clear_transformers_cache_in_ci():
    import transformers

    if os.environ.get("CI") == "true":
        from huggingface_hub import constants

        shutil.rmtree(constants.HF_HUB_CACHE, ignore_errors=True)


_TRANSFORMERS_TRANSLATION_TESTS = [
    (
        "Helsinki-NLP/opus-mt-en-de",
        "▁Hello ▁world ! </s>",
        "",
        "▁Hallo ▁Welt !",
        dict(),
    ),
    (
        "Helsinki-NLP/opus-mt-en-roa",
        ">>ind<< ▁The ▁Prime ▁Minister ▁is ▁coming ▁back ▁tomorrow . </s>",
        "",
        "▁Per da na ▁Men teri ▁akan ▁kembali ▁besok .",
        dict(),
    ),
    (
        "Helsinki-NLP/opus-mt-mul-en",
        "▁Bon jo ur ▁le ▁mo nde </s>",
        "",
        "▁Welcome ▁to ▁the ▁World",
        dict(),
    ),
    (
        "facebook/m2m100_418M",
        "__en__ ▁Hello ▁world ! </s>",
        "__de__",
        "__de__ ▁Hallo ▁der ▁Welt !",
        dict(),
    ),
    (
        "facebook/mbart-large-50-many-to-many-mmt",
        "en_XX ▁Hello ▁world ! </s>",
        "de_DE",
        "de_DE ▁Hallo ▁Welt !",
        dict(),
    ),
    (
        "facebook/mbart-large-en-ro",
        "▁UN ▁Chief ▁Say s ▁There ▁Is ▁No ▁Militar y ▁Solution ▁in ▁Syria </s> en_XX",
        "ro_RO",
        "▁Şe ful ▁ONU ▁de cla ră ▁că ▁nu ▁există ▁o ▁solu ţie ▁militar ă ▁în ▁Siria",
        dict(),
    ),
    (
        "facebook/bart-base",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo <mask> Ġin ĠSyria </s>",
        "",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo ĠWar Ġin ĠSyria",
        dict(),
    ),
    (
        "google/pegasus-xsum",
        "▁PG & E ▁stated ▁it ▁scheduled ▁the ▁blackout s ▁in ▁response ▁to ▁forecasts "
        "▁for ▁high ▁winds ▁amid ▁dry ▁conditions . ▁The ▁aim ▁is ▁to ▁reduce ▁the "
        "▁risk ▁of ▁wildfires . ▁Nearly ▁800 ▁thousand ▁customers ▁were ▁scheduled ▁to "
        "▁be ▁affected ▁by ▁the ▁shutoff s ▁which ▁were ▁expected ▁to ▁last ▁through "
        "▁at ▁least ▁midday ▁tomorrow . </s>",
        "",
        "▁California ' s ▁largest ▁electricity ▁provider ▁has ▁turned ▁off ▁power ▁to "
        "▁hundreds ▁of ▁thousands ▁of ▁customers .",
        dict(length_penalty=0.6),
    ),
    (
        "facebook/nllb-200-distilled-600M",
        ["▁Hello ▁world ! </s> eng_Latn", "</s> eng_Latn"],
        ["fra_Latn", "fra_Latn"],
        ["fra_Latn ▁Bon jour ▁le ▁monde ▁!", "fra_Latn"],
        dict(),
    ),
    (
        "t5-small",
        "▁translate ▁English ▁to ▁German : ▁The ▁house ▁is ▁wonderful . </s>",
        "",
        "▁Das ▁Haus ▁ist ▁wunderbar .",
        dict(),
    ),
    (
        "ml6team/mt5-small-german-query-generation",
        "▁Das ▁Lama ▁( L ama ▁glam a ) ▁ist ▁eine ▁Art ▁der ▁Kam ele . "
        "▁Es ▁ist ▁in ▁den ▁süd amerikanische n ▁And en ▁ver breite t ▁und "
        "▁eine ▁vom ▁Guan ako ▁ab sta mmende ▁ Haustier form . </s>",
        "",
        "▁Was ▁ist ▁ein ▁Lama - L ama ?",
        dict(),
    ),
    (
        "jordimas/t5gemma-s-s-ul2",
        ["Question : ▁Why ▁is ▁the ▁sky ▁blue ? ▁Answer :"],
        "",
        "\n\n Answer : \n\n The ▁sky ▁is ▁blue .",
        dict(),
    ),
    (
        "jordimas/t5gemma-2-270m-270m",
        ["<bos> Question : ▁Why ▁is ▁the ▁sky ▁blue ? ▁Answer :"],
        "",
        "<unused6237> ▁The ▁sky ▁is ▁blue ▁because ▁the ▁sun ▁shines ▁on ▁it . "
        "▁The ▁sun ▁is ▁the ▁source ▁of ▁all ▁the ▁light ▁in ▁the ▁sky .",
        dict(max_decoding_length=50),
    ),
]


@test_utils.only_on_linux
@pytest.mark.parametrize(
    "model,source_tokens,target_tokens,expected_tokens,kwargs",
    _TRANSFORMERS_TRANSLATION_TESTS,
    ids=[args[0] for args in _TRANSFORMERS_TRANSLATION_TESTS],
)
def test_transformers_translation(
    clear_transformers_cache,
    tmp_dir,
    model,
    source_tokens,
    target_tokens,
    expected_tokens,
    kwargs,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    if not isinstance(expected_tokens, list):
        expected_tokens = [expected_tokens]
    if not isinstance(source_tokens, list):
        source_tokens = [source_tokens]
    if target_tokens and not isinstance(target_tokens, list):
        target_tokens = [target_tokens]

    translator = ctranslate2.Translator(output_dir)
    results = translator.translate_batch(
        [line.split() for line in source_tokens],
        [line.split() for line in target_tokens] if target_tokens else None,
        **kwargs,
    )
    output_tokens = [" ".join(result.hypotheses[0]) for result in results]
    assert output_tokens == expected_tokens


_TRANSFORMERS_GENERATION_TESTS = [
    (
        "bigcode/tiny_starcoder_py",
        (
            "<fim_prefix> def Ġprint _ one _ two _ three (): ĊĠĠĠ Ġprint (' one ') "
            "ĊĠĠĠĠ <fim_suffix> ĊĠĠĠ Ġprint (' three ') <fim_middle>"
        ),
        26,
        (
            "<fim_prefix> def Ġprint _ one _ two _ three (): ĊĠĠĠ Ġprint (' one ') "
            "ĊĠĠĠĠ <fim_suffix> ĊĠĠĠ Ġprint (' three ') <fim_middle>"
            " print (' two ')"
        ),
    ),
    (
        "Salesforce/codegen-350M-mono",
        "def Ġhello _ name ( name ):",
        25,
        "def Ġhello _ name ( name ):"
        ' Ċ      print ( f " Hello Ġ{ name } ") Ċ Ċ hello _ name (" John ")',
    ),
    (
        "gpt2",
        "<|endoftext|>",
        10,
        "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof",
    ),
    (
        "facebook/opt-350m",
        "</s>",
        10,
        "Ċ The Ġfollowing Ġis Ġa Ġlist Ġof Ġthe Ġmost Ġpopular",
    ),
    (
        "microsoft/DialoGPT-medium",
        "Hello <|endoftext|>",
        100,
        "Hello <|endoftext|> Hello Ġ! Ġ: D",
    ),
    (
        "bigscience/bloom-560m",
        "Hello , ĠI Ġam",
        20,
        "Hello , ĠI Ġam Ġa Ġnew bie Ġin Ġthe Ġworld Ġof Ġweb Ġdesign Ġand ĠI Ġam "
        "Ġlooking Ġfor Ġa Ġweb Ġdeveloper",
    ),
    (
        "jordimas/gemma-3-1b-it",
        "<bos> Which ▁city ▁hosted ▁the ▁Olympic ▁Games ▁in ▁ 1 9 9 2 ?",
        50,
        "Which ▁city ▁hosted ▁the ▁Olympic ▁Games ▁in ▁ 1 9 9 2 ? \n\n"
        " The ▁answer ▁is ▁** Barcelona **. \n",
    ),
    (
        "Qwen/Qwen3-0.6B",
        "<|im_start|> user Ċ What Ġis Ġthe Ġcapital Ġof ĠPortugal ? Ġ/ no _th ink Ċ <|im_end|> Ċ "
        "<|im_start|> assistant Ċ",
        50,
        "<|im_start|> user Ċ What Ġis Ġthe Ġcapital Ġof ĠPortugal ? Ġ/ no _th ink Ċ <|im_end|> Ċ "
        "<|im_start|> assistant Ċ <think> ĊĊ </think> ĊĊ The Ġcapital Ġof ĠPortugal Ġis ĠLisbon .",
    ),
]


@test_utils.only_on_linux
@pytest.mark.parametrize(
    "model,start_tokens,max_length,expected_tokens",
    _TRANSFORMERS_GENERATION_TESTS,
    ids=[args[0] for args in _TRANSFORMERS_GENERATION_TESTS],
)
def test_transformers_generation(
    clear_transformers_cache,
    tmp_dir,
    model,
    start_tokens,
    max_length,
    expected_tokens,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    results = generator.generate_batch([start_tokens.split()], max_length=max_length)
    output_tokens = " ".join(results[0].sequences[0])
    assert output_tokens == expected_tokens

    # Test empty inputs.
    assert generator.generate_batch([]) == []

    with pytest.raises(ValueError, match="start token"):
        generator.generate_batch([[]])


@test_utils.only_on_linux
def test_transformers_dtype(clear_transformers_cache, tmp_dir):
    converter = ctranslate2.converters.TransformersConverter("facebook/opt-350m")
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    model_b = os.path.getsize(os.path.join(output_dir, "model.bin"))
    model_mb = model_b / (1000**2)

    assert model_mb < 700


@test_utils.only_on_linux
def test_transformers_marianmt_vocabulary(clear_transformers_cache, tmp_dir):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-de"
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    vocabulary_path = os.path.join(output_dir, "shared_vocabulary.json")
    with open(vocabulary_path, encoding="utf-8") as vocabulary_file:
        vocab = json.load(vocabulary_file)

    assert vocab[-1] != "<pad>"


@test_utils.only_on_linux
@pytest.mark.parametrize("beam_size", [1, 2])
def test_transformers_marianmt_disable_unk(
    clear_transformers_cache, tmp_dir, beam_size
):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-roa"
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    tokens = ">>ind<< ▁The ▁Prime <unk> ▁is ▁coming ▁back ▁tomorrow . </s>".split()
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([tokens], beam_size=beam_size, disable_unk=True)
    assert "<unk>" not in output[0].hypotheses[0]


@test_utils.only_on_linux
@test_utils.on_available_devices
@pytest.mark.parametrize(
    "model_name",
    [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "distilbert-base-cased-distilled-squad",
        "typeform/distilbert-base-uncased-mnli",
    ],
)
def test_transformers_encoder(clear_transformers_cache, tmp_dir, device, model_name):
    import torch
    import transformers

    text = ["Hello world!", "Hello, my dog is cute"]

    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)

    inputs.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    mask = inputs.attention_mask.unsqueeze(-1).cpu().numpy()
    ref_last_hidden_state = outputs.last_hidden_state.cpu().numpy()
    ref_pooler_output = (
        outputs.pooler_output.cpu().numpy()
        if hasattr(outputs, "pooler_output")
        else None
    )

    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    encoder = ctranslate2.Encoder(output_dir, device=device)

    ids = [tokenizer(t).input_ids for t in text]
    outputs = encoder.forward_batch(ids)

    last_hidden_state = _to_numpy(outputs.last_hidden_state, device)
    assert last_hidden_state.shape == ref_last_hidden_state.shape

    last_hidden_state *= mask
    ref_last_hidden_state *= mask
    np.testing.assert_array_almost_equal(
        last_hidden_state, ref_last_hidden_state, decimal=5
    )

    if ref_pooler_output is not None:
        pooler_output = _to_numpy(outputs.pooler_output, device)
        assert pooler_output.shape == ref_pooler_output.shape
        np.testing.assert_array_almost_equal(
            pooler_output, ref_pooler_output, decimal=5
        )


def _to_numpy(storage, device):
    import torch

    return (
        np.array(storage)
        if device == "cpu"
        else torch.as_tensor(storage, device=device).cpu().numpy()
    )


@test_utils.only_on_linux_and_intel
def test_transformers_gptbigcode(clear_transformers_cache, tmp_dir):
    import transformers

    _check_generator_logits(
        tmp_dir,
        "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
        transformers.GPTBigCodeForCausalLM,
        transformers.AutoTokenizer,
        "hello",
    )


def _check_generator_logits(
    tmp_dir, model_name, hf_model_class, hf_tokenizer_class, input_text
):
    import torch

    model = hf_model_class.from_pretrained(model_name)
    tokenizer = hf_tokenizer_class.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    ref_logits = outputs.logits.numpy()

    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    logits = generator.forward_batch([tokens])
    logits = np.array(logits)

    assert logits.shape == ref_logits.shape
    np.testing.assert_array_almost_equal(logits, ref_logits)


class TestGeneration:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    def test_transformers_lm_scoring(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        tokens = "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof".split()
        output = generator.score_batch([tokens])[0]
        assert output.tokens == tokens[1:]
        assert len(output.log_probs) == len(output.tokens)

        # Test empty inputs.
        assert generator.score_batch([]) == []

        output = generator.score_batch([[], tokens])[0]
        assert not output.tokens
        assert not output.log_probs

        output = generator.score_batch([["<|endoftext|>"]])[0]
        assert not output.tokens
        assert not output.log_probs

    @test_utils.only_on_linux_and_intel
    @test_utils.on_available_devices
    @pytest.mark.parametrize("return_log_probs", [True, False])
    @pytest.mark.parametrize("tensor_input", [True, False])
    def test_transformers_lm_forward(
        self, tmp_dir, device, return_log_probs, tensor_input
    ):
        import torch
        import transformers

        model_name = "gpt2"

        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir, device=device)

        text = ["Hello world!"]

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            inputs.to(device)
            model.to(device)
            output = model(**inputs)
            ref_output = output.logits
            if return_log_probs:
                ref_output = torch.nn.functional.log_softmax(ref_output, dim=-1)
            ref_output = ref_output.cpu().numpy()

        kwargs = dict(return_log_probs=return_log_probs)

        if tensor_input:
            inputs = tokenizer(text, return_length=True, return_tensors="pt")
            inputs.to(device)
            ids = inputs.input_ids.to(torch.int32)
            lengths = inputs.length.to(torch.int32)

            if device == "cpu":
                ids = ids.numpy()
                lengths = lengths.numpy()

            ids = ctranslate2.StorageView.from_array(ids)
            lengths = ctranslate2.StorageView.from_array(lengths)

            with pytest.raises(ValueError, match="lengths"):
                generator.forward_batch(ids, **kwargs)
            output = generator.forward_batch(ids, lengths, **kwargs)

        else:
            ids = tokenizer(text).input_ids
            output = generator.forward_batch(ids, **kwargs)

        if device == "cpu":
            output = np.array(output)
        else:
            output = torch.as_tensor(output, device=device).cpu().numpy()

        assert output.shape == ref_output.shape
        np.testing.assert_allclose(output, ref_output, rtol=1e-2)

    @test_utils.only_on_linux
    def test_transformers_generator_on_iterables(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        start_tokens = ["<|endoftext|>"]
        tokens = "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof".split()
        output = next(generator.generate_iterable(iter([start_tokens]), max_length=10))
        assert output.sequences[0] == tokens

        output = next(generator.score_iterable(iter([tokens])))
        assert output.tokens == tokens[1:]
        assert len(output.log_probs) == len(output.tokens)

        # Test empty iterables.
        with pytest.raises(StopIteration):
            next(generator.score_iterable(iter([])))
        with pytest.raises(StopIteration):
            next(generator.generate_iterable(iter([])))

    @test_utils.only_on_linux
    def test_transformers_generator_suppress_sequences(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        output = generator.generate_batch(
            [["<|endoftext|>"]],
            max_length=10,
            suppress_sequences=[["Ġfirst", "Ġtime"]],
        )

        expected_tokens = "Ċ The Ġfirst Ġof Ġthe Ġthree Ġnew Ġseries Ġof Ġthe".split()
        assert output[0].sequences[0] == expected_tokens

    @test_utils.only_on_linux
    @pytest.mark.parametrize("beam_size", [1, 2])
    def test_transformers_generator_ignore_prompt(self, tmp_dir, beam_size):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        tokens = "Ċ The Ġfirst Ġtime ĠI".split()

        result_wo_prompt = generator.generate_batch(
            [tokens],
            beam_size=beam_size,
            max_length=max_length - len(tokens),
            return_scores=True,
            include_prompt_in_result=False,
        )[0]

        result_w_prompt = generator.generate_batch(
            [tokens],
            beam_size=beam_size,
            max_length=max_length - 1,
            return_scores=True,
        )[0]

        assert len(result_w_prompt.sequences[0]) == max_length
        assert tokens + result_wo_prompt.sequences[0] == result_w_prompt.sequences[0]

        cum_score_wo_prompt = result_wo_prompt.scores[0] * (
            len(result_wo_prompt.sequences[0])
        )
        cum_score_w_prompt = result_w_prompt.scores[0] * (
            len(result_w_prompt.sequences[0]) - 1
        )

        assert cum_score_wo_prompt == pytest.approx(cum_score_w_prompt, abs=1e-3)

    @test_utils.only_on_linux
    @pytest.mark.parametrize("beam_size", [1, 2])
    def test_transformers_generator_ignore_prompt_batch(self, tmp_dir, beam_size):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        new_tokens = 2
        prompt = [
            "Ċ The Ġfirst Ġtime ĠI".split(),
            "Ċ The Ġfirst".split(),
        ]

        results = generator.generate_batch(
            prompt,
            beam_size=beam_size,
            min_length=new_tokens,
            max_length=new_tokens,
            include_prompt_in_result=False,
        )

        for tokens, result in zip(prompt, results):
            assert len(result.sequences[0]) == new_tokens

    @test_utils.only_on_linux
    def test_transformers_generator_static_prompt(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        prompt = "Ċ The Ġfirst Ġtime ĠI".split()

        expected_result = generator.generate_batch(
            [prompt],
            max_length=max_length,
            include_prompt_in_result=False,
        )[0]

        result = generator.generate_batch(
            [[expected_result.sequences[0][0]]],
            max_length=max_length - 1,
            static_prompt=prompt,
        )[0]

        assert result.sequences[0] == expected_result.sequences[0]

        result = generator.generate_batch(
            [expected_result.sequences[0][:2]],
            max_length=max_length - 2,
            static_prompt=prompt,
            include_prompt_in_result=False,
        )[0]

        assert (
            expected_result.sequences[0][:2] + result.sequences[0]
            == expected_result.sequences[0]
        )

        batch_results = generator.generate_batch(
            [[expected_result.sequences[0][0]], [expected_result.sequences[0][0]]],
            max_length=max_length - 1,
            static_prompt=prompt,
        )

        assert batch_results[0].sequences[0] == expected_result.sequences[0]
        assert batch_results[1].sequences[0] == expected_result.sequences[0]

    @test_utils.only_on_linux
    @pytest.mark.parametrize("return_log_prob", [True, False])
    def test_transformers_generator_token_streaming(self, tmp_dir, return_log_prob):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        prompt = "Ċ The Ġfirst Ġtime ĠI".split()

        expected_result = generator.generate_batch(
            [prompt],
            max_length=max_length,
            return_scores=True,
            include_prompt_in_result=False,
        )[0]

        step_results = generator.generate_tokens(
            prompt, max_length=max_length, return_log_prob=return_log_prob
        )

        assert inspect.isgenerator(step_results)

        tokens = []
        ids = []
        cum_log_probs = 0

        for step_result in step_results:
            assert isinstance(step_result, ctranslate2.GenerationStepResult)

            tokens.append(step_result.token)
            ids.append(step_result.token_id)

            if return_log_prob:
                cum_log_probs += step_result.log_prob
            else:
                assert step_result.log_prob is None

        assert tokens == expected_result.sequences[0]
        assert ids == expected_result.sequences_ids[0]

        if return_log_prob:
            assert cum_log_probs / len(ids) == pytest.approx(
                expected_result.scores[0], abs=1e-5
            )

    @test_utils.only_on_linux
    def test_transformers_generator_token_streaming_early_stop(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        prompt = "Ċ The Ġfirst Ġtime ĠI".split()
        results = generator.generate_tokens(prompt)
        for result in results:
            break

        results.close()


class TestWhisper:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "model_name,prompts,expected_transcriptions,expected_no_speech_probs",
        [
            (
                "openai/whisper-tiny",
                [
                    [
                        "<|startoftranscript|>",
                        "<|en|>",
                        "<|transcribe|>",
                        "<|notimestamps|>",
                    ],
                    [
                        "<|startoftranscript|>",
                        "<|en|>",
                        "<|transcribe|>",
                        "<|notimestamps|>",
                        "ĠAnd",
                        "Ġthus",
                        "Ġmy",
                    ],
                ],
                [
                    " Mr. Quilter is the apostle of the middle classes and we are glad"
                    " to welcome his gospel.",
                    " And thus my fellow Americans ask not what your country can do for you,"
                    " ask what you can do for your country.",
                ],
                [
                    pytest.approx(0.0022832120303064585, abs=1e-4),
                    pytest.approx(0.06885894387960434, abs=1e-3),
                ],
            ),
            (
                "openai/whisper-tiny",
                [
                    ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"],
                    ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"],
                ],
                [
                    " Mr. Quilter is the apostle of the middle classes and we are glad"
                    " to welcome his gospel.",
                    " And so, my fellow Americans, ask not what your country can do for you,"
                    " ask what you can do for your country.",
                ],
                [
                    pytest.approx(0.0022832120303064585, abs=1e-4),
                    pytest.approx(0.06885894387960434, abs=1e-3),
                ],
            ),
            (
                "openai/whisper-tiny.en",
                [["<|startoftranscript|>"], ["<|startoftranscript|>"]],
                [
                    " Mr. Quilter is the apostle of the middle classes, and we are glad"
                    " to welcome his gospel.",
                    " And so, my fellow Americans ask not what your country can do for you"
                    " ask what you can do for your country.",
                ],
                [
                    pytest.approx(0.02644546702504158, abs=1e-4),
                    pytest.approx(0.062380101531744, abs=1e-3),
                ],
            ),
        ],
    )
    def test_transformers_whisper(
        self,
        tmp_dir,
        device,
        model_name,
        prompts,
        expected_transcriptions,
        expected_no_speech_probs,
    ):
        import transformers

        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        audio_paths = [
            os.path.join(test_utils.get_data_dir(), "audio", "mr_quilter.npy"),
            os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy"),
        ]
        audio = list(map(np.load, audio_paths))

        processor = transformers.WhisperProcessor.from_pretrained(model_name)

        def _get_features(audio):
            # Pad after computing the log-Mel spectrogram to match the openai/whisper behavior.
            inputs = processor(audio, padding=False, sampling_rate=16000)
            features = inputs.input_features[0]
            features = np.pad(features, [(0, 0), (0, 3000 - features.shape[-1])])
            return features

        features = np.stack(list(map(_get_features, audio)))
        features = ctranslate2.StorageView.from_array(features)

        model = ctranslate2.models.Whisper(output_dir, device=device)

        assert model.is_multilingual == (not model_name.endswith(".en"))

        if model.is_multilingual:
            for result in model.detect_language(features):
                best_lang, best_prob = result[0]
                assert best_lang == "<|en|>"
                assert best_prob > 0.9
        else:
            with pytest.raises(RuntimeError, match="multilingual"):
                model.detect_language(features)

        results = model.generate(
            features,
            prompts,
            beam_size=2,
            num_hypotheses=2,
            return_no_speech_prob=True,
        )

        timestamp_begin = (
            processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
        )

        for prompt, result, expected_transcription, expected_no_speech_prob in zip(
            prompts, results, expected_transcriptions, expected_no_speech_probs
        ):
            assert len(result.sequences_ids) == 2
            assert result.no_speech_prob == expected_no_speech_prob

            for tokens in result.sequences_ids:
                if "<|notimestamps|>" in prompt:
                    assert all(token < timestamp_begin for token in tokens)
                else:
                    assert tokens[0] >= timestamp_begin
                    assert tokens[-1] >= timestamp_begin
                    assert tokens[-1] > tokens[0]

            token_ids = list(
                filter(lambda token: token < timestamp_begin, result.sequences_ids[0])
            )

            transcription = processor.decode(token_ids)
            assert transcription == expected_transcription

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "test_names", [["jfk"], ["jfk", "jfk"], ["mr_quilter", "jfk"]]
    )
    def test_transformers_whisper_align(self, tmp_dir, device, test_names):
        import transformers

        test_cases = []
        audio = []

        test_dir = os.path.join(test_utils.get_data_dir(), "audio")

        for name in test_names:
            audio_path = os.path.join(test_dir, "%s.npy" % name)
            audio.append(np.load(audio_path))

            test_case_path = os.path.join(test_dir, "%s_alignments.json" % name)
            with open(test_case_path) as test_case_file:
                test_cases.append(json.load(test_case_file))

        model_name = "openai/whisper-tiny.en"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(audio, return_tensors="np", sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        model = ctranslate2.models.Whisper(output_dir, device=device)

        results = model.align(
            features,
            [50257],
            [test_case["text_tokens"] for test_case in test_cases],
            [test_case["num_frames"] for test_case in test_cases],
        )

        for result, test_case in zip(results, test_cases):
            assert np.sum(result.text_token_probs) == pytest.approx(
                test_case["expected_text_token_probs_sum"], abs=1e-3
            )

            assert result.alignments == [
                tuple(pair) for pair in test_case["expected_alignments"]
            ]

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    def test_transformers_whisper_encode(self, tmp_dir, device):
        import transformers

        model_name = "openai/whisper-tiny.en"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        audio_path = os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy")
        audio = np.load(audio_path)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(audio, sampling_rate=16000)
        features = inputs.input_features[0]
        features = np.expand_dims(features, 0)
        features = ctranslate2.StorageView.from_array(features)

        model = ctranslate2.models.Whisper(output_dir, device=device)
        encoded = model.encode(features)
        prompts = [["<|startoftranscript|>", "<|notimestamps|>"]]
        result = model.generate(encoded, prompts)[0]

        transcription = processor.decode(result.sequences_ids[0])

        assert transcription == (
            " And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country."
        )

    @test_utils.only_on_linux
    def test_transformers_whisper_partial_audio_context(self, tmp_dir):
        import transformers

        model_name = "openai/whisper-tiny"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        audio_path = os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy")
        audio = np.load(audio_path)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(
            audio, padding=False, return_tensors="np", sampling_rate=16000
        )
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        model = ctranslate2.models.Whisper(output_dir)
        encoder_output = model.encode(features)

        assert encoder_output.shape == [1, features.shape[2] // 2, 384]

    @test_utils.only_on_linux
    def test_transformers_whisper_include_tokenizer_json(self, tmp_dir):
        model_name = "openai/whisper-tiny"
        converter = ctranslate2.converters.TransformersConverter(
            model_name, copy_files=["tokenizer.json"]
        )
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        assert os.path.isfile(os.path.join(output_dir, "tokenizer.json"))


class TestWav2Vec2:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "model_name,expected_transcriptions",
        [
            (
                "facebook/wav2vec2-large-robust-ft-swbd-300h",
                [
                    "MISTER QUILTER IS THE APOSSEL OF THE MIDDLE CLASSES AND"
                    " WE ARE GLAD TO WELCOME HIS GOSPEL",
                    "MISTER QUILTER IS THE APOSSTEL OF THE MIDDLE CLASSES AND"
                    " WE ARE GLAD TO WELCOME HIS GOSPEL",
                ],
            ),
        ],
    )
    def test_transformers_wav2vec2(
        self,
        tmp_dir,
        device,
        model_name,
        expected_transcriptions,
    ):
        import torch
        import transformers

        converter = ctranslate2.converters.TransformersConverter(
            model_name, load_as_float16="int8"
        )
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        w2v2_processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)
        w2v2_processor.save_pretrained(output_dir + "/wav2vec2_processor")
        processor = transformers.AutoProcessor.from_pretrained(
            output_dir + "/wav2vec2_processor"
        )

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        # cpu_threads = int(os.environ.get("OMP_NUM_THREADS", 0))
        model = ctranslate2.models.Wav2Vec2(
            output_dir,
            device=device,
            device_index=[0],
            compute_type="int8",
            intra_threads=1,
            inter_threads=1,
        )

        speech_array = np.load(
            os.path.join(test_utils.get_data_dir(), "audio", "mr_quilter.npy")
        )
        input_values = processor(
            speech_array,
            padding=True,
            return_tensors="pt",
            sampling_rate=16000,
        ).input_values

        hidden_states = np.ascontiguousarray(input_values.unsqueeze(0))
        hidden_states = ctranslate2.StorageView.from_array(hidden_states)
        to_cpu = model.device == "cuda" and len(model.device_index) > 1
        output = model.encode(hidden_states, to_cpu=to_cpu)
        if model.device == "cuda":
            logits = torch.as_tensor(output, device=model.device)[0]
        else:
            logits = torch.as_tensor(
                np.array(output), dtype=torch.float32, device=model.device
            )[0]

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids, output_word_offsets=True)
        transcription = transcription[0].replace(processor.tokenizer.unk_token, "")

        assert transcription in expected_transcriptions


class TestWav2Vec2Bert:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "model_name,expected_transcription",
        [
            (
                "hf-audio/wav2vec2-bert-CV16-en",
                [
                    "mr quilter is the apostle of the middle classes and"
                    " we are glad to welcome his gospel"
                ],
            ),
        ],
    )
    def test_transformers_wav2vec2bert(
        self,
        tmp_dir,
        device,
        model_name,
        expected_transcription,
    ):
        import torch
        import transformers

        converter = ctranslate2.converters.TransformersConverter(
            model_name, load_as_float16="int8"
        )
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        w2v2_processor = transformers.Wav2Vec2BertProcessor.from_pretrained(model_name)
        w2v2_processor.save_pretrained(output_dir + "/wav2vec2_processor")
        processor = transformers.AutoProcessor.from_pretrained(
            output_dir + "/wav2vec2_processor"
        )

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        # cpu_threads = int(os.environ.get("OMP_NUM_THREADS", 0))
        model = ctranslate2.models.Wav2Vec2Bert(
            output_dir,
            device=device,
            device_index=[0],
            compute_type="int8",
            intra_threads=1,
            inter_threads=1,
        )

        speech_array = np.load(
            os.path.join(test_utils.get_data_dir(), "audio", "mr_quilter.npy")
        )
        input_values = processor(
            [speech_array],
            padding=True,
            return_tensors="pt",
            sampling_rate=16000,
        ).input_features

        hidden_states = np.ascontiguousarray(input_values)
        hidden_states = ctranslate2.StorageView.from_array(hidden_states)
        to_cpu = model.device == "cuda" and len(model.device_index) > 1
        output = model.encode(hidden_states, to_cpu=to_cpu)
        if model.device == "cuda":
            logits = torch.as_tensor(output, device=model.device)[0]
        else:
            logits = torch.as_tensor(
                np.array(output), dtype=torch.float32, device=model.device
            )[0]

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids, output_word_offsets=True)
        transcription = transcription[0].replace(processor.tokenizer.unk_token, "")

        assert transcription == expected_transcription[0]


def _build_tiny_t5_family_model(cls, config_cls, **kwargs):
    import torch

    config = config_cls(
        vocab_size=64,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=3,
        num_decoder_layers=3,
        num_heads=4,
        feed_forward_proj="gated-gelu",
        **kwargs,
    )
    model = cls(config).eval()

    # Make each table trivially identifiable so a mix-up is visible.
    with torch.no_grad():
        for stack in (model.encoder, model.decoder):
            for i, block in enumerate(stack.block):
                attention = block.layer[0].SelfAttention
                if attention.has_relative_attention_bias:
                    attention.relative_attention_bias.weight.fill_(i + 1)

    return model


def _load_stacks(loader, model):
    from ctranslate2.specs import common_spec, transformer_spec

    spec = transformer_spec.TransformerSpec.from_config(
        (model.config.num_layers, model.config.num_decoder_layers),
        model.config.num_heads,
        pre_norm=True,
        activation=common_spec.Activation.GELUTanh,
        ffn_glu=True,
        relative_attention_bias=True,
        rms_norm=True,
    )
    loader.set_stack(spec.encoder, model.encoder)
    loader.set_stack(spec.decoder, model.decoder, is_decoder=True)
    return spec


def _biases(stack_spec):
    return [layer.self_attention.relative_attention_bias for layer in stack_spec.layer]


@test_utils.only_on_linux
def test_umt5_loader_keeps_one_relative_attention_bias_per_layer():
    import transformers

    from ctranslate2.converters.transformers import UMT5Loader

    model = _build_tiny_t5_family_model(
        transformers.UMT5ForConditionalGeneration, transformers.UMT5Config
    )
    spec = _load_stacks(UMT5Loader(), model)

    for stack in (spec.encoder, spec.decoder):
        biases = _biases(stack)
        assert len(biases) == 3
        assert all(bias is not None for bias in biases)
        # UMT5 has a distinct table per layer: the values must not be copied from layer 0.
        assert [float(bias.detach().flatten()[0]) for bias in biases] == [1.0, 2.0, 3.0]


# --- Runtime tests -----------------------------------------------------------------
#
# The loader tests above prove the *spec* carries one bias table per layer. They would
# still pass if the C++ runtime read layer 0's table for every layer, so the tests below
# exercise the runtime itself.
#
# Two things matter when building the synthetic model:
#
# 1. The tables must be non-constant. A table that is uniform over buckets adds the same
#    value to every attention logit, which cancels in the softmax -- so a runtime that
#    wrongly shared layer 0 would still produce identical output and the test would pass.
#
# 2. The rest of the weights must come from Transformers' own initializer. T5 omits the
#    1/sqrt(head_dim) attention scaling and compensates in the `q` initialiser, so a
#    hand-rolled 1/sqrt(fan_in) init makes the attention logits saturate and CTranslate2
#    and Transformers then disagree by ~1e-1 for reasons unrelated to the bias tables.
#    `initializer_factor` is kept small to stay well-conditioned through the gated FFN.


def _tiny_umt5_vocabulary(vocab_size):
    return ["<pad>", "</s>", "<unk>"] + ["tok%d" % i for i in range(3, vocab_size)]


def _build_umt5_with_biases(
    encoder_distinct,
    decoder_distinct,
    vocab_size=64,
    num_layers=3,
    initializer_factor=0.05,
    bias_scale=5.0,
    amplify_decoder_values=1.0,
):
    """Builds a tiny UMT5 whose per-layer bias tables are distinct and non-constant.

    When a stack is not `distinct`, every layer gets layer 0's table, which is what a
    T5-style shared-bias model looks like.
    """
    import torch
    import transformers

    torch.manual_seed(3)
    config = transformers.UMT5Config(
        vocab_size=vocab_size,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=4,
        feed_forward_proj="gated-gelu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        initializer_factor=initializer_factor,
    )

    # Use eager attention for the reference. Under SDPA, Transformers releases
    # v5.15.0 to v5.17.0 leave the UMT5 decoder non-causal
    # (https://github.com/huggingface/transformers/issues/49134, fixed by #49135),
    # which would show up here as a mismatch that has nothing to do with the bias
    # tables. Eager materialises the mask and is correct on every release.
    config._attn_implementation = "eager"

    model = transformers.UMT5ForConditionalGeneration(config).eval()

    # At this weight scale the decoder's value vectors are nearly identical across
    # positions, so a change in attention weights barely moves the output. Amplifying
    # only the self-attention value path makes attention differences observable without
    # destabilising the rest of the stack.
    if amplify_decoder_values != 1.0:
        with torch.no_grad():
            for block in model.decoder.block:
                attention = block.layer[0].SelfAttention
                attention.v.weight.mul_(amplify_decoder_values)
                attention.o.weight.mul_(amplify_decoder_values)

    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for stack, distinct in (
            (model.encoder, encoder_distinct),
            (model.decoder, decoder_distinct),
        ):
            attention = stack.block[0].layer[0].SelfAttention
            shape = attention.relative_attention_bias.weight.shape
            tables = [
                (torch.rand(shape, generator=generator) * 2 - 1) * bias_scale
                for _ in stack.block
            ]
            for i, block in enumerate(stack.block):
                block.layer[0].SelfAttention.relative_attention_bias.weight.copy_(
                    tables[i] if distinct else tables[0]
                )

    return model


def _save_umt5(model, output_dir):
    from ctranslate2.converters.transformers import UMT5Loader

    loader = UMT5Loader()
    spec = loader.get_model_spec(model)
    tokens = _tiny_umt5_vocabulary(model.config.vocab_size)
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<pad>"
    spec.config.eos_token = "</s>"
    spec.config.unk_token = "<unk>"
    spec.config.decoder_start_token = "<pad>"

    # Mirror Converter.convert().
    spec.validate()
    spec.optimize(quantization="float32")
    os.makedirs(output_dir, exist_ok=True)
    spec.save(output_dir)
    return tokens


def _save_umt5_encoder(model, output_dir):
    from ctranslate2.converters.transformers import _SUPPORTED_ACTIVATIONS, UMT5Loader
    from ctranslate2.specs import transformer_spec

    config = model.config
    encoder = transformer_spec.TransformerEncoderSpec(
        config.num_layers,
        config.num_heads,
        pre_norm=True,
        activation=_SUPPORTED_ACTIVATIONS[config.dense_act_fn],
        ffn_glu=config.is_gated_act,
        relative_attention_bias=True,
        rms_norm=True,
    )
    spec = transformer_spec.TransformerEncoderModelSpec(encoder)
    UMT5Loader().set_stack(spec.encoder, model.encoder)

    tokens = _tiny_umt5_vocabulary(config.vocab_size)
    spec.register_vocabulary(tokens)
    spec.config.bos_token = "<pad>"
    spec.config.eos_token = "</s>"
    spec.config.unk_token = "<unk>"

    spec.validate()
    spec.optimize(quantization=None)
    os.makedirs(output_dir, exist_ok=True)
    spec.save(output_dir)
    return tokens


_UMT5_SOURCE = [5, 9, 14, 23, 7, 33, 41, 18, 52, 11, 27, 6, 44, 30, 21, 38]
_UMT5_TARGET = [11, 4, 19, 30, 27, 48, 13, 36, 8, 25, 55, 17, 42, 9, 31, 20]


def _umt5_log_probs(model_dir, tokens, source, target):
    translator = ctranslate2.Translator(
        model_dir,
        device="cpu",
        compute_type="float32",
        inter_threads=1,
        intra_threads=1,
    )
    result = translator.score_batch(
        [[tokens[i] for i in source]], [[tokens[i] for i in target]]
    )[0]
    return np.array(result.log_probs)


def _umt5_reference_log_probs(model, source, target):
    import torch

    # score_batch appends </s> to the target itself.
    target = target + [model.config.eos_token_id]
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([source + [model.config.eos_token_id]]),
            decoder_input_ids=torch.tensor(
                [[model.config.decoder_start_token_id] + target[:-1]]
            ),
        )
        log_probs = torch.log_softmax(outputs.logits[0].float(), -1)
        return log_probs[torch.arange(len(target)), torch.tensor(target)].numpy()


@test_utils.only_on_linux
def test_umt5_runtime_uses_per_layer_bias_in_encoder(tmp_dir):
    """The encoder runtime must read each layer's own bias table.

    Checked on the encoder output rather than on token scores: on a randomly
    initialised model the encoder's contribution to the final scores is numerically
    negligible (measured as exactly 0), so a score-level assertion here would be
    vacuous. The encoder output separates the two cases by ~6 orders of magnitude.
    """
    import torch

    # A larger scale than the decoder test can afford: the encoder alone stays
    # well-conditioned here, and it makes the bias effect unmistakable.
    kwargs = dict(initializer_factor=0.3, bias_scale=5.0)
    distinct = _build_umt5_with_biases(True, True, **kwargs)
    shared = _build_umt5_with_biases(False, False, **kwargs)

    tokens = _save_umt5_encoder(distinct, str(tmp_dir.join("distinct")))
    _save_umt5_encoder(shared, str(tmp_dir.join("shared")))

    def encode(model_dir):
        encoder = ctranslate2.Encoder(
            model_dir,
            device="cpu",
            compute_type="float32",
            inter_threads=1,
            intra_threads=1,
        )
        output = encoder.forward_batch([[tokens[i] for i in _UMT5_SOURCE]])
        return np.array(output.last_hidden_state)[0]

    got_distinct = encode(str(tmp_dir.join("distinct")))
    got_shared = encode(str(tmp_dir.join("shared")))

    with torch.no_grad():
        expected = (
            distinct.encoder(input_ids=torch.tensor([_UMT5_SOURCE]))
            .last_hidden_state[0]
            .numpy()
        )

    # The runtime reproduces Transformers with one table per layer.
    assert np.abs(got_distinct - expected).max() < 1e-5

    # And the per-layer tables actually reach the kernels: a runtime that reused
    # layer 0's table would return the shared-bias output instead.
    assert np.abs(got_distinct - got_shared).max() > 1e-2


@test_utils.only_on_linux
def test_umt5_runtime_uses_per_layer_bias_in_decoder(tmp_dir):
    """The decoder runtime must read each layer's own bias table."""
    kwargs = dict(amplify_decoder_values=30.0)
    distinct = _build_umt5_with_biases(False, True, **kwargs)
    shared = _build_umt5_with_biases(False, False, **kwargs)

    tokens = _save_umt5(distinct, str(tmp_dir.join("distinct")))
    _save_umt5(shared, str(tmp_dir.join("shared")))

    got_distinct = _umt5_log_probs(
        str(tmp_dir.join("distinct")), tokens, _UMT5_SOURCE, _UMT5_TARGET
    )
    got_shared = _umt5_log_probs(
        str(tmp_dir.join("shared")), tokens, _UMT5_SOURCE, _UMT5_TARGET
    )
    expected = _umt5_reference_log_probs(distinct, _UMT5_SOURCE, _UMT5_TARGET)

    assert np.abs(got_distinct - expected).max() < 1e-4
    assert np.abs(got_distinct - got_shared).max() > 1e-4


@test_utils.only_on_linux
def test_umt5_runtime_matches_transformers_with_distinct_biases_in_both_stacks(tmp_dir):
    """Distinct tables in both stacks at once, so a scope mix-up between the
    encoder and the decoder would show up here."""
    model = _build_umt5_with_biases(True, True, amplify_decoder_values=30.0)
    tokens = _save_umt5(model, str(tmp_dir.join("model")))

    got = _umt5_log_probs(
        str(tmp_dir.join("model")), tokens, _UMT5_SOURCE, _UMT5_TARGET
    )
    expected = _umt5_reference_log_probs(model, _UMT5_SOURCE, _UMT5_TARGET)

    assert np.abs(got - expected).max() < 1e-4


@test_utils.only_on_linux
def test_t5_loader_shares_the_first_relative_attention_bias():
    import transformers

    from ctranslate2.converters.transformers import T5Loader

    model = _build_tiny_t5_family_model(
        transformers.T5ForConditionalGeneration, transformers.T5Config
    )
    spec = _load_stacks(T5Loader(), model)

    for stack in (spec.encoder, spec.decoder):
        biases = _biases(stack)
        assert len(biases) == 3
        # T5 only defines the table in the first layer and shares it with the others.
        assert all(bias is biases[0] for bias in biases)
