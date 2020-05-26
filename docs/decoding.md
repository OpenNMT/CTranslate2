# Decoding

This page describes CTranslate2 decoding features. The Python API is used for demonstration but all features are also supported with the C++ API.

The examples use the following symbols that are left unspecified:

* `translator`: a `ctranslate2.Translator` instance
* `tokenize`: a function taking a string and returning a list of string
* `detokenize`: a function taking a list of string and returning a string

This `input` sentence will be used as an example:

> This project is geared towards efficient serving of standard translation models but is also a place for experimentation around model compression and inference acceleration.

## Greedy search

Greedy search is the most basic and fastest decoding strategy. It simply takes the token that has the highest probability at each timestep.

```python
results = translator.translate_batch([tokenize(input)], beam_size=1)
print(detokenize(results[0][0]["tokens"]))
```

> Dieses Projekt ist auf die effiziente Bedienung von Standard-Übersetzungsmodellen ausgerichtet, aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.

If you do not need `results` to include meaningful prediction scores, you can set the flag `return_scores=False`. In this case, we can skip the last softmax layer and increase performance of greedy search.

```python
translator.translate_batch(batch, beam_size=1, return_scores=False)
```

## Beam search

Beam search is a common decoding strategy for sequence models. The algorithm keeps N hypotheses at all times. This negatively impacts decoding speed and memory but allow finding a better final hypothesis.

```python
results = translator.translate_batch([tokenize(input)], beam_size=4)
print(detokenize(results[0][0]["tokens"]))
```

> Dieses Projekt ist auf die effiziente Bedienung von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.

By default, only the final best hypothesis is returned but more hypotheses can be returned by setting the `num_hypotheses` argument.

## Autocompletion

The `target_prefix` argument can be used to force the start of the translation. Let's say we want to replace the first occurence of `die` by `das` in the translation:

```python
results = translator.translate_batch(
    [tokenize(input)], target_prefix=[tokenize("Dieses Projekt ist auf das")])
print(detokenize(results[0][0]["tokens"]))
```

The prefix effectively changes the target context and the rest of the translation:

> Dieses Projekt ist auf das effiziente **Servieren** von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.

## Alternatives at a position

Combining `target_prefix` with the `return_alternatives` flag returns alternative words just after the prefix:

```python
results = translator.translate_batch(
    [tokenize(input)],
    target_prefix=[tokenize("Dieses Projekt ist auf die")],
    num_hypotheses=5,
    return_alternatives=True)
for hypothesis in results[0]:
    print(detokenize(hypothesis["tokens"]))
```

> Dieses Projekt ist auf die **effiziente** Bedienung von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.
>
> Dieses Projekt ist auf die **effektive** Bedienung von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.
>
> Dieses Projekt ist auf die **effizientere** Bedienung von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.
>
> Dieses Projekt ist auf die **effizienten** Dienste von Standard-Übersetzungsmodellen ausgerichtet, aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.
>
> Dieses Projekt ist auf die **Effizienz** des Servierens von Standard-Übersetzungsmodellen ausgerichtet, ist aber auch ein Ort für Experimente rund um Modellkompression und Inferenzbeschleunigung.

## Random sampling

This decoding mode randomly samples tokens from the model output distribution. This strategy is frequently used in back-translation techniques ([Edunov et al. 2018](https://www.aclweb.org/anthology/D18-1045/)). The example below restricts the sampling to the best 10 candidates at each timestep:

```python
all_results = [
    translator.translate_batch([tokenize(input)], beam_size=1, sampling_topk=10),
    translator.translate_batch([tokenize(input)], beam_size=1, sampling_topk=10),
    translator.translate_batch([tokenize(input)], beam_size=1, sampling_topk=10)]
for results in all_results:
    print(detokenize(results[0][0]["tokens"]))
```

> Dieses Programm ist auf eine effiziente Bedienung von Standard-Übersetzungsmodellen ausgerichtet und ermöglicht gleichzeitig einen Einsatzort für Experimente rund um die Modellkompression oder das Beschleunigen der Schlussfolgerung.
>
> Es dient dazu, die standardisierten Übersetzungsmodelle effizient zu bedienen, aber auch zur Erprobung um die Formkomprimierung und die Folgebeschleunigung.
>
> Das Projekt richtet sich zwar auf den effizienten Service von Standard-Übersetzungen-Modellen, ist aber auch ein Ort für Experimente rund um Modellkomprimierung und ineffektive Beschleunigung.

You can increase the randomness of the generation by increasing the value of the argument `sampling_temperature`.
