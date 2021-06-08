import sys
import docker
import sacrebleu

from benchmark import benchmark_image


test_set = "wmt14"
langpair = "en-de"
num_cpus = 4
gpu = len(sys.argv) > 1 and sys.argv[1].lower() == "gpu"
num_samples = 5 if gpu else 3


print("Building the images...")
client = docker.from_env()
ctranslate2, _ = client.images.build(
    path="pretrained_transformer_base/ctranslate2", tag="opennmt/ctranslate2-benchmark"
)
opennmt_py, _ = client.images.build(
    path="pretrained_transformer_base/opennmt_py", tag="opennmt/opennmt-py-benchmark"
)
opennmt_tf, _ = client.images.build(
    path="pretrained_transformer_base/opennmt_tf", tag="opennmt/opennmt-tf-benchmark"
)

print("Downloading the test files...")
source_file = sacrebleu.get_source_file(test_set, langpair=langpair)
target_file = sacrebleu.get_reference_files(test_set, langpair=langpair)[0]

print("")

if gpu:
    print("| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |")
else:
    print("| | Tokens per second | Max. memory | BLEU |")


def run(name, image, env=None):
    result = benchmark_image(
        image.tags[0],
        source_file,
        target_file,
        num_samples=num_samples,
        environment=env,
        num_cpus=num_cpus,
        use_gpu=gpu,
    )
    tokens_per_sec = result.num_tokens / result.translation_time

    if gpu:
        print(
            "| %s | %.1f | %d | %d | %.2f |"
            % (
                name,
                tokens_per_sec,
                int(result.max_gpu_mem),
                int(result.max_cpu_mem),
                result.bleu_score,
            )
        )
    else:
        print(
            "| %s | %.1f | %d | %.2f |"
            % (
                name,
                tokens_per_sec,
                int(result.max_cpu_mem),
                result.bleu_score,
            )
        )


run("OpenNMT-tf", opennmt_tf)

run("OpenNMT-py", opennmt_py)
if not gpu:
    run("- int8", opennmt_py, env={"INT8": "1"})

run("CTranslate2", ctranslate2)
if gpu:
    run("- int8", ctranslate2, env={"COMPUTE_TYPE": "int8"})
    run("- float16", ctranslate2, env={"COMPUTE_TYPE": "float16"})
else:
    run("- int16", ctranslate2, env={"COMPUTE_TYPE": "int16"})
    run("- int8", ctranslate2, env={"COMPUTE_TYPE": "int8"})
    run("- int8 + vmap", ctranslate2, env={"COMPUTE_TYPE": "int8", "USE_VMAP": "1"})
