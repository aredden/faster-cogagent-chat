import time
import warnings
from argparse import ArgumentParser
from csv import DictReader, DictWriter
from pathlib import Path

from loguru import logger
from PIL import Image
from sortedcontainers import SortedSet

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.attention.bias"
)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from transformers import TextStreamer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
START = time.perf_counter()


class COGVLMAgentWorker:
    def __init__(
        self, model_path="THUDM/cogagent-chat-hf", tokenizer_path="lmsys/vicuna-7b-v1.5"
    ) -> None:
        import torch

        from cogagent_model.modeling_cogagent import CogAgentForCausalLM
        from transformers import BitsAndBytesConfig, LlamaTokenizer
        from patch_cogagent_model import hijack_cogagent_linears

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.torch_type = torch.float16

        print(
            "========Use torch type as:{} with device:{}========\n\n".format(
                self.torch_type, self.DEVICE
            )
        )

        self.model: CogAgentForCausalLM = (
            CogAgentForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    bnb_4bit_compute_dtype=self.torch_type,
                    load_in_4bit=True,
                    llm_int8_skip_modules=["cross_attn.query"],
                ),
            )
            .eval()
            .requires_grad_(False)
        )
        logger.debug("Hijacking linear layers")
        hijack_cogagent_linears(self.model.model.layers, self.torch_type)
        logger.debug("Done hijacking linear layers")
        END = time.perf_counter()
        print(f"Time to load model: {END - START}")
        print(self.model.model.layers)

    def get_basic_inputs(self, context_prompt, image_path):
        history = []
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=context_prompt, history=history, images=[image]
            )
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.DEVICE),
                "token_type_ids": input_by_model["token_type_ids"]
                .unsqueeze(0)
                .to(self.DEVICE),
                "attention_mask": input_by_model["attention_mask"]
                .unsqueeze(0)
                .to(self.DEVICE),
                "images": [
                    [input_by_model["images"][0].to(self.DEVICE).to(self.torch_type)]
                ],
            }
            if "cross_images" in input_by_model and input_by_model["cross_images"]:
                inputs["cross_images"] = [
                    [
                        input_by_model["cross_images"][0]
                        .to(self.DEVICE)
                        .to(self.torch_type)
                    ]
                ]
        return inputs

    def generate(self, context_prompt, image_path, stream_output=False):
        history = []
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=context_prompt, history=history, images=[image]
            )
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.DEVICE),
                "token_type_ids": input_by_model["token_type_ids"]
                .unsqueeze(0)
                .to(self.DEVICE),
                "attention_mask": input_by_model["attention_mask"]
                .unsqueeze(0)
                .to(self.DEVICE),
                "images": [
                    [input_by_model["images"][0].to(self.DEVICE).to(self.torch_type)]
                ],
            }
            if "cross_images" in input_by_model and input_by_model["cross_images"]:
                inputs["cross_images"] = [
                    [
                        input_by_model["cross_images"][0]
                        .to(self.DEVICE)
                        .to(self.torch_type)
                    ]
                ]

            # add any transformers params here.
            gen_kwargs = {
                "max_length": 2048,
                "do_sample": False,
            }
            if stream_output:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                gen_kwargs["streamer"] = streamer
            with torch.no_grad():
                st = time.perf_counter()
                outputs = self.model.generate(**inputs, **gen_kwargs)
                nd = time.perf_counter()

                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                print(f"Total tps: {outputs.shape[1]/(nd - st)}")
                response = self.tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]
        return image_path, response


def resolve_actual_paths(image_paths, reference_path):
    reader = DictReader(
        open(reference_path, "r").readlines(), fieldnames=["path", "prompt"]
    )
    already_read = [r["path"] for r in reader]
    already_read = SortedSet(already_read)
    total_discarded = 0
    print(len(already_read), already_read[0])
    for image in list(image_paths):
        if image in already_read:
            total_discarded += 1
            image_paths.remove(image)

    assert total_discarded > 0, "Discarded 0 images, even though path already existed!!"
    logger.debug(f"Skipping {total_discarded} images from caption list!")
    return image_paths, total_discarded


def resolve_actual_paths(image_paths, reference_path):
    reader = DictReader(
        open(reference_path, "r").readlines(), fieldnames=["path", "prompt"]
    )
    already_read = [r["path"] for r in reader]
    already_read = SortedSet(already_read)
    total_discarded = 0
    for image in list(image_paths):
        if image in already_read:
            total_discarded += 1
            image_paths.remove(image)

    logger.debug(f"Skipping {total_discarded} images from caption list!")
    return image_paths, total_discarded


def to_paths(image_list_textfile):
    with open(image_list_textfile, "r") as f:
        image_paths = f.readlines()
    for idx, i in enumerate(image_paths):
        try:
            num = int(i.rsplit(" ")[1])
            if type(num) == int:
                image_paths[idx] = i.rsplit(" ")[0]
        except:
            continue
    image_paths = [p.strip() for p in image_paths if p.strip()]
    for p in image_paths:
        if not Path(p).exists():
            raise ValueError(f"Path {p} in your image list textfile does not exist!")
    return image_paths


args = ArgumentParser()
args.add_argument(
    "-i",
    "--image-list-textfile",
    type=str,
    required=True,
    help="Path to textfile with newline separated list of image paths",
)
args.add_argument(
    "-o", "--output-csv", type=str, required=True, help="Path to output csv"
)
args.add_argument(
    "-p",
    "--context-prompt",
    type=str,
    default="Provide a very detailed description for this image.",
    help="Context prompt",
)
arg1 = args.parse_args()
image_list_textfile = arg1.image_list_textfile
output_csv = arg1.output_csv
context_prompt = arg1.context_prompt
csvpath = Path(output_csv)


csv_writer = DictWriter(csvpath.open("a+"), fieldnames=["path", "prompt"])

image_paths = to_paths(image_list_textfile)

worker = COGVLMAgentWorker()
for image_path in image_paths:
    _, response = worker.generate(context_prompt, image_path, True)
    csv_writer.writerow({"path": image_path, "prompt": response})
