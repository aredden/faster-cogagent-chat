# Faster Cogagent Chat

This repository contains a faster implementation of [`cogagent-chat-hf`](https://github.com/THUDM/CogVLM), using the [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library, my own library [`torch-bnb-fp4`](https://github.com/aredden/torch-bnb-fp4), and fused operations from [NVIDIA Apex](https://github.com/NVIDIA/apex).

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.8+
- PyTorch
- [NVIDIA Apex](https://github.com/NVIDIA/apex)
  > Note: At least on my machine, when installing nvidia apex, I have to edit the setup.py script and comment out the runtime error which occurs when there is a cuda _MINOR_ incompatibility (facepalm) like so:

```py

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    # <- COMMENT OUT THE FOLLOWING ERROR ->
    # if (bare_metal_version != torch_binary_version):
    #     raise RuntimeError(
    #         "Cuda extensions are being compiled with a version of Cuda that does "
    #         "not match the version used to compile Pytorch binaries.  "
    #         "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
    #         + "In some cases, a minor-version mismatch will not cause later errors:  "
    #         "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
    #         "You can try commenting out this check (at your own risk)."
    #     )


```

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/aredden/faster-cogagent-chat.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Simple Usage

To use the `cogagent-chat-hf` model, follow these steps:

1. Create a list of image paths to caption using the tool in `scripts/imagelist.py`

   ```bash
    python scripts/imagelist.py -i /path/to/your/image/directory -o images_to_caption.txt
   ```

2. Then, using that caption list, you can run the example using the `run_cogagent_chat.py` script, which patches the linear layers and runs inference on the list of image paths that you generated in step 1:

   ```bash
   python run_cogagent_chat.py -i images_to_caption.txt -o ./image_captions.csv
   ```

   Optionally, if you want to customize the context prompt, you can do so by providing a context prompt string

   ```
    python run_cogagent_chat.py -i images_to_caption.txt -o ./image_captions.csv --context-prompt "Provide a detailed description of this image from the perspective of captain jack sparrow"
   ```

3. Initialize the chatbot:

   ```python
   chatbot = cogagent_chat_hf.ChatBot(model)
   ```

4. Start a conversation:

   ```python
   user_input = input("User: ")
   response = chatbot.generate_response(user_input)
   print("ChatBot:", response)
   ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
