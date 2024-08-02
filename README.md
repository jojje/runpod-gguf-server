# Runpod image serving using koboldcpp

Downloads and serves GGUF models from huggingface via KoboldCPP upon startup.

Includes CUDA compiled llama.cpp as well for easy quantization or gguf file munging.

Takes the following environment variables as configuration (with defaults shown):

* `MODEL` - Huggingface model URL. Either a repository URL, a directory in the repo or an individual file. **Required**, must be configured on the runpod environment variable overrides (no default value). Example: `https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
* `MODEL_DIR` - directory in the container where the downloaded model is stored (default: `/models`)
* `CTX` - max context length for inference. Lower it if you run into GPU out of memory crash (default: `8192`)
* `ARGS` - additional koboldcpp arguments in addition to the corresponding above ones. (default: `--usecublas mmq --gpulayers 999 --flashattention --ignoremissing --skiplauncher`)
* `PORT` - local bind port for koboldcpp in the container (default: `5002`)
* `DRYRUN` - for development and testing. Runs the container without side-effects.

## How to use:
1. Select a GPU and the template: TBD
2. Change the `MODEL` environment variable to whatever GGUF file you want to serve.
3. (Optional and discouraged) if you want to serve the Koboldcpp web UI or API directly via Cloudflare's or Runpod's MITM attack-as-a-service, add port 5002 to the "Expose HTTP Ports" field.
4. Launch the pod.
5. (Optional but highly recommended instead of step 3) Create a SSH tunnel using the instruction in the pod log.

See below why #3 is discouraged.

## FAQ

### Q1. Why another pod template for kobold?

Because I don't trust the ones I found on runpod. They didn't have source code available when I inquired.
Worse, there were signs of MITM going on, where traffic that should have been routed directly to the pod, took a detour half around the world for some unexplained reason, resulting in very high latency in the container.

So this container probably does nothing new compared to the other ones, but it _is_ completely transparent about _what it does_, _why_, and _how_. It also prints
the SSH server's key fingerprints so you can easily verify that you're indeed connecting _directly_ to the ssh server within the pod, and not some MITM cloudflare proxy or similar that intercepts your private communication with the pod. 

This increased security comes at a small inconvenience, since you have to open an explicit SSH tunnel to it. The pod logs tell you exactly what to type to get connected though, so I regard this a feature and not a bug, since _you_ control _exactly_ how you connect to the model serving endpoint. Barring runpod having gone to great length compromising their hardware or injects malware into the processes launched in their pods in order to exfiltrate crypto keys or plaintext traffic from customer pods (a targeted supply chain attack), then your communication should remain private between your browser and the pod (unless you use google chrome or microsoft edge that is ;)

### Q2. How does it decide what model to serve?

When providing a link to a single model file, it serves that one. When providing a link to a directory or HF repo, it picks the first gguf file it finds in that repo, ordered alphanumerically.

### Q3. Can I change the model being served without restarting the pod?

Yes. You can SSH into the machine and terminate the kobold process. Then start kobold with whatever other settings you like, including different model files, and bind it to the same port or a different one. The exact kobold command used to launch the initial model serving on startup is printed in the logs, so you can use that as a starting point if you're not familiar with the kobold CLI arguments.

Or you could serve using the bundled llama-cpp, or download something completely different instead. Whatever else you like. The pod will live until the "wait" process is terminated, or you kill the pod via the runpod API or admin UI.

This _feature_ is actually one other reason I created this image, so I can quickly experiment with different things, without having to (re)launch a gazillion pods, and re-download tens to hundreds of gigabytes of model files again and again every time.

### Q4. Can I point it to a local file instad of a huggingface URL?

Yes. You can mount volumes and specify the model path to a GGUF file in a mounted directory. E.g. if you mount your model volume at say `/mnt` and have a file "/mnt/tinyllama.gguf" in the volume, simply specify `/mnt/tinyllama.gguf` for the `MODEL` environment variable, and it will be read from disk on container startup.

### Q5. What if I don't know if the model fits into the GPU VRAM, or how large a context the VRAM can hold?

The startup script finds that out automatically. If the koboldcpp startup runs into a crash, the container will perform a binary search to find the largest context size that fits into the VRAM of the (combined) GPU(s), up to and including the specified or default context size (`CTX`). Check the pod logs for the optimal number it discovered. You can then set that value explicitly using the `CTX` environment variable in the pod configuration, so that future startups don't run into the same crash situation and have to go through the same search process again. Without out of memory crashes, the bootup will be faster.

### Q6. The pod reports some error, but does not terminate, why?

In order to avoid infinite crash looping. Runpod restarts a pod when it crashes, which is rather annoying if you're trying to find out _why_ the pod crashed. For instance, if the downloaded model weights are corrupt, or there is some bug in koboldcpp. As such, when an unrecoverable error happens, the startup script will just notify you of this fact in the pod logs, and then pause, giving you the opportunity to figure out what's wrong. Such as SSH:ing into the pod in order to look around for the root cause.
