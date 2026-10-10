# Manual RepE / control-vector setup for Gemma 4

This guide is for a person following terminal commands. No AI coding assistant,
custom gateway, or training API is required. The main route uses a standalone
llama.cpp trainer and a normal llama-server with a fixed control vector.

The accompanying folder includes the small trainer patch, a 64-pair example,
and Python helpers. Keep these files together when sharing the guide.

**Scope:** text-only inference on a Gemma 4 GGUF. Our reference experiment used
`Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf`, a particular community
checkpoint. Its name, layer count, dimensions, and results must not be treated
as properties of every Gemma 4 model. Multimodal input and other variants were
not validated by these experiments.

The extraction method and vectors were tested in the original Windows setup.
The portable build commands below describe a fresh source build; the complete
procedure has not been rerun on a second machine. The patch helper checks the
exact input source before changing it.

## 1. What you are building

For each matched pair of prompts, run the model without generating an answer.
At each selected transformer layer, collect the activation at the final real
prompt token. Compute:

```text
difference[layer] = mean(positive_activation - negative_activation)
direction[layer]  = difference[layer] / norm(difference[layer])
```

During inference the runtime adds `strength * direction[layer]` to the residual
stream. A positive strength targets the positive side of the dataset; a negative
strength targets the negative side. There is no backpropagation, optimizer, or
change to the original model weights.

This is one practical RepE-style method, not the only possible extraction method.
It learns a contrast between prompts; it does not prove that the model has the
named emotion, or that the vector is a reliable detector of that emotion.

## 2. Prerequisites and layout

Install Git, Python 3.10 or later, CMake, and a C++ build toolchain. For NVIDIA
GPU execution also install the CUDA Toolkit, a compatible driver, and a host
compiler supported by that Toolkit. On Windows, use Visual Studio 2022 Build
Tools with C++ development support. On Linux, use a supported GCC/Clang toolchain.

Have enough RAM/VRAM for the model **plus** context and temporary activations.
Use the exact serving GGUF for fitting where practical. Match the tokenizer,
chat template, model variant, and quantization. Record the model hash.

Arrange a new working directory like this:

To obtain this guide and its helpers from GitHub, create the working directory
and clone this repository into a subfolder named `manual-gemma4`:

```text
mkdir gemma-repe
cd gemma-repe
git clone https://github.com/amastrae/gemma4-repE-guide.git manual-gemma4
```

Stay in `gemma-repe/` for the remaining commands; do not change into the cloned
`manual-gemma4/` folder. Place your model in `models/`. The resulting layout is:

```text
gemma-repe/
  manual-gemma4/       # this guide and its companion files
  models/model.gguf    # your actual model; use its real filename if preferred
  llama.cpp/          # created by git clone below
  data/               # generated later
  vectors/            # generated later
```

Run commands from `gemma-repe/`. Commands use relative paths deliberately.
For a split GGUF, supply the first shard and keep all shards together.

Create a Python environment and install the validator's dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy
```

On Linux, activate with `source .venv/bin/activate` instead. If Windows blocks
activation, call `.\.venv\Scripts\python.exe` directly in place of `python`;
changing the machine's execution policy is unnecessary.

## 3. Get the pinned source and patch the trainer

```text
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout d81235049384534c167caea52b85a694f6103d14
python manual-gemma4/patch_trainer.py llama.cpp --check
python manual-gemma4/patch_trainer.py llama.cpp
git -C llama.cpp diff -- tools/cvector-generator/cvector-generator.cpp
```

The helper changes only the standalone generator. It does **not** add our
custom server endpoints, touch inference code, or modify model files. It refuses
unrecognized source and recognizes its own already-applied patch. Do not force
it onto a newer revision by removing its checks.

### Why these fixes matter for this Gemma setup

The patch makes the following extraction choices explicit:

| Issue | What this version does |
| --- | --- |
| Unequal prompt lengths | Extracts each prompt's own final real token, without padding the shorter side. |
| Capturing all prompt activations | Copies only the final token row, substantially reducing host transfer and storage. |
| Final-layer token selection | Accepts the actual `l_out-*` tensor shape; later graph stages can select only output-token rows. It does not require every layer tensor to have the original prompt length. |
| Layer numbering | Captures native layers `1..N-1`, excludes native layer zero, and exports matching `direction.1..direction.N-1` tensor names. This avoids shifting the captured layers by one. |
| Negative differences | Uses absolute magnitude when deciding whether a difference row is all zero. |
| Silent failures | Stops on prompt preparation/decode failures and asserts the expected number of captured layers. |

The pinned generator has comments inherited from its previous extraction
behavior. After this patch, **layer zero is omitted; the final native layer is
included**. Follow the actual indexing, not those older comments.

Gemma 4's graph has model-dependent features such as sliding/full attention,
shared KV layers, per-layer inputs and optional output scaling. In the inspected
implementation the control vector is added near the end of a transformer block,
after those block computations, at the `build_cvec(cur, il)` call immediately
before the `l_out` callback. An attention-head activation, per-layer embedding,
or final normalized embedding is not an interchangeable extraction point.
See the pinned [Gemma graph source](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/gemma4.cpp).

If your model needs a newer runtime to load, use that runtime and port/review
these changes against its source. Do not assume this pin supports every later
Gemma checkpoint.

## 4. Build the trainer and server

For NVIDIA on Windows, run this in a Developer PowerShell with CUDA available:

```powershell
cmake -S llama.cpp -B llama.cpp/build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build llama.cpp/build --config Release --target llama-cvector-generator llama-server --parallel 8
```

The executables should be in `llama.cpp/build/bin/Release/`. Check that both run:

```powershell
.\llama.cpp\build\bin\Release\llama-cvector-generator.exe --help
.\llama.cpp\build\bin\Release\llama-server.exe --help
```

For Linux/NVIDIA, use:

```bash
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build llama.cpp/build --target llama-cvector-generator llama-server --parallel 8
```

Linux executable paths are normally `./llama.cpp/build/bin/llama-server` and
`./llama.cpp/build/bin/llama-cvector-generator`. Substitute those paths in the
remaining Windows examples; the application arguments are the same.

`GGML_BACKEND_DL=OFF` is intentional: at this pin the generator is gated on a
non-dynamic CPU backend and directly references CPU backend symbols. Our
original deployment used a different, machine-specific dynamic-plugin build;
its hardcoded `.lib` paths and copied CUDA DLLs are unnecessary here. Build the
backend and executables together. Do not mix DLLs from unrelated releases.

For CPU-only use, configure a separate build with `GGML_CUDA=OFF` and run with
`-ngl 0`. Expect different timing. Other GPU backends require their own build
options and are outside this guide's tested setup.

## 5. Prepare matched pairs and render the real chat template

`manual-gemma4/pairs.example.json` contains 64 happy/sad pairs. Each pair has
exactly two plain strings. To create a new axis, edit a copy of that file:

```json
{
  "name": "happy_sad_v1",
  "positive_label": "happy",
  "negative_label": "sad",
  "pairs": [
    {
      "positive": "A fictional narrator feels happy while making tea. Write their thoughts.",
      "negative": "A fictional narrator feels sad while making tea. Write their thoughts."
    },
    {
      "positive": "A fictional narrator feels happy while waiting for a train. Write their thoughts.",
      "negative": "A fictional narrator feels sad while waiting for a train. Write their thoughts."
    }
  ]
}
```

That is a valid small example; the supplied 64-pair file is a better starting
experiment. Labels are descriptive. Only the contrasting prompt texts determine
the fitted direction. Keep task, subject, perspective and unrelated instructions
matched. Vary situations so one topic does not dominate the contrast. Reserve
separate prompts for evaluating the result.

Start a temporary **unsteered** server with the model you will train:

```powershell
.\llama.cpp\build\bin\Release\llama-server.exe -m models/model.gguf -ngl all -c 2048 --parallel 1 --host 127.0.0.1 --port 8080 --alias local --jinja
```

If you normally use a template override, add the same
`--chat-template-file path/to/template.jinja` here and when serving later.
Do not select a generic `gemma` template just because the name looks right.
Wait until `/health` reports ready. In a second terminal, from the same working
directory, render the dataset:

```text
python manual-gemma4/prepare_pairs.py manual-gemma4/pairs.example.json --url http://127.0.0.1:8080 --out data
```

The helper uses ordinary llama-server `/apply-template` and `/tokenize`
endpoints. It performs no text generation and needs no custom `/repe/train`
endpoint. If the local server has authentication enabled, set `LLAMA_API_KEY`
in the helper's environment.

It creates `data/positive.txt`, `data/negative.txt`, and `data/rendered.json`.
Inspect the latter before proceeding. Confirm both sides have the same
assistant-generation suffix, differ in the intended instruction, and tokenize
as distinct prompts. The helper checks a 480-token ceiling with special-token
parsing enabled and requests thinking off.

### Gemma-specific template pitfalls

- Our checkpoint's rendered prompt uses `<|turn>user`, `<turn|>`,
  `<|turn>model` and channel markers. Do not transplant older Gemma
  `<start_of_turn>` examples or hand-insert markers into plain pair strings.
- With our template, thinking off still produces an empty thought-channel
  prefix. The presence of a thought marker alone does not mean thinking is on.
  Inspect the template and rendered prefix; test generated behavior separately.
- The extraction point is the end of the **rendered prompt**, including its
  assistant prefix, not the last word in your original user text.
- The generator tokenizes special markers and applies the vocabulary's BOS
  policy. Do not manually prepend another BOS or disable special-token parsing.
- The text files use one physical line per prompt, with escaped newlines. The
  helper escapes literal backslashes too, so text such as `\n` is not silently
  transformed into a different prompt by the generator's escape parser.

Now stop the temporary server with Ctrl+C and wait for it to release the model.
The next step starts a separate model process. This avoids holding two copies
of the weights in VRAM.

## 6. Fit and validate the vector

Create an output directory, then run the patched trainer:

```powershell
New-Item -ItemType Directory -Force vectors
.\llama.cpp\build\bin\Release\llama-cvector-generator.exe -m models/model.gguf -ngl all -fa on -c 512 -b 512 -ub 512 --positive-file data/positive.txt --negative-file data/negative.txt --method mean -o vectors/happy_sad_v1.gguf
```

On Linux, use `mkdir -p vectors`. Check that the process exits successfully.
Do not load a partially written output after an error.

Keep each rendered prompt within the helper's 480-token limit and use the
512-token batch/microbatch settings above. This simple callback expects one
capture per layer per prompt. Splitting a longer prompt into multiple
microbatches can produce extra captures and trip the layer-count assertions.
If you need longer examples, adapt the collector rather than disabling checks.

Validate the GGUF against **your model's actual residual width and layer count**.
For our particular 48-layer, width-3,840 checkpoint:

```text
python manual-gemma4/validate_vector.py vectors/happy_sad_v1.gguf --layers 47 --width 3840
```

The expected exported count is `N-1`, because native layer zero is omitted.
Determine `N` and residual width from your GGUF metadata/model startup output.
The validator checks tensor names, dimensions, finite values and unit norms.
A successful shape check proves file consistency, not a useful behavioral effect.
If a mean difference is zero or nearly zero, normalization can fail; improve the
dataset instead of accepting NaNs or forcing the vector through.

Save the dataset, rendered prompts, template, model hash, source commit,
training command, output hash, layer range and evaluation settings together.
The GGUF's model hint is not a cryptographic compatibility check.

## 7. Serve with a fixed control vector

For our reference checkpoint, native layers 10 through 30 inclusive were a
useful initial range. For another architecture/size, select valid native layer
indices and compare ranges on held-out prompts; do not assume 10–30 is optimal.

```powershell
.\llama.cpp\build\bin\Release\llama-server.exe -m models/model.gguf -ngl all -c 4096 --parallel 1 --host 127.0.0.1 --port 8080 --alias local --jinja --control-vector-scaled vectors/happy_sad_v1.gguf:1 --control-vector-layer-range 10 30
```

**Syntax trap:** at this pin the argument is `FNAME:SCALE`, not two arguments
`FNAME SCALE`. The parser splits on colons. A Windows drive-qualified argument
such as `D:\vectors\happy.gguf:1` is therefore problematic. Run from the working
directory and use a relative path as above. Quote the entire argument if its
path contains spaces; avoid commas in the filename too.

Send an ordinary chat request, for example with Python's standard library:

```python
import json
from urllib.request import Request, urlopen

body = {
    "model": "local",
    "messages": [{"role": "user", "content": "Describe a quiet evening in three sentences."}],
    "temperature": 0,
    "seed": 1234,
    "max_tokens": 256,
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False}
}
request = Request("http://127.0.0.1:8080/v1/chat/completions",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choices"][0]["message"])
print("finish:", result["choices"][0]["finish_reason"])
```

If the server requires a specific model identifier, use its `/v1/models` result
instead of `local`. For this manual workflow, restart the server with a different
coefficient to change strength. Use `:0` or omit both control-vector arguments
for a neutral baseline; use `:-1` for the opposite pole. All requests to this
server share its startup control.

The `steering` JSON field and named-direction registry in the original project
are **custom gateway features**. Sending that field to this ordinary server does
not implement per-request controls. Likewise, this guide does not add a detector
or an adaptive correction loop.

## 8. Evaluate before relying on it

Compare neutral, +1, -1, +2 and -2 on unseen prompts. Start modestly; test larger
magnitudes only if needed. Keep the model, template, prompt, generation cap and
sampling settings fixed. Temperature zero reduces variability, but different
batching and hardware can still change near-tied tokens.

Measure what you actually want: word counts for length, repeated phrases for
repetition, and human judgment for tone. Check factual accuracy, relevance,
instruction-following and grammatical quality too. Include ordinary prompts
that should remain unaffected. Use separate validation and final test prompts
when selecting strength or layer range.

Important limits from our experiments:

- Effects are not guaranteed to be smooth, monotonic or independent. A
  "repetition" vector may also change length; distrust can become hostility
  or fabricated suspicion. Strong steering can overwhelm the task.
- Exact copying and exact word counts are not guaranteed. `max_tokens` is an
  output cap, not a semantic length controller. Record `finish_reason`; a
  length-capped response is incomplete, not naturally concise.
- Fit with thinking off first, then evaluate thinking on separately. Steering
  can affect generated reasoning as well as the answer. It does not selectively
  target only the visible answer by default.
- Combinations add vectors but their behavioral effects need not add neatly.
  Test each direction first, then test the combination separately.
- A reading score along a vector is not an emotion probability or a reliable
  factuality detector. Automatic feedback needs its own calibration and testing.

## 9. Performance, caching and operational pitfalls

**Standalone training loads its own model.** Our original 64-pair standalone run
took about 7.64 seconds including loading, extraction, fitting and export on its
machine. Setup, compilation and evaluation were excluded. Your hardware, model,
offload and prompt lengths will determine your own result.

The roughly 4–5-second runs mentioned in the original project used a separate
custom shared-model training implementation. It reused loaded weights, created
a small training context, paused inference at a decode boundary and resumed it.
One measured run temporarily used about 722 MiB extra VRAM. Those timings and
memory figures do not describe the standalone process in this guide.

Each vector for our model occupies roughly 0.7 MiB. That does not mean inference
latency is unchanged: longer answers increase total time, GPU operations cost
something, and speculative-decoding acceptance may change. Benchmark both
tokens/second and end-to-end latency.

Begin without MTP/speculative decoding. Our custom integration later kept MTP
enabled and applied steering to the target model only; that result does not
establish compatibility for every draft implementation.

Keep the control fixed for the lifetime of a serving process when following this
guide. For a future dynamic implementation, changing vectors can invalidate
cached prompt/KV state, and a global control must not leak between concurrent
requests. Our original custom version serialized requests and cleared relevant
cache state when controls changed. A new implementation must address those
issues explicitly.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| Unknown architecture or tensor-loading error | Your runtime may not support this exact model. Verify the checkpoint and use a compatible runtime; port the trainer changes deliberately if necessary. |
| No `llama-cvector-generator` build target | At this pin, use `GGML_BACKEND_DL=OFF`, `GGML_CPU=ON`, and keep tools/common enabled. Reconfigure a fresh build directory if changing backend configuration. |
| CUDA/toolset or missing DLL error | Build with compatible CUDA and compiler versions; use libraries from that same build. Do not copy arbitrary DLLs from another release. |
| Captured-layer assertion | Check prompt length, microbatch size, model graph/tensor naming and the source revision. Do not simply remove the assertion. |
| Non-finite or zero vector | Check for identical pairs, cancellation, failed extraction or an inappropriate final-token position. |
| No apparent steering | Confirm the server loaded the vector, the layer range is valid, the tensor indices match, and the contrast is learnable. Compare both signs with a fresh server. |
| Broken prose or unrelated behavior | Reduce strength; revisit the pairs, layer range and evaluation. A larger coefficient is not always the solution. |
| Unexpected thought text | Check the actual chat template and thinking options; do not infer behavior from a generic model-family name. |
| VRAM suddenly doubles | A server and standalone trainer may both have loaded weights. Stop the temporary server before fitting. |
| `control-vector-scaled format: FNAME:SCALE` | Use one `relative/path.gguf:strength` argument. Avoid a Windows drive colon. |

## References and companion files

- [llama.cpp control-vector documentation](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14/tools/cvector-generator) — upstream context; some examples may use older flag syntax, so prefer this pin's `--help` and parser.
- [Pinned argument parser](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/common/arg.cpp) — scaled-vector argument parsing.
- [Pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/README.md) — standard template, tokenization and chat endpoints.
- [Representation Engineering paper](https://arxiv.org/abs/2310.01405) — research background.
- `patch_trainer.py` + `trainer-patch.json` — exact standalone extraction patch.
- `pairs.example.json` — 64 synthetic matched happy/sad prompts.
- `prepare_pairs.py` — real-template rendering and token-length checks.
- `validate_vector.py` — exported-vector sanity checks, requiring NumPy.

No model weights, credentials, previous request logs, or private conversation
data are included in this folder. The helpers contact only the server URL you
supply for rendering/tokenization; training and generation remain local to your
chosen runtime.
