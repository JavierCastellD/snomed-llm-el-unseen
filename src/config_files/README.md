# Configuration files for running the script

This folder contains an example configuration run file, [config_run.cfg](config_run.cfg). A single configuration file is needed, with the following sections:

- `[CONF]`: parameters for the pipeline (checkpoints folder, abbreviation disambiguation, reranker, candidate options, etc).
- `[LLM]`: which backend to use (`azure` or `ollama`), the model name, and the temperature.
- `[AZURE]`: the API_KEY and ENDPOINT for Azure OpenAI calls. Only read when `backend = azure`.
- `[OLLAMA]`: the host for the Ollama server. Only read when `backend = ollama`.

The `[LLM]` values can be overridden per-run with the environment variables `LLM_BACKEND`, `LLM_MODEL_NAME`, and `LLM_TEMPERATURE` (used by the SLURM scripts to run several models in parallel from the same config file).
