Hello!

Please refer to the following documentation to reproduce my solution for the NBME - Score Clinical Patient Notes competition.

If you run into any trouble with the setup/code or have any questions please contact me at saun.walker.150892@gmail.com

# ARCHIVE CONTENTS

prod-models : model binaries used to make predictions
training-code : code to rebuild models from scratch
inference-code : code to generate predictions from model binaries
outputs: folder to store training and inference artifacts
submissions: folder to store submission results
deberta_v2_v3_tokenizer: contains scripts fast tokenizers for DeBERTa V2/V3 models

# HARDWARE

Colab Pro + (High RAM + GPU)

The following specs were used to create the original solution
Ubuntu 18.04.5 LTS (Bionic Beaver) with 200GB Disk
8 vCPUs, 56 GB memory
1 x NVIDIA Tesla P100

# SOFTWARE

python packages are detailed separately in `requirements.txt`
Python 3.7
CUDA 11.2

It is assumed that the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed.

Please execute the following command from top level directory i.e. folder containing this file

```
python convert_deberta_v2_v3_tokenizer.py --python_path <path_to_python_env>
```

where `path_to_python_env` is path to folder containing `site-packages` folder e.g.
`/Users/rajabiswas/opt/anaconda3/envs/nbme_env/lib/python3.7/`. This will convert slow tokenizer to fast tokenizer from DeBERTa V2/V3 models.

# MODEL BUILD: There are two options to produce the solution.

1. ordinary prediction
   a) uses binary model in prod-models folder (~8 hours)
2. retrain models
   a) expect this to run around two weeks
   b) trains all models from scratch
   c) follow this with (1) to produce entire solution from scratch

## For option 1:

Please follow the 5 steps detailed in `# Section B: NBME Predictions` of `entry_points.md`
(Overwrites files in the outputs folder)

## For option 2:

Please follow the 5 steps detailed in `# Section A: NBME Training` of `entry_points.md`
(Overwrites files in the prod-models folder)
