# Section A: NBME Training

Please execute the following steps to train the models used for my solution.

## Step 1: Folders Setup

The following commands created the necessary folders for training. The commands should be executed from top level project directory i.e. folder containing this file.

```
mkdir outputs
mkdir dev-models
mkdir -p data/train_data/
```

## Step 2: Data Preparation

This step downloads the NBME training data.

```
cd data/train_data/
kaggle competitions download -c nbme-score-clinical-patient-notes
unzip nbme-score-clinical-patient-notes.zip
rm nbme-score-clinical-patient-notes.zip
cd ../..
```

## Step 3: Task Adaptation using Masked Language Modeling (MLM)

This step performs the task adaptation for the 4 backbones in my solution. The MLM accuracy should be around 76-78%.

```
cd training-code
python mlm.py --config_path ./configs/mlm_config_del.json
python mlm.py --config_path ./configs/mlm_config_dexl.json
python mlm.py --config_path ./configs/mlm_config_dexlv2.json
python mlm.py --config_path ./configs/mlm_config_delv3.json
```

## Step 4: Training of Models

This step involves training of the models used in inference code. The commands are to be executed from `training-code` folder. As a first step, please create the folds by running the command below.

```
python create_folds.py
```

### DeBERTa Large

This section contains code for training of 4 DeBERTa Large models.

#### Model 1

For Model 1, the (soft) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_del_soft.py --config_path ./configs/mpl_del_m1.json
python mpl_del_student_finetune.py --config_path ./configs/mpl_del_m1.json
```

#### Model 2

For Model 2, the (soft) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_del_soft.py --config_path ./configs/mpl_del_m2.json
python mpl_del_student_finetune.py --config_path ./configs/mpl_del_m2.json
```

#### Model 3

Model 3 is trained via knowledge distillation from Model 1 + Model 2 using labeled + unlabelled data.

```
python kd_del.py --config_path ./configs/kd_del_m3.json
```

#### Model 4

Model 4 is trained via knowledge distillation from Model 1 + Model 2 using only pseudo labels.

```
python kd_del.py --config_path ./configs/kd_del_m4.json
```

### DeBERTa XLarge

This section contains code for training of 2 DeBERTa XLarge models.

#### Model 5

For Model 5, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_dexl_hard.py --config_path ./configs/mpl_dexl_m1.json
python mpl_dexl_student_finetune.py --config_path ./configs/mpl_dexl_m1.json
```

#### Model 6

For Model 6, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_dexl_hard.py --config_path ./configs/mpl_dexl_m2.json
python mpl_dexl_student_finetune.py --config_path ./configs/mpl_dexl_m2.json
```

### DeBERTa V2 XLarge

This section contains code for training of 2 DeBERTa V2 XLarge models.

#### Model 7

For Model 7, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_dexlv2_hard.py --config_path ./configs/mpl_dexlv2_m1.json
python mpl_dexlv2_student_finetune.py --config_path ./configs/mpl_dexlv2_m1.json
```

#### Model 8

Model 8 is trained with standard fine-tuning approach.

```
python sft_dexlv2.py --config_path ./configs/sft_dexlv2_m1.json
```

### DeBERTa V3 Large

This section contains code for training of 5 DeBERTa V3 Large models.

#### Model 9

For Model 9, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_delv3_hard.py --config_path ./configs/mpl_delv3_m1.json
python mpl_delv3_student_finetune.py --config_path ./configs/mpl_delv3_m1.json
```

#### Model 10

For Model 10, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data.

```
python meta_st/mpl_delv3_hard.py --config_path ./configs/mpl_delv3_m2.json
python mpl_delv3_student_finetune.py --config_path ./configs/mpl_delv3_m2.json
```

#### Model 11

For Model 11, the (hard) meta-pseudo-labels training is performed. Subsequently, the trained student is fine-tuned with labeled data. During this fine-tuning, SWA is used for better generalization.

```
python meta_st/mpl_delv3_hard.py --config_path ./configs/mpl_delv3_m3.json
python mpl_delv3_student_finetune_swa.py --config_path ./configs/mpl_delv3_m3.json
```

#### Model 12

For Model 12, the (hard) meta-pseudo-labels training is performed. Marker tokens are added in feature text to distinguish different cases present in patient notes. Subsequently, the trained student is fine-tuned with labeled data. During this fine-tuning, SWA is used for better generalization.

```
python meta_st/mpl_delv3_hard_marker.py --config_path ./configs/mpl_delv3_m4_marked.json
python mpl_delv3_student_finetune_swa_marker.py --config_path ./configs/mpl_delv3_m4_marked.json
```

#### Model 13

For Model 13, the (hard) meta-pseudo-labels training is performed. Marker tokens are added in feature text to distinguish different cases present in patient notes. Subsequently, the trained student is fine-tuned with labeled data. During this fine-tuning, SWA is used for better generalization.

```
python meta_st/mpl_delv3_hard_marker.py --config_path ./configs/mpl_delv3_m5_marked.json
python mpl_delv3_student_finetune_swa_marker.py --config_path ./configs/mpl_delv3_m5_marked.json
```

## Step 4: Cleanup

Clean up temporary artifacts in dev-models folder.

```
cd ..
rm -rf ./dev-models/tmp
```

## Step 5: Final Step

Now the contents of `prod-models` can be replaced with contents of `dev-models`.

## Notes:

- I found MLM and Meta Pseudo Labels Training of DeBERTa V2 XLarge models very sensitive to hyper-parameters and may at times diverge. Therefore, a lower learning rate is set for this training.
- For Meta Pseudo Labels Training I continuously monitored the student and teacher losses (stdout print). At the end of training the student loss should be around `0.005` and teacher loss should be around `0.001`
  ################################################################################

# Section B: NBME Predictions

Please execute the following steps for inference on test data and generating the submission.

## Step 1: Folders Setup

The following commands created the necessary folders for inference. The commands should be executed from top level project directory i.e. folder containing this file.

```
mkdir outputs
mkdir submissions
mkdir -p data/inference_data/
```

## Step 2: Data Preparation

Please execute the following shell commands from from the top level directory to download the data required for inference.

```
cd data/inference_data/
kaggle competitions download -c nbme-score-clinical-patient-notes
unzip nbme-score-clinical-patient-notes.zip
rm nbme-score-clinical-patient-notes.zip
cd ../..
```

## Step 3: Create Datasets

The following commands will create the inference datasets.

```
cd inference-code
python sort_data.py
python generate_dataset.py --config_path ./configs/dataset_config_dexl.json
python generate_dataset.py --config_path ./configs/dataset_config_dexlv2.json
python generate_dataset.py --config_path ./configs/dataset_config_delv3.json
python generate_dataset_marked.py --config_path ./configs/dataset_config_delv3_marked.json
```

## Step 4: Inference

Make predictions from the trained models.

```
python predict_lakecity.py --config_path ./configs/del_mpl_1.json --save_path ../outputs/preds_del_mpl_1.pkl

python predict_lakecity.py --config_path ./configs/del_mpl_2.json --save_path ../outputs/preds_del_mpl_2.pkl

python predict_lakecity.py --config_path ./configs/del_kd_1.json --save_path ../outputs/preds_del_kd_1.pkl

python predict_lakecity.py --config_path ./configs/del_kd_2.json --save_path ../outputs/preds_del_kd_2.pkl



python predict_lakecity.py --config_path ./configs/dexl_mpl_1.json --save_path ../outputs/preds_dexl_mpl_1.pkl

python predict_lakecity.py --config_path ./configs/dexl_mpl_2.json --save_path ../outputs/preds_dexl_mpl_2.pkl


python predict_lakecity.py --config_path ./configs/dexlv2_mpl_1.json --save_path ../outputs/preds_dexlv2_mpl_1.pkl

python predict_lakecity.py --config_path ./configs/dexlv2_sft_1.json --save_path ../outputs/preds_dexlv2_sft_1.pkl


python predict_lakecity.py --config_path ./configs/delv3_mpl_1.json --save_path ../outputs/preds_delv3_mpl_1.pkl

python predict_lakecity.py --config_path ./configs/delv3_mpl_2.json --save_path ../outputs/preds_delv3_mpl_2.pkl

python predict_lakecity.py --config_path ./configs/delv3_mpl_3.json --save_path ../outputs/preds_delv3_mpl_3.pkl

python predict_lakecity.py --config_path ./configs/delv3_mpl_4_marked.json --save_path ../outputs/preds_delv3_mpl_4_marked.pkl

python predict_lakecity.py --config_path ./configs/delv3_mpl_5_marked.json --save_path ../outputs/preds_delv3_mpl_5_marked.pkl

python predict_public.py --save_path ../outputs/preds_delv3_public.pkl
```

## Step 5: Submission

Generate the submission file.

```
python generate_submission.py
cd ..
```
