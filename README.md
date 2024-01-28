# Sequential Texts Driven Cohesive Motions Synthesis with Natural Transitions (ICCV 2023)

The official PyTorch implementation of the paper [**"Sequential Texts Driven Cohesive Motions Synthesis with Natural Transitions"**](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Sequential_Texts_Driven_Cohesive_Motions_Synthesis_with_Natural_Transitions_ICCV_2023_paper.html).

Please visit our [**webpage**](https://druthrie.github.io/sequential-texts-to-motion/) for more details.

![teaser](https://github.com/Druthrie/sequential-texts-to-motion/blob/main/fig/teaser_M29.png)

#### Bibtex
If you find this code useful in your research, please cite:

```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Shuai and Zhuang, Sisi and Song, Wenfeng and Zhang, Xinyu and Chen, Hejia and Hao, Aimin},
    title     = {Sequential Texts Driven Cohesive Motions Synthesis with Natural Transitions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9498-9508}
}
```

## Getting started

This code was tested on `Ubuntu 20.04.1 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment
Setup conda env:
```shell
conda env create -f environment.yaml
conda activate ST2M_env
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

### 2. Get data

```shell
mkdir ./dataset/
```

#### BABEL_TEACH

save BABEL_TEACH dataset to ./dataset/

#### STDM

If you want to request our STDM dataset please fill out the ["The STDM Dataset Release Agreement"](https://github.com/Druthrie/SequentialTexts2Motion/blob/main/dataset_agreement.pdf), send the email to sisizhuang@buaa.edu.cn

unzip and place BABEL_TEACH dataset to ./dataset/

### 3. Download the pretrained models
```shell
mkdir ./checkpoints/
```
Download the models of [BABEL_TEACH dataset](https://drive.google.com/file/d/1_KCzpH6BA-7BnL2_QrkJa2CA_RQUAvWR/view?usp=sharing) and [STDM datset](https://drive.google.com/file/d/1q6PgN2Nut7fuAlEXZITA7gBMDehUrkJC/view?usp=sharing), then unzip and place them in `./checkpoints/`, which should be like
```shell
./checkpoints/BABEL_TEACH/
./checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/           # Sequential-text-to-motion generation model
./checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH_slerp/           # Sequential-text-to-motion generation model with slerp
(The model is the same as the model without slerp operation, split into two folders just to facilitate final evaluations.)
./checkpoints/BABEL_TEACH/Decomp_SP001_SM001_H512/ # Motion autoencoder
./checkpoints/BABEL_TEACH/text_mot_match_M10_BABEL_TEACH/          # Motion & Text feature extractors for evaluation

./checkpoints/STDM/
./checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM/           # Sequential-text-to-motion generation model
./checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM_slerp/           # Sequential-text-to-motion generation model with slerp
(The model is the same as the model without slerp operation, split into two folders just to facilitate final evaluations.)
./checkpoints/STDM/Decomp_SP001_SM001_H512/ # Motion autoencoder
./checkpoints/STDM/text_mot_match_M10_STDM/          # Motion & Text feature extractors for evaluation
```

## Training Models

### Train Sequential-text-to-motion model
#### BABEL_TEACH dataset
```shell
python st2m_train.py --name ST2M_model_name --gpu_id 0 --dataset_name BABEL_TEACH
```

#### STDM dataset
```shell
python st2m_train.py --name ST2M_model_name --gpu_id 0 --dataset_name STDM
```


### Training motion & text feature extractors
#### BABEL_TEACH dataset
```shell
python st2m_train_tex_mot_match.py --name match_model_name --gpu_id 0 --dataset_name BABEL_TEACH
```

#### STDM dataset
```shell
python st2m_train_tex_mot_match.py --name match_model_name --gpu_id 0 --dataset_name STDM
```


## Generating and Animating 3D Motions

### BABEL_TEACH dataset
#### without slerp operation
```shell
python st2m_gen_mul_motions_scipy_V2.py --gpu_id 0 --dataset_name BABEL_TEACH --name trainV13_LV1LT1LK001LA01_BABEL_TEACH --text_file ./inputs_texts/BABEL_TEACH/0.txt --ext 0 --repeat_times 3
```

#### with slerp operation
```shell
python st2m_gen_mul_motions_scipy_V2.py --gpu_id 0 --dataset_name BABEL_TEACH --name trainV13_LV1LT1LK001LA01_BABEL_TEACH --text_file ./inputs_texts/BABEL_TEACH/0.txt --ext 0_slerp --repeat_times 3 --do_slerp
```

### STDM dataset
#### without slerp operation
```shell
python st2m_gen_mul_motions_scipy_V2.py --gpu_id 0 --dataset_name STDM --name trainV13_LV1LT1LK001LA01_STDM --text_file ./inputs_texts/STDM/0.txt --ext 0 --repeat_times 3
```

#### with slerp operation
```shell
python st2m_gen_mul_motions_scipy_V2.py --gpu_id 0 --dataset_name STDM --name trainV13_LV1LT1LK001LA01_STDM --text_file ./inputs_texts/STDM/0.txt --ext 0_slerp --repeat_times 3 --do_slerp
```


## Quantitative Evaluations
#### BABEL_TEACH dataset
```shell
python st2m_final_evaluations.py --dataset_name BABEL_TEACH --log_file_name final_contrast
```

#### STDM dataset
```shell
python st2m_final_evaluations.py --dataset_name STDM --log_file_name final_contrast
```
