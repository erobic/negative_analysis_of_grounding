# A negative case analysis of visual grounding methods for VQA (ACL 2020 short paper)
Recent works in VQA attempt to improve visual grounding by training the model to attend to query-relevant visual regions. Such methods
 have claimed impressive gains in challenging datasets such as VQA-CP. However, in this work we show that boosts in performance come from a regularization effect as opposed to proper visual grounding.

This repo is based on [Self-Critical Reasoning codebase](https://github.com/jialinwu17/self_critical_vqa). 


### Install dependencies
We use Anaconda to manage our dependencies. You will need to execute the following steps to install all dependencies:

- Edit the value for `prefix` variable in `requirements.yml` file, by assigning it the path to conda environment

- Then, install all dependencies using:
``conda env create -f requirements.yml``

- Change to the new environment:
``source activate negative_analysis_of_grounding``

- Install:
``python -m spacy download en_core_web_lg``

### Executing scripts
While executing scripts, first ensure that your main project directory is in PYTHONPATH:

``cd ${PROJ_DIR} && export PYTHONPATH=.``
    

### Setting up data
- Inside `scripts/common.sh`, edit `DATA_DIR` variable by assigning it the path where you wish to download all data 
- Download UpDn features from [google drive](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr?usp=sharing) into `${DATA_DIR}` folder
- Download questions/answers for VQAv2 and VQA-CPv2 by executing `./scripts/download.sh`
- Preprocess VQA datasets by executing: `./scripts/preprocess.sh`    
    
### Training baseline model
Execute `./scripts/baseline/vqacp2_baseline.sh`.

- *Note#1: We will be using the pre-trained baseline model to train HINT/SCR and our regularizer.*
- *Note#2: We need to train baselines on 100% of the training set. However, by default, the training script expects to train only on subset with visual hints (e.g., HAT or textual explanations).
So, to train baseline, we need to use the flag `--do_not_discard_items_without_hints`, otherwise it will throw an error message saying that `hint_type` flag is missing.*

### Training models that do well on VQA-CP

#### Setting up data

#### Training HINT [1]
 
 
#### Training SCR [2]

#### Training our regularizer 

### Analysis
#### Computing rank correlation

#### Visualizing with Grad-Cam

### References

[1] Selvaraju, Ramprasaath R., et al. "Taking a hint: Leveraging explanations to make vision and language models more grounded." Proceedings of the IEEE International Conference on Computer Vision. 2019.

[2] Wu, Jialin, and Raymond Mooney. "Self-Critical Reasoning for Robust Visual Question Answering." Advances in Neural Information Processing Systems. 2019.