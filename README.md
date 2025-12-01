# EmoDynamics  
### *Character Emotion Detection, Influence Modeling & Network Visualization from TV Show Dialogues*

EmoDynamics is an end-to-end NLP project that analyzes the **emotional dynamics between characters** using real dialogue data from the TV show **Friends**.  
It extracts emotions, models how emotions propagate from one speaker to the next, and visualizes the full character interaction network with emotional context.

---

# Features

### 1. **Utterance-level Emotion Detection**
- Uses **DistilRoBERTa emotion classifier** (`j-hartmann/emotion-english-distilroberta-base`)
- Fixes missing/ambiguous labels like `"non-neutral"`
- Includes cleaning, lemmatization, POS tagging, and NER extraction

### 2. **Emotion Influence Prediction**
Learns:
> *Given Person A’s utterance + emotion, what emotion will Person B respond with?*

- Trained using A → B conversational pairs  
- Input formatted as:  [SRC_EMO=anger] Why did you do that?
- Built using Transformers Trainer API

### 3. **Character Network Graph**
- Nodes = characters  
- Edges = dialogue interactions  
- Edge weight = number of exchanges  
- Edge color = dominant emotion  
- Node style = circular images of characters  
- Fully interactive via PyVis

### 4. **Analytics Dashboard**
- Emotion distribution
- Utterances per speaker
- Dialogue length insights
- Emotion × speaker cross-sections
- Word clouds & n-grams
- Transition matrices & Sankey-style emotion flows
- Ego Networks

### 5. **Streamlit Web Application**
- Home page + hero section  
- Dashboard (Dataset, Network, Emotion Influence, Text Analytics)  
- Real-time **Emotion Influence Prediction Tool**

---

# Project Structure
```pgsql
character-network-dialogue-sentiment/
│
├── data/
│ ├── Raw/
│ │ └── friends.json
│ ├── friends.csv
│ ├── friends_preprocessed.csv
│ └── friends_pairs_balanced.csv
│
├── models/
│ └── emotion_influence/
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer.json
│ ├── label_mapping.txt
│ └── test_classification_report.txt
│
├── src/
│ ├── preprocessing/
│ │ ├── json_csv.py
│ │ ├── preprocessing.py
│ │ └── pairs.py
│ ├── model/
│ │ └── emotion_influence_model.py
│ └── app/
│ └── streamlit_app.py
│
└── README.md
```

---

# Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/character-network-dialogue-sentiment.git
cd character-network-dialogue-sentiment
```

### 2. Create the Environment
```bash
conda env create -f environment.yml
conda activate nlp_basics
```
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader all
```

### 3. Run files 
#### Convert JSON → CSV
```bash
python src/preprocessing/json_csv.py
```

#### Preprocessing
```bash
python src/preprocessing/preprocessing.py
```

#### Build Conversational Pairs
```bash
python src/preprocessing/pairs.py
```

#### Train Emotion Influence Model
```bash
python src/model/emotion_influence_model.py
```

#### Web Interface
```bash
streamlit run src/app/streamlit_app.py
```

## Visualizations Included
- Emotion distribution (bar + pie)
- Utterances per speaker
- Dialogue lengths
- Emotion × speaker patterns
- Word clouds per emotion
- Top bigrams
- Token length per emotion
- Empirical emotion transition matrix
- Model-based transition results
- Character Network (PyVis)
- Ego Networks
- Adjacency matrices
- Centrality plots (degree, betweenness)

Each visualization answers a key question about emotional behavior in conversations.

## Emotion Influence Model
Input format:
```vbnet
[SRC_EMO=sadness] I don't know why this keeps happening to me.
```
Output:
``` lua
Emotion Detected for Person X: {emotion_x}
Predicted Reaction:
"In reaction to this utterance, it's {confidence_emotion_y} likely that the next person will respond in {emotion_y}."
```

# About the Author
* Sanjana R
* 4th year Student
* B.Tech (Hons) Data Science
* Vidyashilp University, Bengaluru




# Multimodal-to-Unimodal Knowledge Distillation for Sarcasm Detection (MUStARD)

## Abstract
This project builds a lightweight **audio-only sarcasm detection model** by leveraging **knowledge distillation** from a multimodal **Text + Vision teacher model**. 
The teacher is first trained on a large **Meme dataset**, then **fine-tuned on the MUStARD dataset**.
We then distill this knowledge into efficient **Transformer** and **CNN** audio students using several advanced KD techniques.

The project includes:
- Baseline Unimodal, Bimodal, Trimodal models 
- Multimodal teacher models across **6 fusion strategies** 
- Fine-tuning on MUStARD 
- Student distillation with **5-fold cross-validation** 
- Streamlit-based real-time sarcasm detection using audio

The final best model: 
### CNN Student distilled from the **Late Fusion** Teacher (Top performer)


---

# Project Structure
bash
```
MML_Project/
│
├── Multimodal_Final_Datasets/
│ └── MUStARD/
├── Models/
│ ├── Teacher/
│ └── Student/
├── Outputs/
│ ├── Teacher/
│ └── Student/
├── src/
│ ├── Teacher_Model/
│ ├── Student_Model/
│ └── Baselines/
└── apps/
├── app_student_transformer_sum.py
└── app_student_cnn_late.py
```

# Baseline Models

## Summary
We implemented and evaluated five baseline architectures on the MUStARD dataset:

| Model Type | Modalities |
|-----------|-----------|
| Unimodal | Audio |
| Bimodal | Audio+Text, Audio+Vision, Text+Vision |
| Trimodal | Audio+Text+Vision |

These establish performance benchmarks before applying knowledge distillation.

## Run Commands
Move to baselines folder:
bash
```
cd src/Baselines/
```

# Unimodal Audio
bash
```
python Unimodal/Audio_Unimodal.py
```

# Bimodal Models
bash
```
python Bimodal/Audio_Text/Audio_Text_Bimodal.py
python Bimodal/Audio_Vision/Audio_Vision_Bimodal.py
python Bimodal/Text_Vision/Text_Vision_Bimodal.py
```

# Trimodal
bash
```
python Trimodal/trimodal.py
```

### Outputs saved under:
swift
```
Outputs/Baselines/
```

# Teacher Model (Text + Vision)

### Summary

A multimodal teacher model was trained using RoBERTa-base (Text) + ResNet50 (Vision), fused using 6 fusion methods:
 - concat
 - sum
 - prod
 - soft_attn
 - hard_attn
 - late

After Meme dataset training, teachers were fine-tuned on MUStARD.

## Run Commands
Prepare MUStARD images
bash
```
cd src/Teacher_Model/
python prepare_mustard_images.py
```

Train teachers on Meme dataset
bash
```
python Text_Vision/fusion_comparison.py
```

# Fine-tuning Teachers on MUStARD

### Summary

We fine-tuned all 6 fusion teachers on MUStARD using an 80/20 split.
Saved results include:
- Best checkpoint
- Training/validation metrics
- Loss/accuracy/F1 curves

## Run Command
bash
```
cd src/Teacher_Model/
python train_teacher_mustard.py
```

### Checkpoints saved under:
swift
```
Models/Teacher/Text_Vision_MUSTARD/<fusion>/
```
# Student Model (Audio-Only)
## Summary
Two student architectures were trained using Knowledge Distillation:
Transformer Audio Student
- CNN Audio Student
- KD enhancements used:
- Teacher soft probabilities
- Temperature scaling (T=6)
- CE-dominant loss (α = 0.8)
- Warmup (first 5 epochs)
- Label smoothing
- Normalized embeddings
- Gradient clipping
- Early stopping
- 5-Fold Cross Validation

We trained:6 fusions × 2 models × 5 folds = 60 KD experiments

## Run Commands

Audio Preprocessing
bash
```
cd src/Student_Model/
python audio_preprocess_mustard.py
```

Transformer Student (5-Fold CV)
bash
```
python audio_student_kd_transformer_cv.py
```

CNN Student (5-Fold CV)
bash
```
python audio_student_kd_cnn_cv.py
```

All results saved under:
swift
```
Outputs/Student/Audio_MUSTARD/

```

# Model Comparison & Ranking
## Compare Transformer CV Results
bash
```
python src/Student_Model/comparison/compare_transformer_cv_results.py
```

## Compare CNN CV Results
bash
```
python src/Student_Model/comparison/compare_cnn_cv_results.py
```

## Final Cross-Model Comparison
bash
```
python src/evaluation/compare_all_models.py
```

Final files generated:
swift
```
final_model_comparison.csv
final_detailed_foldwise_comparison.csv
```

# Streamlit Inference Applications

### Summary
Two apps for real-time sarcasm prediction from audio:
- Transformer Sum Student
- CNN Late Student (best)

User uploads audio → embedding generated → model predicts sarcasm with probability.

## Run Commands
Transformer Sum App
bash
```
streamlit run apps/app_student_transformer_sum.py
```

CNN Late App (Best Model)
bash
```
streamlit run apps/app_student_cnn_late.py
```
