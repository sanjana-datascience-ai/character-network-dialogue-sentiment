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

