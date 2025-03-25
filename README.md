# Natural Language Understanding (NLU) for Named Entity Recognition (NER)

## 📌 Overview
This project implements **Named Entity Recognition (NER)** using **BERT (Bidirectional Encoder Representations from Transformers)**. It trains a model to identify entities such as **names, locations, organizations, and more** from text data.

## 📚 Project Structure
- **`NLU.py`** – The main script for data preprocessing, model training, evaluation, and inference.
- **`ner_dataset_fixed.csv`** – A labeled dataset used for training the NER model.

## 🔧 Features
✔️ **Preprocessing**: Converts raw text into structured format for training.  
✔️ **BERT-based Model**: Uses `bert-base-uncased` for high-accuracy entity recognition.  
✔️ **NER Chatbot**: Interactive chatbot that detects entities from user input.  
✔️ **Evaluation Metrics**: Uses **F1-score, Accuracy, and Classification Report** to assess model performance.  

## 📊 Dataset
The dataset is a CSV file with:
- `sentence_id`: Groups words into sentences.
- `word`: The actual word/token.
- `tag`: The entity label (e.g., `B-PER`, `I-LOC`, `O` for non-entity words).

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Nimra71/InternIntelligence_NaturalLanguageUnderstanding.git
cd InternIntelligence_NaturalLanguageUnderstanding
```
### **2️⃣ Install Dependencies**
```bash
pip install pandas torch transformers datasets scikit-learn
```
### **3️⃣ Train the Model**
Run the `NLU.py` script to train the NER model:
```bash
python NLU.py
```

## 📊 Model Training
- Uses **Hugging Face Transformers** to train `BERT` for NER.
- Data is tokenized using **BertTokenizerFast**.
- Training is performed using **Trainer API** with `AdamW` optimizer.
- Model is saved to `./ner_model` after training.

## 🤖 Using the NER Chatbot
After training, the chatbot will detect entities in user-provided text:
```bash
python NLU.py
```
Example:
```
You: Elon Musk is the CEO of Tesla.
Entities: [('Elon Musk', 'PER'), ('Tesla', 'ORG')]
```

## 📊 Evaluation
The model is evaluated using:
- **Accuracy**
- **F1-Score**
- **Classification Report**
Run evaluation:
```bash
python NLU.py  # The script includes automatic evaluation
```

## 🚀 Future Improvements
- Expand dataset for better generalization.
- Fine-tune on domain-specific data (e.g., legal, medical texts).
- Deploy as a web API using **FastAPI** or **Flask**.

## 📝 License
This project is licensed under the **MIT License**.

---

### 🔗 **Connect with Me**
🌟 GitHub: [Nimra71](https://github.com/Nimra71)  
📧 Email: [nimrafatima745@gmail.com]  


