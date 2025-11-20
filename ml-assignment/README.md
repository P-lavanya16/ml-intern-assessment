## How to Run

1. **Install Dependencies**
   
   ```bash
   pip install -r requirements.txt
2.**Prepare the Training Data**

Place your training text file inside the data/ directory.
Example:

data/input.txt


3.**Train the Trigram Model**

python src/train.py --input data/input.txt --model outputs/model.pkl


**Generate Text Using the Model**

python src/generate.py --model outputs/model.pkl --seed "your starting words" --length 50


**Evaluation (Optional)**

python src/evaluate.py --model outputs/model.pkl --test data/test.txt
