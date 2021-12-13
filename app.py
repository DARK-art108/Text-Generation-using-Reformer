import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelWithLMHead

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
model = AutoModelWithLMHead.from_pretrained(r'C:\Users\lionh\Desktop\Reformer(Text Generation)\Reformer(Text Generation)\output')
def gen_text(sentence):
    generated_text = tokenizer.decode(model.generate(tokenizer.encode(sentence, return_tensors="pt"), do_sample=True,temperature=0.7, max_length=100)[0])
    return generated_text

@app.route("/", methods=['GET'])
def home():
    return render_template('INDEX.html')

@app.route("/predict", methods=['POST'])
def predictRoute():
    data = request.json['data']
    result = gen_text(data)
    print(result)
    return jsonify({"text" : result })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)