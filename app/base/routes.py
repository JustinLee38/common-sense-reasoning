# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import jsonify, render_template, redirect, request, url_for, make_response
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

from app import db, login_manager
from app.base import blueprint
from app.base.forms import LoginForm, CreateAccountForm
from app.base.models import User
from app.base.models import SubtaskA
from app.base.models import SubtaskB
from app.base.models import SubtaskC

from app.base.util import verify_pass

from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import torch
import os

model = BertForSequenceClassification.from_pretrained("model_save/")
tokenizer = BertTokenizer.from_pretrained("model_save/")

tokenizerC = GPT2Tokenizer.from_pretrained("gpt2-medium")
modelC = GPT2LMHeadModel.from_pretrained("my-model/")

@blueprint.route('/subtaskA', methods=['GET'])
def user_records():

    return render_template(
        'subTaskA.html',
        subtaskA=SubtaskA.query.all()
    )

@blueprint.route('/subtaskB', methods=['GET'])
def subTaskB():
  
    return render_template(
        'subTaskB.html',
        subtaskB=SubtaskB.query.all()
    )

@blueprint.route('/subtaskC', methods=['GET'])
def subTaskC():

    return render_template(
        'subTaskC.html',
        subtaskC=SubtaskC.query.all()
    )

@blueprint.route('/predictA',methods=['POST'])
def predictA():
    sent0 = request.form.get('sent0')
    sent1 = request.form.get('sent1')
    test1 = tokenizer.encode_plus(sent0,sent1, return_tensors="pt")
    test1_logits = model(**test1)[0]
    test1_result = torch.softmax(test1_logits, dim=1).tolist()[0]

    sent0Accuracy = round(test1_result[0]* 100)
    if sent0Accuracy > 50:
        makeSense = 0
        output = f"The model is {sent0Accuracy}% confidence that sentence 1 makes sense."
    else:
        makeSense = 1
        output = f"The model is {100-sent0Accuracy}% confidence that sentence 2 makes sense."

    

    return render_template('subTaskA.html', subtaskA=SubtaskA.query.all(), prediction_text=output, makeSense = makeSense, sent0= sent0, sent1 = sent1)

@blueprint.route('/predictB',methods=['POST'])
def predictB():
    falseSent = request.form.get('falseSent')
    sent0 = request.form.get('sent0')
    sent1 = request.form.get('sent1')
    sent2 = request.form.get('sent2')
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    # output = round(prediction[0], 2)
    flag = 1
    
    return render_template('subTaskB.html', flag= flag, falseSent= falseSent, sent0=sent0, sent1 = sent1, sent2= sent2, subtaskB=SubtaskB.query.all())

@blueprint.route('/predictC',methods=['POST'])
def predictC():
    falseSent = request.form.get('falseSent')
    falseSentE = falseSent + " <SEP>"

    break_mode = "token"
    lst = {}

    if break_mode == "string":
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens': ['<SEP>']}
        
    elif break_mode == "token":
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}

    num_added_toks = tokenizerC.add_special_tokens(special_tokens_dict)
    modelC.resize_token_embeddings(len(tokenizerC))
    lst = run(falseSentE)

    return render_template('subTaskC.html', subtaskC=SubtaskC.query.all(), prediction_text=lst, chosenSent=falseSent)

@blueprint.route('/')
def route_default():
    return render_template(
        'subTaskA.html',
        subtaskA=SubtaskA.query.all()
    )


## Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html'), 403

@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403

@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404

@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500

def run(prompt_text):
    break_mode = "token"

    encoded_prompt = tokenizerC.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
    #encoded_prompt = encoded_prompt.to(device)

    output_sequences = modelC.generate(
        input_ids=encoded_prompt,
        max_length=20 + len(encoded_prompt[0]),
        temperature=0.7,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=10,
    )

 #   Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizerC.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token 
        if break_mode == "string":
          text = text[: text.find('doesn\'t', text.find('doesn\'t')+1)]
          text = text[: text.find('.')+1]
        elif break_mode == "token":
          text = text[: text.find('<SEP>', text.find('<SEP>')+1)]
          text = text[: text.find('<EOS>')]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
         text[len(tokenizerC.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
    return generated_sequences
