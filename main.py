from transformers import pipeline
import stanza

# PyTorch Install command: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

category_propensity_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=0)
# aggregation_strategy could be 'none' or 'simple'
entity_recognition_model = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy='simple', device=0)
stanford_text_analysis = stanza.Pipeline('en', processors='tokenize,pos,ner')


def label_tendencies(sequence: str, labels: list[str]) -> list[tuple]:
    propensities = category_propensity_model(sequence, labels)
    return list(zip(propensities["labels"], propensities["scores"]))


def emotional_tendencies(sequence: str) -> list[tuple]:
    e = emotion_classifier(sequence)[0]
    return [(item['label'], item['score']) for item in e]


def recognized_entities(sequence: str):
    return entity_recognition_model(sequence)


def stanford_stanza(sequence: str) -> list[tuple]:
    doc = stanford_text_analysis(sequence)
    return [(word['text'], word['upos'], word['ner'], f"{word['start_char']}:{word['end_char']}")
            for sentence in doc.sentences
            for word in sentence.to_dict()]


if __name__ == '__main__':
    # text = "Hello, We have tried to send you this email as HTML (pictures and words) but it wasn't possible. In order for you to see what we had hoped to show you please click here to view online in your browser:"
    text = "the egyptians and the greeks were of the opinion that man primarily sought experience consciously and willingly. these peoples did not have a fall in their theology as some other nations had this concept of man being driven out of paradise"
    prediction = label_tendencies(text, ['urgent', 'sensitive', 'threat', 'information', 'financial', 'problem', 'account', 'empty'])
    emotions = emotional_tendencies(text)
    print(f'Sample: {text}\n{prediction}\n{emotions}')
    print(recognized_entities(text))
    print(stanford_stanza(text))

# Sample: the egyptians and the greeks were of the opinion that man primarily sought experience consciously and willingly. these peoples did not have a fall in their theology as some other nations had this concept of man being driven out of paradise
# [('account', 0.29101645946502686), ('information', 0.2796013355255127), ('threat', 0.13327176868915558), ('sensitive', 0.11280672997236252), ('problem', 0.07923632860183716), ('urgent', 0.07758237421512604), ('empty', 0.01471769530326128), ('financial', 0.011767369695007801)]
# [('anger', 0.06187498942017555), ('disgust', 0.015549559146165848), ('fear', 0.03018961474299431), ('joy', 0.03844471648335457), ('neutral', 0.7493038773536682), ('sadness', 0.039836011826992035), ('surprise', 0.06480120122432709)]
# [{'entity_group': 'MISC', 'score': 0.9006633, 'word': 'gre', 'start': 22, 'end': 25}]
# [('the', 'DET', 'O', '0:3'), ('egyptians', 'NOUN', 'O', '4:13'), ('and', 'CCONJ', 'O', '14:17'), ('the', 'DET', 'O', '18:21'), ('greeks', 'NOUN', 'O', '22:28'), ('were', 'AUX', 'O', '29:33'), ('of', 'ADP', 'O', '34:36'), ('the', 'DET', 'O', '37:40'), ('opinion', 'NOUN', 'O', '41:48'), ('that', 'SCONJ', 'O', '49:53'), ('man', 'NOUN', 'O', '54:57'), ('primarily', 'ADV', 'O', '58:67'), ('sought', 'VERB', 'O', '68:74'), ('experience', 'NOUN', 'O', '75:85'), ('consciously', 'ADV', 'O', '86:97'), ('and', 'CCONJ', 'O', '98:101'), ('willingly', 'ADV', 'O', '102:111'), ('.', 'PUNCT', 'O', '111:112'), ('these', 'DET', 'O', '113:118'), ('peoples', 'NOUN', 'O', '119:126'), ('did', 'AUX', 'O', '127:130'), ('not', 'PART', 'O', '131:134'), ('have', 'VERB', 'O', '135:139'), ('a', 'DET', 'O', '140:141'), ('fall', 'NOUN', 'O', '142:146'), ('in', 'ADP', 'O', '147:149'), ('their', 'PRON', 'O', '150:155'), ('theology', 'NOUN', 'O', '156:164'), ('as', 'SCONJ', 'O', '165:167'), ('some', 'DET', 'O', '168:172'), ('other', 'ADJ', 'O', '173:178'), ('nations', 'NOUN', 'O', '179:186'), ('had', 'VERB', 'O', '187:190'), ('this', 'DET', 'O', '191:195'), ('concept', 'NOUN', 'O', '196:203'), ('of', 'ADP', 'O', '204:206'), ('man', 'NOUN', 'O', '207:210'), ('being', 'AUX', 'O', '211:216'), ('driven', 'VERB', 'O', '217:223'), ('out', 'ADP', 'O', '224:227'), ('of', 'ADP', 'O', '228:230'), ('paradise', 'NOUN', 'O', '231:239')]
