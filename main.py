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

