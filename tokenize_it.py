from pathlib import Path
from spacy.symbols import NOUN, PROPN, ADJ, VERB
from spacy import Language
from tqdm.auto import tqdm
import itertools
from argparse import ArgumentParser
import spacy
from spacy.tokens import Span, Doc
from gensim.models import Phrases
import pandas as pd
from gensim.models.phrases import Phraser, ENGLISH_CONNECTOR_WORDS

# Adapted from McLevey https://github.com/UWNETLAB/dcss_supplementary

def tokenize_span(text_span: Span | Doc) -> list[str]:
    return [
        token.text.lower()
        for token in text_span
        if token.text != "\n" and token.is_alpha
    ]

def bigram_process(
        texts,
        nlp,
        threshold=0.75,
        scoring="npmi",
        detokenize: bool | str = True,
        n_process=1,
        pre_trained_model=None,
    ):
        sentences = []
        docs = []

        # sentence segmentation doesn't need POS tagging or lemmas.
        for i, doc in enumerate(
            nlp.pipe(
                texts, disable=["tagger", "lemmatizer", "ner"], n_process=n_process
            )
        ):
            if i % 1000 == 0:
                print(f"Training Phraser: Processed {i:,}/{len(texts):,}")
            doc_sents = [tokenize_span(sent) for sent in doc.sents]
            # the flat list of tokenized sentences for training
            sentences.extend(doc_sents)
            # the nested list of documents, with a list of tokenized sentences in each
            docs.append(doc_sents)

        if pre_trained_model is None:
            model = Phrases(
                sentences,
                min_count=5,
                threshold=threshold,
                scoring=scoring,
                connector_words=ENGLISH_CONNECTOR_WORDS,
            )  # train the model
            # create more memory and processing efficient applicator of trained model
            bigrammer = Phraser(
                model
            )  # bigrammer = model.freeze() # the same as above but for gensim 4.0 and higher
        else:
            bigrammer = pre_trained_model
        bigrammed_list = [
            [bigrammer[sent] for sent in doc] for doc in docs
        ]  # apply the model to the sentences in each doc

        if detokenize == True:
            # rejoin the tokenized sentences into strings
            bigrammed_list = [
                [" ".join(sent) for sent in doc] for doc in bigrammed_list
            ]
            # rejoin the sentences to strings in each document
            bigrammed_list = [" ".join(doc) for doc in bigrammed_list]
        elif detokenize == "sentences":
            # rejoin the tokenized sentences into strings, returning a list of documents that are each a list of sentence strings
            bigrammed_list = [
                [" ".join(sent) for sent in doc] for doc in bigrammed_list
            ]
            bigrammed_list = [
                [sent for sent in doc if len(sent.strip()) > 0]
                for doc in bigrammed_list
            ]
        else:
            # return a tokenized list of documents
            bigrammed_list = list(itertools.chain(*bigrammed_list))
        return model, bigrammed_list

def spacy_stream(texts: list[str], nlp: Language, n_process: int = 1):
        for doc in nlp.pipe(texts, disable=["ner", "parser"], n_process=n_process):
            yield doc

def tokenize(doc: Doc, detokenize: bool = True) -> list[str] | str:
        # parts of speech to tell spaCy to keep
        # https://spacy.io/usage/linguistic-features
        allowed_postags = [
            NOUN, PROPN, ADJ, VERB
        ]

        processed = [
            token.lemma_
            for token in doc
            if token.is_stop == False
            and len(token) > 1
            and token.pos in allowed_postags
        ]

        if detokenize:
            processed = " ".join(processed)

        return processed

def quick_token(documents: list, save_path:str= None) -> list[str]:
    nlp = spacy.load("en_core_web_md")
    print('Training Phraser')
    _, bigrammed_sents = bigram_process(documents, nlp=nlp, detokenize="sentences", n_process=-1)
    docs = list(map(" ".join, bigrammed_sents))

    processed_docs = []
    with tqdm(total=len(docs), desc='Phrasing and Tokenizing') as bar:
        for doc in spacy_stream(docs, nlp, n_process=-1):
            tokens = tokenize(doc, detokenize=True)
            processed_docs.append(tokens)
            bar.update(1)
    if save_path:
        with open(save_path, "w") as f:
            f.writelines("%s\n" % l for l in processed_docs)
    return processed_docs

def load_quick_tokens(path: str) -> list[str]:
    with open(path, 'r') as f:
        token_strings = f.read().split('\n')[:-1]
    return token_strings


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=str, required=True, help='.csv file containing the texts')
    argparser.add_argument('--output', type=str, required=False, default='tokenized_documents.txt', help='Path of output .txt file')
    argparser.add_argument('--text_col', type=str, required=False, default='text')
    args = argparser.parse_args()

    MESSAGE_PATH = Path(args.input)
    TOKENISED_DOCS_PATH = args.output

    df = pd.read_csv(MESSAGE_PATH)
    texts = df[args.text_col]

    quick_token(texts, save_path=TOKENISED_DOCS_PATH)

