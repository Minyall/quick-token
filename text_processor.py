from pathlib import Path
from typing import List, Union

from spacy import Language
from tqdm import tqdm

if __name__ == "__main__":
    import itertools
    from argparse import ArgumentParser
    import spacy
    from spacy.tokens import Span, Doc, DocBin
    from gensim.models import Phrases
    import pandas as pd
    from gensim.models.phrases import Phraser, ENGLISH_CONNECTOR_WORDS
    from utilities import build_logger, flatten

    argparser = ArgumentParser()
    argparser.add_argument('--input', type=str, required=True, help='.csv file containing the texts and unique IDs')
    argparser.add_argument('--output', type=str, required=True, help='name of output dir')
    argparser.add_argument('--text_col', type=str, required=False, default='text')
    argparser.add_argument('--id_col', type=str, required=False, default='id')
    args = argparser.parse_args()

    DATA_DIR = Path(args.output)

    MESSAGE_PATH = Path(args.input)
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)
    logger = build_logger("main", "log.log")

    TOKENISED_DOCS_PATH = Path(DATA_DIR, "tokenised_documents.txt")
    TOKENISED_SENTS_PATH = Path(DATA_DIR, "tokenised_sentences.txt")
    SENTENCE_INDEX_PATH = Path(DATA_DIR, "sentence_index.parquet")

    PHRASER_PATH = str(Path(MODELS_DIR, "phraser.bin"))

    nlp = spacy.load("en_core_web_md")

    # Adapted from McLevey https://github.com/UWNETLAB/dcss_supplementary

    def tokenize_span(text_span: Union[Span, Doc]) -> List[str]:
        return [
            token.text.lower()
            for token in text_span
            if token.text != "\n" and token.is_alpha
        ]

    def bigram_process(
        texts,
        nlp=nlp,
        threshold=0.75,
        scoring="npmi",
        detokenize: Union[bool, str] = True,
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
                logger.info(f"Training Phraser: Processed {i:,}/{len(texts):,}")
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

    def spacy_stream(texts: List[str], nlp: Language, n_process: int = 1):

        for doc in nlp.pipe(texts, disable=["ner", "parser"], n_process=n_process):
            yield doc

    def tokenize(doc: Doc, detokenize: bool = True) -> Union[List[str], str]:
        # parts of speech to tell spaCy to keep
        allowed_postags = [
            92,
            96,
            84,
        ]  # equivalent to: allowed_postags = ['NOUN', 'PROPN', 'ADJ']

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

    df = pd.read_csv(MESSAGE_PATH)
    # Convert ID fields to nullable integers
    id_columns = [args.id_col]

    # Using convert_dtypes to automatically convert to the most suitable nullable types
    df[id_columns] = df[id_columns].convert_dtypes()


    texts = df[args.text_col]
    id_lookup = df.index.tolist()

    model, bigrammed_sents = bigram_process(texts, detokenize="sentences", n_process=-1)

    sentence_directory = [
        (doc_idx, sent_idx, id_lookup[doc_idx])
        for doc_idx, doc in enumerate(bigrammed_sents)
        for sent_idx, sent in enumerate(doc)
    ]

    docs_total = len(bigrammed_sents)
    docs = map(" ".join, bigrammed_sents)

    sentences_total = len(list(flatten(bigrammed_sents)))
    sentences = flatten(bigrammed_sents)

    docs_tracker = tqdm(total=docs_total)
    store = DocBin()
    file_iter = 1

    processed_docs = []

    for i, doc in enumerate(spacy_stream(docs, nlp, n_process=-1), start=1):
        if i % 10000 == 0:
            file_iter += 1
        # store.add(doc)
        tokens = tokenize(doc, detokenize=True)
        processed_docs.append(tokens)
        docs_tracker.update(1)

    with open(TOKENISED_DOCS_PATH, "w") as f:
        f.writelines("%s\n" % l for l in processed_docs)

    sents_tracker = tqdm(total=sentences_total)
    processed_sents = []
    for i, sent in enumerate(spacy_stream(sentences, nlp, n_process=-1), start=1):
        tokens = tokenize(sent, detokenize=True)
        processed_sents.append(tokens)
        sents_tracker.update(1)

    model.save(PHRASER_PATH)

    data = pd.DataFrame(sentence_directory, columns=["doc", "sent", "unique_id"])
    data["text"] = processed_sents
    data = data[data["text"].str.len() > 0]
    data.reset_index(inplace=True, drop=True)
    data[["doc", "sent", "unique_id", "text"]].to_parquet(SENTENCE_INDEX_PATH)

    with open(TOKENISED_SENTS_PATH, "w") as f:
        f.writelines("%s\n" % l for l in data["text"].tolist())
