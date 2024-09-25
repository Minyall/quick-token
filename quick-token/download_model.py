from spacy.cli import download

def download_it():
    print('Downloading Spacy Model "en_core_web_md"')
    download('en_core_web_md')