# Quick Token

A set of functions to rapidly generate phrased tokens for document representation.

## Installation
Two steps

### 1. Install via Pip as usual
```
pip install quick-token

```
### 2. Install the Spacy Model
PyPi prohibits direct inclusion of this model in the requirements, so you'll need to run this commmand to ensure you have the spacy model.

```
python -m spacy download en_core_web_md

```
# Basic Example
```
from quick_token import quick_token
texts = ["Hello there", "how are you today", "I'm feeling great thank you!"]
tokens = quick_token(texts)
> ['', 'today', 'feel great thank']
```