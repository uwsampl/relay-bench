import glob
import os
import requests
import string
import unicodedata
import zipfile

__DATA__ = None
DATA_URL = 'https://download.pytorch.org/tutorial/data.zip'
DATA_PATH = 'data'

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS) + 1
N_CATEGORIES = None
ALL_CATEGORIES = []

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def get_data():
    global N_CATEGORIES
    global __DATA__
    if __DATA__ is not None:
        return __DATA__

    if not os.path.exists(DATA_PATH):
        resp = requests.get(DATA_URL)
        with open('data.zip', 'wb') as zip_file:
            zip_file.write(resp.content)

        zip_file = zipfile.ZipFile('data.zip')
        zip_file.extractall('.')

    languages = {}
    for language in glob.glob(os.path.join(DATA_PATH , 'names', "*")):
        with open(language, encoding='utf-8') as language_file:
            category = os.path.splitext(os.path.basename(language))[0]
            ALL_CATEGORIES.append(category)
            lines = language_file.read().strip().split('\n')
            names = [unicode_to_ascii(line) for line in lines]
            languages[category] = names

    # Modify global state.
    __DATA__ = languages
    N_CATEGORIES = len(__DATA__.keys())
    return __DATA__
