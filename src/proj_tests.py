import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, Model as KModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def _unwrap(m):
    """Retourne l'objet Keras nu (déballe un wrapper qui expose .model)."""
    return getattr(m, "model", m)

def _loss_is_sparse_cce(model):
    # TF/Keras peut stocker la loss comme string, fonction, ou instance
    loss = model.loss
    if loss is None:
        return False
    if isinstance(loss, str):
        return loss.lower() in ("sparse_categorical_crossentropy", "sparsecategoricalcrossentropy")
    if loss is sparse_categorical_crossentropy:
        return True
    if isinstance(loss, SparseCategoricalCrossentropy):
        return True
    # fallback par nom
    name = getattr(loss, "__name__", loss.__class__.__name__)
    return "sparse" in name.lower() and "categorical" in name.lower() and "crossentropy" in name.lower()

def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    model = _unwrap(model)

    # Accepte Sequential OU Functional
    assert isinstance(model, (Sequential, KModel)), \
        f"Expected a Keras model, got {type(model)}"

    assert model.input_shape == (None, *input_shape[1:]), \
        f'Wrong input shape. Found {model.input_shape} using parameter input_shape={input_shape}'

    assert model.output_shape == (None, output_sequence_length, french_vocab_size), \
        f'Wrong output shape. Found {model.output_shape} using parameters output_sequence_length={output_sequence_length} and french_vocab_size={french_vocab_size}'

    # Vérif de la loss (compatible TF 2.x)
    assert model.loss is not None, 'No loss function set. Apply the `compile` function to the model.'
    assert _loss_is_sparse_cce(model), 'Not using `sparse_categorical_crossentropy` function for loss.'

def test_tokenize(tokenize):
    sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '

def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        f'Pad returned the wrong type. Found {type(padded_tokens)}, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(
        padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)
    ), 'Using length argument return incorrect results'

def test_simple_model(simple_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344
    model = simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)

def test_embed_model(embed_model):
    input_shape = (137861, 21)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344
    model = embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)

def test_encdec_model(encdec_model):
    input_shape = (137861, 15, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344
    model = encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)

def test_bd_model(bd_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344
    model = bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)

def test_model_final(model_final):
    input_shape = (137861, 15)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344
    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)
