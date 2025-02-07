import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"


import keras_nlp
import tensorflow as tf
import keras

# Download pretraining data.
wiki_dir = os.path.expanduser("datasets/wikitext-103/")
# Download finetuning data.
sst_dir = os.path.expanduser("datasets/SST-2/")
#Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# Training params.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3

# Load SST-2.
sst_train_ds = tf.data.experimental.CsvDataset(
    sst_dir + "train.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)

# Take a peak at the sst-2 dataset.
print(sst_train_ds.unbatch().batch(4).take(1).get_single_element())

# This layer will turn our input sentence into a list of 1s and 0s the same size
# our vocabulary, indicating whether a word is present in absent.
multi_hot_layer = keras.layers.TextVectorization(
    max_tokens=4000, output_mode="multi_hot"
)
multi_hot_layer.adapt(sst_train_ds.map(lambda x, y: x))
multi_hot_ds = sst_train_ds.map(lambda x, y: (multi_hot_layer(x), y))
multi_hot_val_ds = sst_val_ds.map(lambda x, y: (multi_hot_layer(x), y))

# We then learn a linear regression over that layer, and that's our entire
# baseline model!

inputs = keras.Input(shape=(4000,), dtype="int32")
outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
baseline_model = keras.Model(inputs, outputs)
baseline_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
baseline_model.fit(multi_hot_ds, validation_data=multi_hot_val_ds, epochs=5)



# Setting sequence_length will trim or pad the token outputs to shape
# (batch_size, SEQ_LENGTH).
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)
# Setting mask_selection_length will trim or pad the mask outputs to shape
# (batch_size, PREDICTIONS_PER_SEQ).
masker = keras_nlp.layers.MaskedLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=MASK_RATE,
    mask_selection_length=PREDICTIONS_PER_SEQ,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
)


def preprocess(inputs):
    inputs = tokenizer(inputs)
    outputs = masker(inputs)
    # Split the masking layer outputs into a (features, labels, and weights)
    # tuple that we can use with keras.Model.fit().
    features = {
        "token_ids": outputs["token_ids"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    weights = outputs["mask_weights"]
    return features, labels, weights


# We use prefetch() to pre-compute preprocessed batches on the fly on the CPU.
pretrain_ds = wiki_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pretrain_val_ds = wiki_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
# The masks will change each time you run the cell.
#print(pretrain_val_ds.take(1).get_single_element())

inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# Embed our tokens with a positional embedding.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM,
)
outputs = embedding_layer(inputs)

# Apply layer normalization and dropout to the embedding.
outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(outputs)
outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

# Add a number of encoder blocks
for i in range(NUM_LAYERS):
    outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        layer_norm_epsilon=NORM_EPSILON,
    )(outputs)

encoder_model = keras.Model(inputs, outputs)
encoder_model.summary()

# Create the pretraining model by attaching a masked language model head.
inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "mask_positions": keras.Input(
        shape=(PREDICTIONS_PER_SEQ,), dtype="int32", name="mask_positions"
    ),
}

# Encode the tokens.
encoded_tokens = encoder_model(inputs["token_ids"])

# Predict an output word for each masked input token.
# We use the input token embedding to project from our encoded vectors to
# vocabulary logits, which has been shown to improve training efficiency.
outputs = keras_nlp.layers.MaskedLMHead(
    token_embedding=embedding_layer.token_embedding,
    activation="softmax",
)(encoded_tokens, mask_positions=inputs["mask_positions"])

# Define and compile our pretraining model.
pretraining_model = keras.Model(inputs, outputs)
pretraining_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(PRETRAINING_LEARNING_RATE),
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)

# Pretrain the model on our wiki text dataset.
pretraining_model.fit(
    pretrain_ds,
    validation_data=pretrain_val_ds,
    epochs=PRETRAINING_EPOCHS,
)

# Save this base model for further finetuning.
encoder_model.save("encoder_model.keras")

def preprocess(sentences, labels):
    return tokenizer(sentences), labels


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
finetune_ds = sst_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
finetune_val_ds = sst_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
print(finetune_val_ds.take(1).get_single_element())

# Reload the encoder model from disk so we can restart fine-tuning from scratch.
encoder_model = keras.models.load_model("encoder_model.keras", compile=False)

# Take as input the tokenized input.
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# Encode and pool the tokens.
encoded_tokens = encoder_model(inputs)
pooled_tokens = keras.layers.GlobalAveragePooling1D()(encoded_tokens[0])

# Predict an output label.
outputs = keras.layers.Dense(1, activation="sigmoid")(pooled_tokens)

# Define and compile our fine-tuning model.
finetuning_model = keras.Model(inputs, outputs)
finetuning_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(FINETUNING_LEARNING_RATE),
    metrics=["accuracy"],
)

# Finetune the model for the SST-2 task.
finetuning_model.fit(
    finetune_ds,
    validation_data=finetune_val_ds,
    epochs=FINETUNING_EPOCHS,
)

