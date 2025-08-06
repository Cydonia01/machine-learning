import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # Use legacy Keras for compatibility

import ktrain
from ktrain import text

# Creating the training and test sets
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir="aclImdb", classes=['pos', 'neg'], maxlen=500, train_test_names=['train', 'test'], preprocess_mode='bert')

# Building the BERT model
model = text.text_classifier(name='bert', train_data=(x_train, y_train), preproc=preproc)

# Training the model
learner = ktrain.get_learner(model=model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)
learner.fit_onecycle(lr=2e-5, epochs=1)

# Validating the model
learner.validate(val_data=(x_test, y_test), class_names=preproc.get_classes())

# Saving the model
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save("bert_imdb_predictor")

