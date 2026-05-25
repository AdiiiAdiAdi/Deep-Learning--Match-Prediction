import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional,
    Dropout, BatchNormalization, Concatenate, Layer, Embedding, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Custom Attention Layer ────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attn_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attn_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e       = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha   = tf.keras.activations.softmax(e, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


# ── Custom Loss ───────────────────────────────────────────────────────────────
# FIX: focal_loss must be a registered top-level callable so Keras can
# serialize and deserialize it when saving/loading the .keras model.
# Using a nested closure (the old approach) produces an unregistered
# 'function' named 'loss', which causes:
#   TypeError: Could not locate function 'loss'
# on load_model() in evaluate.py and app.py.
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    y_true        = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=3)
    epsilon       = 1e-7
    y_pred        = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true_onehot * tf.math.log(y_pred)
    weight        = tf.pow(1 - y_pred, gamma)
    return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))


# ── Build model ───────────────────────────────────────────────────────────────
def build_train_model(n_timesteps, n_features, n_static):
    lstm_input   = Input(shape=(n_timesteps, n_features), name='lstm_input')
    static_input = Input(shape=(n_static,),               name='static_input')
    home_input   = Input(shape=(1,),                      name='home_team')
    away_input   = Input(shape=(1,),                      name='away_team')

    num_teams = 100  # Max number of teams

    home_emb = Embedding(num_teams, 16)(home_input)
    away_emb = Embedding(num_teams, 16)(away_input)

    home_emb = Flatten()(home_emb)
    away_emb = Flatten()(away_emb)

    # LSTM branch
    x        = Bidirectional(LSTM(128, return_sequences=True))(lstm_input)
    x        = Dropout(0.15)(x)
    lstm_seq = Bidirectional(LSTM(64, return_sequences=True))(x)

    context     = AttentionLayer(name='attention')(lstm_seq)
    x           = Dense(64, activation='relu')(context)
    x           = BatchNormalization()(x)
    lstm_branch = Dropout(0.3)(x)

    # ANN branch
    s          = Dense(64, activation='relu')(static_input)
    s          = BatchNormalization()(s)
    s          = Dropout(0.3)(s)
    s          = Dense(32, activation='relu')(s)
    ann_branch = Dropout(0.2)(s)

    # Merge
    merged = Concatenate()([
        lstm_branch, 
        ann_branch,
        home_emb,
        away_emb
    ])
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(64, activation='relu')(merged)

    main_output = Dense(3, activation='softmax', name='main_output')(merged)

    model = Model(
        inputs=[lstm_input, static_input, home_input, away_input],
        outputs=main_output,
        name='football_predictor'
    )
    return model


def get_attention_weights(model, X_seq, X_static, home_team, away_team):
    lstm_seq_layer  = model.get_layer('bidirectional_1')
    attn_layer      = model.get_layer('attention')

    lstm_seq_output = lstm_seq_layer.output
    attn_scores = tf.keras.layers.Lambda(
        lambda z: tf.keras.activations.softmax(
            tf.keras.activations.tanh(
                tf.tensordot(z, attn_layer.W, axes=1) + attn_layer.b
            ), axis=1
        ), name='attn_scores_output'
    )(lstm_seq_output)

    attn_extractor = Model(
        inputs=model.inputs,
        outputs=attn_scores,
        name='attn_extractor'
    )

    weights = attn_extractor.predict(
        {
            'lstm_input': X_seq, 
            'static_input': X_static,
            'home_team': home_team,
            'away_team': away_team
        },
        verbose=0
    )
    return weights


# ── Training ──────────────────────────────────────────────────────────────────
def train_model():

    # 1. Load data
    print("1. Loading preprocessed data...")
    X_seq_train    = np.load(os.path.join(PROCESSED_DIR, 'X_seq_train.npy'),    allow_pickle=True)
    X_seq_val      = np.load(os.path.join(PROCESSED_DIR, 'X_seq_val.npy'),      allow_pickle=True)
    X_seq_test     = np.load(os.path.join(PROCESSED_DIR, 'X_seq_test.npy'),     allow_pickle=True)
    X_static_train = np.load(os.path.join(PROCESSED_DIR, 'X_static_train.npy'), allow_pickle=True)
    X_static_val   = np.load(os.path.join(PROCESSED_DIR, 'X_static_val.npy'),   allow_pickle=True)
    X_static_test  = np.load(os.path.join(PROCESSED_DIR, 'X_static_test.npy'),  allow_pickle=True)
    home_train = np.load(os.path.join(PROCESSED_DIR, 'home_team_train.npy'))
    home_val   = np.load(os.path.join(PROCESSED_DIR, 'home_team_val.npy'))
    home_test  = np.load(os.path.join(PROCESSED_DIR, 'home_team_test.npy'))
    away_train = np.load(os.path.join(PROCESSED_DIR, 'away_team_train.npy'))
    away_val   = np.load(os.path.join(PROCESSED_DIR, 'away_team_val.npy'))
    away_test  = np.load(os.path.join(PROCESSED_DIR, 'away_team_test.npy'))

    y_train        = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_val          = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
    y_test         = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

    X_seq_train    = X_seq_train.astype(np.float32)
    X_seq_val      = X_seq_val.astype(np.float32)
    X_seq_test     = X_seq_test.astype(np.float32)
    X_static_train = X_static_train.astype(np.float32)
    X_static_val   = X_static_val.astype(np.float32)
    X_static_test  = X_static_test.astype(np.float32)
    y_train        = y_train.astype(np.int32)
    y_val          = y_val.astype(np.int32)
    y_test         = y_test.astype(np.int32)

    home_train = home_train.reshape(-1)
    home_val   = home_val.reshape(-1)
    home_test  = home_test.reshape(-1)
    away_train = away_train.reshape(-1)
    away_val   = away_val.reshape(-1)
    away_test  = away_test.reshape(-1)

    print(home_train.shape, away_train.shape)

    n_timesteps = X_seq_train.shape[1]
    n_features  = X_seq_train.shape[2]
    n_static    = X_static_train.shape[1]

    print(f"   Sequence shape : {X_seq_train.shape}")
    print(f"   Static shape   : {X_static_train.shape}")
    print(f"   Label counts   : {np.bincount(y_train.astype(int))}  (0=Away / 1=Draw / 2=Home)")

    # 2. Sample weights
    print("2. Computing class weights...")
    classes       = np.unique(y_train)
    weights       = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"   {class_weights}")
    sample_weights_train = np.array([class_weights[int(l)] for l in y_train])

    # 3. Build
    print("3. Building Hybrid LSTM + ANN model...")
    model = build_train_model(n_timesteps, n_features, n_static)

    # 4. Compile
    # FIX: Pass focal_loss directly (registered top-level function).
    # The old nested closure approach saved an unregistered inner function
    # named 'loss', crashing load_model() with "Could not locate function 'loss'".
    print("4. Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=focal_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_acc")
        ]
    )
    model.summary()

    # 5. Callbacks
    print("5. Setting up callbacks...")
    best_model_path = os.path.join(MODELS_DIR, 'best_model.keras')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(MODELS_DIR, 'training_history.csv'))
    ]

    # 6. Train
    print("6. Training model...")
    history = model.fit(
        x={
            'lstm_input': X_seq_train, 
            'static_input': X_static_train,
            'home_team': home_train,
            'away_team': away_train
        },
        y=y_train,
        validation_data=(
            {
                'lstm_input': X_seq_val, 
                'static_input': X_static_val,
                'home_team': home_val,
                'away_team': away_val
            },
            y_val
        ),
        epochs=100,
        batch_size=32,
        sample_weight=sample_weights_train,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Save model
    print("7. Saving model...")
    final_path = os.path.join(MODELS_DIR, 'final_model.keras')
    model.save(final_path)
    print(f"   Saved → {final_path}")

    model_info = {
        'n_timesteps'    : int(n_timesteps),
        'n_features'     : int(n_features),
        'n_static'       : int(n_static),
        'class_map'      : {'0': 'Away Win', '1': 'Draw', '2': 'Home Win'},
        'model_path'     : final_path,
        'best_model_path': best_model_path
    }
    with open(os.path.join(MODELS_DIR, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    # 8. Attention weights
    print("8. Extracting attention weights on test set...")
    attn_weights = get_attention_weights(model, X_seq_test, X_static_test, home_test, away_test)
    np.save(os.path.join(MODELS_DIR, 'test_attention_weights.npy'), attn_weights)
    print(f"   Attention weights shape: {attn_weights.shape}")

    # 9. Plot training curves
    print("9. Plotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    best_ep = int(np.argmin(history.history['val_loss']))

    axes[0].plot(history.history['loss'],     label='Train Loss',  color='#2563eb')
    axes[0].plot(history.history['val_loss'], label='Val Loss',    color='#dc2626')
    axes[0].axvline(x=best_ep, color='gray', linestyle='--', linewidth=1,
                    label=f'Best epoch ({best_ep + 1})')
    axes[0].set_title('Loss over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'],     label='Train Accuracy', color='#2563eb')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#dc2626')
    axes[1].set_title('Accuracy over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(MODELS_DIR, 'training_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"   Curves saved → {curve_path}")

    # 10. Test evaluation
    print("10. Evaluating on test set...")
    results   = model.evaluate(
        x={
            'lstm_input': X_seq_test, 
            'static_input': X_static_test,
            'home_team': home_test,
            'away_team': away_test
        },
        y=y_test,
        verbose=0
    )
    test_loss = results[0]
    test_acc  = results[1]
    test_top2 = results[2]

    best_val_acc = float(np.max(history.history['val_accuracy']))

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Best epoch (lowest val loss) : {best_ep + 1}")
    print(f"  Best validation accuracy     : {best_val_acc * 100:.2f}%")
    print(f"  Final test accuracy          : {test_acc * 100:.2f}%")
    print(f"  Final test Top-2 accuracy    : {test_top2 * 100:.2f}%")
    print(f"  Final test loss              : {test_loss:.4f}")
    print(f"  Models saved in              : {MODELS_DIR}/")
    print("=" * 50)
    print("\nNext step → python src/evaluate.py")


if __name__ == "__main__":
    train_model()