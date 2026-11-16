import os, shutil
P = "src/classifiers/lite.py"
s = open(P).read()
if "LITE_EARLY_STOPPING" in s:
    raise SystemExit("already patched")
shutil.copy(P, P + ".bak")

def sub(old, new):
    global s
    assert s.count(old) == 1, "anchor not found exactly once:\n" + old[:90]
    s = s.replace(old, new)

sub("from codecarbon import track_emissions\n",
    "from codecarbon import track_emissions\nimport zlib\nfrom sklearn.model_selection import train_test_split\n")

sub("        self.n_epochs = n_epochs\n\n        self.build_model()\n",
'''        self.n_epochs = n_epochs

        # early-stopping experiment; defaults reproduce the original behaviour
        self.early_stopping = os.environ.get("LITE_EARLY_STOPPING", "0") == "1"
        self.patience = int(os.environ.get("LITE_PATIENCE", "100"))
        self.val_split = float(os.environ.get("LITE_VAL_SPLIT", "0.2"))

        self.build_model()
''')

sub("        self.callbacks = [reduce_lr, model_checkpoint]\n",
'''        self.callbacks = [reduce_lr, model_checkpoint]
        if self.early_stopping:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=50, min_lr=1e-4
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_directory + "best_model.keras",
                monitor="val_loss",
                save_best_only=True,
            )
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
            self.callbacks = [reduce_lr, model_checkpoint, early_stop]
''')

sub('''    def fit_and_track_emissions(
        self, xtrain, ytrain, xval=None, yval=None, plot_test=False
    ):
        @track_emissions(project_name="LITE", output_dir=self.output_directory)
        def _fit(xtrain, ytrain, xval, yval, plot_test):

            ytrain = np.expand_dims(ytrain, axis=1)
            ohe = OHE(sparse_output=False)
            ytrain = ohe.fit_transform(ytrain)

            if plot_test:

                yval = np.expand_dims(yval, axis=1)
                ohe = OHE(sparse_output=False)
                yval = ohe.fit_transform(yval)
''',
'''    def _split_validation(self, xtrain, ytrain):
        seed = zlib.crc32("/".join(os.path.normpath(self.output_directory).split(os.sep)[-2:]).encode()) % (2**31 - 1)
        try:
            return train_test_split(
                xtrain, ytrain, test_size=self.val_split, stratify=ytrain, random_state=seed
            )
        except ValueError:  # a class too small to stratify
            return train_test_split(
                xtrain, ytrain, test_size=self.val_split, random_state=seed
            )

    def fit_and_track_emissions(
        self, xtrain, ytrain, xval=None, yval=None, plot_test=False
    ):
        self._all_classes = np.unique(ytrain)
        if self.early_stopping:
            # validation comes from the TRAIN set; the test set is not used for any decision
            xtrain, xval, ytrain, yval = self._split_validation(xtrain, ytrain)
            plot_test = True

        @track_emissions(project_name="LITE", output_dir=self.output_directory)
        def _fit(xtrain, ytrain, xval, yval, plot_test):

            categories = [self._all_classes] if self.early_stopping else "auto"

            ytrain = np.expand_dims(ytrain, axis=1)
            ohe = OHE(sparse_output=False, categories=categories)
            ytrain = ohe.fit_transform(ytrain)

            if plot_test:

                yval = np.expand_dims(yval, axis=1)
                ohe = OHE(sparse_output=False, categories=categories)
                yval = ohe.fit_transform(yval)
''')

sub("            self.train_duration = time.time() - start_time\n",
'''            self.train_duration = time.time() - start_time
            key = "val_loss" if self.early_stopping else "loss"
            self.epochs_run = len(hist.history["loss"])
            self.best_epoch = int(np.argmin(hist.history[key])) + 1
''')

sub('            "duration": self.train_duration,\n',
'''            "duration": self.train_duration,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "early_stopping": bool(self.early_stopping),
''')

open(P, "w").write(s)
print("patched; backup in", P + ".bak")
