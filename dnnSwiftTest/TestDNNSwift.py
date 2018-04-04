import dnnSwift
import numpy as np
import os
import pickle
import shutil


def test_from_scratch():
    test_dir = os.path.join(base_dir, "test_from_scratch")
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    test_hdf5_fn = os.path.join(test_dir, hdf5_fn)
    test_index_fn = os.path.join(test_dir, index_fn)

    # Copy data file
    shutil.copy2(src=os.path.join(base_dir, hdf5_fn), dst=test_hdf5_fn)

    my_dnn = dnnSwift.DNNWrapper(
        categories=categories, layout=dnn_layout)
    my_dnn.initialize_training_data(
        filename=test_hdf5_fn, outfile=test_index_fn)
    my_dnn.train_dnn(
        num_epochs=1, batch_size=128, weights_dir="weights",
        verbose=False)


def test_continue_training():
    test_dir = os.path.join(base_dir, "test_continue_training")
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    test_hdf5_fn = os.path.join(test_dir, hdf5_fn)
    test_index_fn = os.path.join(test_dir, index_fn)

    # Copy data file
    shutil.copy2(src=os.path.join(base_dir, hdf5_fn), dst=test_hdf5_fn)

    my_dnn = dnnSwift.DNNWrapper(
        categories=categories, layout=dnn_layout)
    my_dnn.initialize_training_data(
        filename=test_hdf5_fn, outfile=test_index_fn)
    my_dnn.train_dnn(
        num_epochs=1, batch_size=128, weights_dir="weights",
        verbose=False)

    # Continue without reset
    my_dnn.train_dnn(
        num_epochs=2, batch_size=128, weights_dir="weights",
        start_epoch=1, verbose=False)

    # Continue with reset
    del my_dnn
    my_dnn = dnnSwift.DNNWrapper(
        categories=categories, layout=dnn_layout)
    my_dnn.initialize_training_data(
        filename=test_hdf5_fn, outfile=test_index_fn)
    my_dnn.train_dnn(
        num_epochs=3, batch_size=128, weights_dir="weights",
        start_epoch=2, verbose=False)


def test_apply_dnn():
    print(
        "Training verbosity turned off for this test "
        "for easier to parse output. Be patient.")
    test_dir = os.path.join(base_dir, "test_apply_dnn")
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    test_hdf5_fn = os.path.join(test_dir, hdf5_fn)
    test_index_fn = os.path.join(test_dir, index_fn)

    # Copy data file
    shutil.copy2(src=os.path.join(base_dir, hdf5_fn), dst=test_hdf5_fn)

    # Train on dataset
    my_dnn = dnnSwift.DNNWrapper(
        categories=categories, layout=dnn_layout)
    my_dnn.initialize_training_data(
        filename=test_hdf5_fn, outfile=test_index_fn)
    my_dnn.train_dnn(
        num_epochs=2, batch_size=128, weights_dir="weights",
        verbose=False)

    # Load images
    img_dat = my_dnn.get_images(list_name="test")
    images = img_dat["images"]
    labels = img_dat["labels"]

    # Apply without reinitializing the DNN by keeping parameters identical
    output = my_dnn.apply_dnn(images=images)
    output = np.squeeze(a=output)
    output_cat = np.argmax(a=output, axis=1)
    labels_cat = np.argmax(a=labels, axis=1)
    acc_no_reinit = np.mean(np.equal(output_cat, labels_cat))
    print("Accuracy: %s" % str(acc_no_reinit))

    # Apply with reinitializing the DNN by loading a new set of weights
    weights_file = os.path.join(test_dir, "weights", "weights_0.pkl")
    with open(weights_file, "r") as f:
        weights = pickle.load(f)

    output = my_dnn.apply_dnn(images=images, weights=weights)
    output = np.squeeze(a=output)
    output_cat = np.argmax(a=output, axis=1)
    labels_cat = np.argmax(a=labels, axis=1)
    acc_with_reinit = np.mean(np.equal(output_cat, labels_cat))
    print("Accuracy: %s" % str(acc_with_reinit))


if __name__ == "__main__":
    base_dir = "."
    categories = {0: 0, 5: 1, 7: 2}
    hdf5_fn = "MNISTDemo.h5"
    index_fn = "MNISTDemo_indexSplit.pkl"

    # Load layout
    with open(os.path.join(base_dir, "dnn_layout"), "r") as layout_file:
        dnn_layout = np.safe_eval("[" + layout_file.read() + "]")

    print("Training verbosity has been turned off for the tests to make the "
          "output easier \nto parse. These tests will take several minutes "
          "to complete.")
    print("------------------------------------------------------------------"
          "--------------")
    print("TEST: test_from_scratch()")
    test_from_scratch()

    print("TEST: test_continue_training()")
    test_continue_training()

    print("TEST: test_apply_dnn()")
    test_apply_dnn()
