import os
# Force TensorFlow to use CPU only
os.environ['TENSORFLOW_TEST_ONLY'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Try importing TensorFlow and running a simple operation
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("Creating a simple tensor...")
    tensor = tf.constant([[1, 2], [3, 4]])
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor: {tensor.numpy()}")
    print("TensorFlow is working correctly in CPU-only mode.")
except Exception as e:
    print(f"Error importing or using TensorFlow: {e}") 