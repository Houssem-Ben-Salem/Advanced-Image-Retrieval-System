import os
import tensorflow as tf
import warnings
from transformers import BertTokenizer, BertModel

# Suppress all TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Alternatively, you can also control TensorFlow's logger directly:
tf.get_logger().setLevel('ERROR')

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Your original code
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)