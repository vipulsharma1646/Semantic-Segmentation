# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from PIL import Image

# Load the image using PIL (Pillow) and convert to a NumPy array
image = Image.open('/content/input image for ss.png').convert('RGB')  # Ensure 3 channels (RGB)
image_array = np.array(image)
print(image_array.shape)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add a batch dimension (expand dims)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, channels)

# Create a list to hold the processed tensors
i_t = []

# Append the original image tensor (if needed)
i_t.append(image_tensor)

# Run the loop 6 times and process each tensor
for i in range(7):
    # Apply Conv2D layer
    output_tensor = tf.keras.layers.Conv2D(filters=2**(4+i), kernel_size=(3, 3), padding='same', activation='relu')(image_tensor)

    # Apply MaxPooling
    out_tensor = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(output_tensor)

    # Remove the batch dimension if needed (optional, use squeeze)
    final_tensor = tf.squeeze(out_tensor)  # Removes batch size of 1, making it 3D (height, width, channels)

    # Append the final processed tensor
    i_t.append(final_tensor)

    # Print the shape of each output tensor
    print(final_tensor.shape)

print(final_tensor)

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Create a simple CNN using Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

])

# Apply the model to the image tensor
output_tensor = model(image_tensor)

# If you want to remove the batch dimension and return to 3D, use:
final_tensor = tf.squeeze(output_tensor)  # Removes dimensions of size 1

# Print the shape of the output tensor
print(final_tensor.shape)


# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Create a simple CNN using Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    # UpSample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
])

# Apply the model to the image tensor
output_tensor = model(image_tensor)

# If you want to remove the batch dimension and return to 3D, use:
final_tensor = tf.squeeze(output_tensor)  # Removes dimensions of size 1

# Print the shape of the output tensor
print(final_tensor.shape)

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Define the first part of the CNN
model_downsample = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
])

# Apply the downsampling model to the image tensor
output_tensor = model_downsample(image_tensor)

# Remove the batch dimension to return to 3D (optional)
downsampled_tensor = tf.squeeze(output_tensor)

# Print the shape of the output tensor after downsampling
print("Downsampled tensor shape:", downsampled_tensor.shape)

# Second part: Upsample after applying Conv2D layers
model_upsample = tf.keras.Sequential([


    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same'),
    tf.keras.layers.Softmax(axis=-1)
])

# Apply the upsampling model to the downsampled tensor
upsampled_tensor = model_upsample(output_tensor)

# Remove the batch dimension to return to 3D (optional)
final_tensor = tf.squeeze(upsampled_tensor)

# Print the shape of the output tensor after upsampling
print("Upsampled tensor shape:", final_tensor.shape)

import matplotlib.pyplot as plt

# Example random tensor output (assuming 2D single-channel image)
output_image = tf.random.uniform([64, 64], minval=0, maxval=255, dtype=tf.int32)

# Convert the tensor to numpy array for visualization
output_image_np = output_image.numpy()

# Plot the image using matplotlib
plt.imshow(output_image_np, cmap='gray')  # Use 'gray' for grayscale image
plt.title('Output Image')
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Define the first part of the CNN
model_downsample = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
])

# Apply the downsampling model to the image tensor
output_tensor = model_downsample(image_tensor)

# Remove the batch dimension to return to 3D (optional)
downsampled_tensor = tf.squeeze(output_tensor)

# Print the shape of the output tensor after downsampling
print("Downsampled tensor shape:", downsampled_tensor.shape)

# Second part: Upsample after applying Conv2D layers
model_upsample = tf.keras.Sequential([


    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),

    # Upsample the tensor
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same'),
    tf.keras.layers.Softmax(axis=-1)
])

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Apply the upsampling model to the downsampled tensor
upsampled_tensor = model_upsample(output_tensor)

# Remove the batch dimension to return to 3D (height, width, 3)
final_tensor = tf.squeeze(upsampled_tensor)

# Clip the values to [0, 255] and convert to uint8 for proper visualization
final_tensor = tf.clip_by_value(final_tensor, 0, 255)
final_image = tf.cast(final_tensor, tf.uint8)

# Convert the final tensor to NumPy array for visualization
final_image_np = final_image.numpy()

# Visualize the final image using matplotlib
plt.imshow(final_image_np)  # No need for 'cmap' since it's an RGB image
plt.title('Upsampled RGB Image')
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Define the downsampling part of the model
model_downsample = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
])

# Apply the downsampling model to the image tensor
output_tensor = model_downsample(image_tensor)

# Second part: Upsample after applying Conv2D layers
model_upsample = tf.keras.Sequential([
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same'),  # Final output with 3 channels (RGB)
    tf.keras.layers.Softmax(axis=-1)  # Apply softmax for classification over the channels
])

# Apply the upsampling model to the downsampled tensor
upsampled_tensor = model_upsample(output_tensor)

# Remove the batch dimension to return to 3D (height, width, 3)
final_tensor = tf.squeeze(upsampled_tensor)

# Clip the values to [0, 255] and convert to uint8 for proper visualization
final_tensor = tf.clip_by_value(final_tensor * 255, 0, 255)
final_image = tf.cast(final_tensor, tf.uint8)

# Convert the final tensor to NumPy array for visualization
final_image_np = final_image.numpy()

# Visualize the final RGB image using matplotlib
plt.imshow(final_image_np)
plt.title('Upsampled RGB Image')
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Define the downsampling part of the model
model_downsample = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
])

# Apply the downsampling model to the image tensor
output_tensor = model_downsample(image_tensor)

# Second part: Upsample after applying Conv2D layers
model_upsample = tf.keras.Sequential([
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 1), padding='same'),  # Output 10 channels for segmentation classes
    tf.keras.layers.Softmax(axis=-1)  # Apply softmax for classification over the channels
])

# Apply the upsampling model to the downsampled tensor
upsampled_tensor = model_upsample(output_tensor)

# Remove the batch dimension to return to 3D (height, width, classes)
final_tensor = tf.squeeze(upsampled_tensor)

# Get the predicted class for each pixel (argmax across the class dimension)
segmentation_output = tf.argmax(final_tensor, axis=-1)

# Convert to NumPy array for visualization
segmentation_output_np = segmentation_output.numpy()

# Apply a color map (you can customize this for your specific classes)
color_map = plt.get_cmap('jet')

# Visualize the segmentation result using a color map
plt.imshow(color_map(segmentation_output_np / 10.0))  # Normalize class indices for color map
plt.title('Semantic Segmentation Result')
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image using PIL (Pillow) and convert to RGB to ensure 3 channels
image = Image.open('/content/input image for ss.png').convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image)

# Convert the NumPy array to a tensor
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Add batch dimension to make it 4D (1, height, width, channels)
image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)

# Define the downsampling part of the model
model_downsample = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_tensor.shape[1], image_tensor.shape[2], 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(5, 5), padding='same', activation='relu'),
])

# Apply the downsampling model to the image tensor
output_tensor = model_downsample(image_tensor)

# Second part: Upsample after applying Conv2D layers
model_upsample = tf.keras.Sequential([
    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(size=(2, 2)),  # Upsample by a factor of 2
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 1), padding='same'),  # Output 10 channels for segmentation classes
    tf.keras.layers.Softmax(axis=-1)  # Apply softmax for classification over the channels
])

# Apply the upsampling model to the downsampled tensor
upsampled_tensor = model_upsample(output_tensor)

# Remove the batch dimension to return to 3D (height, width, classes)
final_tensor = tf.squeeze(upsampled_tensor)

# Get the predicted class for each pixel (argmax across the class dimension)
segmentation_output = tf.argmax(final_tensor, axis=-1)

# Convert to NumPy array for visualization
segmentation_output_np = segmentation_output.numpy()

# Apply a color map (you can customize this for your specific classes)
color_map = plt.get_cmap('jet')

# Visualize the segmentation result using a color map
plt.imshow(color_map(segmentation_output_np / 10.0))  # Normalize class indices for color map
plt.title('Semantic Segmentation Result')
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

