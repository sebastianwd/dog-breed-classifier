This a dog breed image classifier using a custom model from a custom dataset of images.

# Notes

- Convert a Tensorflow Keras model to a graph model so it can be used by tf.js

```bash
tensorflowjs_converter --input_format keras --output_format tfjs_graph_model my_model.h5  model
```
