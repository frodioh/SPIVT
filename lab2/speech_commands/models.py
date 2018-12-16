from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

#Вычисление минимальной степени двойки, большей или равной переданному числу
def _next_power_of_two(x):
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()

#Возвращает словарь, содержащий настройки для модели
#label_count: Как много классов будет распознаваться
#sample_rate: Количество сэмплов на секунду
#clip_duration_ms: Длина отрезка аудио в миллисекундах
#window_size_ms: Размер окна в миллисекундах
#window_stride_ms: Сдвиг между окнами в миллисекундах
#feature_bin_count: Количество частотных блоков, используемых для анализа
def prepare_model_settings(label_count, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count):
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

  average_window_width = -1
  fingerprint_width = feature_bin_count
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': 'mfcc',
      'average_window_width': average_window_width,
  }

#fingerprint_input: Узел TensorFlow, который будет выводить векторы признаков.
#model_settings: Словарь с информацией о модели.
#is_training: Будет ли модель использоваться для обучения.
#Вектор признаков получается из спектрограммы
#с использованием Мел-частотных кепстральных коэффициентов

#Возвращается узел тензорфлоу, выдающий результаты.
def create_model(fingerprint_input, model_settings, is_training, runtime_settings=None):
  return create_conv_model(fingerprint_input, model_settings, is_training)

#Функция для восстановления из сохранённого чекпоинта
def load_variables_from_checkpoint(sess, start_checkpoint):
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

#Строит стандартную свёрточную сеть
#структура приведена в следующей работе
#http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
def create_conv_model(fingerprint_input, model_settings, is_training):
  """
  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable(
      name='second_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.get_variable(
      name='second_bias',
      initializer=tf.zeros_initializer,
      shape=[second_filter_count])
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
